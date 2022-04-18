import os

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.jit.annotations import List, Dict
from torchvision.ops.misc import FrozenBatchNorm2d

from domain_utils.disentangle import CGM, DomainClassifier, NormStats
from .adv_net import grad_reverse, adv_loss
from .layer_getter import IntermediateLayerGetter
from .resnet50 import ResNet, Bottleneck
from .feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork

def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps



class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodule that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks: ExtraFPNBlock
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, hparams, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()



        self.algorithm = hparams.algorithm

        self.dbs = hparams.batch_size
        self.out_channels = out_channels

        if self.algorithm == 'AlignSource':
            self.adv_layer = hparams.adv_layer
            self.domain_num = hparams.domain_num
            self.domain_classifier = DomainClassifier(in_channel=1024, domain_num=self.domain_num)
            self.adv_w = hparams.adv_w

        if self.algorithm == 'CGMDRL':
            self.domain_num = hparams.domain_num

            # todo correct
            self.cgm = CGM(gl_w=hparams.gl_w,
                           dis_w=hparams.dis_w,
                           cls_w=hparams.cls_w,
                           gl_m=hparams.gl_m,
                           domain_num=self.domain_num,
                           s_init=hparams.s_init)

        if self.algorithm == 'CSDRL':
            self.domain_num = hparams.domain_num
            self.ns = NormStats(scale_num=5,
                                domain_num=self.domain_num,
                                channel_num=self.out_channels,
                                dbs=self.dbs)


        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )


    def forward(self, x, targets):
        """
        :param targets:         targets
        :param x:(OrderDict)    0: torch.Size([16, 256, 112, 200])
                                1: torch.Size([16, 512, 56, 100])
                                2: torch.Size([16, 1024, 28, 50])
                                3: torch.Size([16, 2048, 14, 25])
        :return:
        """


        domain_labels = None

        if targets is not None and self.algorithm != 'Stitch':
            domain_labels = [t['domain'] for t in targets]                      # get domain labels
            domain_labels = torch.cat(domain_labels)

        backbone_loss = {}

        inter_feats = self.body(x)
        out_feats = self.fpn(inter_feats)
        if self.algorithm == 'CGMDRL':
            out_feats, backbone_loss = self.cgm(out_feats, domain_labels)

        elif self.algorithm == 'AlignSource' and self.training:
            adv_feats = grad_reverse(inter_feats[str(self.adv_layer)])

            p = self.domain_classifier(adv_feats)
            backbone_loss = {"adversarial_loss": self.adv_w * adv_loss(p, domain_labels)}

        # todo complete
        elif self.algorithm == 'CSDRL':
            out_feats = self.ns(out_feats)

        return out_feats, backbone_loss



def resnet50(
        returned_layers,
        hparams,
        pretrain_path="",
        norm_layer=FrozenBatchNorm2d,  # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
        trainable_layers=3,
        extra_blocks=None,
):
    """
    搭建resnet50_fpn——backbone
    Args:
        pretrain_path: resnet50的预训练权重，如果不使用就默认为空
        norm_layer: 官方默认的是FrozenBatchNorm2d，即不会更新参数的bn层(因为如果batch_size设置的很小会导致效果更差，还不如不用bn层)
                    如果自己的GPU显存很大可以设置很大的batch_size，那么自己可以传入正常的BatchNorm2d层
                    (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers: the layers to train
        returned_layers: return the intermediate layers output
        extra_blocks: add extra layer to the output
        hparams: hyper-parameters


    Returns: backbone model

    """

    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    out_channels = 256

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],
                             include_top=False,
                             norm_layer=norm_layer)

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
        # 载入预训练权重
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        # 只训练不在layers_to_train列表中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    # if use_adv:
    #     return AdvNet(resnet_backbone, return_layers=return_layers, domain_num=4)

    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # in_channel 为layer4的输出特征矩阵channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # 记录resnet50提供给fpn的特征层channels
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # 通过fpn后得到的每个特征层的channel


    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels,
                           extra_blocks=extra_blocks, hparams=hparams)




