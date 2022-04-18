import torch

from backbone import resnet50, LastLevelP6P7
from nets.fasterrcnn import FasterRCNN, FastRCNNPredictor
from nets.retinanet import RetinaNet
from nets.ssd.ssd import SSDBackbone, SSD300


def create_model(hparams):
    if hparams.net == 'retinanet':
        backbone = resnet50(norm_layer=torch.nn.BatchNorm2d,
                            returned_layers=hparams.return_layers,
                            extra_blocks=LastLevelP6P7(256, 256),
                            trainable_layers=3,
                            hparams=hparams)
        model = RetinaNet(backbone, hparams.num_classes)

        # load pretraining weight
        # https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
        weights_dict = torch.load(hparams.pretraining_weight, map_location='cpu')
        # 删除分类器部分的权重，因为自己的数据集类别与预训练数据集类别(91)不一定致，如果载入会出现冲突
        del_keys = ["head.classification_head.cls_logits.weight", "head.classification_head.cls_logits.bias"]
        for k in del_keys:
            del weights_dict[k]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    elif hparams.net == 'fasterrcnn':
        backbone = resnet50(norm_layer=torch.nn.BatchNorm2d,
                            trainable_layers=3,
                            returned_layers=hparams.return_layers,
                            hparams=hparams)

        # load pretraining weight which class num is 91
        model = FasterRCNN(backbone=backbone, num_classes=91, hparams=hparams)

        weights_dict = torch.load(hparams.pretraining_weight, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, hparams.num_classes + 1)

    elif hparams.net == 'ssd':
        backbone = SSDBackbone(pretrain_path=None)
        model = SSD300(backbone=backbone, num_classes=hparams.num_classes)
        pre_model_dict = torch.load(hparams.pretraining_weight, map_location='cpu')
        pre_weights_dict = pre_model_dict["model"]

        # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
        del_conf_loc_dict = {}
        for k, v in pre_weights_dict.items():
            split_key = k.split(".")
            if "conf" in split_key:
                continue
            del_conf_loc_dict.update({k: v})

        missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        return model

    else:
        model = None
        # model = Darknet(cfg)

    return model
