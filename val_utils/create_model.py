
import torch

from backbone import resnet50, LastLevelP6P7
from nets.fasterrcnn import FasterRCNN
from nets.retinanet import RetinaNet


def create_model(hparams):

    if hparams.net == 'retinanet':


        backbone = resnet50(norm_layer=torch.nn.BatchNorm2d,
                            returned_layers=[2, 3, 4],
                            extra_blocks=LastLevelP6P7(256, 256),
                            hparams=hparams)
        model = RetinaNet(backbone, hparams.num_classes)

    else:
        # backbone = resnet50(norm_layer=torch.nn.BatchNorm2d, returned_layers=[1, 2, 3, 4], use_fpn=True)
        backbone = resnet50(norm_layer=torch.nn.BatchNorm2d,
                            returned_layers=hparams.return_layers,
                            hparams=hparams)
        model = FasterRCNN(backbone=backbone, num_classes=hparams.num_classes + 1, hparams=hparams)

    return model
