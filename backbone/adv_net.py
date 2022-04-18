import torch
import torch.nn.functional as F
import torch.nn as nn

from .layer_getter import IntermediateLayerGetter



def adv_loss(p, gt):
    """
    domain classification loss
    :param p: the predicted tensor
    :param gt: the ground truth label
    :return:
    """
    # gt = [t['domain'] for t in targets]
    # gt = torch.cat(gt)
    return F.cross_entropy(p, gt)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class AdvNet(nn.Module):
    def __init__(self, backbone, return_layers, domain_num, in_channels=1024, out_channels=256, adv_layer=0):
        super(AdvNet, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        classifier = [nn.Linear(in_channels, int(in_channels / 2), bias=True),
                      nn.ReLU(), nn.BatchNorm1d(int(in_channels / 2)),
                      nn.Linear(int(in_channels / 2), domain_num, bias=True)]
        self.domain_classifier = nn.Sequential(*classifier)

        self.conv = nn.Conv2d(2048, out_channels, 3, padding=1)

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        feat = self.conv(x['1'])
        x = grad_reverse(x['0'])
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(start_dim=1)
        pd = self.domain_classifier(x)

        return feat, pd


