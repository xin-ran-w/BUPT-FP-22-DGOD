from collections import OrderedDict
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from vis_utils.vis_ca import vis_channel_mask


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class DomainClassifier(nn.Module):

    def __init__(self, in_channel, domain_num, dense=False):
        super(DomainClassifier, self).__init__()

        if not dense:
            classifier = [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_channel, int(in_channel / 2)), nn.ReLU(), nn.BatchNorm1d(int(in_channel / 2)),
                nn.Linear(int(in_channel / 2), domain_num)
            ]
        else:
            classifier = [
                nn.Conv2d(in_channel, int(in_channel / 2), kernel_size=3, padding=1),  # C, H, W
                nn.ReLU(), nn.BatchNorm2d(128),
                nn.Conv2d(int(in_channel / 2), domain_num, kernel_size=1)
            ]

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        return self.classifier(x)


class CGM(nn.Module):

    def __init__(self, domain_num, gl_w=0.1, cls_w=0.01, dis_w=0.1, gl_m=0.01, dense=True, s_init=True):
        super(CGM, self).__init__()

        self.in_channel_list = [256] * 5
        self.gl_weight = gl_w
        self.domain_num = domain_num
        self.domain_classifiers = nn.ModuleList()
        self.domain_discriminators = nn.ModuleList()
        self.cls_weight = cls_w
        self.dis_weight = dis_w
        self.dense = dense
        self.gl_margin = torch.tensor(gl_m).cuda()
        self.index = 0
        self.s_init = s_init


        self.channel_attention = [
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.InstanceNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.InstanceNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.InstanceNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)
        ]

        self.channel_attention = nn.Sequential(*self.channel_attention)

        if self.s_init:
            for i, m in enumerate(self.channel_attention.modules()):
                if i == 10:
                    m.bias.data.fill_(9.2)

        for in_channel in self.in_channel_list:
            self.domain_classifiers.append(self.get_classifier(in_channel=in_channel))
            self.domain_discriminators.append(self.get_classifier(in_channel=in_channel))

    def forward(self, features, domain_labels=None):
        di_features = OrderedDict()
        ds_features = OrderedDict()
        cls_losses = []
        dis_losses = []
        losses = {}
        di_list, ds_list, gl_loss = self.get_attention(features)

        # # todo refine the code
        # vis_channel_mask(di_list, ds_list, self.index)
        # self.index += 1

        for i, k in enumerate(features.keys()):
            di_features[k] = di_list[i][:, :, None, None] * features[k]
            ds_features[k] = ds_list[i][:, :, None, None] * features[k]

            if self.training:
                cls_losses.append(
                    self.classification_loss(self.domain_classification(di_features[k], i), domain_labels))
                dis_losses.append(
                    self.discrimination_loss(self.domain_discrimination(di_features[k], i), domain_labels))

        if self.training:
            losses = {
                "gate_loss": gl_loss,
                "cgm_cls_loss": sum(cls_losses),
                "cgm_dis_loss": sum(dis_losses)
            }
        return di_features, losses

    def gate_loss(self, di, ds):

        di = di.unsqueeze(1)                        # shape [B, 1, C]
        ds_T = ds.unsqueeze(1).permute(0, 2, 1)     # shape [B, C, 1]

        l = di @ ds_T  # shape [B, 1, 1]
        l = l.flatten()  # shape [B]
        l = l.sum()  # shape [1]

        if l < self.gl_margin:
            l = self.gl_margin

        return self.gl_weight * l

    def get_attention(self, features):
        features = list(features.values())

        di_list = []
        ds_list = []
        gl_loss = []

        for i, feature in enumerate(features):
            di = self.channel_attention(feature)
            di = di.flatten(start_dim=1)
            ds = (1 - di)
            di_list.append(di)
            ds_list.append(ds)
            if self.training:
                gl_loss.append(self.gate_loss(di, ds))

        return di_list, ds_list, sum(gl_loss)

    def domain_classification(self, ds_feature, idx):
        out = self.domain_classifiers[idx](ds_feature)
        return out

    def domain_discrimination(self, di_feature, idx):

        di_feature = grad_reverse(di_feature)  # reverse the gradient
        out = self.domain_discriminators[idx](di_feature)
        return out

    def get_classifier(self, in_channel):
        return DomainClassifier(in_channel=in_channel,
                                domain_num=self.domain_num,
                                dense=self.dense)

    def classification_loss(self, pt, gt):

        if self.dense:
            pt, gt = self.process_tensor(pt, gt)
        return self.cls_weight * F.cross_entropy(pt, gt)

    def discrimination_loss(self, pt, gt):
        if self.dense:
            pt, gt = self.process_tensor(pt, gt)
        return self.dis_weight * F.cross_entropy(pt, gt)

    def process_tensor(self, pt, gt):
        b, h, w = pt.shape[0], pt.shape[2], pt.shape[3]
        pt = pt.transpose(1, 2).transpose(2, 3).reshape(-1, self.domain_num)
        gt = gt.unsqueeze(1)
        gt = gt.expand(b, h * w).flatten()
        return pt, gt


def norm_kl_divergence(m1, v1, m2, v2):
    return (v2 / v1).log() + (v1 ** 2 + (m1 - m2) ** 2) / (2 * v2 ** 2) - 0.5


class NormStats(nn.Module):

    def __init__(self, scale_num, domain_num, channel_num, dbs, top=0.7, momentum=0.9):
        super(NormStats, self).__init__()
        self.channel_num = channel_num                                      # channel num
        self.scale_num = scale_num                                          # scale num
        self.domain_num = domain_num                                        # domain num
        self.dbs = dbs                                                      # domain batch size
        self.momentum = momentum

        self.only_stats = True
        self.top_k = int(self.channel_num * top)                            # top k similarity
        self.register_buffer('moving_mean_mat', self._init_mean())
        self.register_buffer('moving_var_mat', self._init_var())
        self.sm_index = self._init_sm_index()

        self.drop_prob = 0.5


    def is_only_stats(self, x: bool):
        self.only_stats = x


    def _init_sm_index(self):

        domain_index = [i for i in range(self.domain_num)]
        dx, dy = np.meshgrid(domain_index, domain_index)
        dx, dy = dx.reshape(-1), dy.reshape(-1)
        sm_index = [(x, y) for x, y in zip(dx, dy) if x != y]
        return sm_index

    def _init_similarity_matrix(self):
        return

    def _init_mean(self):
        return torch.zeros(self.scale_num, self.domain_num, self.channel_num)

    def _init_var(self):
        return torch.ones(self.scale_num, self.domain_num, self.channel_num)

    def forward(self, features):


        channel_masks = None

        if not self.only_stats:
            sm = self.cal_channel_similarity()
            channel_masks = self.get_mask(sm)
            di_features = OrderedDict()
        else:
            di_features = features

        for s, k in enumerate(features.keys()):
            feature = features[k]
            f_m = feature.mean(dim=(2, 3))
            f_v = feature.var(dim=(2, 3))

            for d in range(self.domain_num):
                start = d * self.dbs
                end = (d + 1) * self.dbs


                if not self.only_stats:
                    di_mask = channel_masks[s, d, :]
                    di_mask = di_mask[None, :, None, None]
                    di_mask = di_mask.expand(self.dbs, self.channel_num, 1, 1)
                    di_features[k] = di_mask * feature[start: end, :, :, :]

                g_m = self.moving_mean_mat[s, d, :]
                g_v = self.moving_var_mat[s, d, :]

                g_m = self.momentum * g_m + (1 - self.momentum) * f_m[start:end, :].mean(dim=0)
                g_v = self.momentum * g_v + (1 - self.momentum) * f_v[start:end, :].mean(dim=0)

                self.moving_mean_mat[s, d, :] = g_m
                self.moving_var_mat[s, d, :] = g_v

        return di_features

    def cal_channel_similarity(self):
        sm = torch.zeros((self.domain_num, self.domain_num, self.channel_num))
        for s in range(self.scale_num):
            for (d1, d2) in self.sm_index:
                sm[s, d1, d2, :] = norm_kl_divergence(m1=self.moving_mean_mat[s, d1, :],
                                                      v1=self.moving_var_mat[s, d1, :],
                                                      m2=self.moving_mean_mat[s, d2, :],
                                                      v2=self.moving_var_mat[s, d2, :])
        return sm


    def get_mask(self, sm):
        channel_mask = torch.zeros((self.scale_num, self.domain_num, self.channel_num))
        sm = sm.mean(dim=2)
        _, top_ids = sm.topk(k=self.top_k, dim=-1, largest=False)
        channel_mask[top_ids] = 1
        channel_mask = self.randomly_drop(channel_mask=channel_mask)
        return channel_mask

    def randomly_drop(self, channel_mask):
        sp = channel_mask.size()
        channel_mask = channel_mask.reshape(-1)
        for i in range(channel_mask.shape[0]):
            if channel_mask[i] == 0:
                if random.random() > self.drop_prob:
                    channel_mask[i] = 1
        channel_mask = channel_mask.reshape(*sp)
        return channel_mask

    def print_stats(self):
        for mm, vm in zip(self.moving_mean_mat, self.moving_var_mat):
            for d in range(self.domain_num):
                print(f"Domain {d}: \nchannel_mean{mm[d, :]} \nchannel_var {vm[d, :]}\n")
