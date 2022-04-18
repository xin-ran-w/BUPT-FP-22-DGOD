import itertools
import math
import random
import sys

import numpy as np
import torch
import copy

import train_utils.distributed_utils as distributed_utils
from aug_utils import aug_utils
from train_utils import train_eval_utils as utils
from .create_model import create_model
from torchvision.transforms import ToPILImage
from vis_utils.draw_box_utils import vis_stitch_img

from aug_utils.aug_utils import CropImages, StitchImages

ALGORITHMS = [
    'Baseline',
    'Stitch',
    'AlignSource',
    'CGMDRL',
    'CSDRL',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(object):

    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.domainbus = domainbus
        self.device = device
        self.print_freq = hparams.print_freq
        self.warmup = warmup
        self.scaler = scaler
        self.lr = None
        self.lr_scheduler = None
        self.mean_loss = None
        self.header = 'Epoch: [{}]'
        self.metric_logger = distributed_utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', distributed_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.valid_train_set = hparams.valid_train_set
        self.cls = hparams.cls
        self.aug_prob = hparams.aug_prob

    @classmethod
    def get_model(cls, hparams):
        return create_model(hparams)


    @classmethod
    def get_transforms(cls, hparams):

        if hparams.net == 'ssd':
            data_transform = {
                "train": aug_utils.Compose([aug_utils.SSDCropping(),
                                            aug_utils.Resize(),
                                            aug_utils.ColorJitter(),
                                            aug_utils.ToTensor(),
                                            aug_utils.RandomHorizontalFlip(),
                                            aug_utils.Normalization(),
                                            aug_utils.AssignGTtoDefaultBox()]),
                "val": aug_utils.Compose([aug_utils.Resize(),
                                          aug_utils.ToTensor(),
                                          aug_utils.Normalization()])
            }

        else:
            data_transform = {
                "train": aug_utils.Compose([aug_utils.ToTensor(),
                                            aug_utils.RandomHorizontalFlip(0.5)]),
                "val": aug_utils.Compose([aug_utils.ToTensor()])
            }

        return data_transform

    @classmethod
    def get_hparams(cls, hparams):
        return hparams

    def train_one_epoch(self, epoch):
        raise NotImplementedError

    def update(self, iter_i, images, targets):
        raise NotImplementedError

    def reset_domainbus(self, epoch, distributed=False):
        self.domainbus.reset()
        if distributed:
            self.domainbus.set_epoch(epoch)

    def eval_one_epoch(self, source_domains, val_dataloaders, train_val_dataloaders):

        for sd, vdl in zip(source_domains, val_dataloaders):
            _ = utils.evaluate(self.model, vdl, device=self.device, domain_name=sd)

        if self.valid_train_set:

            for sd, vdl in zip(source_domains, train_val_dataloaders):
                _ = utils.evaluate(self.model, vdl, device=self.device, domain_name=sd)


class Baseline(Algorithm):

    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):
        super(Baseline, self).__init__(model, optimizer, domainbus, device,
                                       hparams, warmup=warmup, scaler=scaler)

    @classmethod
    def get_hparams(cls, hparams):
        if hparams.net == 'fasterrcnn':
            hparams.return_layers = [1, 2, 3, 4]
        elif hparams.net == 'retinanet':
            hparams.return_layers = [2, 3, 4]

        return hparams

    def train_one_epoch(self, epoch):
        self.model.train()

        if epoch == 0 and self.warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.domainbus) - 1)

            self.lr_scheduler = distributed_utils.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)
        else:
            self.lr_scheduler = None

        self.mean_loss = torch.zeros(1).to(self.device)  # mean losses
        for iter_i, [images, targets] in enumerate(self.metric_logger.log_every(
                self.domainbus, self.print_freq, self.header.format(epoch))
        ):
            images, targets = self.aug(images, targets)
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.update(iter_i, images, targets)

        return self.mean_loss, self.lr

    def update(self, iter_i, images, targets):

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        self.mean_loss = (self.mean_loss * iter_i + loss_value) / (iter_i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:  # 第一轮使用warmup训练方式
            self.lr_scheduler.step()

        self.lr = self.optimizer.param_groups[0]["lr"]
        self.metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        self.metric_logger.update(lr=self.lr)

    def aug(self, images, targets):
        return images, targets


class Stitch(Baseline):

    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):

        super().__init__(model, optimizer, domainbus, device, hparams=hparams, warmup=warmup, scaler=scaler)

        self.crop_imgs = CropImages(resize=True)
        self.stitch_imgs = StitchImages()
        self.d_batch_size = hparams.batch_size
        self.domain_num = len(hparams.sdi)
        self.d_idx = [i for i in range(self.domain_num)]
        if self.domain_num < 2:
            raise ValueError("To use stitch method the domain num must >= 2")


    def reset_idx(self):
        self.d_idx = [i for i in range(self.domain_num)]

    def aug(self, images, targets):
        if random.random() > self.aug_prob:
            images, targets = self.crop_then_stitch(images, targets)

        return images, targets

    def crop_then_stitch(self, images, targets):
        all_imgs = []
        all_trgs = []
        s_imgs = []
        s_trgs = []

        combination_list = []

        self.reset_idx()

        for di in self.d_idx:
            start = di * self.d_batch_size
            end = start + self.d_batch_size

            d_c_imgs, d_c_trgs = self.crop_imgs(images[start: end],
                                                targets[start: end])
            d_c_imgs, d_c_trgs = self.supplement(d_c_imgs, d_c_trgs)

            all_imgs.append(d_c_imgs)
            all_trgs.append(d_c_trgs)


        for i in range(self.d_batch_size * 2):
            random.shuffle(self.d_idx)
            combination_list.extend(self.d_idx)

        combination_list = [combination_list[i: i + 2] for i in range(0, len(combination_list), 2)]

        for i, comb in enumerate(combination_list):
            s_img, s_trg = self.stitch_imgs(all_imgs[comb[0]].pop(0),
                                            all_trgs[comb[0]].pop(0),
                                            all_imgs[comb[1]].pop(0),
                                            all_trgs[comb[1]].pop(0))
            s_imgs.append(s_img)
            s_trgs.append(s_trg)

            # visualization
            # img = ToPILImage()(s_img)
            # vis_stitch_img(img, copy.deepcopy(s_trg), class_dict_path=self.cls)
            # img.save(f"/data/wangxinran/img/stitch_img_{i}.jpg")

        return s_imgs, s_trgs

    def supplement(self, c_imgs, c_trgs):

        num = len(c_imgs)

        while num < self.d_batch_size * 2:
            si = random.randint(0, num - 1)
            c_imgs.append(c_imgs[si].clone())
            c_trgs.append(copy.deepcopy(c_trgs[si]))
            num += 1

        return c_imgs, c_trgs


class TwoLevelAug(Baseline):

    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):
        super(TwoLevelAug, self).__init__(model, optimizer, domainbus, device, hparams, warmup, scaler)


class TwoLevelErase(TwoLevelAug):
    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):
        super(TwoLevelErase, self).__init__(model, optimizer, domainbus, device, hparams, warmup, scaler)

        self.mask_w_ratio = [0.1, 0.2, 0.4]
        self.mask_h_ratio = [0.1, 0.2, 0.4]

        self.h_ratios, self.w_ratios = np.meshgrid(self.mask_h_ratio, self.mask_w_ratio)
        self.h_ratios = self.h_ratios.reshape(-1)
        self.w_ratios = self.w_ratios.reshape(-1)
        self.mask_aspect_ratios = [(h_r, w_r) for h_r, w_r in zip(self.h_ratios, self.w_ratios)]

        self.l_choices = ['img', 'ins', 'both']
        self.level = self.l_choices[2]

    def aug(self, images, targets):
        if random.random() > self.aug_prob:
            pass

        return images, targets

    def aug_img_level(self, images, targets):
        return images, targets

    def aug_ins_level(self, images, targets):
        return images, targets



class AlignSource(Baseline):

    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):
        super(AlignSource, self).__init__(model, optimizer, domainbus, device, hparams, warmup, scaler)

    @classmethod
    def get_hparams(cls, hparams):
        hparams.adv_w = 0.2                          # the weight of adv_loss

        """
            adv_layer: which layer use to do domain classification 
            choices [0, 1, 2, 3]
            
            0   represent the layer 1 of resnet 50 
            1   represent the layer 2 of resnet 50
            2   represent the layer 3 of resnet 50
            3   represent the layer 4 of resnet 50
            
        """

        # returned layers, no change or there will be wrong when loading state dict
        if hparams.net == 'fasterrcnn':
            hparams.return_layers = [1, 2, 3, 4]
            hparams.adv_layer = 2

        elif hparams.net == 'retinanet':
            hparams.return_layers = [2, 3, 4]
            hparams.adv_layer = 1

        return hparams


class CGMDRL(Baseline):

    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):
        super(CGMDRL, self).__init__(model, optimizer, domainbus, device, hparams, warmup, scaler)

    @classmethod
    def get_hparams(cls, hparams):
        hparams.gl_w = 0.1
        hparams.cls_w = 0.01
        hparams.dis_w = 0.1
        hparams.gl_m = 0.05
        hparams.s_init = True
        # returned layers, no change or there will be wrong when loading state dict
        if hparams.net == 'fasterrcnn':
            hparams.return_layers = [1, 2, 3, 4]
        elif hparams.net == 'retinanet':
            hparams.return_layers = [2, 3, 4]
        return hparams


class CSDRL(Baseline):

    def __init__(self, model, optimizer, domainbus, device, hparams, warmup=False, scaler=None):
        super(CSDRL, self).__init__(model, optimizer, domainbus, device, hparams, warmup, scaler)

    @classmethod
    def get_hparams(cls, hparams):
        hparams.adv_layer = -1

        if hparams.net == 'fasterrcnn':
            hparams.return_layers = [1, 2, 3, 4]
        elif hparams.net == 'retinanet':
            hparams.return_layers = [2, 3, 4]

        return hparams

