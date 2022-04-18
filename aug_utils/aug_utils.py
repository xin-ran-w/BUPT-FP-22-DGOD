import random

import torchvision
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as t
from nets.ssd.utils import calc_iou_tensor, dboxes300_coco, Encoder
from train_utils.distributed_utils import is_main_process


class Normalization(object):
    """对图像标准化处理,该方法应放在ToTensor后"""
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.normalize = t.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target

class SSDCropping(object):
    """
    根据原文，对图像进行裁剪,该方法应放在ToTensor前
    Cropping for SSD, according to original paper
    Choose between following 3 conditions:
    1. Preserve the original image
    2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
    3. Random crop
    Reference to https://github.com/chauhan-utk/src.DomainAdaptation
    """
    def __init__(self):
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )
        self.dboxes = dboxes300_coco()

    def __call__(self, image, target):
        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:  # 不做随机裁剪处理
                return image, target

            htot, wtot = target['height_width']

            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('+inf') if max_iou is None else max_iou

            # Implementation use 5 iteration to find possible candidate
            for _ in range(5):
                # 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w / h < 0.5 or w / h > 2:  # 保证宽高比例在0.5-2之间
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                # boxes的坐标是在0-1之间的
                bboxes = target["boxes"]
                ious = calc_iou_tensor(bboxes, torch.tensor([[left, top, right, bottom]]))

                # tailor all the bboxes and return
                # all(): Returns True if all elements in the tensor are True, False otherwise.
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                # 查找所有的gt box的中心点有没有在采样patch中的
                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                # 如果所有的gt box的中心点都不在采样的patch中，则重新找
                if not masks.any():
                    continue

                # 修改采样patch中的所有gt box的坐标（防止出现越界的情况）
                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                # 虑除不在采样patch中的gt box
                bboxes = bboxes[masks, :]
                # 获取在采样patch中的gt box的标签
                labels = target['labels']
                labels = labels[masks]

                # 裁剪patch
                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                image = image.crop((left_idx, top_idx, right_idx, bottom_idx))

                # 调整裁剪后的bboxes坐标信息
                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                # 更新crop后的gt box坐标信息以及标签信息
                target['boxes'] = bboxes
                target['labels'] = labels

                return image, target


class Resize(object):
    """对图像进行resize处理,该方法应放在ToTensor前"""
    def __init__(self, size=(300, 300)):
        self.resize = t.Resize(size)

    def __call__(self, image, target):
        image = self.resize(image)
        return image, target


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ResizeImageAndBox(object):
    def __init__(self, short_side):
        self.scaler = torchvision.transforms.Resize(short_side)


    def __call__(self, image, target):

        height, width = image.shape[1], image.shape[2]
        scaled_image = self.scaler(image)


        if target:
            bbox = target["boxes"]

            # scale the bbox with [width_ratio, height_ratio]
            bbox = self.resize_boxes(bbox, [width, height], [scaled_image.shape[2], scaled_image.shape[1]])
            target["boxes"] = bbox

        return scaled_image, target

    @classmethod
    def resize_boxes(cls, boxes, original_size, new_size):
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratios_width, ratios_height = ratios[0], ratios[1]

        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratios_width
        xmax = xmax * ratios_width
        ymin = ymin * ratios_height
        ymax = ymax * ratios_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)



class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target


class CropImages(object):


    def __init__(self, c_h=500, c_w=400, resize=False):

        self.c_imgs = []
        self.c_trgs = []

        self.c_h = c_h
        self.c_w = c_w

        self.resize = resize

        if self.resize:
            self.scaler = ResizeImageAndBox(short_side=c_h)

    def __call__(self, images, targets):

        self.reset()

        for img, trg in zip(images, targets):

            if self.resize:
                img, trg = self.scaler(img, trg)

            c_img, c_trg = self.crop(img, trg)

            self.c_imgs.extend(c_img)
            self.c_trgs.extend(c_trg)

        return self.c_imgs, self.c_trgs


    def crop(self, image, target):

        c_imgs = []
        c_trgs = []

        r_img = image
        r_trg = target
        r_img_w = image.shape[2]

        while r_img_w >= 400:

            c_img, c_trg, r_img, r_trg = self.__crop_img(image=r_img, target=r_trg)
            if c_trg["boxes"] is not None:
                c_imgs.append(c_img)
                c_trgs.append(c_trg)
            if r_trg["boxes"] is None:
                break

            r_img_w = r_img.shape[2]

        return c_imgs, c_trgs

    def __crop_img(self, image, target):

        width = image.shape[2]
        c_trg = dict()
        r_trg = dict()

        # crop from right side
        if random.random() > 0.5:

            bound = width - self.c_w

            # crop image
            c_img = image[:, :, bound:]
            r_img = image[:, :, :bound]

            # crop target
            r_trg["boxes"], r_trg["labels"], c_trg["boxes"], c_trg["labels"] = self.__crop_trg(target=target, bound=bound)


        # crop from left side
        else:
            bound = self.c_w

            # crop image
            c_img = image[:, :, :bound]
            r_img = image[:, :, bound:]

            # crop target
            c_trg["boxes"], c_trg["labels"], r_trg["boxes"], r_trg[
                "labels"] = self.__crop_trg(target=target, bound=bound)

        return c_img, c_trg, r_img, r_trg


    def __crop_trg(self, target, bound):
        bboxes = target["boxes"]
        labels = target["labels"]

        left_bboxes = list()
        left_labels = list()
        right_bboxes = list()
        right_labels = list()

        for index, bbox in enumerate(bboxes):

            if bbox[2] >= bound:

                # bbox is on the boundary
                if bbox[0] < bound:
                    left_bbox, right_bbox, left_bool, right_bool = self.split_bbox(bbox, bound)

                    if left_bool:
                        left_bboxes.append(left_bbox.unsqueeze(0))
                        left_labels.append(labels[index].unsqueeze(0))

                    if right_bool:
                        right_bboxes.append(right_bbox.unsqueeze(0))
                        right_labels.append(labels[index].unsqueeze(0))

                # bbox is on the right side
                else:
                    right_bboxes.append(bbox.unsqueeze(0))
                    right_labels.append(labels[index].unsqueeze(0))

            # bbox is on the left side
            else:
                left_bboxes.append(bbox.unsqueeze(0))
                left_labels.append(labels[index].unsqueeze(0))

        # transform the list to 2D Tensor

        if len(left_bboxes):
            left_bboxes = torch.cat(left_bboxes, dim=0)
            left_labels = torch.cat(left_labels, dim=0)

        else:
            left_bboxes = left_labels = None

        if len(right_bboxes):
            right_bboxes = torch.cat(right_bboxes, dim=0)
            right_labels = torch.cat(right_labels, dim=0)

            # correct the right bboxes by subtract the bound
            right_bboxes[:, 0::2] -= bound

        else:
            right_bboxes = right_labels = None

        return left_bboxes, left_labels, right_bboxes, right_labels


    def split_bbox(self, bbox, bound):
        """
        When the bbox is cross the boundary, then split the bbox into two bbox
        :param bbox: Tensor(x_min, y_min, x_max, y_max)
        :param bound: int the split boundary
        :return: left_bbox, right_bbox, left_bool, right_bool
        """

        # cannot use detach since it will share memory of the tensor
        right_bbox = bbox.clone()
        left_bbox = bbox.clone()

        # new x_min of right_bbox
        right_bbox[0] = bound

        # new x_max of left_bbox
        left_bbox[2] = bound - 1

        left_bool = self.is_box_valid(left_bbox)
        right_bool = self.is_box_valid(right_bbox)

        return left_bbox, right_bbox, left_bool, right_bool

    def is_box_valid(self, box):

        width = box[2].item() - box[0].item()
        height = box[3].item() - box[1].item()

        if width > 0 and height > 0:
            return True

        return False

    def reset(self):
        self.c_imgs = []
        self.c_trgs = []



class StitchImages(object):

    def __init__(self, width=400, stitch_num=2):
        """
        :param width: (Int) the image width.
        :param stitch_num: (Int) the num of stitch image.
        """

        self.img_width = width
        self.size = stitch_num


    def __call__(self, origin_img, origin_target, new_img, new_target):
        """
        :param origin_img: Tensor[N, C, H, W]
        :param origin_target: Dict
        :param new_img: Tensor[N, C, H, W]
        :param new_target: Dict
        :return: Tensor[N, C, H, W], Dict
        """

        # stitch the new image on the right side
        if random.random() > 0.5:

            # There are three dims [C, H, W] for one image Tensor stitch the img on the dim W
            img = torch.cat([origin_img, new_img], dim=2)
            target = self.stitch_target(origin_target, new_target, right_side=True)

        # stitch the new image on the left side
        else:
            img = torch.cat([new_img, origin_img], dim=2)
            target = self.stitch_target(origin_target, new_target, right_side=False)

        return img, target

    def stitch_target(self, origin_target, new_target, right_side):

        origin_bboxes = origin_target["boxes"]
        origin_labels = origin_target["labels"]

        new_bboxes = new_target["boxes"]
        new_labels = new_target["labels"]

        if right_side:
            new_bboxes[:, 0::2] += self.img_width
        else:
            origin_bboxes[:, 0::2] += self.img_width

        target_bboxes = torch.cat([origin_bboxes, new_bboxes], dim=0)
        target_labels = torch.cat([origin_labels, new_labels], dim=0)

        target = {
            "boxes": target_bboxes,
            "labels": target_labels,
        }

        return target

class AssignGTtoDefaultBox(object):
    """将DefaultBox与GT进行匹配"""

    def __init__(self):
        self.default_box = dboxes300_coco()
        self.encoder = Encoder(self.default_box)

    def __call__(self, image, target):
        boxes = target['boxes']
        labels = target["labels"]
        # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        bboxes_out, labels_out = self.encoder.encode(boxes, labels)
        target['boxes'] = bboxes_out
        target['labels'] = labels_out

        return image, target



class ColorJitter(object):
    """对图像颜色信息进行随机调整,该方法应放在ToTensor前"""
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = t.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self.trans(image)
        return image, target

