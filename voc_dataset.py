import warnings

from torch.utils.data.dataset import Dataset
import os
import os.path as op
import torch
import json
from PIL import Image
from lxml import etree

from aug_utils.aug_utils import Compose


class VOCDataSet(Dataset):

    def __init__(self,
                 data_dir: str,
                 img_dir: str = 'JPEGImages',
                 anno_dir: str = 'Annotations',
                 split_dir: str = 'ImageSets',
                 num_classes: int = 8,
                 split: str = 'train',
                 class_file: str = './class_dict/bcf.json',
                 transforms: Compose = None,
                 dataset_name: str = None,
                 max_samples: int = -1,
                 d_seq: int = 0):

        num_classes = str(num_classes)

        self.data_dir = data_dir
        self.img_dir = op.join(data_dir, img_dir)
        self.anno_dir = op.join(data_dir, anno_dir + '_' + num_classes)
        self.split_dir = op.join(data_dir, split_dir, num_classes)

        assert op.exists(self.data_dir), "the data dir doesn't exists"
        assert op.exists(self.img_dir), "the img dir doesn't exists"
        assert op.exists(self.anno_dir), "the anno dir doesn't exists"
        assert op.exists(self.split_dir), "the split dir doesn't exists"

        self.name = op.basename(data_dir)

        self.split = split

        assert self.split in ['train', 'val'], 'split has two choices [train, val], but not {}'.format(self.split)

        self.split_file = os.path.join(self.split_dir, self.split + '.txt')

        self.class_file = class_file

        self.transforms = transforms

        self.dataset_name = dataset_name

        self.xml_list = self._read_split_file()

        self.class_dict = self._read_class_dict()

        self.max_samples = None

        if 0 < max_samples < len(self.xml_list):
            self.xml_list = self.xml_list[:max_samples]

        self.d_seq = d_seq

    def _read_split_file(self):
        with open(self.split_file) as read:
            anno_list = [op.join(self.anno_dir, line.strip())
                         for line in read.readlines() if len(line.strip()) > 0]
        return anno_list

    def _read_class_dict(self):

        assert os.path.exists(self.class_file), "{} file not exist.".format(self.class_file)
        json_file = open(self.class_file, 'r')
        return json.load(json_file)

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        # data_height = int(data["size"]["height"])
        # data_width = int(data["size"]["width"])
        # height_width = [data_height, data_width]
        img_path = os.path.join(self.img_dir, data["filename"])
        image = Image.open(img_path)
        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            if ymax > ymin and xmax > xmin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_dict[obj["name"]])
            else:
                warnings.warn(f"The image {img_path} has invalid bounding box!")
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        d_seq = torch.tensor([self.d_seq])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        target = {"boxes": boxes, "labels": labels,
                  "image_id": image_id, "area": area,
                  "iscrowd": iscrowd, "domain": d_seq}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
