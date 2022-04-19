# RetinaNet

## 该项目主要是来自pytorch官方torchvision模块中的源码
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练
* 详细环境配置见```requirements.txt```

## 文件结构：
```
  ├── backbone: 特征提取网络(ResNet50+FPN)
  ├── nets: RetinaNet网络
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── voc_dataset.py: 自定义dataset用于读取VOC格式的数据集
  ├── train.py: 单卡训练启动脚本
  ├── train_multi_GPU.py: 多卡训练启动脚本
  ├── validation.py: 评估脚本
  └── class_dict: 标签文件
```

## 预训练权重下载地址（下载后放入backbone文件夹中）：
* ResNet50+FPN backbone: https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
* 注意，下载的预训练权重记得要重命名，比如在train.py中读取的是```retinanet_resnet50_fpn_coco.pth```文件，
  不是```retinanet_resnet50_fpn_coco-eeacb38b.pth```


## 数据集，本例程使用的是PASCAL VOC2012数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* 如果不了解数据集或者想使用自己的数据集进行训练，请参考我的bilibili：https://b23.tv/F1kSCK
* 基于迁移学习在PASCAL VOC2012训练集训练得到的权重： 链接: https://pan.baidu.com/s/1mqrBFWuJ_lfDloCfVjWqaA  密码: sw0t
* 在PASCAL VOC2012验证集上结果：
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.563
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.798
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.688
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.707
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
```



