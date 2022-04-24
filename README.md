# BUPT-22-FP-DGOD

### 1. Introduction

**Codes of BUPT 2022 Final Project: Research on Domain Generalization for Object Detection**

This project compared three DG algorithms under the Object Detection task. They are:

* Cross-domain Stitch
* Source Domains Feature Alignment
* GDIFD

### 2. Benchmarks

Using five datasets building two benchmarks: BCF, BCFKS. You have to download the five original datasets: **BDD100k**, **Cityscapes**, **Foggy Cityscapes**, **KITTI**, **SIM10k**, and preprocess them to PASCAL VOC format. Then filter the common categories according to the following two benchmarks. I will upload a script of pre-processing of datasets in another repo soon.

#### 2.1. BCF

Original Datasets: **BDD100k**, **Cityscapes**, **Foggy Cityscapes**

common categories: **person**, **car**, **train**, **rider**, **truck**, **motor**, **bike**, **bus**

#### 2.2. BCFKS

Original Datasets: **BDD100k**, **Cityscapes**, **Foggy Cityscapes**, **KITTI**, **SIM10k**

common categories: **car**

### 3. Installation

1. Main Requirements

   `python 3.7.11`

   `pycocotools 2.0`

   `pytorch 1.7.1`

   `torchvision 0.82`

   `fiftyone` (optional, used in visualization)

2. Install

   ```bash
   # create conda env
   conda create -n [env] python=3.7
   conda activate [env]
   
   # install pytorch
   conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
   
   # install pycocotools
   ## before install pycocotools, you should install cython
   
   pip install cython
   
   git clone https://github.com/cocodataset/cocoapi.git
   cd cocoapi/PythonAPI
   python setup.py build_ext install
   
   # install fiftyone, this package is used to visualize the dataset and predict results by different algorithms, to see more details in https://voxel51.com/
   
   pip install fiftyone
   
   ## if your ubuntu's version <= 18.04, install following package
   pip install fiftyone-db-ubuntu1604
   
   # Download pre-training weights and rename it according to the basic configs
   ## Faster R-CNN
   wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
   ## RetinaNet
   wget https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
   ```

### 4. Basic Configs

The basic, train, valid parameters all listed in the params package.

```python
self.domain_info = {
    "data_dir": [
        "path/to/bdd100k",            # domain 0
        "path/to/cityscapes",         # domain 1
        "path/to/foggycity",          # domain 2
        "path/to/kitti",              # domain 3
        "path/to/sim10k",             # domain 4
    ],

    "class_dict": [
        './class_dict/bcf.json',                                        # class_dict 0
        './class_dict/bcfks.json',                                      # class_dict 1
        './class_dict/bcf_b.json',                                      # class_dict 2 with background
        './class_dict/bcfks_b.json',                                    # class_dict 3 with background
    ],
}

self.net_info = {
    "networks": [
        'fasterrcnn',
        'retinanet',
    ],

    "weights": [
        '/path/to/fasterrcnn_resnet50_fpn_coco.pth',
        '/path/to/retinanet_resnet50_fpn_coco.pth',
    ]
}
```

### 5. Train

Use `train.py` to train on a single GPU, or `train_multi_GPU.py` to train on multiple GPUs. 

In the following commands: 

`CUDA_VISIBLE_DEVICES=gpu_id` choose which gpu/gpus to be used. 

`--algorithm` choose the algorithm, total three choices: `CGMDRL`, `Baseline`, `AlignSource`, `Stitch`. 

`--num-classes` is 1 if you choose the BCFKS benchmark, 8 if you choose BCF benchmark.

`--amp` indicate whether use mixed precision training in PyTorch.

`--ni` indicate which network to use: 0 for Faster R-CNN, 1 for RetinaNet. 

`--cfi` indicate the class dict

`--sdi` indicate the source domains in training

`--tdi` indicate the target domains in training

```bash
cd /path/to/workspace
conda activate [env]

# train on a single GPU
CUDA_VISIBLE_DEVICES=0 python train.py --num-classes 1 --amp True --batch-size 4 --sdi 1 2 3 4 --tdi 0 --ni 0 --cfi 3 --algorithm Stitch

# train on multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py --num-classes 1 --amp True --batch-size 4 --sdi 1 2 3 4 --tdi 0 --ni 0 --cfi 3 --algorithm Stitch
```

### 6. Test

```bash
CUDA_VISIBLE_DEVICES=0 python validation.py --cfi 2 --ni 0 --sdi 0 2 --tdi 1 --num-classes 8 --model-path /path/to/weight --algorithm Stitch
```

### 7. Infer & Visualization

I use **fiftyone** in this part.

```bash
python infer.py --cfi 1 --ni 1 --dataset bdd100k --num-classes 1 --model-path /path/to/weight --algorithm Stitch
```

### 8. Acknowledgement

1. [Pytorch torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models/detection)
2. [WZMIAOMIAO's deep-learning-for-image-processing repo](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/commits?author=WZMIAOMIAO)

3. [WZMIAOMIAO's bilibili channel for deep learning](https://space.bilibili.com/18161609/channel/index)
