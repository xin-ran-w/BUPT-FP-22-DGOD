# BUPT-22-FP-DGOD

### 1. Introduction

**Codes of BUPT 2022 Final Project: Research on Domain Generalization for Object Detection**

This project compared three DG algorithms under the Object Detection task. They are:

* Cross-domain Stitch
* Source Domains Feature Alignment
* GDIFD

### 2. Benchmarks

Using five datasets building two benchmarks: BCF, BCFKS. You have to download the five original datasets: **BDD100k**, **Cityscapes**, **Foggy Cityscapes**, **KITTI**, **SIM10k**, and preprocess them to PASCAL VOC format. Then filter the common categories according to the following two benchmarks.

#### 2.1. BCF

Original Datasets: BDD100k, Cityscapes, Foggy Cityscapes

common categories: **person, car, train, rider, truck, motor, bike, bus**

#### 2.2. BCFKS

Original Datasets: BDD100k, Cityscapes, Foggy Cityscapes, KITTI, SIM10k

common categories: **car**

### 3. Installation

1. Main Requirements

   `python 3.7.11`

   `pycocotools 2.0`

   `pytorch 1.7.1`

   `torchvision 0.82`

   `fiftyone` (optional, used in visualization)

2. Install

   ```
   
   ```

### 4. Train

Use `train.py` to train on a single GPU, or `train_multi_GPU.py` to train on multiple GPUs. 

In the following commands, the `CUDA_VISIBLE_DEVICES=gpu_id` choose which gpu/gpus to be used. 

`--algorithm` choose the algorithm, total three choices: `CGMDRL`, `Baseline`, `AlignSource`, `Stitch`. 

`--num-classes` is 1 if you choose the BCFKS benchmark, 8 if you choose BCF benchmark.



```bash
conda activate env

# train on a single GPU

CUDA_VISIBLE_DEVICES=0 python train.py --num-classes 1 --amp True --batch-size 4 --sdi 1 2  --tdi 0 --ni 0 --cfi 4 --algorithm CGMDRL

# train on multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py --num-classes 1 --amp True --batch-size 4 --sdi 1 2 3 4 --tdi 0 --ni 0 --cfi 4 --algorithm CGMDRL
```

### 5. Test

