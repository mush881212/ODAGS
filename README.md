# [Siggraph Asia 2024 TC] ODA-GS: Occlusion- and Distortion-aware Gaussian Splatting for Indoor Scene Reconstruction

### [[Paper](https://dl.acm.org/doi/abs/10.1145/3681758.3697997)]

<img width="2662" height="831" alt="teaser_6" src="https://github.com/user-attachments/assets/9c9de5d1-5e2c-45be-9f95-72bb186a2259" />

In this work, we aim to address the quality degradation issues of indoor scene reconstruction using 3D Gaussian Splatting (3DGS). Existing methods enhance reconstruction quality by exploiting learned geometric priors like Signed Distance Functions (SDF), but these come with significant computational costs. We analyze the traditional 3DGS training process and identify key factors contributing to quality degradation: over-reconstruction and gradient dilution during the densification stage, and the occurrence of distorted/redundant Gaussians during the post-optimization stage. To tackle these issues, we introduce ODA-GS, a novel framework that modifies 3DGS with tailored modules. During densification, we employ occlusion-aware gradient accumulation to prevent gradient dilution and use homo-directional gradients to mitigate over-reconstruction. In the post-optimization stage, we introduce post-pruning to eliminate distorted and redundant Gaussians, thereby enhancing visual quality and reducing computational overhead. Tested on the ScanNet++ and Replica datasets, ODA-GS outperforms several baselines both qualitatively and quantitatively.

## Prerequisites
* Linux or Windows
* CUDA == 11.6
* Python3

## Getting Started
1. Clone this repo:
```sh
git clone https://github.com/mush881212/ODAGS.git --recursive
cd ODAGS
```
2. Install [conda](https://www.anaconda.com/).
3. Install all the dependencies
```sh
conda env create --file environment.yml
```
4. Switch to the conda environment
```sh
conda activate odags
```
## Dataset
We evaluate our method on the [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) dataset and a manually rendered version of the [Replica](https://github.com/facebookresearch/Replica-Dataset) dataset. For the data split, we select every eighth image from the original training set (specified in transforms_train.json) to construct the test set, ensuring higher image quality. Please refer to [relevant code in scene/dataset_readers.py](scene/dataset_readers.py#L225) for additional details.

## Training and Evaluation
### Training
```sh
python train.py -s path/to/data_root --iterations 30000 --data_device cpu --model_path path/to/output_folder --eval -r 1
```
#### --iterations
Number of total iterations to train for, 30_000 by default.
#### --data_device
Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training.
#### --model_path / -m
Path to the trained model directory.
#### --eval
Enable this flag to split every 8th image for evaluation; otherwise, all images will be used for training.
#### --resolution / -r
Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect.

### Evaluation
```sh
python render.py -m path/to/output_folder --skip_train # Generate renderings
python metrics.py -m path/to/output_folder # Compute error metrics on renderings
```
#### --skip_train
Flag to skip rendering the training set.
#### --model_path / -m
Path to the trained model directory you want to create renderings for.

You can also reference the [run.sh](run.sh) script to see more commands.

For more flexible usage, please refer to [the file](arguments/__init__.py) to see the arguments.

## Citation
If you find our code/models useful, please consider citing our paper:
```
@inproceedings{10.1145/3681758.3697997,
author = {Lee, Chai-Rong and Yen, Ting-Yu and Hsiao, Kai-Wen and Hung, Shih-Hsuan and Hsu, Sheng-Chi and Hu, Min-Chun and Yao, Chih-Yuan and Chu, Hung-Kuo},
title = {ODA-GS: Occlusion- and Distortion-aware Gaussian Splatting for Indoor Scene Reconstruction},
year = {2024}
}
```

## Acknowledgment
This project references [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). Thanks for the amazing work!
