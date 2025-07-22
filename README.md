# [Siggraph Asia 2024] ODA-GS: Occlusion- and Distortion-aware Gaussian Splatting for Indoor Scene Reconstruction

[Paper](https://dl.acm.org/doi/abs/10.1145/3681758.3697997)


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

## Training and Evaluation
1. Simple example:
```sh
python train.py -s data_path --iterations 30000 --data_device cpu --model_path path/to/output_folder --eval -r 1
python render.py -m path/to/output_folder --skip_train
python metrics.py -m path/to/output_folder
```
You can also refence the [run.sh](run.sh) script to see more commands.

2. For more flexible usage, please refer to [the file](arguments/__init__.py) to see the arguments.

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
