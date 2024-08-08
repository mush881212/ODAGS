# Occluson-Aware Gaussian Splatting

## Prerequisites
* Linux or Windows
* CUDA == 11.6
* Python3

## Getting Started
1. Clone this repo:
```sh
git clone https://github.com/mush881212/SAOGS.git --recursive
cd SAOGS
```
2. Install [conda](https://www.anaconda.com/).
3. Create conda environment
```sh
conda env create --file environment.yml
```
4. Switch to the conda environment
```sh
conda activate occlusionawaregs
```
## Dataset
1. Download on [CGV Nas](https://cgv.cs.nthu.edu.tw:5001/). The full path of the training data is "GS_Densification/Datasets".

## Training and Evaluation
1. Simple example:
```sh
python train.py -s data_path --iterations 30000 --data_device cpu --model_path output_path --eval -r 1
python render.py -m  output_path --skip_train
python metrics.py -m output_path
```
You can also refence the [run.sh](run.sh) script to see the commands.

2. For more flexible usage, please refer to [the file](arguments/__init__.py) to see the arguments.

## Acknowledgment
This project references [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). Thanks for the amazing work!
