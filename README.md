# DnCNN

This repo is for Jinan University 2023 Spring Mathematical Modeling Project

Original paper: [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981)

Original repo: [cszn](https://github.com/cszn)/[DnCNN](https://github.com/cszn/DnCNN)

## Training platform

### Hardware

We are using ModelArts AI platform provided by [广州人工智能公共算力中心](https://aipcc-gz.com/).

```bash
 __  __               _          _                      _         
|  \/  |             | |        | |     /\             | |        
| \  / |   ___     __| |   ___  | |    /  \     _ __   | |_   ___ 
| |\/| |  / _ \   / _  |  / _ \ | |   / /\ \   |  __| | __ | / __|
| |  | | | (_) | | (_| | |  __/ | |  / ____ \  | |     | |_  \__ \
|_|  |_|  \___/   \__ _|  \___| |_| /_/    \_\ |_|      \__| |___/
Using user ma-user
EulerOS 2.0 (SP8), CANN-6.3.RC1.alpha001
```
- 24 ARM cores
- 96 GB RAM
- Ascend 910 (32GB HBM) AI Processor (npu) * 1

### Environment


- Python 3.7.10
- numpy 1.21.6
- opencv-contrib-python 4.7.0.72
- torch 1.11
- torchvision 0.12.0
- torch-npu 1.11


## File description
- **create_patch.py** : create patches for training (default 40*40)

- **train_color.py** : train model for color image denoising

- **dataset_color.py** : load the data for training

- **denoise_color.py** : dednoise the noisy image

- **psnr_experiment.py** : calculata the average psnr between denoised images and original images

You can check the folder **dataset** for the download link for the datasets we used for training or testing.

