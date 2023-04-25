# Unofficial MVSS-Net (ICCV 2021) reproducing with PyTorch
![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) ![GitHub repo size](https://img.shields.io/github/repo-size/dddb11/MVSS-Net?logo=hack%20the%20box) ![GitHub](https://img.shields.io/github/license/Sunnyhaze/ManTra-Net_Pytorch?logo=license)  [![Ask Me Anything !](https://img.shields.io/badge/Official%20-No-1abc9c.svg)](https://GitHub.com/Sunnyhaze) ![visitors](https://visitor-badge.glitch.me/badge?page_id=dddb11.MVSS-Net)


Unofficial implementation of the MVSS-Net, which was proposed in ICCV 2021 by Chengbo Dong et al, includes code for training! This unofficial implementation is supported by the [DICA Lab of Sichuan University](https://dicalab.cn/).

The original repo is lacking the training code, links are here: [OFFICIAL MVSS-Net link](https://github.com/dong03/MVSS-Net/) we tried our best to reproduce the result of the model.

> We need to point out that many researchers reported that they failed to reproduce the result in their paper. This is also our goal when building this repo, but in the end, our re-implementation also fails and the metrics are far from the metrics in the paper on the Columbia and COVER datasets. 
> You could discuss this phenomenon with researchers through these links: [Zhihu(a Chinese forum)](https://zhuanlan.zhihu.com/p/422549140); [Issues of the official repo](https://github.com/dong03/MVSS-Net/issues).

>Besides there is a known issue in the official repo where there is ambiguity in the calculation of the Image-level F1 score. However, they suspiciously benefit a lot from this ambiguous f1 score: [Link](https://github.com/dong03/MVSS-Net/issues/30).

## Enviroment
Ubuntu 
PyTorch 1.10.0 + cu111

## Quick Start
- Prepare your datasets:
  - All dataset images is managed with a `txt` file recording the path of `images`, `groundtruths` and `edge masks`(if have) and `image label`. This file follows the following format:
    - Each line of this file should contain 4 elements, separate with a space:
      - <path_to_input_image>
      - <path_to_groundtruth_mask> (For authentic image, here is `None`)
      - <path_to_edge_mask> (If not generated in advance, here could be `None`)
      - <image_level_label> (0 or 1. 1 refers manipulated image, 0 for authentic image)
    
    - For example, each line in txt file should like this:
      - Authentic image:
        ```
        ./Casiav2/authentic/Au_ani_00001.jpg None None 0
        ``` 
      - Manipulated image with pre-generated edge mask: 
        ```
        ./Casiav2/tampered/Tp_D_CND_M_N_ani00018_sec00096_00138.tif ./Casiav2/mask/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png ./Casiav2/edge/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png 1
        ```
      - Manipulated image without pre-generated edge mask: 
        ```
        ./Casiav2/tampered/Tp_D_CND_M_N_ani00018_sec00096_00138.tif ./Casiav2/mask/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png None 1
        ``` 
  - You should follow the format and generate your own "path file" in a `xxxx.txt`.
> You could only use this Repo to generate the edge mask during training at this time. This will be a little bit slow. Since every Epoch you will generate a edge mask for each image, however, they are always the same edge mask. Better choice should be generate the edge mask from the ground truth mask before start training. This script will release later......

- Then you could start to run this work, the main entrance is [train_launch.py](./train_launch.py). Since `torch.nn.parallel.DistributedDataParallel` is used, you need to use the following command to start the training:
  ```bash
  torchrun \
    --standalone \
    --nproc_per_node=<Number of your GPU> \
  train_launch.py
    --paths_file <your_own_path_txt file> \
    --lr 1e-4
  ```
  - The content in the \<xxx\> above needs to be replaced with your personalized string or number
  - The above instructions are only the most basic ones. If you want to further adjust the parameters, please refer to [here](https://github.com/dddb11/MVSS-Net/blob/09c589e19e01dfaf97151f9ee246be371863005c/train_base.py#L46).
- You could use `Tensorboard` to monitor the progress of the model during training. Logs should under `./save/` path.

## Introduction
Still on Working...

## Files in the repo
Still on Working...

## Some Comments
Still on Working...

## Links
If you want to train this Model with the CASIAv2 dataset, we provide a revised version of CASIAv2 datasets, which corrected several mistakes in the original datasets provided by the author. Details can find in the [link](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth) shown below:

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=Sunnyhaze&repo=CASIA2.0-Corrected-Groundtruth)](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth)

## Cite
[1] Chen, X., Dong, C., Ji, J., Cao, J., & Li, X. (2021). Image Manipulation Detection by Multi-View Multi-Scale Supervision. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 14165–14173. https://doi.org/10.1109/ICCV48922.2021.01392


