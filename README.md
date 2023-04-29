# Unofficial MVSS-Net (ICCV 2021) reproducing with PyTorch
![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) ![GitHub repo size](https://img.shields.io/github/repo-size/dddb11/MVSS-Net?logo=hack%20the%20box) ![GitHub](https://img.shields.io/github/license/Sunnyhaze/ManTra-Net_Pytorch?logo=license)  [![Ask Me Anything !](https://img.shields.io/badge/Official%20-No-1abc9c.svg)](https://GitHub.com/Sunnyhaze) ![visitors](https://visitor-badge.glitch.me/badge?page_id=dddb11.MVSS-Net)


Unofficial implementation of the MVSS-Net, which was proposed in ICCV 2021 by Chengbo Dong et al, includes code for training! This unofficial implementation is supported by the [DICA Lab of Sichuan University](https://dicalab.cn/).

The original repo is lacking the training code, links are here: [OFFICIAL MVSS-Net link](https://github.com/dong03/MVSS-Net/) we tried our best to reproduce the result of the model.

## Foreword
**ALERT**: Although this method may seem like the current SOTA model, but the current results indicate that there are many doubts in this paper. We **do not recommend** using it as a replication model for the target or a model for learning when entering the field for the following reasons:

- Facts:
  - We need to point out that many researchers reported that they failed to reproduce the result in their paper. Reaching the result in their paper is also our goal when building this repo, but in the end, our re-implementation could reach the pixel-level fixed threshold F1 score on CASIAv1 and NIST16 and Defacto, but far from Coverage and Columbia. Many existing researchers have made inferences that there may be data leakage on this two dataset.
  - And for optimal F1, many researchers questioned for this absured metrics and send issues to the author, but the lead author of the article can only give ambiguous answers finally([issue](https://github.com/dong03/MVSS-Net/issues/11)), and unable to correctly clarify how the optimal f1 used in the paper comes from. The value of this optimal F1 can only reach a maximum of `0.62` by reserchers all over the Internet, while far below the 0.7 in the paper.
  - Besides there is a known issue in the official repo where there is ambiguity in the calculation of the Image-level F1 score. However, **they suspiciously benefit a lot from this ambiguous f1 score**, detailed proof is provided in the following [Link](https://github.com/dong03/MVSS-Net/issues/30). 


>You could discuss all phenomenon and stand together with researchers through these links: [Zhihu(a Chinese forum)](https://zhuanlan.zhihu.com/p/422549140); [Issues of the official repo](https://github.com/dong03/MVSS-Net/issues).

- Our comments:
  - From their official issue page, it can be seen that when faced with numerous researchers asking about training codes, there is no response at all. However, once the severe [F1 issue](https://github.com/dong03/MVSS-Net/issues/30) is pointed out and friendly discussions are conducted, **the author's arrogant treatment was received, and the issue was quickly closed with "nothing to do with honesty"**, which is full of arrogance and disrespect toward scientific research. 
  - If you have reproduced this paper, you can discuss the reproduced indicators and inferred conclusions together in the issue of our repo. If you have suffered a lot due to their irresponsible articles, we also welcome everyone to discuss in the issue section of our repo.

## Enviroment
Ubuntu 18.04.5 LTS

Python 3.9.15

PyTorch 1.10.0 + cuda11.1

Detail python librarys can found in [requirements.txt](./requirements.txt)

## Quick Start
- Prepare your datasets:
  - All dataset images is managed with a `txt` file recording the path of `images`, `groundtruths` and `edge masks`(if have) and `image label`. This file follows the following format:
    - Each line of this file should contain 4 elements, separate with a space:
      - <path_to_input_image>
      - <path_to_groundtruth_mask> (For authentic image, here is `None`)
      - <path_to_edge_mask> (If not generated in advance, here could be `None`)
      - <image_level_label> (0 or 1; 1 refers manipulated image, 0 for authentic image)
    
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
> Limits: At this time, the edge mask can only be generated during training and cannot be pre generated.   This will be a little bit slow. Since every Epoch you will generate a edge mask for each image, however, they are always the same edge mask. Better choice should be generate the edge mask from the ground truth mask before start training. Script for pre-generate the edge mask will release later...

- Then you could start to run this work, the main entrance is [train_launch.py](./train_launch.py). Since `torch.nn.parallel.DistributedDataParallel` is used, you need to use the following command to start the training:
  ```bash
  torchrun \
    --standalone \
    --nproc_per_node=<Number of your GPU> \
  train_launch.py \
    --paths_file <your own path txt file> 
  ```
  - The content in the \<xxx\> above **needs to be replaced** with your personalized string or number.
  - The above commands are only the most basic ones. If you want to further adjust the parameters, please refer to [here](https://github.com/dddb11/MVSS-Net/blob/09c589e19e01dfaf97151f9ee246be371863005c/train_base.py#L46).
- You could use `Tensorboard` to monitor the progress of the model during training. Logs should under `./save/` path.

## Introduction
Still on Working...

## Files in the repo
Still on Working...

## Some Comments
Still on Working...

## Result
We trained for 200 epochs and used decay, and finally selected the best data for each epoch on each dataset.Here's the result.Please note that these are only approximate results and we did not make any further adjustments, so they should be taken as a reference only.\
CASIAv1：
{'pixel_f1': 0.43, 'acc': 0.69, 'sen': 0.75, 'spe': 0.63, 'imagelevel_f1': 0.68, 'img_auc': 0.78, 'com_f1': 0.53, 'epoch': '11_end.pth'}
{'pixel_f1': 0.4, 'acc': 0.74, 'sen': 0.55, 'spe': 0.95, 'imagelevel_f1': 0.7, 'img_auc': 0.8, 'com_f1': 0.51, 'epoch': '40_end.pth'}

COVERAGE：
{'pixel_f1': 0.33, 'acc': 0.52, 'sen': 0.96, 'spe': 0.08, 'imagelevel_f1': 0.15, 'img_auc': 0.56, 'com_f1': 0.2, 'epoch': '11_end.pth'}
{'pixel_f1': 0.13, 'acc': 0.56, 'sen': 0.59, 'spe': 0.53, 'imagelevel_f1': 0.56, 'img_auc': 0.59, 'com_f1': 0.21, 'epoch': '21_end.pth'}
{'pixel_f1': 0.22, 'acc': 0.54, 'sen': 0.81, 'spe': 0.28, 'imagelevel_f1': 0.42, 'img_auc': 0.55, 'com_f1': 0.29, 'epoch': '7_end.pth'}

Columbia：
{'pixel_f1': 0.44, 'acc': 0.66, 'sen': 0.98, 'spe': 0.36, 'imagelevel_f1': 0.52, 'img_auc': 0.84, 'com_f1': 0.48, 'epoch': '11_end.pth'}
{'pixel_f1': 0.2, 'acc': 0.81, 'sen': 0.86, 'spe': 0.77, 'imagelevel_f1': 0.81, 'img_auc': 0.88, 'com_f1': 0.32, 'epoch': '35_end.pth'}

NIST16：
{'pixel_f1': 0.2, 'acc': 0.66, 'sen': 0.66, 'spe': 0.0, 'imagelevel_f1': 0.0, 'img_auc': 0.0, 'com_f1': 0.0, 'epoch': '3_end.pth'}

## Links
If you want to train this Model with the CASIAv2 dataset, we provide a revised version of CASIAv2 datasets, which corrected several mistakes in the original datasets provided by the author. Details can find in the [link](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth) shown below:

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=Sunnyhaze&repo=CASIA2.0-Corrected-Groundtruth)](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth)

## Cite
[1] Chen, X., Dong, C., Ji, J., Cao, J., & Li, X. (2021). Image Manipulation Detection by Multi-View Multi-Scale Supervision. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 14165–14173. https://doi.org/10.1109/ICCV48922.2021.01392


