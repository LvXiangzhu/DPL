# Diverse Text-Prompt Generation for Remote Sensing Image Classification

This repository includes introductions and implementation of ***Diverse Text-Prompt Generation for Remote Sensing Image Classification*** in PyTorch.


# Datasets

We conduct experiments using three remote sensing datasets: **[NWPU VHR-10 (Cheng, Zhou, and Han 2016)](https://gcheng-nwpu.github.io/#Datasets), [DOTA (Xia et al. 2018)](https://captain-whu.github.io/DOTA/)** and **[HRRSD (Zhang et al. 2019)](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset)**.

Remote sensing objects are cut out from object detection ground truth.

Ten common object categories among three datasets are reserved for experiments, i.e., ***baseball-diamond, basketball-court, bridge, ground-track-field, harbor, airplane, ship, vehicle, storage-tank and tennis-court*** .

Specifically, folder index and categories are as follows:

>01 baseball-diamond  
02 basketball-court  
03 bridge  
04 ground-track-field  
05 harbor  
06 airplane  
07 ship  
08 vehicle  
09 storage-tank  
10 tennis-court  

You can download post-processed datasets from this link:  **[datasets](https://pan.baidu.com/share/init?surl=STwAU_M2sC23xXe6yNGXJg&pwd=fldi)** 

You can put these datasets in the path "DPL/DPL".
 

## File Structure

The codes are organized into two folders:

1. [Dassl.ProGrad.pytorch](Dassl.pytorch-master/) is the modified toolbox of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).
2. [DPL](DPL/) is our method.

# Models

All model parameters can be obtained from this link: [**DPL.pth.tar**](https://pan.baidu.com/share/init?surl=oYWJIEd5iFYJ76MgOYGT4g&pwd=ds2g).

# Requirements

>PyTorch >= 1.3.1  
>TorchVision >= 0.4.2  
>
>>Recommended  
>>tqdm >= 4.61  
>>matplotlib >= 1.5.1
>>ftfy
>>regex

# Train and Eval

## Train
For detailed argparse params, go to the DPL/train_prompts5.py and modify "--eval-only" to "True" and run
> python train_prompts5.py

## Eval
For detailed argparse params, go to the DPL/train_prompts5.py and modify "--eval-only" to "False" and run
> python train_prompts5.py
