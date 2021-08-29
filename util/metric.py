import torch 
import numpy as np
import pandas as pd 
import torchvision.transforms as transforms
import piq
from piq import FID

def PSNR(gt_imgs, pred_imgs, reduction='none'):
    gt_imgs_unnormalize = gt_imgs * 0.5 + 0.5 # unnormalize images
    pred_imgs_unnormalize = pred_imgs * 0.5 + 0.5 # unnormalize images
    psnr_value = piq.psnr(gt_imgs_unnormalize, pred_imgs_unnormalize, data_range=1.0, reduction=reduction)
    return psnr_value

def SSIM(gt_imgs, pred_imgs, reduction='none'):
    gt_imgs_unnormalize = gt_imgs * 0.5 + 0.5 # unnormalize images
    pred_imgs_unnormalize = pred_imgs * 0.5 + 0.5 # unnormalize images
    ssim_value = piq.ssim(gt_imgs_unnormalize, pred_imgs_unnormalize, data_range=1.0, reduction=reduction)
    return ssim_value

# def 

