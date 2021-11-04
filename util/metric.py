import torch 
import numpy as np
import pandas as pd 
import torchvision.transforms as transforms
import piq
from piq import FID
from torch.utils.data import DataLoader, TensorDataset

# class TestDataset(torch.utils.data.Dataset):
#     def __init__(self, input_range=(0.0, 1.0)):
#         self.data = torch.FloatTensor(15, 3, 256, 256).uniform_(*input_range)
#         self.mask = torch.rand(15, 3, 256, 256)

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.mask[index]

#         return {'images': x, 'mask': y}

#     def __len__(self):
#         return len(self.data)

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

def FID_v(real_b_dict, fake_b_dict, reduction='none'):
    fid_metric = FID()
    # first_dl, second_dl = DataLoader(), DataLoader()
    # first_feats = fid_metric.compute_feats(first_dl)
    # second_feats = fid_metric.compute_feats(second_dl)
    tensor_real = torch.Tensor(real_b_dict)
    tensor_fake = torch.Tensor(fake_b_dict)

    real_dataset = TensorDataset(tensor_real)
    fake_dataset = TensorDataset(tensor_fake)

    real_dataloader = DataLoader(real_dataset)
    fake_dataloader = DataLoader(real_dataset)
    print()

    first_feats = fid_metric.compute_feats(real_dataloader)
    second_feats = fid_metric.compute_feats(fake_dataloader)
    fid_value = fid_metric(first_feats, second_feats)
    return fid_value


# def 

