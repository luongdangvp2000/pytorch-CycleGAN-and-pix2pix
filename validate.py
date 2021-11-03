import torch
import numpy as np
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, Logger
from util.metric import PSNR, SSIM, FID_v
from copy import deepcopy

def validate(model, dataset, epoch, logger):
    print("Valiating at epoch ", epoch)
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    
    # psnr_list = []
    # ssim_list = []
    fid_list = []
    mae_list = []
    l1_loss = torch.nn.L1Loss(reduction='none')
    # model.netG.eval() # switch netG to eval model for validating
    model.eval() 
    with torch.no_grad():
        epoch_start_time = time.time() # calculate time
        for i, data in enumerate(dataset):  # inner loop within one epoch
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.forward()
            visuals = model.get_current_visuals()
            # print(len(PSNR(visuals['real_B'], visuals['fake_B'], reduction='none').detach().cpu().numpy()))
            # psnr_list.append(PSNR(visuals['real_B'], visuals['fake_B'], reduction='none').detach().cpu().numpy())
            # ssim_list.append(SSIM(visuals['real_B'], visuals['fake_B'], reduction='none').detach().cpu().numpy())
            fid_list.append(FID_v(visuals['real_B'], visuals['fake_B'], reduction='none').detach().cpu().numpy())
            mae_list.append(l1_loss(visuals['real_B'], visuals['fake_B']).detach().cpu().numpy()) # -1  1 
        
    # psnr_list = np.concatenate(psnr_list, 0)
    # ssim_list = np.concatenate(ssim_list, 0)
    fid_list = np.concatenate(fid_list, 0)
    mae_list = np.concatenate(mae_list, 0)
    # psnr_value = psnr_list.mean()
    # ssim_value = ssim_list.mean()
    fid_value = fid_list.mean()
    mae_value = mae_list.mean()
    # logger.scalar_summary("val_psnr", psnr_value, epoch)
    # logger.scalar_summary("val_ssim", ssim_value, epoch)
    logger.scalar_summary("val_fid", fid_value, epoch)
    logger.scalar_summary('val_mae', mae_value, epoch)
    # print("PSNR score in validating phase is ", psnr_value, " at epoch ", epoch)
    # print("SSIM score in validating phase is ", ssim_value, " at epoch ", epoch)
    print("FID score in validating phase is ", fid_value, " at epoch ", epoch)
    print("MAE score in validating phase is ", mae_value, " at epoch ", epoch)
    print("Time for validating is %d sec" % (epoch_start_time - time.time()))

    model.train()
    # model.netG.train() # after validating, set netG to training model
    # return psnr_value, ssim_value, mae_value
    return fid_value, mae_value
 
