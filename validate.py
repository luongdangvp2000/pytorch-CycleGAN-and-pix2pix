import torch
import numpy as np
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, Logger
from util.metric import PSNR, SSIM, FID_v
from copy import deepcopy
from util.util import tensor2im
from util.fid import *

def validate(model, dataset, epoch, logger):
    print("Valiating at epoch ", epoch)
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    
    psnr_list = []
    ssim_list = []
    fid_list = []
    mae_list = []
    
    # print(type(real_b_dict))

    l1_loss = torch.nn.L1Loss(reduction='none')
    # model.netG.eval() # switch netG to eval model for validating
    model.eval() 

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    modelFID = InceptionV3([block_idx])
    modelFID = modelFID.cuda()
    with torch.no_grad():
        epoch_start_time = time.time() # calculate time
        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print(i)
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.forward()
            visuals = model.get_current_visuals()            

            # real_b_dict.append({'images':normalize_im(visuals['real_B'])})
            # fake_b_dict.append({'images':normalize_im(visuals['fake_B'])})

            # real_b_dict.append(tensor2im(visuals['real_B']))
            # fake_b_dict.append(tensor2im(visuals['fake_B']))

            psnr_list.append(PSNR(visuals['real_B'], visuals['fake_B'], reduction='none').detach().cpu().numpy())
            ssim_list.append(SSIM(visuals['real_B'], visuals['fake_B'], reduction='none').detach().cpu().numpy())
            mae_list.append(l1_loss(visuals['real_B'], visuals['fake_B']).detach().cpu().numpy()) # -1  1 
            # print(visuals['real_B'].shape)
            # tmp_real_B = visuals['real_B']
            fretchet_dist=calculate_fretchet(visuals['real_B'].repeat(1,3,1,1), visuals['fake_B'].repeat(1,3,1,1), modelFID)
            # print(fretchet_dist)

            # fid_list = FID_v(real_b_dict, fake_b_dict, reduction='none').detach().cpu().numpy()
            fid_list.append(fretchet_dist)

    # print(type(real_b_dict))
    # fid_list = FID_v(real_b_dict, fake_b_dict, reduction='none').detach().cpu().numpy()

    psnr_list = np.concatenate(psnr_list, 0)
    ssim_list = np.concatenate(ssim_list, 0)
    fid_list = np.array(fid_list)
    mae_list = np.concatenate(mae_list, 0)

    psnr_value = psnr_list.mean()
    ssim_value = ssim_list.mean()
    fid_value = fid_list.mean()
    mae_value = mae_list.mean()

    logger.scalar_summary("val_psnr", psnr_value, epoch)
    logger.scalar_summary("val_ssim", ssim_value, epoch)
    logger.scalar_summary("val_fid", fid_value, epoch)
    logger.scalar_summary('val_mae', mae_value, epoch)

    print("PSNR score in validating phase is ", psnr_value, " at epoch ", epoch)
    print("SSIM score in validating phase is ", ssim_value, " at epoch ", epoch)
    print("FID score in validating phase is ", fid_value, " at epoch ", epoch)
    print("MAE score in validating phase is ", mae_value, " at epoch ", epoch)
    print("Time for validating is %d sec" % (epoch_start_time - time.time()))

    model.train()
    # model.netG.train() # after validating, set netG to training model
    # return psnr_value, ssim_value, mae_value
    return psnr_value, ssim_value, fid_value, mae_value


# def normalize_im(image):
#     gt_imgs_unnormalize = image * 0.5 + 0.5
#     return gt_imgs_unnormalize
 
