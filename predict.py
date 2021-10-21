import torch
import numpy as np
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, Logger
from util.util import tensor2im
from util.metric import PSNR, SSIM
from copy import deepcopy
import os 


if __name__ == '__main__':
    out_dir = './predict/'
    os.makedirs(out_dir, exist_ok=True)

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    opt.num_test = len(dataset.dataset)
    scores_dict = {}
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.forward()           # run inference
        fake_B = tensor2im(model.fake_B, imtype=np.float32)[:, :, 0]
        print(data['A_paths'])
        path = data['A_paths'][0]
        name = path.split("/")[-1]
        new_path = os.path.join(out_dir, name)
        np.savez_compressed(new_path, data=fake_B)
        

