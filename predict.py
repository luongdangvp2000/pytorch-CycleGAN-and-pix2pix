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
import argparse
import cv2
from tqdm import tqdm

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--name', type=str, help='name of trained model')
# opt, _ = parser.parse_known_args()


if __name__ == '__main__':
    

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
    

    out_dir = opt.checkpoints_dir
    model_name = opt.name
    trained_model_path = os.path.join(out_dir, model_name)
    predict_path = os.path.join(trained_model_path, 'predict')

    os.makedirs(predict_path, exist_ok=True)

    if opt.eval:
        model.eval()
    opt.num_test = len(dataset.dataset)
    scores_dict = {}
    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        # print(data)
        # exit(0)
        model.set_input(data)  # unpack data from data loader
        model.forward()           # run inference
        temp_fake_B, temp_real_B = model.get_current_visuals()['fake_B'], model.get_current_visuals()['real_B']
        temp_fake_B = tensor2im(temp_fake_B, to_RGB=False)
        temp_real_B = tensor2im(temp_real_B, to_RGB=False)
        # print("fake B predict ", temp_fake_B.shape, temp_fake_B.min(), temp_fake_B.max())
        # print("real B ", temp_real_B.shape, temp_real_B.min(), temp_real_B.max())
        # fake_B = tensor2im(model.fake_B, imtype=np.float32)[:, :, 0]
        # print(data['B_paths'])
        path = data['B_paths'][0]
        origin_name = path.split("/")[-1].replace(".npz","_origin.png")
        predict_name = path.split("/")[-1].replace(".npz","_predict.png")


        new_predict_path = os.path.join(predict_path, predict_name)
        new_orign_path = os.path.join(predict_path, origin_name)
        cv2.imwrite(new_predict_path, temp_fake_B[:, :, 0])
        cv2.imwrite(new_orign_path, temp_real_B[:, :, 0])

        
        # np.savez_compressed(new_path, data=fake_B)
        

