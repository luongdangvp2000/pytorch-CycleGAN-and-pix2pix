import time
from copy import deepcopy
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, Logger
from util.metric import PSNR, SSIM
from validate import validate
import torch

torch.set_printoptions(precision=10)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt_val = deepcopy(opt) # copy opt and change phase to test for creating val dataset
    opt_val.phase = 'val'
    opt_val.batch_size = 1 # use batch size 64 for fast validating  
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # print(opt)
    # print(dataset)                    
    data = dataset.dataset
    # print(opt.preprocess)
    A = data[0]['A']
    # print(A.sum(), A.mean(), A.max(), A.min(), A.shape, B.sum())
    for i, data in enumerate(dataset):
        # print(i)
        # print(data)
        # print(data['A'].mean(), data['B'].mean())
        break
    