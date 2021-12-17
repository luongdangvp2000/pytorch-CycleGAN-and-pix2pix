"""
src/networks.py provides the generator and discriminator architectures.

src/utils.py provides two training APIs train_i2i_UNet3headGAN and train_i2i_Cas_UNet3headGAN. 
The first API is to be used to train the primary GAN, whereas the second API is to be used to train the subsequent GANs.

An example command to use the first API is:
"""
import torch
from utils import *
from networks import *
from ds import *
import argparse
# from data import create_dataset


if __name__ == "__main__":
    # print("start")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/', help='models are saved here')
    parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--phase', type=str, default='train')
    
    args = parser.parse_args()


    # train_loader, test_loader = 
    dataroot = args.dataroot
    train_phase = "train"
    val_phase = "val"
    max_dataset_size = args.max_dataset_size

    # train_loader = PairedImages_PETCT(dataroot, val_phase, max_dataset_size)
    # test_loader = PairedImages_PETCT(dataroot, val_phase, max_dataset_size)

    train_loader = create_dataset(dataroot, train_phase, max_dataset_size)
    test_loader = create_dataset(dataroot, val_phase, max_dataset_size)

    # first load the prior Generators 
    netG_A1 = CasUNet_3head(1,1)
    netG_A1.load_state_dict(torch.load(args.ckpt_path + "i2i_0_UNet3headGAN_eph49_G_A.pth"))

    netG_A2 = UNet_3head(4,1)
    netD_A = NLayerDiscriminator(1, n_layers=4)

        #train the cascaded framework
    list_netG_A, list_netD_A = train_i2i_Cas_UNet3headGAN(
        [netG_A1, netG_A2], [netD_A],
        train_loader, test_loader,
        dtype=torch.cuda.FloatTensor,
        device='cuda',
        num_epochs=50,
        init_lr=1e-5,
        ckpt_path=args.ckpt_path + "i2i_2_UNet3headGAN",
    )

    # netG_A, netD_A = train_i2i_UNet3headGAN(
    #     netG_A, netD_A,
    #     train_loader, test_loader,
    #     dtype=torch.cuda.FloatTensor,
    #     device='cuda',
    #     num_epochs=50,
    #     init_lr=1e-5,
    #     ckpt_path=args.ckpt_path,
    # )

# """
# This will save checkpoints in ../ckpt/ named as i2i_0_UNet3headGAN_eph*.pth

# An example command to use the second API (here we assumed the primary GAN and first subsequent GAN are trained already)
# """


# # first load the prior Generators 
# netG_A1 = CasUNet_3head(1,1)
# netG_A1.load_state_dict(torch.load('../ckpt/i2i_0_UNet3headGAN_eph49_G_A.pth'))
# netG_A2 = UNet_3head(4,1)
# netG_A2.load_state_dict(torch.load('../ckpt/i2i_1_UNet3headGAN_eph49_G_A.pth'))

# #initialize the current GAN
# netG_A3 = UNet_3head(4,1)
# netD_A = NLayerDiscriminator(1, n_layers=4)

# #train the cascaded framework
# list_netG_A, list_netD_A = train_uncorr2CT_Cas_UNet3headGAN(
#     [netG_A1, netG_A2, netG_A3], [netD_A],
#     train_loader, test_loader,
#     dtype=torch.cuda.FloatTensor,
#     device='cuda',
#     num_epochs=50,
#     init_lr=1e-5,
#     ckpt_path='../ckpt/i2i_2_UNet3headGAN',
# )