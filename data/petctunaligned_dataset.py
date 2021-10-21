import os
from data.base_dataset import BaseDataset, get_transform, get_transform_for_petct
from data.image_folder import make_dataset, is_image_file
import random
import numpy as np
import torch 

"""
TODO: need to be fullfill 
"""


class PetctUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # create a path '/path/to/data/train'

        self.AB_paths = sorted(self._make_dataset(self.dir_AB, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_size = len(self.AB_paths)  # get the size of dataset A
        self.B_size = len(self.AB_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform_for_petct(self.opt)
        self.transform_B = get_transform_for_petct(self.opt)
        print(self.transform_A, self.transform_B)
        print(self.dir_AB)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = os.path.join(self.dir_AB,f"CT__{self.AB_paths[index % self.A_size]}.npz")
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = os.path.join(self.dir_AB,f"PET__{self.AB_paths[index_B]}.npz")
        A = np.load(A_path)['data'].astype(np.float32) 
        B = np.load(B_path)['data'].astype(np.float32)
        print("Befor transform ", A.max(), A.min(), A.mean(), B.max(), B.min(), B.mean())
        # apply image transformation
        A = self.transform_A(A) 
        B = self.transform_B(B)
        print("After transform ", A.max(), A.min(), A.mean(), B.max(), B.min(), B.mean())


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    
    def _make_dataset(self, dir,  max_dataset_size):
        images = set()
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    # path = os.path.join(root, fname)
                    name = fname.split("__")[-1].replace(".npz", "") # CT__UID_IDX.jpg
                    images.add(name)

        return list(images)[:min(max_dataset_size, len(images))]
