import os
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_for_petct
from data.image_folder import make_dataset, is_image_file
import numpy as np
import torch 


class PetctAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the number of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=1, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(self._make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = os.path.join(self.dir_AB,f"CT__{self.AB_paths[index]}.npz")
        B_path = os.path.join(self.dir_AB,f"PET__{self.AB_paths[index]}.npz")
        A = np.load(A_path)['data'] 
        B = np.load(B_path)['data'] 

        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, B.shape)
        A_transform = get_transform_for_petct(self.opt, transform_params)
        B_transform = get_transform_for_petct(self.opt, transform_params)

        A = A_transform(A)
        B = B_transform(B)                                                                                                                   

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

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

