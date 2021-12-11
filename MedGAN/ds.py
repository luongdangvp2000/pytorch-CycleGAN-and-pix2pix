import torch.utils.data as data
import os.path
import numpy as np
import random
import torch
import albumentations as A
import cv2
import torchvision.transforms as transforms
random.seed(0)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
    '.npz',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Albumentations:
    def __init__(self, augmentations):
        self.augmentations = A.Compose(augmentations)
    
    def __call__(self, image):
        image = self.augmentations(image=image)['image']
        return image

def get_transform_for_petct(preprocess='resize_and_crop', load_size=256, crop_size=256, no_flip=False):
    # parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    transform_list = []
    # load_size=256 #scale images to this size
    # crop_size=256 #'then crop to this size')
    # no_flip=False #', action='store_true', help='if specified, do not flip the images for data augmentation')

    if 'resize' in preprocess:
        transform_list.append(A.Resize(load_size, load_size, interpolation=cv2.INTER_NEAREST))
    if 'crop' in preprocess:
        transform_list.append(A.RandomCrop(crop_size, crop_size))
    if not no_flip:
        transform_list.append(A.HorizontalFlip())

    return transforms.Compose([
        Albumentations(transform_list),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img / img.max()),
        transforms.Normalize ((0.5,), (0.5,))
        ])

class PairedImages_w_nameList(data.Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, root1, root2, flist1, flist2, transform1=None, transform2=None, do_aug=False):
        self.root1 = root1
        self.root2 = root2
        self.flist1 = flist1
        self.flist2 = flist2
        self.transform1 = transform1
        self.transform2 = transform2
        self.do_aug = do_aug
    def __getitem__(self, index):
        impath1 = self.flist1[index]
        img1 = np.load(os.path.join(self.root1, impath1)) #load numpy array
        impath2 = self.flist2[index]
        img2 = np.load(os.path.join(self.root2, impath2))
        if self.transform1 is not None:
            img1 = self.transform1(img1)
            img2 = self.transform2(img2)
        if self.do_aug:
            p1 = random.random()
            if p1<0.5:
                img1, img2 = torch.fliplr(img1), torch.fliplr(img2)
            p2 = random.random()
            if p2<0.5:
                img1, img2 = torch.flipud(img1), torch.flipud(img2)
        return img1, img2
    def __len__(self):
        return len(self.flist1)

        

class PairedImages_PETCT(data.Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, dataroot, phase, max_dataset_size):
        self.dir_AB = os.path.join(dataroot, phase)  # get the image directory
        self.AB_paths = sorted(self._make_dataset(self.dir_AB, max_dataset_size))  # get image paths
        # print(self.AB_paths)
        
    def __getitem__(self, index):

        A_path = os.path.join(self.dir_AB,f"CT__{self.AB_paths[index]}.npz")
        B_path = os.path.join(self.dir_AB,f"PET__{self.AB_paths[index]}.npz")
        A = np.load(A_path)['data'] 
        B = np.load(B_path)['data'] 

        A_transform = get_transform_for_petct()
        B_transform = get_transform_for_petct()

        A = A_transform(A)
        B = B_transform(B)
        return A, B

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

def create_dataset(dataroot, phase, max_dataset_size):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(dataroot, phase, max_dataset_size)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, dataroot, phase, max_dataset_size):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        max_dataset_size = float("inf")
        batch_size = 1
        serial_batches = True
        num_threads = 2

        self.dataset = PairedImages_PETCT(dataroot, phase, max_dataset_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), float("inf"))

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # if i * batch_size >= max_dataset_size
            if i * 1 >= float("inf"):
                break
            yield data

if __name__ == "__main__":
    dataroot = "/home/anhtrangchip/code/lab/petct/data/suv_npz_split_stratify_threshold"
    phase = "val"
    max_dataset_size = float("inf")
    AB = PairedImages_PETCT(dataroot, phase, max_dataset_size)
    for i, batch in enumerate(AB):
        print(batch[0])
        # print(np.max(batch[0]))
        # print(np.min(batch[0]))
        print(torch.max(batch[0]))
        print(torch.min(batch[0]))
        exit(0)
