import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

from utils.mypath import path_data
from utils.gaussian_blur import GaussianBlur

# Code from https://github.com/sthalles/PyTorch-BYOL/blob/master/data/multi_view_data_injector.py
class MultiViewDataInjector():
    def __init__(self, args, dict_transform):
        self.transforms = dict_transform
        self.args = args

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transforms]
        return output

# Code from https://github.com/facebookresearch/moco/blob/master/moco/loader.py
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class load_data():
    def __init__(self, args, split, transform):
        self.args = args
        self.split = split
        self.transform = transform

    def load_dataset(self):
        data_set = ImageFolder(root=f'{path_data + self.split}',
                               transform=self.transform)
        return data_set


# augmentation set
# color jitter
s = 1
color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)

# covid augmentation
augmentation_dictionary = {
    'covid': transforms.Compose(
        [transforms.RandomResizedCrop(size=(480,480), ratio=(0.3, 1.0)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([color_jitter], p=0.8),
         transforms.RandomRotation(degrees= 15),
         # transforms.RandomGrayscale(p=0.2),
         # GaussianBlur(kernel_size=int(0.1*480)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.469, 0.469, 0.469],
                              std=[0.285, 0.285, 0.285])]
    ),
}