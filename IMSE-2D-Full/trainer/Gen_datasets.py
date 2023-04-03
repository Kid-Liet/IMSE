import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.deformation import shuffle_remap, affine,non_affine_2d


class ImageDataset(Dataset):
    def __init__(self, root, transforms_, opt):
        self.transforms = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob("%s/T1/*" % root))
        self.files_B = sorted(glob.glob("%s/T2/*" % root))
        self.opt = opt
        self.affine = affine
        self.non_affine = non_affine_2d


    def __getitem__(self, index):
        item_A = self.transforms(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        item_B = self.transforms(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_A_ = self.affine(random_numbers=random_numbers, imgs=[item_A], padding_modes=['border'], opt=self.opt)
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_B_ = self.affine(random_numbers=random_numbers, imgs=[item_B], padding_modes=['border'], opt=self.opt)
        item_A_ = self.non_affine(imgs=[item_A_], padding_modes=['border'], opt=self.opt)
        item_B_= self.non_affine(imgs=[item_B_], padding_modes=['border'], opt=self.opt)
        return item_A_, item_B_

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class TestDataset(Dataset):
    def __init__(self, config):
        self.files_A = sorted(glob.glob("%s/T1/*" % root))
        self.files_B = sorted(glob.glob("%s/T2/*" % root))   
    def __getitem__(self, index):
        item_A = torch.from_numpy(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)).unsqueeze(0)
        item_B = torch.from_numpy(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.load(self.files_mask[index % len(self.files_mask)]).astype(np.float32)).unsqueeze(0)
        return {'A': item_A, 'B': item_B, 'mask': mask}#

    def __len__(self):
        return len(self.files_A)
