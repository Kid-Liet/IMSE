import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn
import torch.nn.functional as F
from .utils import shuffle_remap
from .utils import _Affine, _NonAffine

class ImageDataset(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        self.files_root = sorted(glob.glob("%s/*" % root))
        self.opt = opt
        self._Affine = _Affine
        self._NonAffine = _NonAffine
    def __getitem__(self, index):
        patients = self.files_root[index % len(self.files_root)]
        
       
        item_A = self.transforms(np.load(patients+"/T1.npy").astype(np.float32)) 
        item_B = self.transforms(np.load(patients+"/T2.npy").astype(np.float32)) 
        #### make different affine to A and B
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A = self._Affine(random_numbers=random_numbers,imgs = [item_A],padding_modes=['border'],opt = self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_B = self._Affine(random_numbers=random_numbers,imgs = [item_B],padding_modes=['border'],opt = self.opt)
        ############ 
        # make different non-affine to A and B
        item_A = self._NonAffine(imgs = [item_A],padding_modes=['border'],opt = self.opt)
        #keep same deformation for A and B
        item_B = self._NonAffine(imgs = [item_B],padding_modes=['border'],opt = self.opt)
        return item_A, item_B
    def __len__(self):
        return len(self.files_root)

    
    
    
    




##############test################
class TestDataset(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        self.files_root = sorted(glob.glob("%s/*" % root))[:3]
        self.opt = opt
        self._Affine = _Affine
        self._NonAffine = _NonAffine
    
    def __getitem__(self, index):
        patients = self.files_root[index % len(self.files_root)]
        
        # Only use one modal data for evaluator training
        item_A = self.transforms(np.load(patients+"/T1.npy").astype(np.float32)) 
        item_B = self.transforms(np.load(patients+"/T2.npy").astype(np.float32)) 

        item_Mask_A = self.transforms(np.load(patients+"/label.npy").astype(np.float32)) # MASK
        item_Mask_B = self.transforms(np.load(patients+"/label.npy").astype(np.float32)) # MASK
        item_Mask_A[item_Mask_A > 0] = 1
        item_Mask_B[item_Mask_B > 0] = 1

        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A ,item_Mask_A = self._Affine(random_numbers=random_numbers,imgs = [item_A, item_Mask_A], 
                                            padding_modes=['border', 'zeros'],opt=self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_B, item_Mask_B = self._Affine(random_numbers=random_numbers,imgs = [item_B, item_Mask_B],
                                            padding_modes=['border', 'zeros'],opt=self.opt)
        
#         ############ 
#         # deformation
        item_A, item_Mask_A = self._NonAffine(imgs = [item_A,item_Mask_A],padding_modes=['border', 'zeros'],opt=self.opt)
        item_B, item_Mask_B = self._NonAffine(imgs = [item_B,item_Mask_B],padding_modes=['border', 'zeros'],opt=self.opt)
       
        return item_A, item_B, item_Mask_A, item_Mask_B
    
    
    
    
    def __len__(self):
        return len(self.files_root)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    