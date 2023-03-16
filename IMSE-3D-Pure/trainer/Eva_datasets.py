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
        
        # Only use one modal data for evaluator training
        item_A1 = self.transforms(np.load(patients+"/T1.npy").astype(np.float32)) 
        item_A2 = item_A1
       
        #### make different affine to A and B
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A1 = self._Affine(random_numbers=random_numbers,imgs = [item_A1],padding_modes=['border'],opt = self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A2 = self._Affine(random_numbers=random_numbers,imgs = [item_A2],padding_modes=['border'],opt = self.opt)

        
        ############ 
        # make different non-affine to A and B
        item_A1 = self._NonAffine(imgs = [item_A1],padding_modes=['border'],opt = self.opt)
        # make different non-affine to A and B
        item_A2 = self._NonAffine(imgs = [item_A2],padding_modes=['border'],opt = self.opt)
        
        
        label = (item_A2 + 1) / 2 - (item_A1 + 1) / 2  # make (A2-A1 ) as label and keep range (-1,1)
          
        #### add shuffle remap
        item_A1_noise = shuffle_remap(item_A1)
      
       
        return item_A1_noise, item_A2, label
    
    
    
    
    def __len__(self):
        return len(self.files_root)
    
    
    
    


