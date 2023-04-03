import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import utils.deformation as deformation
from utils.deformation import shuffle_remap, affine,non_affine_2d


class ImageDataset(Dataset):
    def __init__(self, root, transforms_, opt):
        self.transforms = transforms.Compose(transforms_)
        
        # # Only use one modal data for evaluator training
        self.files_A = sorted(glob.glob("%s/T1/*" % root))
        self.opt = opt
        self.affine = affine
        self.non_affine = non_affine_2d

    def __getitem__(self, index):
        
        item_A = self.transforms(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))  # RA

        # make different spatial affine transform  T1 and T2 
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_A_1 = self.affine(random_numbers=random_numbers, imgs=[item_A], padding_modes=['border'], opt=self.opt)
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_A_2 = self.affine(random_numbers=random_numbers, imgs=[item_A], padding_modes=['border'], opt=self.opt)



        # make different spatial deformation transform  T1 and T2 
        item_A_1 = self.non_affine(imgs=[item_A_1], padding_modes=['border'], opt=self.opt)
        item_A_2 = self.non_affine(imgs=[item_A_2], padding_modes=['border'], opt=self.opt)
        
        
       
        label = (item_A_2 + 1) / 2 - (item_A_1 + 1) / 2  # make (A2-A1 ) as label and keep range (-1,1)
        # add shuffle remap
        item_A_1_noise = shuffle_remap(item_A_1)
  
        return item_A_1_noise, item_A_2 ,label

    def __len__(self):
        return len(self.files_A)
    

    
# test imse for image2image translation   
class TestDataset(Dataset):
    def __init__(self, root, transforms_, opt):
        self.transforms = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob("%s/T1/*" % root))
        self.files_B = sorted(glob.glob("%s/T2/*" % root))
        self.opt = opt
        self.affine = affine
        self.non_affine = non_affine_2d
        
    def __getitem__(self, index):
        ### B1 is the source image 
        ###  A1 is label
        #### A2 is reference image
        
        item_A = self.transforms(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))  # RA
        item_B = self.transforms(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))  # RB
        
        # 
        # keep A1,B1 same spatial
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_A_1,item_B_1 = self.affine(random_numbers=random_numbers, imgs=[item_A,item_B], padding_modes=['border','border'], opt=self.opt)
        item_A_1,item_B_1 = self.non_affine(imgs=[item_A_1,item_B_1], padding_modes=['border','border'], opt=self.opt)
        
        
        #  make sure A2 has different spatial with A1
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_A_2 = self.affine(random_numbers=random_numbers, imgs=[item_A], padding_modes=['border'], opt=self.opt)
        item_A_2= self.non_affine(imgs=[item_A_2], padding_modes=['border'], opt=self.opt)
       
  
        return item_A_1,item_B_1,item_A_2



    def __len__(self):
        return len(self.files_A)   
    