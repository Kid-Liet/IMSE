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
from .utils import create_affine_transformation_matrix,shuffle_remap,histgram_shift


class Transformer_3D_cpu(nn.Module):
    def __init__(self):
        super(Transformer_3D_cpu, self).__init__()

    def forward(self, src, flow):
        b = flow.shape[0]
        d = flow.shape[2]
        h = flow.shape[3]
        w = flow.shape[4]
        size = (d, h, w)
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1, 1)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        warped = F.grid_sample(
            src, new_locs, align_corners=True, padding_mode="border")

        return warped



class ImageDataset(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        self.files_root = sorted(glob.glob("%s/*" % root))
        self.opt = opt
    
    
    def _NonAffine(self,imgs, elastic_random=None):
        if elastic_random is None:
            elastic_random = torch.rand([3,imgs[0].shape[2],imgs[0].shape[3],imgs[0].shape[4]]).numpy()*2-1#.numpy()
            
        sigma =  self.opt["gaussian_smoothing"]        #需要根据图像大小调整
        alpha =  self.opt["non_affine_alpha"]  #需要根据图像大小调整
        
        dz = gaussian_filter(elastic_random[0], sigma) * alpha
        dx = gaussian_filter(elastic_random[1], sigma) * alpha
        dy = gaussian_filter(elastic_random[2], sigma) * alpha
        
        dz = np.expand_dims(dz, 0)
        dx = np.expand_dims(dx, 0)
        dy = np.expand_dims(dy, 0)
        
        flow = np.concatenate((dz,dx,dy), 0)
        flow = np.expand_dims(flow, 0)
        flow = torch.from_numpy(flow).to(torch.float32)
        
        res_img = []
        for i in range(len(imgs)):
            res_img.append(Transformer_3D_cpu()(imgs[i],flow))
        return res_img
    
    def _Affine(self, random_numbers,imgs):
        D, H, W = imgs[0].shape[2:]
        n_dims = 3
        tmp = np.ones(3)
        tmp[0:3] = random_numbers[0:3]
        scaling = tmp * self.opt['scaling'] + 1
        tmp[0:3] = random_numbers[3:6]
        rotation = tmp * self.opt['rotation']
        tmp[0:2] = random_numbers[6:8]
        tmp[2] = random_numbers[8]/2
        translation = tmp * self.opt['translation'] 
        theta = create_affine_transformation_matrix(
            n_dims=n_dims, scaling=scaling, rotation=rotation, shearing=None, translation=translation)
        theta = theta[:-1, :]
        theta = torch.from_numpy(theta).to(torch.float32)
        size = torch.Size((1, 1, D, H, W))
        grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True)
        
        res_img = []
        for i in range(len(imgs)):
            res_img.append(F.grid_sample(imgs[i], grid, align_corners=True, padding_mode="border"))
        return res_img
      
    def __getitem__(self, index):
        patients = self.files_root[index % len(self.files_root)]
        
        # Only use one modal data for evaluator training
        item_A = self.transforms(np.load(patients+"/T1.npy").astype(np.float32)) 
        item_B = self.transforms(np.load(patients+"/T2.npy").astype(np.float32)) 
       
        #### make different affine to A and B
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A = self._Affine(random_numbers=random_numbers,imgs = [item_A])[0]
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_B = self._Affine(random_numbers=random_numbers,imgs = [item_B])[0]

        
        ############ 
        # make different non-affine to A and B
        item_A = self._NonAffine(imgs = [item_A])[0].squeeze(0)
        #keep same deformation for A and B
        item_B = self._NonAffine(imgs = [item_B])[0].squeeze(0)

      
       
        return item_A, item_B
    
    
    
    
    
    def __len__(self):
        return len(self.files_root)

    
    
    
    




##############test################
class TestDataset(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        
        self.files_root = sorted(glob.glob("%s/*" % root))#[0:10]
        self.opt = opt
    def _NonAffine(self,imgs, elastic_random=None):
        if elastic_random is None:
            elastic_random = torch.rand([3,imgs[0].shape[2],imgs[0].shape[3],imgs[0].shape[4]]).numpy()*2-1#.numpy()
            
        sigma =  12#需要根据图像大小调整
        alpha =  300 #需要根据图像大小调整
        
        dz = gaussian_filter(elastic_random[0], sigma) * alpha
        dx = gaussian_filter(elastic_random[1], sigma) * alpha
        dy = gaussian_filter(elastic_random[2], sigma) * alpha
        
        dz = np.expand_dims(dz, 0)
        dx = np.expand_dims(dx, 0)
        dy = np.expand_dims(dy, 0)
        
        flow = np.concatenate((dz,dx,dy), 0)
        flow = np.expand_dims(flow, 0)
        flow = torch.from_numpy(flow).to(torch.float32)
        
        res_img = []
        for i in range(len(imgs)):
            res_img.append(Transformer_3D_cpu()(imgs[i],flow))
        return res_img

    
    def _Affine(self, random_numbers,imgs):
        D, H, W = imgs[0].shape[2:]
        n_dims = 3
        tmp = np.ones(3)
        tmp[0:3] = random_numbers[0:3]
        scaling = tmp * self.opt['scaling'] + 1
        tmp[0:3] = random_numbers[3:6]
        rotation = tmp * self.opt['rotation']
        tmp[0:2] = random_numbers[6:8]
        tmp[2] = 0
        translation = tmp * self.opt['translation'] 
        theta = create_affine_transformation_matrix(
            n_dims=n_dims, scaling=scaling, rotation=rotation, shearing=None, translation=translation)
        theta = theta[:-1, :]
        theta = torch.from_numpy(theta).to(torch.float32)
        size = torch.Size((1, 1, D, H, W))
        grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True)
        
        res_img = []
        for i in range(len(imgs)):
            res_img.append(F.grid_sample(imgs[i], grid, align_corners=True, padding_mode="border"))
        return res_img
 
    def __getitem__(self, index):
         # make a seed with numpy generator 
        item_FA = torch.zeros((1,1,1))
        item_FB = torch.zeros((1,1,1))
        patients = self.files_root[index % len(self.files_root)]
        patients_data = sorted(glob.glob("%s/*" % patients))  # RA RB LA LB
        item_RA =  self.transforms(np.load(patients+"/A.npy").astype(np.float32))  # RA
        item_RB =  self.transforms(np.load(patients+"/B.npy").astype(np.float32))  # RB
        item_Mask_A = self.transforms(np.load(patients+"/Label_A.npy").astype(np.float32)) # MASK
        item_Mask_B = self.transforms(np.load(patients+"/Label_B.npy").astype(np.float32)) # MASK
        item_Mask_A[item_Mask_A > 0] = 1
        item_Mask_B[item_Mask_B > 0] = 1
        
        
        if self.opt['mode'] == "Gen":
           
            item_FA = self.transforms(np.load(patients+"/FT1.npy").astype(np.float32))
            item_FB = self.transforms(np.load(patients+"/FT2.npy").astype(np.float32))
            if self.opt['affine']:
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RA, item_FB, item_Mask_A = self._Affine(random_numbers=random_numbers,imgs = [item_RA, item_FB, item_Mask_A])
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RB, item_FA, item_Mask_B = self._Affine(random_numbers=random_numbers,imgs = [item_RB, item_FA, item_Mask_B])
            
            item_RA, item_FB, item_Mask_A = self._NonAffine(imgs = [item_RA, item_FB, item_Mask_A])
            item_RB, item_FA, item_Mask_B = self._NonAffine(imgs = [item_RB, item_FA, item_Mask_B])
 
               
#         if self.opt['affine']:
#             random_numbers = torch.rand(8).numpy() * 2 - 1
#             item_RA ,item_Mask_A = self._Affine(random_numbers=random_numbers,imgs = [item_RA, item_Mask_A])
#             random_numbers = torch.rand(8).numpy() * 2 - 1
#             item_RB, item_Mask_B = self._Affine(random_numbers=random_numbers,imgs = [item_RB, item_Mask_B])

#         ############ 
#         # deformation
#         item_RA, item_Mask_A = self._NonAffine(imgs = [item_RA,item_Mask_A])
#         item_RB, item_Mask_B = self._NonAffine(imgs = [item_RB,item_Mask_B])

                
        return item_RA.squeeze(0), item_RB.squeeze(0), item_FA.squeeze(0), item_FB.squeeze(0), item_Mask_A.squeeze(0), item_Mask_B.squeeze(0)
    
    
    
    
    def __len__(self):
        return len(self.files_root)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##############test################
class TestDataset2(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        
        self.files_root = sorted(glob.glob("%s/*" % root))#[0:10]
        self.opt = opt
    def _NonAffine(self,imgs, elastic_random=None):
        if elastic_random is None:
            elastic_random = torch.rand([3,imgs[0].shape[2],imgs[0].shape[3],imgs[0].shape[4]]).numpy()*2-1#.numpy()
            
        sigma =  12#需要根据图像大小调整
        alpha =  300 #需要根据图像大小调整
        
        dz = gaussian_filter(elastic_random[0], sigma) * alpha
        dx = gaussian_filter(elastic_random[1], sigma) * alpha
        dy = gaussian_filter(elastic_random[2], sigma) * alpha
        
        dz = np.expand_dims(dz, 0)
        dx = np.expand_dims(dx, 0)
        dy = np.expand_dims(dy, 0)
        
        flow = np.concatenate((dz,dx,dy), 0)
        flow = np.expand_dims(flow, 0)
        flow = torch.from_numpy(flow).to(torch.float32)
        
        res_img = []
        for i in range(len(imgs)):
            res_img.append(Transformer_3D_cpu()(imgs[i],flow))
        return res_img

    
    def _Affine(self, random_numbers,imgs):
        D, H, W = imgs[0].shape[2:]
        n_dims = 3
        tmp = np.ones(3)
        tmp[0:3] = random_numbers[0:3]
        scaling = tmp * self.opt['scaling'] + 1
        tmp[0:3] = random_numbers[3:6]
        rotation = tmp * self.opt['rotation']
        tmp[0:2] = random_numbers[6:8]
        tmp[2] = 0
        translation = tmp * self.opt['translation'] 
        theta = create_affine_transformation_matrix(
            n_dims=n_dims, scaling=scaling, rotation=rotation, shearing=None, translation=translation)
        theta = theta[:-1, :]
        theta = torch.from_numpy(theta).to(torch.float32)
        size = torch.Size((1, 1, D, H, W))
        grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True)
        
        res_img = []
        for i in range(len(imgs)):
            res_img.append(F.grid_sample(imgs[i], grid, align_corners=True, padding_mode="border"))
        return res_img
 
    def __getitem__(self, index):
         # make a seed with numpy generator 
        item_FA = torch.zeros((1,1,1))
        item_FB = torch.zeros((1,1,1))
        patients = self.files_root[index % len(self.files_root)]
        patients_data = sorted(glob.glob("%s/*" % patients))  # RA RB LA LB
        item_RA =  self.transforms(np.load(patients+"/A.npy").astype(np.float32)).unsqueeze(0) # RA
        item_RB =  self.transforms(np.load(patients+"/B.npy").astype(np.float32)).unsqueeze(0)   # RB
        item_Mask_A = self.transforms(np.load(patients+"/Label_A.npy").astype(np.float32)).unsqueeze(0)  #[:,:,:,:,-2].unsqueeze(0)  # MASK
        item_Mask_B = self.transforms(np.load(patients+"/Label_B.npy").astype(np.float32)).unsqueeze(0)  #[:,:,:,:,-2].unsqueeze(0) # MASK
        #print (item_RA.shape,item_Mask_A.shape)
       
#         item_Mask_A[item_Mask_A > 0] = 1
#         item_Mask_B[item_Mask_B > 0] = 1
        
        
        if self.opt['mode'] == "Gen":
            item_FA = self.transforms(np.load(patients+"/FT1.npy").astype(np.float32)).unsqueeze(0)
            item_FB = self.transforms(np.load(patients+"/FT2.npy").astype(np.float32)).unsqueeze(0)
            
            if self.opt['affine']:
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RA, item_FB, item_Mask_A = self._Affine(random_numbers=random_numbers,imgs = [item_RA, item_FB, item_Mask_A])
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RB, item_FA, item_Mask_B = self._Affine(random_numbers=random_numbers,imgs = [item_RB, item_FA, item_Mask_B])
            
            item_RA, item_FB, item_Mask_A = self._NonAffine(imgs = [item_RA, item_FB, item_Mask_A])
            item_RB, item_FA, item_Mask_B = self._NonAffine(imgs = [item_RB, item_FA, item_Mask_B])
 
               
#         if self.opt['affine']:
#             random_numbers = torch.rand(8).numpy() * 2 - 1
#             item_RA ,item_Mask_A = self._Affine(random_numbers=random_numbers,imgs = [item_RA, item_Mask_A])
#             random_numbers = torch.rand(8).numpy() * 2 - 1
#             item_RB, item_Mask_B = self._Affine(random_numbers=random_numbers,imgs = [item_RB, item_Mask_B])

#         ############ 
#         # deformation
#         item_RA, item_Mask_A = self._NonAffine(imgs = [item_RA,item_Mask_A])
#         item_RB, item_Mask_B = self._NonAffine(imgs = [item_RB,item_Mask_B])

                
        return item_RA.squeeze(0), item_RB.squeeze(0), item_FA.squeeze(0), item_FB.squeeze(0), item_Mask_A.squeeze(0), item_Mask_B.squeeze(0)
    
    
    
    
    def __len__(self):
        return len(self.files_root)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    