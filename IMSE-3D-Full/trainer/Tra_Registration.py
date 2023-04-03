#!/usr/bin/python3

import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from utils import LambdaLR,Logger,ReplayBuffer,ToTensor,Resize3D,Crop3D
import torch.nn.functional as F
from utils import Logger
import numpy as np
from utils import Transformer_3D,smooth_loss, MIND, MI, NCC,neg_Jdet_loss,HD,jacobian_determinant
import yaml
import shutil
import glob,sys
sys.path.append("..")
from model.Eva_model import Evaluator,Evaluator2
#from model.Eva_model import Evaluator
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
class Resize3D():
    def __init__(self, size_tuple, use_cv = True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv


    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor
        tensor = F.interpolate(tensor, size = [self.size_tuple[0],self.size_tuple[1],self.size_tuple[2]],align_corners=True, mode='trilinear')

        #tensor = tensor.squeeze(0)
 
        return tensor


class Tra():
    def registration(self, config):
        self.config = config
        dataset = sorted(glob.glob(os.path.join('%s' % config['testroot'], '*')))[:10]
       
        ####################### Multimodal operator:  MIND   NCC   MI

        if config['sim'] == "MIND":
            self.sim_loss = MIND().loss
        elif config['sim'] == "NCC":
            self.sim_loss = NCC().loss
        elif config['sim'] == "MI":
            self.sim_loss = MI().loss
        elif config['sim'] == "L1":
            self.sim_loss = torch.nn.L1Loss()
        else:
            net_E = Evaluator2(config['input_nc'], config['output_nc']).cuda()
            net_E.load_state_dict(torch.load(self.config['evaluator_root']))
            self.sim_loss = torch.nn.L1Loss()
          
        self.trans = Transformer_3D().cuda()    
        Befor_DICE, DICE , Befor_HD,HD95 ,SMOOTH = 0,0,0,0,0
        num = 0
        
        for i in range(len(dataset)):
        
            RA = torch.from_numpy(np.load(dataset[i]+"/"+"/ct.npy").astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
            RB = torch.from_numpy(np.load(dataset[i]+"/"+"/mr.npy").astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
            A_mask = torch.from_numpy(np.load(dataset[i]+"/"+"/ct_label.npy").astype(np.float32)).cuda()[:,:,:,2].unsqueeze(0).unsqueeze(0)
            B_mask = torch.from_numpy(np.load(dataset[i]+"/"+"/mr_label.npy").astype(np.float32)).cuda()[:,:,:,2].unsqueeze(0).unsqueeze(0)
            RA = F.interpolate(RA, size = [48,128,128],align_corners=True, mode='trilinear')
            RB = F.interpolate(RB, size = [48,128,128],align_corners=True, mode='trilinear')
            A_mask = F.interpolate(A_mask, size = [48,128,128],align_corners=True, mode='trilinear')
            B_mask = F.interpolate(B_mask, size = [48,128,128],align_corners=True, mode='trilinear')
            
          
            
            #A_mask[A_mask > 0] = 1
            #B_mask[B_mask > 0] = 1
            
            displacement = torch.zeros(1,3,48,128,128).float().cuda()
            displacement.requires_grad = True
            optimizer = torch.optim.Adam([displacement],lr = config["lr"])
            
            
            for _ in range(config["max_iter"]):
                optimizer.zero_grad()
                # forward
                
                warp = self.trans(RA, displacement)
                if config['sim'] == "Eva":
                    AB = torch.cat([warp, RB], 1)
                    error_map = net_E(AB)
                    error_map = torch.abs(error_map)
                    loss_sim = torch.mean(error_map)
                else:   
                    loss_sim = self.sim_loss(warp,RB)
                loss_smooth = smooth_loss(displacement)
                total_loss = self.config["sim_w"]*loss_sim+self.config["smooth_w"]*loss_smooth
               
                #print ("loss_sim:",loss_sim)
                #print ("loss_smooth:",loss_smooth)
                total_loss.backward()
                optimizer.step()
            
            displacement.requires_grad = False
            warp_mask = self.trans(A_mask, displacement)
            warp_mask = (warp_mask > 0.5).float()
            befor_dice = self.cal_dice(A_mask, B_mask)
            befor_hd = HD(A_mask.squeeze(), B_mask.squeeze())
            dice = self.cal_dice(warp_mask, B_mask)
            hd95 = HD(warp_mask.squeeze(), B_mask.squeeze())
            print ("befor dc:",i,befor_dice)
            print ("after dc:",i,dice)
            print ("befor hd:",i,befor_hd)
            print ("after hd:",i,hd95)
            print ("smooth:",i,loss_smooth)  
            wqer
            
#             fff
            Befor_DICE += befor_dice
            Befor_HD += befor_hd
            DICE += dice
            HD95 += hd95
            SMOOTH += loss_smooth
            num += 1
            

            

        print ('Befor DC:',Befor_DICE/num)        
        print ('DC:',DICE/num)
        print ('Befor HD:',Befor_HD/num) 
        print ('HD95:',HD95/num)
        print ('SMOOTH:',SMOOTH/num)
        
       
    def cal_dice(self, A, B):
        A = A.round()
        B = B.round()
        num = A.size(0)
        A_flat = A.view(num, -1)
        B_flat = B.view(num, -1)
        inter = (A_flat * B_flat).sum(1)
        return (2.0 * inter) / (A_flat.sum(1) + B_flat.sum(1))
    



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../Yaml/Tra_Reg.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = get_config(opts.config)
print (config)
tradition = Tra()
tradition.registration(config)

    
    
    
    
    
    
    
    
    
    
    