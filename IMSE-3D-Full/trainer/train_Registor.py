#!/usr/bin/python3

import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from .utils import LambdaLR,Logger,ReplayBuffer,ToTensor,Resize3D,Crop3D
import torch.nn.functional as F
from .utils import Logger
import numpy as np
from .Reg_datasets import ImageDataset,TestDataset
from model.Eva_model import Evaluator
from model.Reg_model import VxmDense
from .utils import Transformer_3D,smooth_loss, MIND, MI, NCC,neg_Jdet_loss,HD,jacobian_determinant


class Reg_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.net_R = VxmDense(cfg = config).cuda()
        self.trans = Transformer_3D().cuda()
        self.optimizer_R = torch.optim.Adam(self.net_R.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['mode'] == "Gen":
            self.sim_loss = torch.nn.L1Loss()
        elif config['mode'] == "Eva":
            self.net_E = Evaluator(config['input_nc'], config['output_nc']).cuda()
            self.net_E.load_state_dict(torch.load(self.config['evaluator_root']))
        ####################### Multimodal operator:  MIND   NCC   MI
        else:
            if config['sim'] == "MIND":
                self.sim_loss = MIND().loss
            elif config['sim'] == "NCC":
                self.sim_loss = NCC().loss
            elif config['sim'] == "MI":
                self.sim_loss = MI().loss
            else:
                self.sim_loss = torch.nn.L1Loss()
   
        self.transforms_1 =  [
                       ToTensor(),
                       Resize3D(size_tuple = (config['depth'],config['size'], config['size']))
                       ]
      

        self.dataloader = DataLoader(ImageDataset(config['dataroot'],transforms_=self.transforms_1,opt = config, unaligned=False),   batch_size=config['batchSize'],shuffle=True, num_workers=config['n_cpu'])
        
        self.test_dataloader = DataLoader(TestDataset(config['testroot'],transforms_=self.transforms_1,opt = config, unaligned=False),                         batch_size=1,shuffle=False, num_workers=config['n_cpu'])
        
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))
    
    def cal_dice(self, A, B):
        A = A.round()
        B = B.round()
        num = A.size(0)
        A_flat = A.view(num, -1)
        B_flat = B.view(num, -1)
        inter = (A_flat * B_flat).sum(1)
        return (2.0 * inter) / (A_flat.sum(1) + B_flat.sum(1))#,(A_flat.sum(1) + B_flat.sum(1))
    
    def cal_dice_intersection(self, A, B):
        A = A.round()
        B = B.round()
        num = A.size(0)
        A_flat = A.view(num, -1)
        B_flat = B.view(num, -1)
        inter = (A_flat * B_flat).sum(1)
        return inter
    
    
    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for A,B in self.dataloader:
                A = A.cuda()
                B = B.cuda()
                Depth = A.shape[2]
                self.optimizer_R.zero_grad()
                flow = self.net_R(A, B)
                warp = self.trans(A, flow)

                
                flow_ = self.net_R(B, A)
                warp_ = self.trans(B, flow_)
                # loss
                loss_smooth = self.config["smooth_w"] * (smooth_loss(flow)+smooth_loss(flow_))
          
                
                
                if self.config['mode'] == "Eva":
                    
                    error_map = self.net_E(torch.cat([B,warp], 1))
                    
                    error_map_ = self.net_E(torch.cat([warp_,A], 1))
                    
                    error_map = torch.abs(error_map)
                    error_map_ = torch.abs(error_map_)
                    
                    loss_sim = torch.mean(error_map) + torch.mean(error_map_)
                    

                    
                else:
                    
                    loss_sim = self.config["sim_w"] * self.sim_loss(warp,B)
                
                loss_reg = loss_sim + loss_smooth
                loss_reg.backward()
                self.optimizer_R.step()


                self.logger.log({'L_Sim': loss_sim, 'L_Smooth': loss_smooth},
                           images={'mov': A[0,:,int(Depth/2),:,:], 'fix': B[0,:,int(Depth/2),:,:],                                                 'warp': warp[0,:,int(Depth/2),:,:]})#'error_map': error_map[0,:,int(Depth/2),:,:]
            
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
                
            if self.config['mode'] == "Gen":
                torch.save(self.net_R.state_dict(), self.config['save_root'] + 'net_R_Gen.pth')
            elif self.config['mode'] == "Eva":
                torch.save(self.net_R.state_dict(), self.config['save_root'] + 'net_R_Eva.pth')
            else:
                torch.save(self.net_R.state_dict(), self.config['save_root'] + 'net_R_Tra_'+self.config['sim']+'.pth')

            
    def test_all(self):
        # choose registration pth
        self.net_R.load_state_dict(torch.load(self.config['save_root'] + 'net_R_Eva.pth'))
        self.net_R.eval()
        
        Befor_DICE,Befor_HD= 0, 0 
        #the result of after registration
        A2B_DICE, B2A_DICE = 0, 0, 
        A2B_Jac, B2A_Jac = 0, 0, 
        A2B_HD95, B2A_HD95 = 0, 0
        
        for RA, RB, A_mask, B_mask in self.test_dataloader:
            RA = RA.cuda()
            RB = RB.cuda()
            A_mask = A_mask.cuda()
            B_mask = B_mask.cuda()
            
            befor_mask = self.cal_dice(A_mask, B_mask)
            befor_hd = HD(A_mask.squeeze(), B_mask.squeeze())
            



            flow = self.net_R(RA, RB)
            flow_ = self.net_R(RB, RA)

            warp2B_mask = self.trans(A_mask, flow)
            warp2A_mask = self.trans(B_mask, flow_)

            a2bjdet = smooth_loss(flow)
            b2ajdet = smooth_loss(flow_)

            a2b_dice = self.cal_dice(warp2B_mask, B_mask)
            b2a_dice = self.cal_dice(warp2A_mask, A_mask) 

            a2b_HD95 = HD(warp2B_mask.squeeze(), B_mask.squeeze())
            b2a_HD95= HD(warp2A_mask.squeeze(), A_mask.squeeze())


            Befor_DICE += befor_mask
            Befor_HD += befor_hd

            A2B_DICE += a2b_dice
            B2A_DICE += b2a_dice 

            A2B_Jac += a2bjdet
            B2A_Jac += b2ajdet



            A2B_HD95 += a2b_HD95
            B2A_HD95 += b2a_HD95

            num += 1




        print ('Befor DC:',Befor_DICE/num)
        print ('Befor HD:',Befor_HD/num)
        print ('A2B DC:',A2B_DICE/num)
        print ('B2A DC:',B2A_DICE/num)
        print ('A2B HD95:',A2B_HD95/num)
        print ('B2A HD95:',B2A_HD95/num)
        print ('A2B Jac:',A2B_Jac)
        print ('B2A Jac:',B2A_Jac)

if __name__ == '__main__':
    main()