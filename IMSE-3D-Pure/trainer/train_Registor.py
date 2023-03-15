#!/usr/bin/python3

import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from .utils import LambdaLR,Logger,ReplayBuffer,ToTensor,Resize3D
import torch.nn.functional as F
from .utils import Logger
import numpy as np
from .Reg_datasets import ImageDataset,TestDataset,TestDataset2
from model.Eva_model import Evaluator
from model.Reg_model import VxmDense

from .utils import Transformer_3D,smooth_loss,HD


class Reg_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.net_R = VxmDense(cfg = config).cuda()
        self.trans = Transformer_3D().cuda()
        self.optimizer_R = torch.optim.Adam(self.net_R.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.net_E = Evaluator(config['input_nc'], config['output_nc']).cuda()
        self.net_E.load_state_dict(torch.load(self.config['evaluator_root']))
        
   
        self.transforms_1 = [ ToTensor(),
                       Resize3D(size_tuple = (config['size']))
                       ]
        
        self.dataloader = DataLoader(ImageDataset(config['dataroot'],transforms_=self.transforms_1,
                           opt = config, unaligned=False),batch_size=config['batchSize'],shuffle=True,num_workers=config['n_cpu'])
        
        
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))

    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):

            for A,B in self.dataloader:
                # use B replace the A1_noise 
                A = A.cuda()
                B = B.cuda()
                Depth = A.shape[2]
                self.optimizer_R.zero_grad()
               
                # A regist to B
                flow = self.net_R(A, B)
                A_warp = self.trans(A, flow)
                error_map = self.net_E(torch.cat([B, A_warp], 1))
                error_map = torch.abs(error_map)
                
                # B regist to A
                flow_ = self.net_R(B, A)
                B_warp = self.trans(B, flow_)
                error_map_ = self.net_E(torch.cat([B_warp, A], 1))
                error_map_ = torch.abs(error_map_)
                
                # loss
                loss_sim = self.config["sim_w"] *(torch.mean(error_map)+torch.mean(error_map_))                             
                loss_smooth = self.config["smooth_w"] * (smooth_loss(flow)+smooth_loss(flow_))
                loss_reg = loss_sim + loss_smooth
   
                
                loss_reg.backward()
                self.optimizer_R.step()
                
                

                self.logger.log({'L_Sim': loss_sim, 'L_Smooth': loss_smooth},
                           images={'A': A[0,:,int(Depth/2),:,:], 'B': B[0,:,int(Depth/2),:,:],                                                               'A_warp': A_warp[0,:,int(Depth/2),:,:],'B_warp': B_warp[0,:,int(Depth/2),:,:],
                                 'error_map': error_map[0,:,int(Depth/2),:,:],
                                 'error_map_': error_map_[0,:,int(Depth/2),:,:]})
            torch.save(self.net_R.state_dict(), self.config['save_root'] + 'Registration.pth')

 
            
    def test_regisatration(self):
        self.net_R.load_state_dict(torch.load(self.config['save_root'] + 'Registration.pth'))
        with torch.no_grad():
                #the dice and hd95 befor deformation registration 
                Befor_DICE,Befor_HD= 0, 0 
                #the result of after registration
                A2B_DICE, B2A_DICE = 0, 0, 
                A2B_Jac, B2A_Jac = 0, 0, 
                A2B_HD95, B2A_HD95 = 0, 0
                num = 0
                for RA, RB, FA, FB, A_mask, B_mask in self.test_dataloader:
                    RA = RA.cuda()
                    RB = RB.cuda()
                    A_mask = A_mask.cuda()
                    B_mask = B_mask.cuda()
                    befor_mask = self.cal_dice(A_mask, B_mask)
                    befor_hd = HD(A_mask.squeeze(), B_mask.squeeze())
                    print (befor_mask)
                    if self.config['mode'] == "Gen":
                        FA = FA.cuda()
                        FB = FB.cuda()
                        flow = self.net_R(FB, RB)  # RA 2 FB-->RB
                        flow_ = self.net_R(FA, RA)  # RB 2 FA-->RA

                    else:
                      
                        flow = self.net_R(RA, RB)
                        flow_ = self.net_R(RB, RA)
                        
                    a2bjdet = smooth_loss(flow)
                    b2ajdet = smooth_loss(flow_)
                    a2bnum_jet = jacobian_determinant(flow)    
                    b2anum_jet = jacobian_determinant(flow_)
                    
                    
                    warp2B_mask = self.trans(A_mask, flow)
                    warp2A_mask = self.trans(B_mask, flow_)
                    pos_dice = self.cal_dice(warp2B_mask, B_mask)
                    neg_dice = self.cal_dice(warp2A_mask, A_mask) 
                    
                    a2b_HD95 = HD(warp2B_mask.squeeze(), B_mask.squeeze())
                    b2a_HD95= HD(warp2A_mask.squeeze(), A_mask.squeeze())

                    
                    Befor_DICE += befor_mask
                    Befor_HD += befor_hd
                    Pos_DICE += pos_dice
                    Neg_DICE += neg_dice 
                    A2B_Jac += a2bjdet
                    B2A_Jac += b2ajdet
                    A2Bnum_jet += a2bnum_jet
                    B2Anum_jet += b2anum_jet
                    
                    
                    A2B_HD95 += a2b_HD95
                    B2A_HD95 += b2a_HD95

                    num += 1
                    
                    
                    
                
                print ('Befor DC:',Befor_DICE/num)
                print ('Befor HD:',Befor_HD/num)
                print ('A2B DC:',Pos_DICE/num)
                print ('B2A DC:',Neg_DICE/num)
                print ('A2B HD95:',A2B_HD95/num)
                print ('B2A HD95:',B2A_HD95/num)
                print ('A2B Jac:',A2B_Jac/num, A2Bnum_jet/num)
                print ('B2A Jac:',B2A_Jac/num, B2Anum_jet/num)
                
               
                
                
    
###################################
if __name__ == '__main__':
    main()