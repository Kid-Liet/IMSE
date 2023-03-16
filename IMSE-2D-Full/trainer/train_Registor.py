#!/usr/bin/python3
import os
from torch.utils.data import DataLoader
import torch
from utils.utils import ToTensor, Resize, Logger, smooth_loss, cal_dice
from .Reg_datasets import ImageDataset, TestDataset,TestDataset2
from model.Eva_model import Evaluator
from model.Reg_model import Reg,Reg2
from model.Gen_model import Generator, Discriminator
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import measure
from utils.deformation import Transformer2D
from utils.utils import smooth_loss, MIND, MI, NCC,neg_Jdet_loss,HD,jacobian_determinant
def maxContour(contours):
    cnt_list = np.zeros(len(contours))
    for i in range(0,len(contours)):
        cnt_list[i] = cv2.contourArea(contours[i])

    max_value = np.amax(cnt_list)
    max_index = np.argmax(cnt_list)
    cnt = contours[max_index]

    return cnt, max_index
class Reg_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        # def networks
        self.trans = Transformer2D().cuda()
        self.net_R = Reg().cuda()
        self.optimizer_R = torch.optim.Adam(self.net_R.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['mode'] == "Gen":
            self.sim_loss = torch.nn.L1Loss()
        elif config['mode'] == "Eva":
            self.net_E = Evaluator(config['input_nc'], config['output_nc'], ndims=config['dim']).cuda()
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

        self.transforms_1 = [
            ToTensor(),
            Resize(size_tuple=(config['size']))
        ]
        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], transforms_=self.transforms_1, opt=config),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'],drop_last = True)
        self.test_dataloader = DataLoader(
            TestDataset(config['testroot'], transforms_=self.transforms_1, opt=config),
            batch_size=1, shuffle=False, num_workers=config['n_cpu'])

        self.logger = Logger(config['name'] + '_' + config['mode'], config['port'], config['n_epochs'],
                             len(self.dataloader))

    def train(self):
        # Training
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for A, B in self.dataloader:
                A = A.cuda()
                B = B.cuda()
                self.optimizer_R.zero_grad()
                flow = self.net_R(A, B)
                A_warp = self.trans(A, flow)
                
                flow_ = self.net_R(B, A)
                B_warp = self.trans(B, flow_)
                # loss
                loss_smooth = self.config["smooth_w"] * (smooth_loss(flow)+smooth_loss(flow_))
                
                if self.config['mode'] == "Eva":

                    error_map = self.net_E(torch.cat([B, A_warp], 1))
                    error_map_ = self.net_E(torch.cat([B_warp, A], 1))
                    
                    error_map = torch.abs(error_map)
                    error_map_ = torch.abs(error_map_)
                    
                    loss_sim = self.config["sim_w"] *(torch.mean(error_map)+torch.mean(error_map_))
                else:
                    
                    loss_sim = self.config["sim_w"] * (self.sim_loss(A_warp,B)+self.sim_loss(A,B_warp))
                    
                    
                loss_reg = loss_sim + loss_smooth
                loss_reg.backward()
                self.optimizer_R.step()
  
                self.logger.log({'L_Sim': loss_sim, 'L_Smooth': loss_smooth},
                images={'A': A, 'A_warp': A_warp, 'B': B,"B_warp":B_warp})
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
                
            if self.config['mode'] == "Gen":
                torch.save(self.net_R.state_dict(), self.config['save_root'] + 'net_R_Gen.pth')
            elif self.config['mode'] == "Eva":
                torch.save(self.net_R.state_dict(), self.config['save_root'] + 'net_R_Eva_.pth')
            else:
                torch.save(self.net_R.state_dict(), self.config['save_root'] + 'net_R_Tra_'+self.config['sim']+'.pth')

    def test_all(self):
        
        self.net_R.load_state_dict(torch.load(self.config['save_root'] + 'net_R_Eva_.pth'))
        with torch.no_grad():
                #the dice and hd95 befor deformation registration 
                Befor_DICE,Befor_HD= 0, 0 
                #the result of after registration
                A2B_DICE, B2A_DICE = 0, 0, 
                A2B_Jac, B2A_Jac = 0, 0, 
                A2B_HD95, B2A_HD95 = 0, 0
                num = 0
                for RA, RB, FB, A_mask, B_mask in self.test_dataloader:
                    if A_mask.max() == 0:
                        continue 
                          
                    RA = RA.cuda()
                    RB = RB.cuda()
                    A_mask = A_mask.cuda()
                    B_mask = B_mask.cuda()
                    befor_mask = cal_dice(A_mask, B_mask)
                    befor_hd = HD(A_mask, B_mask)
                    if self.config['mode'] == "Gen":
                        FB = FB.cuda()
                        flow = self.net_R(FB, RB)  # RA 2 FB-->RB
                        flow_ = self.net_R(RB,FB)  # RB 2 FA-->RA

                    else:
                        flow = self.net_R(RA, RB)
                        flow_ = self.net_R(RB, RA)
                    
                    warp2B_mask = self.trans(A_mask, flow)
                    warp2A_mask = self.trans(B_mask, flow_)   
                    
                    a2bjdet = smooth_loss(flow)
                    b2ajdet = smooth_loss(flow_)

                    a2b_dice = cal_dice(warp2B_mask, B_mask)
                    b2a_dice = cal_dice(warp2A_mask, A_mask) 
                    
                    a2b_HD95 = HD(warp2B_mask, B_mask)
                    b2a_HD95= HD(warp2A_mask, A_mask)

                    
                     
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
        
        

        
        