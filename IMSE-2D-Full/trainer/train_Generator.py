#!/usr/bin/python3
import itertools
import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from model.Reg_model import Reg
from utils.utils import ReplayBuffer, ToTensor, Resize, Logger
from .Gen_datasets import ImageDataset, TestDataset
from model.Gen_model import Generator, Discriminator
import cv2
from utils.deformation import Transformer2D
from utils.utils import smooth_loss
class Gen_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        # def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc'], ndims=config['dim']).cuda()
        self.netG_B2A = Generator(config['input_nc'], config['output_nc'], ndims=config['dim']).cuda()
        self.netD_A = Discriminator(config['input_nc'], ndims=config['dim']).cuda()
        self.netD_B = Discriminator(config['input_nc'], ndims=config['dim']).cuda()
        self.net_R = Reg().cuda()
        self.trans = Transformer2D().cuda()
        self.transforms = [
            ToTensor(),
            Resize(size_tuple=config['size'])
        ]

        
        self.optimizer_G_A = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_G_B = torch.optim.Adam(self.netG_B2A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
      
        self.optimizer_R = torch.optim.Adam(self.net_R.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        # Loss
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.target_real = Variable(Tensor(config['batchSize'], 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(config['batchSize'], 1).fill_(0.0), requires_grad=False)
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], transforms_=self.transforms, opt=config),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'],drop_last = True)
        self.logger = Logger('%s_%dD' % (config['name'], config['dim']), config['port'], config['n_epochs'],
                             len(self.dataloader))

    def train(self):
        # Training
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for A, B in self.dataloader:
                real_A = A.cuda()
                real_B = B.cuda()
                ####training CycleGan
                if 'CycleGAN' in self.config['name']:   
                    self.optimizer_G_A.zero_grad()
                    self.optimizer_G_B.zero_grad()
                    # GAN loss
                    fake_B = self.netG_A2B(real_A)
                    pred_fake = self.netD_B(fake_B)
                    loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                    fake_A = self.netG_B2A(real_B)
                    pred_fake = self.netD_A(fake_A)
                    loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                    # Cycle loss
                    recovered_A = self.netG_B2A(fake_B)
                    loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)
                    recovered_B = self.netG_A2B(fake_A)
                    loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)
                    # idt loss
                    idt_A = self.netG_B2A(real_A)
                    loss_A2A = self.config['Iden_lamda'] * self.L1_loss(idt_A, real_A)
                    idt_B = self.netG_A2B(real_B)
                    loss_B2B = self.config['Iden_lamda'] * self.L1_loss(idt_B, real_B)
                    # Total loss
                    loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_A2A + loss_B2B
                    loss_G.backward()
                    self.optimizer_G_A.step()
                    self.optimizer_G_B.step()
                    ###### Discriminator A ######
                    self.optimizer_D_A.zero_grad()
                    self.optimizer_D_B.zero_grad()
                    # Real loss
                    pred_real = self.netD_A(real_A)
                    loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                    # Fake loss
                    fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                    pred_fake = self.netD_A(fake_A.detach())
                    loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                    loss_D_A = (loss_D_real + loss_D_fake)
                    ###### Discriminator B ######
                    # Real loss
                    pred_real = self.netD_B(real_B)
                    loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                    # Fake loss
                    fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                    pred_fake = self.netD_B(fake_B.detach())
                    loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                    loss_D_B = (loss_D_real + loss_D_fake)
                    # Total loss
                    loss_D = loss_D_A + loss_D_B
                    loss_D.backward()
                    self.optimizer_D_A.step()
                    self.optimizer_D_B.step()
                    self.logger.log({'loss_D': loss_D, 'loss_G': loss_G},
                                    images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B, 'fake_A': fake_A})
               
               
                ####training RegGan
                else:
                    self.optimizer_G_A.zero_grad()
                    self.optimizer_R.zero_grad()
                    #### corr loss
                    fake_B = self.netG_A2B(real_A)
                    Trans = self.net_R(fake_B,real_B) 
                    SysRegist_A2B = self.trans(fake_B,Trans)
                    SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                    pred_fake0 = self.netD_B(fake_B)
                    adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                    ####smooth loss
                    SM_loss = self.config['Smooth_lamda'] * smooth_loss(Trans)
                    toal_loss = SM_loss+adv_loss+SR_loss
                    toal_loss.backward()
                    self.optimizer_R.step()
                    self.optimizer_G_A.step()
                    
                    
                    self.optimizer_D_B.zero_grad()
                    with torch.no_grad():
                        fake_B = self.netG_A2B(real_A)
                    pred_fake0 = self.netD_B(fake_B)
                    pred_real = self.netD_B(real_B)
                    loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] *                                       self.MSE_loss(pred_real, self.target_real)
                    loss_D_B.backward()
                    self.optimizer_D_B.step()
                    self.logger.log({'loss_D_B': loss_D_B,'SR_loss':SR_loss},
                           images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})                
    
            
            torch.save(self.netG_A2B.state_dict(),
                           self.config['save_root'] + 'netG_A2B_' + str(self.config['dim']) + 'D.pth')