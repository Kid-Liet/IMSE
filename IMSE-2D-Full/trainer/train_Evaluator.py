#!/usr/bin/python3

import os
from torch.utils.data import DataLoader
import torch
from utils.utils import ToTensor, Resize
from utils.utils import Logger
from .Eva_datasets import ImageDataset,TestDataset
from model.Eva_model import Evaluator
from model.Gen_model import Discriminator
import cv2
import numpy as np
from PIL import Image
import SimpleITK as sitk
from torch.autograd import Variable

    

class Eva_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        # def networks
        self.net_E = Evaluator(config['input_nc'], config['output_nc'], ndims=config['dim']).cuda()
        self.optimizer_E = torch.optim.Adam(self.net_E.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        self.loss = torch.nn.L1Loss()
        self.transforms = [
            ToTensor(),
            Resize(size_tuple=(config['size']))
        ]

        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], transforms_=self.transforms, opt=config),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'],drop_last = True)
        
        self.testdataloader = DataLoader(
            TestDataset(config['testroot'], transforms_=self.transforms, opt=config),
            shuffle=False,batch_size=1, num_workers=0)

        self.logger = Logger(config['name'], config['port'], config['n_epochs'] - config['epoch'], len(self.dataloader))
        
    def train(self):
        # Training
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for A1_noise,A2,Label in self.dataloader:
                A1_noise = A1_noise.cuda()
                A2 = A2.cuda()
                Label = Label.cuda()
                
                self.optimizer_E.zero_grad()
                
                inputs = torch.cat([A1_noise, A2], 1)
                pred = self.net_E(inputs)  # 
                loss_E  = self.loss(pred, Label)
                loss_E .backward()
                self.optimizer_E.step()

                self.logger.log({'L_AB': loss_E, },
                images={'A1_noise': A1_noise, 'A2': A2,'Label': Label,'pred': pred})
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])

            torch.save(self.net_E.state_dict(), self.config['save_root'] + 'net_%s.pth' % self.config['name'])
            
    def test_translation(self):
        # test  translation 
        self.net_E.load_state_dict(torch.load(self.config['save_root'] + 'net_%s.pth' % self.config['name']))
        num = 0
        self.net_E.eval()
        
        
        # B1 translation to A1
        ### B1 is the source image 
        ###  A1 is label
        #### A2 is reference image
        for A1,B1,A2 in self.testdataloader:
            A1 = A1.cuda()
            B1 = B1.cuda()
            A2 = A2.cuda()    
            inputs = torch.cat([B1, A2], 1)
            pred = self.net_E(inputs)# 
            fake_A1 = A2-2*pred   # (reference image) - (pred) = translated_image
            fake_A1 = torch.clamp(fake_A1,-1,1)
            
            A1 = A1.squeeze().detach().cpu().numpy()
            B1 = B1.squeeze().detach().cpu().numpy()
            A2 = A2.squeeze().detach().cpu().numpy()
            fake_A1 = fake_A1.squeeze().detach().cpu().numpy()
            
            cv2.imwrite("/data/klk/BraTS2019/visual/A1/"+str(num)+".png", (A1+1)*127.5)
            cv2.imwrite("/data/klk/BraTS2019/visual/B1/"+str(num)+".png", (B1+1)*127.5)
            cv2.imwrite("/data/klk/BraTS2019/visual/A2/"+str(num)+".png", (A2+1)*127.5)
            cv2.imwrite("/data/klk/BraTS2019/visual/fake_A1/"+str(num)+".png", (fake_A1+1)*127.5)
            
            num += 1
