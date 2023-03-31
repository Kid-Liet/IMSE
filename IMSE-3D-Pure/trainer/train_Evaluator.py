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
from .Eva_datasets import ImageDataset
from model.Eva_model import Evaluator


class Eva_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks

        self.net_E = Evaluator(config['input_nc'], config['output_nc']).cuda()
        self.optimizer_E = torch.optim.Adam(self.net_E.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.loss = torch.nn.L1Loss()
        
        self.transforms_1 = [ ToTensor(),
                       Resize3D(size_tuple = (config['size']))
                       ]
        
        self.dataloader = DataLoader(ImageDataset(config['dataroot'],transforms_=self.transforms_1,
                           opt = config, unaligned=False),batch_size=config['batchSize'],shuffle=True,num_workers=config['n_cpu'])
        
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))

    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            # B is CT, A is other modal images
            for A1_noise,A2,Label in self.dataloader:
                
                A1_noise = A1_noise.cuda()
                A2 = A2.cuda()
                Label = Label.cuda()
                Depth = A1_noise.shape[2]

                self.optimizer_E.zero_grad()

                inputs = torch.cat([A1_noise, A2], 1)
                pred = self.net_E(inputs) 
                
                loss_E = self.loss(pred, Label)
                loss_E.backward()
                self.optimizer_E.step()

                self.logger.log({'loss_E': loss_E,},
                           images={'A1_noise': A1_noise[0,:,int(Depth/2),:,:], 'A2': A2[0,:,int(Depth/2),:,:],                                                 'Label': Label[0,:,int(Depth/2),:,:],'pred':pred[0,:,int(Depth/2),:,:],                                            })
            
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.net_E.state_dict(), self.config['save_root'] + 'Evaluator.pth')
                    

###################################
if __name__ == '__main__':
    main()