import torch.nn as nn
import torch.nn.functional as F
import functools
import torch
import numpy as np
class BaseModel(nn.Module):
    def __init__(self, ndims):
        super(BaseModel, self).__init__()

        assert ndims in [2, 3], 'ndims should be one of 2, or 3. found: %d' % ndims
        #InstanceNorm2d
        self.conv = getattr(nn, 'Conv%dd' % ndims)
        self.pad = getattr(nn, 'ReflectionPad%dd' % ndims)
        self.norm = getattr(nn, 'InstanceNorm%dd' % ndims)
        self.pool = getattr(F, 'avg_pool%dd' % ndims)
        self.transpose = getattr(nn, 'ConvTranspose%dd' % ndims)

class BaseModel2(nn.Module):
    def __init__(self, ndims):
        super(BaseModel2, self).__init__()

        assert ndims in [2, 3], 'ndims should be one of 2, or 3. found: %d' % ndims

        self.conv = getattr(nn, 'Conv%dd' % ndims)
        self.pad = getattr(nn, 'ReflectionPad%dd' % ndims)
        self.norm = getattr(nn, 'InstanceNorm%dd' % ndims)
        self.pool = getattr(F, 'avg_pool%dd' % ndims)
        self.transpose = getattr(nn, 'ConvTranspose%dd' % ndims)
        
class ResidualBlock(BaseModel):
    def __init__(self, in_features, ndims):
        super().__init__(ndims)

        conv_block = [
            self.pad(1),
            self.conv(in_features, in_features, 3),
            self.norm(in_features),
            nn.ReLU(inplace=True),
            self.pad(1),
            self.conv(in_features, in_features, 3),
            self.norm(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    
    
    