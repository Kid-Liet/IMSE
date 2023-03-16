import torch.nn as nn
from .base_model import BaseModel,ResidualBlock
    
class Evaluator(BaseModel):
    def __init__(self, input_nc, output_nc, ndims, ngf=64, n_residual_blocks=9):
        super().__init__(ndims)

        # Initial convolution block
        model = [self.pad(3),
                 self.conv(input_nc, ngf, 7),
                 self.norm(ngf),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [self.conv(in_features, out_features, 3, stride=2, padding=1),
                      self.norm(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features, ndims)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [   nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                 self.norm(out_features),
                 nn.ReLU(inplace=True)]
                
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [self.pad(3),
                  self.conv(ngf, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)