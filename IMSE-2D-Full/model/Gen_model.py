import torch.nn as nn
from .base_model import BaseModel, ResidualBlock


class Generator(BaseModel):
    def __init__(self, input_nc, output_nc, ndims, ngf=64, n_residual_blocks=9):
        super().__init__(ndims)

        # Initial convolution block
        model_head = [self.pad(3),
                      self.conv(input_nc, ngf, 7),
                      self.norm(ngf),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model_head += [self.conv(in_features, out_features, 3, stride=2, padding=1),
                           self.norm(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features, ndims)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [
                self.transpose(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                self.norm(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [self.pad(3),
                       self.conv(ngf, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        return x


class Discriminator(BaseModel):
    def __init__(self, input_nc, ndims):
        super().__init__(ndims)

        # A bunch of convolutions one after another
        model = [self.conv(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [self.conv(64, 128, 4, stride=2, padding=1),
                  self.norm(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [self.conv(128, 256, 4, stride=2, padding=1),
                  self.norm(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [self.conv(256, 512, 4, padding=1),
                  self.norm(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [self.conv(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return self.pool(x, x.size()[2:]).view(x.size()[0], -1)
