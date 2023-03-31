
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock3D(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock3D, self).__init__()

        conv_block = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_features, in_features, 3),
            nn.InstanceNorm3d(in_features),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_features, in_features, 3),
            nn.InstanceNorm3d(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Evaluator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_residual_blocks=9):
        super(Evaluator, self).__init__()

        # Initial convolution block
        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_nc, ngf, 7),
                 nn.InstanceNorm3d(ngf),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock3D(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.Upsample(scale_factor=2, mode='trilinear'),
                        nn.ReplicationPad3d(1), 
                        nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=0, bias=False),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(True),
                        
                      ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReplicationPad3d(3),
                  nn.Conv3d(ngf, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)