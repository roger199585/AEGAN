import torch
import torch.nn as nn
from .autoencoder import Interpolate

class Simulator(nn.Module):
    def __init__(self, input_dim, cnum):
        super(Simulator, self).__init__()
        self.input_dim = input_dim
        self.cnum = cnum
        
        self.block1 = nn.Sequential(
            # conv1
            nn.Conv2d(self.input_dim + 1, self.cnum, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.cnum),
            nn.LeakyReLU(0.2, inplace=True),

            # conv2
            nn.Conv2d(self.cnum, self.cnum, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.cnum),
            nn.LeakyReLU(0.2, inplace=True),

            # conv3
            nn.Conv2d(self.cnum, self.cnum*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.cnum*2),
            nn.LeakyReLU(0.2, inplace=True),

            # conv4
            nn.Conv2d(self.cnum*2, self.cnum*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.cnum*2),
            nn.LeakyReLU(0.2, inplace=True),

            # conv5
            nn.Conv2d(self.cnum*2, self.cnum*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.cnum*4),
            nn.LeakyReLU(0.2, inplace=True),

            # conv6
            nn.Conv2d(self.cnum*4, self.cnum*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.cnum*4),
            nn.LeakyReLU(0.2, inplace=True),

            # conv7
            nn.Conv2d(self.cnum*4, self.cnum*4, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.cnum*4),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(128, 128), mode='bilinear'),
            # conv8
            nn.Conv2d(self.cnum*4, self.cnum*2, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.cnum*2),
            nn.LeakyReLU(0.2, inplace=True),

            # conv9
            nn.Conv2d(self.cnum*2, self.cnum, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.cnum),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(256, 256), mode='bilinear'),

            # conv10
            nn.Conv2d(self.cnum, 3, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        x = self.block1(x)

        x = torch.clamp(x, -1., 1.)
        return x