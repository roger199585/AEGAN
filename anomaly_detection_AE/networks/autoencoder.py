import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.channel_mult = 16

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.channel_mult, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult, self.channel_mult*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult*2, self.channel_mult*3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult*3),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.channel_mult*3, self.channel_mult*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult*2, self.channel_mult, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult, self.channel_mult // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(self.channel_mult / 2)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult // 2, self.channel_mult // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_mult // 4, self.channel_mult // 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult // 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            Interpolate(size=(6, 6), mode='bilinear'),
            nn.Conv2d(self.channel_mult // 8, self.channel_mult // 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult // 4),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(10, 10), mode='bilinear'),
            nn.Conv2d(self.channel_mult // 4, self.channel_mult // 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult // 2),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(18, 18), mode='bilinear'),
            nn.Conv2d(self.channel_mult // 2, self.channel_mult, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(34, 34), mode='bilinear'),
            nn.Conv2d(self.channel_mult, self.channel_mult * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(66, 66), mode='bilinear'),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(130, 130), mode='bilinear'),
            nn.Conv2d(self.channel_mult, self.channel_mult // 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult // 2),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(258, 258), mode='bilinear'),
            nn.Conv2d(self.channel_mult // 2, self.channel_mult // 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.channel_mult // 4),
            nn.LeakyReLU(0.2, inplace=True),

            Interpolate(size=(258, 258), mode='bilinear'),
            nn.Conv2d(self.channel_mult // 4, 3, kernel_size=3, stride=1),
            nn.Sigmoid()
        )        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
