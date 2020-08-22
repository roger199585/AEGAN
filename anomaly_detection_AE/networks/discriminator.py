import torch
import torch.nn as nn
from .autoencoder import Interpolate

class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            torch.nn.utils.spectral_norm(
                nn.Conv2d(inputSize, hiddenSize, kernel_size=5, stride=2, padding=2, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True),

            torch.nn.utils.spectral_norm(
                nn.Conv2d(hiddenSize, hiddenSize*2, kernel_size=5, stride=2, padding=2, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True),

            torch.nn.utils.spectral_norm(
                nn.Conv2d(hiddenSize*2, hiddenSize*4, kernel_size=5, stride=2, padding=2, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True),

            torch.nn.utils.spectral_norm(
                nn.Conv2d(hiddenSize*4, hiddenSize*4, kernel_size=5, stride=2, padding=2, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(hiddenSize*4*16*16, 1),
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x