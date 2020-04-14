import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(115200, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 32))

        self.decoder = nn.Sequential(nn.Linear(32, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 115200))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2