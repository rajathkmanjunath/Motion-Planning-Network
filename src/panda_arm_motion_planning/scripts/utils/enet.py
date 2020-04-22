import torch.nn as nn


class ENet(nn.Module):
    def __init__(self, input_size):
        super(ENet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 256),
                                     nn.PReLU(),
                                     nn.Linear(256, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 64),
                                     nn.PReLU(),
                                     nn.Linear(64, 32))

        self.decoder = nn.Sequential(nn.Linear(32, 64),
                                     nn.PReLU(),
                                     nn.Linear(64, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 256),
                                     nn.PReLU(),
                                     nn.Linear(256, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, input_size))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2
