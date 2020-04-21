import torch.nn as nn


class ENet(nn.Module):
    def __init__(self, input_size):
        super(ENet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 28800),
                                     nn.PReLU(),
                                     nn.Linear(28800, 3200),
                                     nn.PReLU(),
                                     nn.Linear(3200, 1024),
                                     nn.PReLU(),
                                     nn.Linear(1024, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 256))

        self.decoder = nn.Sequential(nn.Linear(256, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 1024),
                                     nn.PReLU(),
                                     nn.Linear(1024, 3200),
                                     nn.PReLU(),
                                     nn.Linear(3200, 28800),
                                     nn.PReLU(),
                                     nn.Linear(28800, input_size))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2
