import torch.nn as nn


class ENet(nn.Module):
    def __init__(self, input_size):
        super(ENet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 40000),
                                     nn.PReLU(),
                                     nn.Linear(40000, 20000),
                                     nn.PReLU(),
                                     nn.Linear(20000, 10000),
                                     nn.PReLU(),
                                     nn.Linear(10000, 5000),
                                     nn.PReLU(),
                                     nn.Linear(5000, 2500),
                                     nn.PReLU(),
                                     nn.Linear(2500, 1250),
                                     nn.PReLU(),
                                     nn.Linear(1250, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 256),
                                     nn.PReLU(),
                                     nn.Linear(256, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 32))

        self.decoder = nn.Sequential(nn.Linear(32, 128),
                                     nn.PReLU(),
                                     nn.Linear(128, 256),
                                     nn.PReLU(),
                                     nn.Linear(256, 512),
                                     nn.PReLU(),
                                     nn.Linear(512, 1250),
                                     nn.PReLU(),
                                     nn.Linear(1250, 2500),
                                     nn.PReLU(),
                                     nn.Linear(2500, 5000),
                                     nn.PReLU(),
                                     nn.Linear(5000, 10000),
                                     nn.PReLU(),
                                     nn.Linear(10000, 20000),
                                     nn.PReLU(),
                                     nn.Linear(20000, 40000),
                                     nn.PReLU(),
                                     nn.Linear(40000, input_size))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2
