import torch
from torch import nn


class MPNet(nn.Module):
    def __init__(self, pc_size, states_size, output_size):
        super(MPNet, self).__init__()
        self.ENet = nn.Sequential(nn.Linear(pc_size, 512),
                                  nn.PReLU(),
                                  nn.Linear(512, 128),
                                  nn.PReLU(),
                                  nn.Linear(128, 32))

        self.PNet = nn.Sequential(nn.Linear(states_size + 32, 1280), nn.PReLU(), nn.Dropout(),
                                  nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
                                  nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
                                  nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
                                  nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                                  nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
                                  nn.Linear(64, 32), nn.PReLU(), nn.Dropout(),
                                  nn.Linear(32, output_size), nn.PReLU(), nn.Dropout())

    def forward(self, point_cloud, states):
        Z = self.ENet(point_cloud)
        ones = torch.ones(states.shape[0], 1)
        # ones = ones.cuda()
        Z = Z * ones
        # print(Z.shape, states.shape)
        S = torch.cat((Z, states), dim=1)
        return self.PNet(S)
