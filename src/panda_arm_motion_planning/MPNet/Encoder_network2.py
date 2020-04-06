import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
from torch.utils.data import Dataset
from torch.utils import data
import numpy as np
import pickle
from dataset_class import points_dataset
from torch.backends import cudnn
import os
import argparse
from cautoencoder import CAE

# ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/MPNet/dataset/'

# CUDA for PyTorch
mse_loss = nn.MSELoss()
autoencoder = CAE()

def combined_loss_function(W, x, h, recons_x, lam):
    mse = mse_loss(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(Variable(W) ** 2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)

def main(args):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    # Parameters
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    # max_epochs = 100
    # numfiles = 200

    partition = {'train':[i+1 for i in range(int(0.9*args.num_files))],
                 'test':[i+1 for i in range(int(0.9*args.num_files), args.num_files)]}

    training_set = points_dataset(partition['train'], args.path)
    train_loader = data.DataLoader(training_set, **params)
    #
    # batch_size = 16
    # num_epochs = 100
    # learning_rate = 0.001

# train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size,
#                                            shuffle=True)

# class CAE(nn.Module):
#     def __init__(self):
#         super(CAE, self).__init__()
#         self.encoder = nn.Sequential(nn.Linear(115200, 512),
#                                      nn.PReLU(),
#                                      nn.Linear(512, 128),
#                                      nn.PReLU(),
#                                      nn.Linear(128, 32))
#
#         self.decoder = nn.Sequential(nn.Linear(32, 128),
#                                      nn.PReLU(),
#                                      nn.Linear(128, 512),
#                                      nn.PReLU(),
#                                      nn.Linear(512, 115200))
#
#     def forward(self, x):
#         h1 = self.encoder(x)
#         h2 = self.decoder(h1)
#         return h1, h2

    if(args.cuda == 'cuda'):
        autoencoder.cuda()

    parameters = autoencoder.parameters()
    optimizer = torch.optim.SGD(parameters, lr = args.learning_rate)

    n_total_steps = len(train_loader)

    for epoch in range(args.num_epochs):
        for i, points in enumerate(train_loader):
            points = points.float()
            points = Variable(points)

            if(args.cuda == 'cuda'):
                points = points.cuda()
            encoded, decoded = autoencoder(points)
            W = autoencoder.state_dict()['encoder.4.weight']
            loss = combined_loss_function(W,points,encoded,decoded,args.lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % 1 == 0):
                print('epoch {0}/{1}, step {2}/{3}, loss = {4:4f}'.format(epoch + 1, args.num_epochs, i + 1, n_total_steps,
                                                                      loss.item()))

        torch.save(autoencoder.state_dict(), os.path.curdir+'weights.pt')

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for points in test_loader:
#         points = points.float()
#         if(torch.cuda.is_available()):
#             points = points.cuda()
#         _, decoded = autoencoder(points)
#         n_samples += labels.shape[0]
#         n_correct = (decoded == points).sum().item()
#
#     acc = 100.0 * n_correct / n_samples
#     print('accuracy = {0}'.format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/scratch/$USER/dataset/', help='location of dataset directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_files', type=int, default=50000, help='num of files')
    parser.add_argument('--num_workers', type=int, default=6, help='number of sub processes for loading data')
    parser.add_argument('--lam', type=float, default=0.001, help='lambda value for the CAE network')
    parser.add_argument('--cuda', type=str, default='cuda', help='Cuda for processing the network')
    args = parser.parse_args()
    main(args)