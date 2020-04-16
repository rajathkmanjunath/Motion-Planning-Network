import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils import data

from utils.dataset_class import points_dataset
from utils.enet import ENet


# CUDA for PyTorch
def combined_loss_function(W, mse, lam):
    contractive_loss = torch.sum(Variable(W) ** 2, dim=1).sum().mul_(lam)
    return mse + contractive_loss


def main(args):
    cudnn.benchmark = True
    # Parameters
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': args.num_workers}

    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': args.num_workers}

    partition = {'train': [i + 1 for i in range(int(0.9 * args.num_files))],
                 'test': [i + 1 for i in range(int(0.9 * args.num_files), args.num_files)]}

    training_set = points_dataset(partition['train'], args.path)
    train_loader = data.DataLoader(training_set, **train_params)

    test_set = points_dataset(partition['train'], args.path)
    test_loader = data.DataLoader(training_set, **test_params)

    mse_loss = nn.MSELoss()
    autoencoder = ENet(90036)

    if (args.cuda == 'cuda'):
        autoencoder = autoencoder.cuda()

    parameters = autoencoder.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    n_total_steps = len(train_loader)

    for epoch in range(args.num_epochs):
        for i, points in enumerate(train_loader):
            points = points.float()
            points = Variable(points)

            if (args.cuda == 'cuda'):
                points = points.cuda()

            _, decoded = autoencoder(points)
            mse = mse_loss(decoded, points)

            W = autoencoder.state_dict()['encoder.4.weight']

            loss = combined_loss_function(W, mse, args.lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % 50 == 0):
                print('epoch {0}/{1}, step {2}/{3}, loss = {4:4f}'.format(epoch + 1, args.num_epochs, i + 1,
                                                                          n_total_steps,
                                                                          loss.item()))

        torch.save(autoencoder.state_dict(), os.path.join(os.path.curdir, 'weights.pt'))

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for points in test_loader:
            points = points.float()
            if (torch.cuda.is_available()):
                points = points.cuda()
            _, decoded = autoencoder(points)
            n_samples += points.shape[0]
            n_correct = (decoded == points).sum().item()
            print(points[0][:10], decoded[0][10])

        acc = 100.0 * n_correct / n_samples
        print('accuracy = {0}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/scratch/rkm350/enet/dataset/',
                        help='location of dataset directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_files', type=int, default=50000, help='num of files')
    parser.add_argument('--num_workers', type=int, default=6, help='number of sub processes for loading data')
    parser.add_argument('--lam', type=float, default=0.001, help='lambda value for the CAE network')
    parser.add_argument('--cuda', type=str, default='cuda', help='Cuda for processing the network')
    args = parser.parse_args()
    main(args)
