import argparse
import os
import sys

import torch
from torch.utils import data

from utils.dataset_class import points_dataset
from utils.enet import ENet


def main(args):
    # stdoutOrigin=sys.stdout

    partition = {'test': [i + 1 for i in range(args.num_files)]}

    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': args.num_workers}

    test_set = points_dataset(partition['test'], args.path)
    test_loader = data.DataLoader(test_set, **test_params)

    autoencoder = ENet(3456)

    if (args.cuda == 'cuda'):
        autoencoder = autoencoder.cuda()

    autoencoder.load_state_dict(torch.load(os.path.join(os.curdir, 'enet_weights.pt')))

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for points in test_loader:
            points = points.float()
            if (args.cuda == 'cuda'):
                points = points.cuda()
            _, decoded = autoencoder(points)
            n_samples += points.shape[0]
            n_correct = ((decoded - points) ** 2 < 0.01).sum().item()
            print(points[0][:8].data.tolist())
            print(decoded[0][:8].data.tolist())
            print('-----------------------------------------------------------------------')

        acc = 100.0 * n_correct / (n_samples * points.shape[1])
        print('accuracy = {0}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/scratch/rkm350/enet/dataset/',
                        help='location of dataset directory')
    # parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_files', type=int, default=50000, help='num of files')
    parser.add_argument('--num_workers', type=int, default=6, help='number of sub processes for loading data')
    parser.add_argument('--lam', type=float, default=0.001, help='lambda value for the CAE network')
    parser.add_argument('--cuda', type=str, default='cuda', help='Cuda for processing the network')
    args = parser.parse_args()
    sys.stdout = open("log.txt", "w")
    main(args)
    sys.stdout.close()
