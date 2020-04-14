import argparse
import numpy as np
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils import data

from utils.mpnet import MPNet
from utils.plan_class import plan_dataset


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

    point_cloud = np.load(os.path.join(args.path, 'point_cloud.npy'))
    point_cloud = torch.from_numpy(point_cloud)

    if (args.cuda == 'cuda'):
        point_cloud = point_cloud.cuda().float()

    training_set = plan_dataset(partition['train'], args.path)
    train_loader = data.DataLoader(training_set, **train_params)

    test_set = plan_dataset(partition['test'], args.path)
    test_loader = data.DataLoader(test_set, **test_params)

    mse = nn.MSELoss()
    planner = MPNet(88920, 14, 7)

    if (args.cuda == 'cuda'):
        planner = planner.cuda()

    parameters = planner.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    n_total_steps = len(train_loader)

    for epoch in range(args.num_epochs):
        for i, (states, plan) in enumerate(train_loader):
            states = states.float()
            plan = plan.float()
            states = Variable(states)
            ones = torch.ones(states.shape[0], 1)

            if (args.cuda == 'cuda'):
                states = states.cuda()
                plan = plan.cuda()
                ones = ones.cuda()

            pc = point_cloud * ones

            prediction = planner(pc, states)
            loss = mse(plan, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) % 1 == 0):
                print('epoch {0}/{1}, step {2}/{3}, loss = {4:4f}'.format(epoch + 1, args.num_epochs, i + 1,
                                                                          n_total_steps,
                                                                          loss.item()))

    torch.save(planner.state_dict(), os.path.join(os.path.curdir, 'end_to_end_weights.pt'))

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for (states, plan) in test_loader:
            states = states.float()
            plan = plan.float()
            ones = torch.ones(states.shape[0], 1)

            if (args.cuda == 'cuda'):
                states = states.cuda()
                plan = plan.cuda()
                ones = ones.cuda()

            pc = point_cloud * ones

            prediction = planner(pc, states)
            print(prediction[0], plan[0])
            n_samples += plan.shape[0]
            n_correct = (abs(prediction - plan) <= 0.01).sum().item()

        acc = 100.0 * n_correct / n_samples
        print('accuracy = {0}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../dataset', help='location of dataset directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_files', type=int, default=10000, help='num of files')
    parser.add_argument('--num_workers', type=int, default=6, help='number of sub processes for loading data')
    parser.add_argument('--lam', type=float, default=0.001, help='lambda value for the CAE network')
    parser.add_argument('--cuda', type=str, default='cuda', help='Cuda for processing the network')
    args = parser.parse_args()
    main(args)

