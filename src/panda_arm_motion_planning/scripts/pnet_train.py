import argparse
import numpy as np
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils import data

from utils.plan_class import plan_dataset
from utils.pnet import PNet


def main(args):
    cudnn.benchmark = True
    # Parameters
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': args.num_workers}

    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': args.num_workers}

    partition = {'train': [i + 1 for i in range(args.num_files)],
                 'test': [i + 1 for i in range(int(0.9 * args.num_files), args.num_files)]}

    training_set = plan_dataset(partition['train'], args.path)
    train_loader = data.DataLoader(training_set, **train_params)

    test_set = plan_dataset(partition['test'], args.path)
    test_loader = data.DataLoader(test_set, **test_params)
    lossarr = []
    mse = nn.MSELoss()
    planner = PNet(14, 7).double()
    if (os.path.isfile(os.path.join(os.path.curdir, 'pnet_weights.pt'))):
        planner.load_state_dict(torch.load(os.path.join(os.path.curdir, 'pnet_weights.pt')))

    if (args.cuda == 'cuda'):
        planner.cuda()

    parameters = planner.parameters()
    optimizer = torch.optim.Adagrad(parameters, lr=args.learning_rate)
    n_total_steps = len(train_loader)

    for epoch in range(args.num_epochs):
        for i, (states, plan) in enumerate(train_loader):
            # print(states[:7][0])
            # print(plan[0])
            # states = states.float()
            # plan = plan.float()
            states = Variable(states)

            if (args.cuda == 'cuda'):
                states = states.cuda()
                plan = plan.cuda()

            prediction = planner(states)
            loss = mse(plan, prediction)
            lossarr.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i + 1) == 1):
                print('epoch {0}/{1}, step {2}/{3}, loss = {4:4f}'.format(epoch + 1, args.num_epochs, i + 1,
                                                                          n_total_steps,
                                                                          loss.item()))

    torch.save(np.array(lossarr), os.path.join(os.path.curdir, 'loss.npy'))
    torch.save(planner.state_dict(), os.path.join(os.path.curdir, 'pnet_weights.pt'))

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for (states, plan) in test_loader:
            # states = states.float()
            # plan = plan.float()

            if (args.cuda == 'cuda'):
                states = states.cuda()
                plan = plan.cuda()

            prediction = planner(states)
            print(prediction[0])
            print(plan[0])
            n_samples += plan.shape[0]
            n_correct = ((prediction - plan) ** 2 <= 0.1).sum().item()

        acc = 100.0 * n_correct / n_samples
        print('accuracy = {0}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/scratch/rkm350/pnet',
                        help='location of dataset directory')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_files', type=int, default=10000, help='num of files')
    parser.add_argument('--num_workers', type=int, default=6, help='number of sub processes for loading data')
    parser.add_argument('--cuda', type=str, default='cuda', help='Cuda for processing the network')
    args = parser.parse_args()
    main(args)
