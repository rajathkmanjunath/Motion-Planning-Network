import argparse
import numpy as np
import os
import random
import time
import torch


def main(args):
        start_time = time.time()
        # xrange = (-100, 100)
        # yrange = (-100, 100)
        # zrange = (-100, 100)
        # step_size = 0.01

        for index in range(args.num_samples):
            lis = []
            for i in range(8):
                x = random.randint(args.xrange[0], args.xrange[1]) * args.step_size
                y = random.randint(args.yrange[0], args.yrange[1]) * args.step_size
                z = random.randint(args.zrange[0], args.zrange[1]) * args.step_size
                xd, yd, zd = 0.4, 0.4, 0.1
                for i in np.arange(x - xd / 2, x + xd / 2, args.step_size):
                    for j in np.arange(y - yd / 2, y + yd / 2, args.step_size):
                        lis.append(np.array([i, j, z - zd / 2]))
                        lis.append(np.array([i, j, z + zd / 2]))

                for i in np.arange(x - xd / 2, x + xd / 2, args.step_size):
                    for j in np.arange(z - zd / 2, z + zd / 2, args.step_size):
                        lis.append(np.array([i, y - yd / 2, j]))
                        lis.append(np.array([i, y + yd / 2, j]))

                for i in np.arange(y - yd / 2, y + yd / 2, args.step_size):
                    for j in np.arange(z - zd / 2, z + zd / 2, args.step_size):
                        lis.append(np.array([x - xd / 2, i, j]))
                        lis.append(np.array([x + xd / 2, i, j]))
                pt = torch.Tensor(lis[:38400]).view(-1)
            if(not os.path.exists(args.path)):
                os.makedirs(args.path)
            torch.save(pt, os.path.join(args.path,'points'+str(index+1)+'.pt'))

        print("the time of execution is: {0}".format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/scratch/$USER/dataset/', help='Location for storing dataset')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples to generate')
    parser.add_argument('--xrange', type=tuple, default=(-100, 100), help='The range of x')
    parser.add_argument('--yrange', type=tuple, default=(-100, 100), help='The range of y')
    parser.add_argument('--zrange', type=tuple, default=(-100, 100), help='The range of z')
    parser.add_argument('--step_size', type=float, default=0.001, help='step size of samples')

    args = parser.parse_args()
    main(args)
