import argparse
import numpy as np
import os
import time


# def scaling_factor(value, lower_limit, upper_limit):
#     return (value - lower_limit) * (upper_limit - lower_limit)


def main(args):
    start_time = time.time()
    xd, yd, zd = int(0.4 / args.step_size), int(0.4 / args.step_size), int(0.1 / args.step_size)
    # print(xd, yd, zd)
    for index in range(args.num_samples):
        lis = []
        for _ in range(6):
            x = np.random.randint(args.xrange[0], args.xrange[1]) * args.step_size
            y = np.random.randint(args.yrange[0], args.yrange[1]) * args.step_size
            z = np.random.randint(args.zrange[0], args.zrange[1]) * args.step_size
            # print(x,y,z)
            for i in np.arange(xd):
                for j in range(yd):
                    xtemp = (x + i * args.step_size) - xd / 2 * args.step_size
                    ytemp = (y + j * args.step_size) - yd / 2 * args.step_size
                    # print(xtemp, ytemp)
                    lis.append(np.array([xtemp, ytemp, z - zd / 2]))
                    lis.append(np.array([xtemp, ytemp, z + zd / 2]))
            for i in range(xd):
                for j in range(zd):
                    xtemp = (x + i * args.step_size) - xd / 2 * args.step_size
                    ztemp = (z + j * args.step_size) - zd / 2 * args.step_size
                    lis.append(np.array([xtemp, y - yd / 2, ztemp]))
                    lis.append(np.array([xtemp, y + yd / 2, ztemp]))
            for i in np.arange(yd):
                for j in np.arange(zd):
                    ytemp = (y + i * args.step_size) - yd / 2 * args.step_size
                    ztemp = (z + j * args.step_size) - zd / 2 * args.step_size
                    lis.append(np.array([x - xd / 2, ytemp, ztemp]))
                    lis.append(np.array([x + xd / 2, ytemp, ztemp]))
        points = np.array(lis).reshape(-1)
        print(points.shape)
        if (not os.path.exists(args.path)):
            os.makedirs(args.path)
        np.save(os.path.join(args.path, 'points' + str(index + 1) + '.npy'), points)

    print("the time of execution is: {0}".format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/scratch/rkm350/enet/dataset/',
                        help='Location for storing dataset')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples to generate')
    parser.add_argument('--xrange', type=tuple, default=(-100, 100), help='The range of x')
    parser.add_argument('--yrange', type=tuple, default=(-100, 100), help='The range of y')
    parser.add_argument('--zrange', type=tuple, default=(-100, 100), help='The range of z')
    parser.add_argument('--step_size', type=float, default=0.01, help='step size of samples')

    args = parser.parse_args()
    main(args)
