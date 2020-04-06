import numpy as np
import random
import torch
import os
import time

start_time = time.time()

xrange = (-50, 59)
yrange = (-50, 50)
zrange = (0, 100)
step_size = 0.01

ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/MPNet/dataset/'

for index in range(5000):
    lis = []
    for i in range(8):
        x= random.randint(xrange[0],xrange[1])/100
        y = random.randint(yrange[0],yrange[1])/100
        z= random.randint(zrange[0], zrange[1])/100
        xd, yd, zd = 0.4, 0.4, 0.1
        for i in np.arange(x - xd / 2, x + xd / 2, step_size):
            for j in np.arange(y - yd / 2, y + yd / 2, step_size):
                lis.append(np.array([i, j, z - zd / 2]))
                lis.append(np.array([i, j, z + zd / 2]))

        for i in np.arange(x - xd / 2, x + xd / 2, step_size):
            for j in np.arange(z - zd / 2, z + zd / 2, step_size):
                lis.append(np.array([i, y - yd / 2, j]))
                lis.append(np.array([i, y + yd / 2, j]))

        for i in np.arange(y - yd / 2, y + yd / 2, step_size):
            for j in np.arange(z - zd / 2, z + zd / 2, step_size):
                lis.append(np.array([x - xd / 2, i, j]))
                lis.append(np.array([x + xd / 2, i, j]))
        pt = torch.Tensor(lis[:38400]).view(-1)
    torch.save(pt, ws_location+'points'+str(index+1)+'.pt')

print("the time of execution is: {0}".format(time.time()-start_time))