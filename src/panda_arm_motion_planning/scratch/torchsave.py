import torch
import numpy as np

ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/MPNet/dataset/'


for i in range(10):
    ar =  np.random.randint(1,100, size=5)
    t = torch.from_numpy(ar)
    torch.save(t, ws_location+'torch'+str(i)+'.pt')