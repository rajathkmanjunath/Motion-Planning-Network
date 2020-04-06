import torch

ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/MPNet/dataset/'

for i in range(10):
    loc = ws_location+'torch'+str(i)+'.pt'
    print(torch.load(loc))