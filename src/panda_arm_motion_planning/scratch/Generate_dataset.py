import torch
import torch.utils.data as data
import pickle
import glob
import numpy as np

ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/MPNet/dataset/'

npd = []

for file in glob.glob(ws_location+'points*.npy'):
    narray = np.load(file)
    npd.append(narray)

dbfile = open(ws_location+'pickle_test.pickle', 'ab+')
pickle.dump(npd, dbfile)
dbfile.close()