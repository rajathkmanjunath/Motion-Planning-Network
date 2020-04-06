import pickle
import numpy as np

ws_location = '/home/rajath/project_workspace/src/panda_arm_motion_planning/dataset/'

with open(ws_location+'plan0.pickle', 'rb+') as depickle_file:
    df = pickle.load(depickle_file, encoding='latin1')
    print(df)
depickle_file.close()