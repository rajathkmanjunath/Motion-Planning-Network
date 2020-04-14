import numpy as np
import os

import torch
from torch.utils.data import Dataset


class plan_dataset(Dataset):
    def __init__(self, list_IDs, path):
        self.list_IDs = list_IDs
        self.path = path

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        states = np.load(os.path.join(self.path, 'states' + str(ID) + '.npy'))
        plan = np.load(os.path.join(self.path, 'plan' + str(ID) + '.npy'))
        states = torch.from_numpy(states)
        plan = torch.from_numpy(plan)
        return states, plan
