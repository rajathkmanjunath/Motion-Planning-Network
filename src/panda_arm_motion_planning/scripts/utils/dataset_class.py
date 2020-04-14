import numpy as np
import os

import torch
from torch.utils.data import Dataset


class points_dataset(Dataset):
    def __init__(self, list_IDs, path):
        self.list_IDs = list_IDs
        self.path = path

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        x = np.load(os.path.join(self.path, 'points' + str(ID) + '.npy'))
        x = torch.from_numpy(x)
        return x
