import torch
from torch.utils.data import Dataset
import os

class points_dataset(Dataset):
    def __init__(self, list_IDs, path):
        self.list_IDs = list_IDs
        self.path = path

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        x = torch.load(os.path.join(self.path,'points'+str(ID)+'.pt'))
        return x
