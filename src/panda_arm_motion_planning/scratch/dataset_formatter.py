import torch
from torch.utils.data import Dataset

class points_dataset(Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs

    def __ len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        x = torch.load(ws_location + 'points'+ID+'.npy')
        return x