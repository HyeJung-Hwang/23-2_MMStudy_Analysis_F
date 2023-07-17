import torch
from torch.utils.data import Dataset
from torch import Tensor

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[1,2,3],[4,5,6],[7,8,9]]
        self.y_data = [[12],[18],[11]]
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
