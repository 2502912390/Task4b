import numpy as np
from torch.utils.data import Dataset


class maestroDataset(Dataset):
    def __init__(self, data, target, data_lass):
        """Initialize the dataset loading."""
        self.data = data
        self.target = target
        self.data_lass = data_lass
    

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i):
        #print(f'get item: {i}')
        data = self.data[i].astype(np.float32)
        target = self.target[i].astype(np.float32)
        data_lass = self.data_lass[i].astype(np.float32)
        return data, target, data_lass


