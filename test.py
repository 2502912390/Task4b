import torch
import numpy as np
from torch.utils.data import Dataset

class maestroDataset(Dataset):
    def __init__(self, data, target):
        """Initialize the dataset loading."""
        self.data = data
        self.target = target
    def __len__(self) -> int:
        return self.data.shape[0]
    def __getitem__(self, i):
        # print(f'get item: {i}')
        data = self.data[i].astype(np.float32)
        target = self.target[i].astype(np.float32)

        return data, target

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs)]
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data

def preprocess_data(_X, _Y, _X_val, _Y_val, _seq_len):# （17 29648 64）
    X = []
    for i in range(17):# 遍历每一个类别 对每一个类别进行划分
        X_cls = split_in_seqs(_X[i], _seq_len)#(148 200 64)
        X.append(X_cls)
    _X = np.stack(X, axis=1) #(148 17 200 64)

    X_val = []
    for i in range(17):
        X_val_cls = split_in_seqs(_X_val[i], _seq_len)
        X_val.append(X_val_cls)
    _X_val = np.stack(X_val, axis=1)

    _Y = split_in_seqs(_Y, _seq_len)
    _Y_val = split_in_seqs(_Y_val, _seq_len)

    return _X, _Y, _X_val, _Y_val

if __name__ == '__main__':

    X = torch.randn(17,29648,64)
    Y = torch.randn(29648,17)
    X_val = torch.randn(17,29648,64)
    Y_val = torch.randn(29648,17)
    X = X.numpy()
    Y = Y.numpy()
    X_val = X_val.numpy()
    Y_val = Y_val.numpy()

    #(148, 17, 200, 64)  (148, 200, 17)
    X, Y, X_val, Y_val = preprocess_data(X, Y, X_val, Y_val, 200)

    train_dataset = maestroDataset(X, Y)
    validate_dataset = maestroDataset(X_val, Y_val)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=3, shuffle=True,
                                                num_workers=1, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=3, shuffle=True,
                                                num_workers=1, pin_memory=True)

    for (batch_data, batch_target) in train_loader:
        print(batch_data.shape) #torch.Size([bs, 17, 200, 64])
        print(batch_target.shape) #torch.Size([3, 200, 17])