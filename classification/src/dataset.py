from torch.utils.data import Dataset
import torch
import wfdb
import numpy as np


class PTBDataset(Dataset):
    def __init__(self, file_paths, labels):
        super().__init__()
        self.file_paths = file_paths
        self.labels = labels

    def __getitem__(self, item):
        x, _ = wfdb.rdsamp(self.file_paths[item])
        x = np.sum(x, axis=1)
        x = x[np.newaxis, :]
        x = torch.from_numpy(x)
        y = self.labels[item]
        y = torch.tensor(y, dtype=torch.int8)
        return x, y

    def __len__(self):
        return len(self.labels)
