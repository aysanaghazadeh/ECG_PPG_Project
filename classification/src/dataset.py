from torch.utils.data import Dataset
import torch
import wfdb
import numpy as np


class PTBDataset(Dataset):
    def __init__(self, file_paths, labels, config):
        super().__init__()
        self.config = config
        self.file_paths = file_paths
        self.labels = labels
        if self.config.model == 'generative':
            self.labels = np.array(self.labels)
            self.file_paths = np.array(self.file_paths)
            indexes = np.where(self.labels == 0)[0]
            self.labels = self.labels[indexes]
            self.file_paths = self.file_paths[indexes]

    def __getitem__(self, item):
        if self.config.data_read == 'hard':
            x, _ = wfdb.rdsamp(self.file_paths[item])
            x = np.sum(x, axis=1)
            x = x[np.newaxis, :]
            x = torch.from_numpy(x.astype(np.double))
            if self.config.model == 'generative':
                y = x[:, int(x.size()[1] / 2):]
                x = x[:, : int(x.size()[1] / 2)]
        else:
            x = self.file_paths[item]
            mean_val = torch.mean(x)
            std_val = torch.std(x)
            x = (x - mean_val) / std_val
        y = self.labels[item]
        y = torch.tensor(y, dtype=torch.int8)
        return x, y

    def __len__(self):
        return len(self.labels)
