from .config import Config
from sklearn.model_selection import train_test_split
from .dataset import *
import os
import pandas as pd
import ast
from torch.utils.data import DataLoader
import wfdb


def read_signal_file(file_path):
    x, _ = wfdb.rdsamp(file_path)
    x = torch.from_numpy(x.astype(np.double))
    return x


def prepare_ptb_dataset(config: Config):
    config = config
    label_filename = 'ptbxl_database.csv'
    label_file = os.path.join(config.path_to_dataset, label_filename)
    label = pd.read_csv(label_file, index_col='ecg_id')
    label.scp_codes = label.scp_codes.apply(lambda x: ast.literal_eval(x))
    if config.sampling_rate == 100:
        record_paths = [os.path.join(config.path_to_dataset, f) for f in label.filename_lr]
    else:
        record_paths = [config.path_to_dataset + f for f in label.filename_hr]
    X = [read_signal_file(file_path) for file_path in record_paths] if config.data_read == 'memory' else record_paths
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = [[1 if c in l else 0 for c in classes] for l in label.scp_codes]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size)
    train_set = PTBDataset(X_train, y_train, config)
    test_set = PTBDataset(X_test, y_test, config)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size, num_workers=os.cpu_count())
    test_loader = DataLoader(test_set, shuffle=False, batch_size=8, num_workers=os.cpu_count())
    return train_loader, test_loader


def prepare_dataset(config: Config):
    datasets = {
        'ptb': prepare_ptb_dataset
    }
    dataset = datasets[config.dataset]
    return dataset(config)
