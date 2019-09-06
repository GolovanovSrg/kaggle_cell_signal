import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def train_test_split(data_csv_path, n_test_experiments=1):
    data_csv = pd.read_csv(data_csv_path)
    cell_types = data_csv.experiment.str.split("-").apply(lambda x: x[0]).unique()
    test_experiments = []
    for t in cell_types:
        cell_experiments_mask = data_csv.experiment.str.startswith(t)
        cell_experiments = data_csv.experiment[cell_experiments_mask].unique()
        np.random.shuffle(cell_experiments)
        test_experiments.extend(cell_experiments[:n_test_experiments].tolist())

    test_mask = data_csv.experiment.apply(lambda x: x in test_experiments)
    test_data_csv = data_csv.loc[test_mask]

    train_mask = ~test_mask
    train_data_csv = data_csv.loc[train_mask]

    return train_data_csv, test_data_csv


def get_data(data_csv, is_train=True):
    def get_id(row, site):
        experiment, well, plate = row['experiment'], row['well'], row['plate']
        return os.path.join(experiment, f'Plate{plate}', f'{well}_s{site}')

    def get_label(row):
        if is_train:
            return row['sirna']
        return row['id_code']

    data = [(get_id(row, 1), get_label(row)) for _, row in data_csv.iterrows()] + \
        [(get_id(row, 2), get_label(row)) for _, row in data_csv.iterrows()]

    return tuple(zip(*data))
