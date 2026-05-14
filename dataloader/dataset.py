import pandas as pd
import numpy as np
from config.path import get_data_file
from dataloader.preprocess import XScaler, YScaler


class MyDataset(object):

    def __init__(self, data_file):
        self.data_file = get_data_file(data_file)
        self.data_x, self.data_y = None, None

    def load(self, task):
        data = pd.read_csv(self.data_file)
        x_scaler = XScaler(data, task)
        self.data_x = x_scaler.scale()

        y_scaler = YScaler(data, task)
        self.data_y = y_scaler.scale()
        return self

    def prepare(self, train_ratio):
        data_x = self.data_x.to_numpy()
        data_y = self.data_y.to_numpy()
        indexes = np.arange(data_x.shape[0])
        np.random.shuffle(indexes)
        train_part = int(len(indexes) * train_ratio)
        train_indexes, test_indexes = indexes[:train_part], indexes[train_part:]
        train_x, train_y = data_x[train_indexes, :], data_y[train_indexes, :]
        test_x, test_y = data_x[test_indexes, :], data_y[test_indexes, :]
        return train_indexes, train_x, train_y, test_indexes, test_x, test_y
