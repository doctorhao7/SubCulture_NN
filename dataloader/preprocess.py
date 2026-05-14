from config.tasks import TASKS
import numpy as np


class MinMaxScaler(object):

    def __init__(self, data, range_dict):
        self.range_dict = range_dict
        self.data = data.loc[:, list(self.range_dict.keys())]

    def scale(self):
        scaled = self.data[:]
        for k in self.range_dict.keys():
            scaled.loc[:, k] = (scaled.loc[:, k] - self.range_dict[k][0]) / \
                               (self.range_dict[k][1] - self.range_dict[k][0])
        return scaled

    def rescale(self, data):
        rescaled = data[:]
        for i, item in enumerate(self.range_dict.items()):
            k, v = item
            rescaled[:, i] = round((rescaled[:, i] * self.range_dict[k][1]) + self.range_dict[k][0])
        return rescaled


class XScaler(MinMaxScaler):

    def __init__(self, data, task):
        super(XScaler, self).__init__(data, range_dict=TASKS[task]['x'])


class YScaler(MinMaxScaler):

    def __init__(self, data, task):
        super(YScaler, self).__init__(data, range_dict=TASKS[task]['y'])
