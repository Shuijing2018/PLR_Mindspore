import mindspore.dataset as ds
import numpy as np

class AbaloneDataSet:
    def __init__(self, dataset, candidate_set, num_set, number):
        self.feature = dataset[:, :dataset.shape[1] - 1]
        self.candidate_set = candidate_set
        self.target = dataset[:, dataset.shape[1] - 1]
        self.num_set = num_set
        self.number = number

    def __getitem__(self, index):
        return self.feature[index], self.candidate_set[index], self.target[index], self.num_set[index], self.number[index]
        # return self.feature[index], self.target[index]

    def __len__(self):
        return len(self.feature)