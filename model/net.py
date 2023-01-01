import mindspore.nn as nn
from mindspore import Model

class MlpModel(nn.Cell):
    def __init__(self, input_dim, num_blocks: [int]):
        super(MlpModel, self).__init__()
        layers = [nn.Dense(input_dim, num_blocks[0])]
        for i in range(1, len(num_blocks)):
            layers.append(nn.Dense(num_blocks[i - 1], num_blocks[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Dense(num_blocks[-1], 1))
        self.linear = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.linear(x)


class LinearModel(nn.Cell):
    def __init__(self, input_dim, num_blocks: [int]):
        super(LinearModel, self).__init__()
        layers = [nn.Linear(input_dim, num_blocks[0])]
        for i in range(1, len(num_blocks)):
            layers.append(nn.Linear(num_blocks[i - 1], num_blocks[i]))
        layers.append(nn.Linear(num_blocks[-1], 1))
        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)
