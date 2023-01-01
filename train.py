from losses import *
from model import *
from parameter import *
import copy
from mindspore.train.callback import Callback, LossMonitor


class CustomWithLossCell(nn.Cell):
    """Connect the forward network and the loss function"""

    def __init__(self, backbone, loss_fn):
        """There are 2 inputs, forward network backbone and loss_fn"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, feature, candidate, target, num_set, num):
        output = self._backbone(feature)                 # Obtain network output through forward caculation
        return self._loss_fn(output, candidate)  # Obtain the multi-label loss value

class CustomWithEvalCell(nn.Cell):
    """Custom multi-label evaluation networks"""

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, feature, candidate, target, num_set, num):
        output = self.network(feature)
        return output, candidate, target

def train_dataset_model(dataset, loss_type, model_type="mlp", print_show=True):
    if model_type == "mlp":
        net = MlpModel(10, dataset["model_structure"])
    elif model_type == "linear":
        net = LinearModel(len(dataset["train_dataset"].__getitem__(0)[0]), dataset["model_structure"])
    else:
        print("invalid model")
        sys.exit()

    optim = nn.Adam(net.trainable_params(), learning_rate=0.001)
    history = [[], []]
    custom_model = CustomWithLossCell(net, net_loss)
    eval_net = CustomWithEvalCell(net)
    mae1 = nn.MSE()
    mae2 = nn.MSE()
    mae1.set_indexes([0, 1])
    mae2.set_indexes([0, 2])
    model = Model(custom_model, optimizer=optim, eval_network=eval_net, metrics={"mae1": mae1, "mae2": mae2})

    # max_loss = MAX_INT
    model.train(dataset["epoch"], dataset["train_dataset"], callbacks=[LossMonitor(10), ], dataset_sink_mode=True)
    # model.eval(dataset["verify_dataset"])


