import warnings
from mindspore import nn
warnings.filterwarnings("ignore")

from mindspore.nn.loss.loss import LossBase
import mindspore.ops as ops


net_loss = nn.loss.MSELoss()