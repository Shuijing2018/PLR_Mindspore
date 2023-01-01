import sys
from data import *
from sklearn.model_selection import train_test_split
import mindspore.dataset as ds
def up_seed(rand_seed):
    """"update seed"""
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_INT = sys.maxsize

batch_size = 256
MAX_SET = [2, 4, 8, 16]
OPTIM_RATE = [0.001, 0.01]
LOSS_BALANCE = [10, 100, 500, 1000, 10000]
GEN_DATA = gen_uniform

epoch = 100
max_set = MAX_SET[3]
std = 10
optim_rate = OPTIM_RATE[0]
weights_extreme = -0.5
gen_data = GEN_DATA
loss_balance = LOSS_BALANCE[0]

seed = 2022
up_seed(2022)

datasets = {
    "abalone": {"fun_data_processing": abalone_data_processing, "class_dataset": AbaloneDataSet, "max_set": max_set,
                "std": std, "model_structure": [20, 30, 10], "optim_rate": optim_rate, "epoch": epoch,
                "gen_data_fun": gen_data, "uniform_range": 10, "weights_extreme": weights_extreme,
                "loss_balance": loss_balance, "huber_delta": 1, "c": 4.685, "max_loss_balance": 10000},
}


def up_dataset(data_name, slice=1):
    """update dataset"""
    data_detail = datasets[data_name]
    data, candidate, num_set, number = data_detail["fun_data_processing"](
        "./data/dataset/" + data_name + ".data",
        data_detail["max_set"],
        data_detail["gen_data_fun"],
        data_detail["std"],
        data_detail["uniform_range"])
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.4)
    verify_idx, test_idx = train_test_split(test_idx, test_size=0.5)
    data_detail["train_dataset"] = ds.GeneratorDataset(
        data_detail["class_dataset"](data[train_idx, :], candidate[train_idx, :], num_set[train_idx],
                                     number[train_idx]), ['feature', 'candidate', 'target', 'num_set', 'num'])
    data_detail["test_dataset"] = ds.GeneratorDataset(
        data_detail["class_dataset"](data[test_idx, :], candidate[test_idx, :], num_set[test_idx], number[test_idx]),
        ['feature', 'candidate', 'target', 'num_set', 'num'])
    data_detail["verify_dataset"] = ds.GeneratorDataset(
        data_detail["class_dataset"](data[verify_idx, :], candidate[verify_idx, :], num_set[verify_idx],
                                     number[verify_idx]), ['feature', 'candidate', 'target', 'num_set', 'num'])

    data_detail["train_dataset"] = data_detail["train_dataset"].batch(batch_size)
    data_detail["test_dataset"] = data_detail["test_dataset"].batch(batch_size)
    data_detail["verify_dataset"] = data_detail["verify_dataset"].batch(144)