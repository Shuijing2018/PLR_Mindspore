from train import *

model_type = "mlp"
dataset_name = "abalone"
datasets[dataset_name]["max_set"] = MAX_SET[2]
datasets[dataset_name]["optim_rate"] = OPTIM_RATE[0]
loss_type = "real"

datasets[dataset_name]["loss_balance"] = 10000
up_seed(seed)
up_dataset(dataset_name)
train_dataset_model(datasets[dataset_name], loss_type, model_type=model_type,
                               print_show=True)
