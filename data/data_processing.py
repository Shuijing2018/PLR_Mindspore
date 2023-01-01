import pandas as pd
import numpy as np
import random

import torch


def gen_normal_noise(target, max_set, *args):
    """Generating weakly supervised information through normal noise"""
    args = args[0]
    candidate_set = np.zeros(max_set)
    num_set = random.randint(0, max_set)
    candidate_set[:num_set] = target + np.random.normal(0, args[0], num_set)
    return candidate_set, num_set


def gen_uniform_noise(target, max_set, *args):
    """Generating weakly supervised information through uniform noise"""
    args = args[0]
    candidate_set = np.zeros(max_set)
    num_set = random.randint(0, max_set)
    bias = args[3] - args[2]
    candidate_set[:num_set] = target + np.random.uniform(-bias, bias, num_set)
    return candidate_set, num_set


def gen_uniform(target, max_set, *args):
    """Generating Weakly Supervised Information via Uniform Distribution"""
    args = args[0]
    candidate_set = np.zeros(max_set)
    num_set = random.randint(0, max_set)
    candidate_set[:num_set] = np.random.uniform(args[2], args[3], num_set)
    return candidate_set, num_set


def t_test(mu1, mu2, std1, std2, num1, num2):
    return abs((mu1 - mu2) / (((std1 ** 2) / num1 + (std2 ** 2) / num2) ** 0.5))


def abalone_data_processing(path, max_set, gen_data_fun, *args):
    data = pd.read_table(path, sep=',')
    data.Rings = data.Rings.astype(float)
    dict_sex = {"M": 0, "F": 1, "I": 2}
    data["Sex"] = data["Sex"].map(dict_sex)
    data["Sex"] = data["Sex"].astype('category')
    data = data.values
    data = np.append(np.eye(3)[data[:, 0].astype(int)], data[:, 1:], axis=1)

    data[:, 3:data.shape[1] - 1] = (data[:, 3:data.shape[1] - 1] - np.mean(data[:, 3:data.shape[1] - 1],
                                                                           axis=0)) / np.std(
        data[:, 3:data.shape[1] - 1], axis=0)

    args = args + (data[:, -1].min(), data[:, -1].max())
    candidate = [gen_data_fun(x, max_set, args) for x in data[:, -1]]
    candidate_set = np.array([x for x, _ in candidate])
    num_set = np.array([x for _, x in candidate]) + 1
    candidate_set = np.append(np.expand_dims(data[:, -1], axis=1), candidate_set, axis=1)

    num_set = num_set
    return np.float32(data), np.float32(candidate_set), num_set, np.array(range(candidate_set.shape[0]))


def auto_mpg_data_processing(path, max_set, gen_data_fun, *args):
    data = pd.read_table(path, sep='\t')
    data = data.drop(["car name", "Unnamed: 9"], axis=1)
    data = data.dropna()
    dict_cylinders = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4}
    dict_origin = {1: 0, 2: 1, 3: 2}
    data["cylinders"] = data["cylinders"].map(dict_cylinders)
    data["origin"] = data["origin"].map(dict_origin)
    data = data.values
    data = np.append(np.delete(data, 1, axis=1), np.eye(len(dict_cylinders))[data[:, 1].astype(int)], axis=1)
    data = np.append(np.delete(data, 6, axis=1), np.eye(len(dict_origin))[data[:, 6].astype(int)], axis=1)

    data[:, 1:5] = (data[:, 1:5] - np.mean(data[:, 1:5], axis=0)) / np.std(data[:, 1:5], axis=0)

    args = args + (data[:, 0].min(), data[:, 0].max())
    candidate = [gen_data_fun(x, max_set, args) for x in data[:, 0]]
    candidate_set = np.array([x for x, _ in candidate])
    num_set = np.array([x for _, x in candidate]) + 1
    candidate_set = np.append(np.expand_dims(data[:, 0], axis=1), candidate_set, axis=1)
    num_set = torch.tensor(num_set)
    return np.float32(data), np.float32(candidate_set), num_set, torch.tensor(range(candidate_set.shape[0])).type(torch.long)


def housing_data_processing(path, max_set, gen_data_fun, *args):
    data = pd.read_table(path, sep='\t')
    data = data.values
    data = np.append(np.eye(2)[data[:, 3].astype(int)], np.delete(data, 3, axis=1), axis=1)

    data[:, 2:14] = (data[:, 2:14] - np.mean(data[:, 2:14], axis=0)) / np.std(data[:, 2:14], axis=0)

    args = args + (data[:, -1].min(), data[:, -1].max())
    candidate = [gen_data_fun(x, max_set, args) for x in data[:, -1]]
    candidate_set = np.array([x for x, _ in candidate])
    num_set = np.array([x for _, x in candidate]) + 1
    candidate_set = np.append(np.expand_dims(data[:, -1], axis=1), candidate_set, axis=1)
    num_set = torch.tensor(num_set)
    return np.float32(data), np.float32(candidate_set), num_set, torch.tensor(range(candidate_set.shape[0])).type(torch.long)


def airfoil_data_processing(path, max_set, gen_data_fun, *args):
    data = pd.read_table(path, sep='\t')
    data = data.values

    data[:, :5] = (data[:, :5] - np.mean(data[:, :5], axis=0)) / np.std(data[:, :5], axis=0)

    args = args + (data[:, -1].min(), data[:, -1].max())
    candidate = [gen_data_fun(x, max_set, args) for x in data[:, -1]]
    candidate_set = np.array([x for x, _ in candidate])
    num_set = np.array([x for _, x in candidate]) + 1
    candidate_set = np.append(np.expand_dims(data[:, -1], axis=1), candidate_set, axis=1)
    num_set = torch.tensor(num_set)
    return np.float32(data), np.float32(candidate_set), num_set, torch.tensor(range(candidate_set.shape[0])).type(torch.long)


def concrete_data_processing(path, max_set, gen_data_fun, *args):
    data = pd.read_table(path, sep=',')
    data = data.values
    data[:, :8] = (data[:, :8] - np.mean(data[:, :8], axis=0)) / np.std(data[:, :8], axis=0)

    args = args + (data[:, -1].min(), data[:, -1].max())
    candidate = [gen_data_fun(x, max_set, args) for x in data[:, -1]]
    candidate_set = np.array([x for x, _ in candidate])
    num_set = np.array([x for _, x in candidate]) + 1
    candidate_set = np.append(np.expand_dims(data[:, -1], axis=1), candidate_set, axis=1)
    num_set = torch.tensor(num_set)
    return np.float32(data), np.float32(candidate_set), num_set, torch.tensor(range(candidate_set.shape[0])).type(torch.long)


def power_plant_data_processing(path, max_set, gen_data_fun, *args):
    data = pd.read_table(path, sep=',')
    data = data.values

    data[:, :4] = (data[:, :4] - np.min(data[:, :4], axis=0)) / (
            np.max(data[:, :4], axis=0) - np.min(data[:, :4], axis=0))

    args = args + (data[:, -1].min(), data[:, -1].max())
    candidate = [gen_data_fun(x, max_set, args) for x in data[:, -1]]
    candidate_set = np.array([x for x, _ in candidate])
    num_set = np.array([x for _, x in candidate]) + 1
    candidate_set = np.append(np.expand_dims(data[:, -1], axis=1), candidate_set, axis=1)
    num_set = torch.tensor(num_set)
    return np.float32(data), np.float32(candidate_set), num_set, torch.tensor(range(candidate_set.shape[0])).type(torch.long)


def cpu_act_data_processing(path, max_set, gen_data_fun, *args):
    data = pd.read_table(path, sep=',', header=None)
    data = data.values

    data[:, :data.shape[1] - 1] = (data[:, :data.shape[1] - 1] - np.mean(data[:, 0:data.shape[1] - 1],
                                                                         axis=0)) / np.std(
        data[:, 0:data.shape[1] - 1], axis=0)

    args = args + (data[:, -1].min(), data[:, -1].max())
    candidate = [gen_data_fun(x, max_set, args) for x in data[:, -1]]
    candidate_set = np.array([x for x, _ in candidate])
    num_set = np.array([x for _, x in candidate]) + 1
    candidate_set = np.append(np.expand_dims(data[:, -1], axis=1), candidate_set, axis=1)
    num_set = torch.tensor(num_set)

    return np.float32(data), np.float32(candidate_set), num_set, torch.tensor(range(candidate_set.shape[0])).type(torch.long)
