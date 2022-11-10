import os
from matplotlib import pyplot as plt

import numpy as np
import jittor as jt
from jittor import distributions


def test_data_type():
    a = jt.float32([1,2,3])
    b = a.numpy()
    c = jt.float32(b)
    print(type(a), a)
    print(type(b), b)
    print(type(c), c)

def test_func(mean = 1, std = 2):
    gaussian = distributions.Normal(mean, std)
    pdf = lambda xx: gaussian.log_prob(xx).exp()
    print(gaussian.log_prob(3))
    print(gaussian.sample([2,3]))

def get_data(mean=0, std=1, interval=[-5, 5], num_samples=1000, split=0.8):
    gaussian = distributions.Normal(mean, std)
    pdf = lambda xx: gaussian.log_prob(xx).exp()

    x = jt.rand(num_samples) * (interval[1] - interval[0]) + interval[0]
    y = pdf(x)

    # Split the data
    train_data = (x[:int(num_samples*split)], y[:int(num_samples*split)])
    test_data = (x[int(num_samples*split):], y[int(num_samples*split):])

    return train_data, test_data, pdf



if __name__ == "__main__":
    jt.set_global_seed(0)

    data_kwargs = {
        'mean': 0,
        'std': 1,
        'interval': [-5, 5],
        'num_samples': 1000,
        'split': 0.8
    }
    num_epoch = 1000
    lr = 3e-4
    
    train_data, test_data, pdf = get_data(**data_kwargs)
    
    print(len(test_data[0]))
    print(type(data_kwargs))
    test_data_type()
    test_func()
