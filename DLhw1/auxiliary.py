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


if __name__ == "__main__":
    x = jt.rand(8)
    y = x**3
    a = (x,y)
    print(type(a))
    print(a[0])
    print(x)

