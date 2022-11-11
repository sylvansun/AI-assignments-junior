from jittor.dataset.dataset import Dataset
import jittor as jt
import numpy as np

from model import Regressor

class RandomData(Dataset):
    def __init__(self, batch_size = 16, sample_size = 800, interval = [-5, 5]):
        super().__init__(batch_size = batch_size, shuffle = True, drop_last= True)
        self.set_attrs(total_len = sample_size)
        self.sample_size = sample_size
        self.interval = interval
    
    def __getitem__(self, x):
        x = jt.rand(1) * (self.interval[1] - self.interval[0]) + self.interval[0]
        return x, self.f(x)

    def f(self, x):
        return x ** 3

    