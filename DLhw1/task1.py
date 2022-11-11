from jittor.dataset.dataset import Dataset
import jittor as jt
import numpy as np

from model import Regressor
from dataset import RandomData

def task1():
    traindata = RandomData(sample_size=800)
    testdata = RandomData(sample_size=200)
    for batch_idx, (inputs, targets) in enumerate(traindata):
        print(batch_idx)
    print(inputs.shape, targets.shape)
    
    
    
if __name__ == "__main__":
    
    task1()