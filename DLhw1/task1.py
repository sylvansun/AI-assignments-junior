from jittor import nn
import jittor as jt
import os 
import numpy as np
import matplotlib.pyplot as plt

from jittor import distributions
from model import Regressor
from dataset import RandomData

def task1(to_fit, num_epoch = 1000, lr = 1e-2):

    train_data = RandomData(to_fit, sample_size=800, batch_size=800)
    test_data = RandomData(to_fit, sample_size=200)
    model = Regressor()
    optim = jt.optim.Adam(model.parameters(), lr)
    loss = nn.MSELoss()
    
    for epoch_idx in range(1, num_epoch + 1):
        train(epoch_idx, model, optim, train_data, loss)
    plot(to_fit, model)

def train(epoch_idx, model, optimizer, train_data, loss_func):
    model.train()

    for batch_idx, (x, y) in enumerate(train_data):
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        optimizer.step(loss)
        print(loss.item())

def plot(pdf, model, interval=[-5, 5], num_samples=1000):

    plt.figure(figsize=(10, 5))
    x = np.linspace(*interval, num_samples)
    x_in = jt.array(x).reshape(-1, 1)
    y = pdf(x_in)
    y_pred = model(x_in)
    y, y_pred = y.data, y_pred.data
    plt.plot(x, y, label='True', linewidth=3)
    plt.plot(x, y_pred, label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(os.path.join('figs', 'fit.png'))


if __name__ == "__main__":
    jt.set_seed(0)
    gaussian = distributions.Normal(0, 1)
    function_to_fit = lambda x : x**3 - x**2 + x 
    task1(function_to_fit)