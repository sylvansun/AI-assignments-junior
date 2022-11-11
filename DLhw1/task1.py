from jittor import nn
import jittor as jt
import numpy as np

from model import Regressor
from dataset import RandomData
import utils

def task1(function_to_fit, num_epoch = 800, lr = 5e-3, interval = [-5, 5]):

    train_data = RandomData(function_to_fit, sample_size=800, interval = interval)
    test_data = RandomData(function_to_fit, sample_size=200, interval = interval)
    model = Regressor()
    optim = jt.optim.Adam(model.parameters(), lr)
    loss = nn.MSELoss()
    
    train_losses, test_losses = [], []
    for epoch_idx in range(1, num_epoch + 1):
        train_loss = train(model, optim, train_data, loss)
        test_loss = test(model, test_data, loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch_idx} \t Train Loss: {train_loss:.4f} \t Test Loss: {test_loss:.4f}')
    utils.plot_fit(function_to_fit, model, interval)
    utils.plot_loss(train_losses, test_losses)

def train(model, optimizer, train_data, loss_func):
    model.train()

    train_losses = []
    for _, (x, y) in enumerate(train_data):
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        train_loss = loss.item()
        train_losses.append(train_loss)
        optimizer.step(loss)
    return np.mean(train_losses)


def test(model, test_data, loss_func):
    model.eval()

    test_losses = []
    with jt.no_grad():
        for _, (x, y) in enumerate(test_data):
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            test_loss = loss.item()
            test_losses.append(test_loss)
    return np.mean(test_losses)


if __name__ == "__main__":
    jt.set_seed(0)
    function_to_fit = lambda x : x ** 3 - 2 * x ** 2 + x - 1
    task1(function_to_fit, interval=[-4, 4])