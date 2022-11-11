import os
import jittor as jt
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_losses, test_losses):
    
    num_epoch = len(train_losses)
    epochs = np.arange(1, num_epoch+1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, test_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 20)
    plt.legend()
    plt.savefig(os.path.join('figs', 'loss.png'))

def plot_fit(f, model, interval, num_samples=1000):

    plt.figure(figsize=(10, 5))
    x = np.linspace(*interval, num_samples)
    x_jt = jt.array(x).reshape(-1, 1)
    y, y_pred = f(x_jt), model(x_jt)
    y, y_pred = y.data, y_pred.data
    plt.plot(x, y, label='True', linewidth=3)
    plt.plot(x, y_pred, label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(os.path.join('figs', 'fit.png'))