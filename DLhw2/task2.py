import numpy as np
from jittor import nn
import jittor as jt

from model import Classifier
from dataset import CIFAR10
from loss import cross_entropy_loss, weighted_cross_entropy_loss




def train(model, train_loader, optimizer, epoch_idx, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        labels_pred = model(inputs)
        loss = weighted_cross_entropy_loss(labels_pred, labels)
        optimizer.step (loss)
        losses.append(loss.numpy()[0])
        losses_idx.append(epoch_idx * lens + batch_idx)
        print(loss.item())
        

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch_idx, loss.numpy()[0]))

def val(model, test_loader, epoch):
    model.eval()
    
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy()==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
    print(f'Test Epoch: {epoch} \tAcc: {acc:.6f}')    	
    

def task2(mode = "default", batch_size = 64, learning_rate = 1e-4, weight_decay = 1e-4, num_epoch = 200):
    losses = []
    losses_idx = []
    train_loader = CIFAR10(train=True, batch_size=batch_size, shuffle=True, data_choice=mode)
    test_loader = CIFAR10(train=False, batch_size=1, shuffle=False)

    model = Classifier()
    optimizer = nn.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch_idx in range(1, num_epoch + 1):
        train(model, train_loader, optimizer, epoch_idx, losses, losses_idx)
        val(model, test_loader, epoch_idx)


if __name__ == "__main__":
    task2(num_epoch=20)