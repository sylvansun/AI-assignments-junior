import numpy as np
from jittor import nn
from model import Classifier
from dataset import CIFAR10


def train(model, train_loader, optimizer, epoch_idx, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step (loss)
        losses.append(loss.numpy()[0])
        losses_idx.append(epoch_idx * lens + batch_idx)
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_idx, batch_idx, len(train_loader) ,
                100. * batch_idx / len(train_loader), loss.numpy()[0]))

def val(model, test_loader, epoch):
    model.eval()
    
    test_loss = 0
    correct = 0
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
        print(f'Test Epoch: {epoch} [{batch_idx}/{len(test_loader)}]\tAcc: {acc:.6f}')    	
        print('Test Acc =', total_acc / total_num)
    

def task2(batch_size = 64, learning_rate = 0.05, weight_decay = 1e-4, num_epochs = 200):
    losses = []
    losses_idx = []
    train_loader = CIFAR10(train=True, batch_size=batch_size, shuffle=True)
    test_loader = CIFAR10(train=False, batch_size=1, shuffle=False)

    model = Classifier()
    optimizer = nn.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch_idx in range(num_epochs):
        train(model, train_loader, optimizer, epoch_idx, losses, losses_idx)
        val(model, test_loader, epoch_idx)


if __name__ == "__main__":

    task2(num_epochs=2)
