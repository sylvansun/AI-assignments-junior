import numpy as np
from jittor import nn
import jittor as jt

from model import Classifier
from dataset import CIFAR10
from loss import make_loss
from utils.parser import make_parser



def train(model, train_loader, optimizer, epoch_idx, loss_function):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        labels_pred = model(inputs)
        loss = loss_function(labels_pred, labels)
        optimizer.step (loss)
        print(loss.item())
        

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch_idx, loss.item()))

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
    

def task2(args):
    mode, batch_size, learning_rate, weight_decay, num_epoch, debug = args.dataset, args.bs, args.lr, args.wd, args.ne, args.debug
    if debug:
        num_epoch = 1
    loss_function = make_loss()[args.loss]
    
    train_loader = CIFAR10(train=True, batch_size=batch_size, shuffle=True, data_choice=mode)
    test_loader = CIFAR10(train=False, batch_size=1, shuffle=False)

    model = Classifier()
    optimizer = nn.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch_idx in range(1, num_epoch + 1):
        train(model, train_loader, optimizer, epoch_idx, loss_function)
        val(model, test_loader, epoch_idx)


if __name__ == "__main__":
    parser = make_parser()
    task2(parser.parse_args())