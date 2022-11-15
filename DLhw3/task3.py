import numpy as np
import os
from jittor import nn
import jittor as jt
import pygmtools as pygm
from tqdm import tqdm

from model import PermNet, PermuteClassifier
from dataset import CIFAR10P, CIFAR10
from utils.parser import make_parser


def train(model, train_loader, optimizer, epoch_idx, file):
    model.train()
    batch_size = train_loader.batch_size
    num_data = len(train_loader)

    train_loss = []
    for batch_idx, (inputs, labels, _) in enumerate(train_loader):
        outputs = model(inputs)
        outputs = outputs.reshape(-1, 4, 4)
        outputs = pygm.sinkhorn(outputs, backend="jittor", max_iter=1)
        loss = (outputs-labels).sqr().sum() / batch_size
        optimizer.step(loss)
        train_loss.append(loss.item())
        if batch_idx % 100 == 0:
            file.write(
                "Train epoch: {}  {:.2f}%\tLoss:{:.6f}\n".format(
                    epoch_idx, 100 * batch_idx * batch_size / num_data, loss.item()
                )
            )
    return np.mean(train_loss)


def val(model, test_loader, epoch_idx, file):
    model.eval()

    test_loss = []
    test_acc = []
    for _, (inputs, labels, orders) in enumerate(test_loader):
        outputs = model(inputs)
        outputs = outputs.reshape(-1, 4, 4)
        outputs = pygm.sinkhorn(outputs, backend="jittor", max_iter=1)
        loss = (outputs-labels).sqr().sum() / labels.shape[0]
        acc = (outputs.argmax(dim=2)[0] == orders).float().mean()
        test_loss.append(loss.item())
        test_acc.append(acc.item())
    total_acc = np.mean(test_acc)
    file.write(f"Test Epoch: {epoch_idx} \t Total Acc: {total_acc:.4f}\n")
    return np.mean(test_loss)



def task3(args):
    batch_size, learning_rate, weight_decay, num_epoch, debug = (
        args.bs,
        args.lr,
        args.wd,
        args.ne,
        args.debug,
    )
    if debug:
        num_epoch = 1

    train_loader = CIFAR10P(train=True, batch_size=batch_size, shuffle=True)
    test_loader = CIFAR10P(train=False, batch_size=batch_size, shuffle=False)

    model = PermNet()
    optimizer = nn.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    folder_name = f"bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}_ne_{num_epoch}"
    if not os.path.exists(f"./checkpoint/{folder_name}"):
        os.mkdir(f"./checkpoint/{folder_name}")
    file_name = f"./checkpoint/{folder_name}/log.txt"
    file = open(file_name, "w")
    file.write(f"{folder_name}\n")
    train_losses, test_losses = [], []
    for epoch_idx in tqdm(range(1, num_epoch + 1)):
        train_loss = train(model, train_loader, optimizer, epoch_idx, file)
        test_loss = val(model, test_loader, epoch_idx, file)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

def train2(model, train_loader, optimizer, epoch_idx, loss_function, file):
    model.train()
    batch_size = train_loader.batch_size
    num_data = len(train_loader)

    train_loss = []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        optimizer.step(loss)
        train_loss.append(loss.item())
        if batch_idx % 100 == 0:
            file.write(
                "Train epoch: {}  {:.2f}%\tLoss:{:.6f}\n".format(
                    epoch_idx, 100 * batch_idx * batch_size / num_data, loss.item()
                )
            )
    return np.mean(train_loss)


def val2(model, test_loader, epoch_idx, loss_function, file):
    model.eval()
    num_data = len(test_loader)

    test_loss = []
    total_correct = 0
    for _, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        test_loss.append(loss.item())
        pred = np.argmax(outputs.numpy(), axis=1)
        total_correct += np.sum(labels.numpy() == pred)
    total_acc = total_correct / num_data
    file.write(f"Test Epoch: {epoch_idx} \t Total Acc: {total_acc:.4f}\n")
    return np.mean(test_loss)

def pretrain(args):
    batch_size, learning_rate, weight_decay, num_epoch, debug = (
        args.bs,
        args.lr,
        args.wd,
        args.ne,
        args.debug,
    )
    if debug:
        num_epoch = 1

    pretrain_loader = CIFAR10P(train=True, batch_size=batch_size, shuffle=True)

    model = PermuteClassifier()
    optimizer = nn.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    folder_name = f"pretrain_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}_ne_{num_epoch}"
    if not os.path.exists(f"./checkpoint/{folder_name}"):
        os.mkdir(f"./checkpoint/{folder_name}")
    file_name = f"./checkpoint/{folder_name}/pretrainlog.txt"
    file = open(file_name, "w")
    file.write(f"{folder_name}\n")

    for epoch_idx in tqdm(range(1, num_epoch + 1)):
        train(model, pretrain_loader, optimizer, epoch_idx, file)
    print("Pretrain Done!")
    file.close()
    model.classify()
    file_name = f"./checkpoint/{folder_name}/trainlog.txt"
    file = open(file_name, "w")
    file.write(f"{folder_name}\n")
    train_loader = CIFAR10(train=True, batch_size=batch_size, shuffle=True)
    test_loader = CIFAR10(train=False, batch_size=batch_size, shuffle=False)
    train_losses, test_losses = [], []
    for epoch_idx in tqdm(range(1, num_epoch + 1)):
        train_loss = train2(model, train_loader, optimizer, epoch_idx,nn.CrossEntropyLoss, file)
        test_loss = val2(model, test_loader, epoch_idx, nn.CrossEntropyLoss, file)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    

if __name__ == "__main__":
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    parser = make_parser()
    ## task3 main function
    task3(parser.parse_args())
    ## pretrain and test task2 directly
    pretrain(parser.parse_args())