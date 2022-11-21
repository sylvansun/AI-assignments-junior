import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import SNN, ANN

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()                    
    for batch_index, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        pred = output.argmax(dim=1)
        loss.backward()
        optimizer.step()
        if batch_index % 10 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))

def test_model(model, device, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad(): 
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, label).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test Average loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, 100.0 * correct / len(test_loader.dataset)))


def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net
    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

def batch_accuracy(train_loader, net, num_steps, device='cpu'):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total

def snn_train(train_loader, test_loader, device='cpu'):
    # neuron and simulation parameters
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 50
    
    snnnet = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(snnnet.parameters(), lr=1e-2, betas=(0.9, 0.999))
    num_epochs = 100
    test_acc_list = []

    # training loop
    for epoch in range(1,num_epochs+1):

        avg_loss = backprop.BPTT(snnnet, train_loader, optimizer=optimizer, criterion=loss_fn,
                            num_steps=num_steps, time_var=False, device=device)

        print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")

        # Test set accuracy
        test_acc = batch_accuracy(test_loader, snnnet, num_steps)
        test_acc_list.append(test_acc)

        print(f"Epoch {epoch}, Test Acc: {test_acc * 100:.2f}%\n")
    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_list)
    plt.title("Test Set Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
        
def ann_train(train_loader, test_loader, device):
    num_epochs = 100
    annnet = ANN().to(device)
    optimizer = torch.optim.Adam(annnet.parameters(), lr=1e-2, betas=(0.9, 0.999)) 
    
    for epoch in range(1, num_epochs + 1):
        train_model(annnet, device, train_loader, optimizer, epoch)
        test_model(annnet, device, test_loader)


def main():
    # dataloader arguments
    batch_size = 128
    data_path='./mnist'
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define a transform
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    snn_train(train_loader, test_loader, device=device)
    ann_train(train_loader, test_loader, device=device)

if __name__ == "__main__":
    main()