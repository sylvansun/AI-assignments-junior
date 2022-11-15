from jittor import Module, nn
import numpy as np
import jittor as jt
import pygmtools as pygm


class Extractor(Module):
    def __init__(self, linear_size=256 * 8 * 8):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(64)
        self.conv2 = nn.Conv(64, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(256)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(linear_size, 512)

    def execute(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.flatten(x, 1)
        x = nn.relu(self.fc(x))
        return x

class MLP(Module):
    def __init__(self, in_channel=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channel * 4, 4096)
        self.fc2 = nn.Linear(4096, 16)

    def execute(self, x):
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPPermuter(Module):
    def __init__(self, in_channel=256 * 8 * 8 * 4):
        super(MLPPermuter, self).__init__()
        self.fc1 = nn.Linear(in_channel, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 16)

    def execute(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class MLPClassifier(Module):
    def __init__(self, in_channel=65536, num_classes=10):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(in_channel , 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 96)
        self.fc4 = nn.Linear(96, num_classes)

    def execute(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.relu(self.fc3(x))
        x = self.fc4(x)
        return x 


class PermNet(Module):
    def __init__(self):
        super(PermNet, self).__init__()
        self.extractor = Extractor()
        self.mlp = MLP()
    def execute(self, x):
        batch_size = x.shape[0]
        feature = self.extractor(x.reshape(-1, *x.shape[-3:]))
        output = self.mlp(feature.reshape(batch_size, -1))
        return output
class ConvEncoder(Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(64)
        self.conv2 = nn.Conv(64, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(256)
        self.pool = nn.AvgPool2d(2, 2)
        
    def execute(self, x):
        x = nn.relu(self.bn1(self.conv1(x.reshape(-1, *x.shape[-3:]))))
        x = self.pool(x)
        x = nn.relu(self.bn2(self.conv2(x)))
        feature = nn.flatten(x, 1)
        return feature
class PermuteClassifier(Module):
    def __init__(self, mode="pretrain"):
        super(PermuteClassifier, self).__init__()
        self.extractor = ConvEncoder()
        self.mlp2order = MLPPermuter()
        self.mlp2class = MLPClassifier()
        self.mode = mode
        
    def execute(self, x):
        batch_size = x.shape[0]
        feature = self.extractor(x)
        if self.mode == "pretrain":
            output = self.mlp2order(feature.reshape(batch_size, -1))
        else:
            output = self.mlp2class(feature.reshape(batch_size, -1))
        return output
    
    def pretrain(self):
        self.mode = "pretrain"
        self.extractor.train()
        self.mlp2order.train()
        self.mlp2class.eval()
    def classify(self):
        self.mode = "classify"
        self.extractor.eval()
        self.mlp2order.eval()
        self.mlp2class.train()
        return
        

if __name__ == "__main__":
    model = PermuteClassifier()
    model.train()
    input = np.ones((64, 4, 3, 16, 16))
    input = jt.float32(input)
    output = model(input)
    print(output.shape)
