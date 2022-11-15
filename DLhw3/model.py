from jittor import Module, nn
import numpy as np
import jittor as jt


class Extractor(Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(64)
        self.conv2 = nn.Conv(64, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(256)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(256 * 8 * 8, 512)

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


if __name__ == "__main__":
    model = PermNet()

    input = np.ones((64, 4, 3, 16, 16))
    input = jt.float32(input)
    output = model(input)
    print(output.shape)