from jittor import Module, nn

class Classifier(Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv(in_channels=3, out_channels=64, kernel_size=5)
        self.bn1 = nn.BatchNorm(num_features=64)
        
        self.conv2 = nn.Conv(in_channels=64, out_channels=256, kernel_size=5)
        self.bn2 = nn.BatchNorm(num_features=256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(256 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 96)
        self.fc3 = nn.Linear(96, num_classes)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.relu(x)
        x = self.pool(x)
        
        x = nn.flatten(x, 1)
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        y = self.fc3(x)
        return y


if __name__ == "__main__":
    model = Classifier()
    print(model)