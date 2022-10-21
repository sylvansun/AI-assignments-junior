import paddle
import numpy as np
import math
import paddle.fluid as fluid
from paddle.io import Dataset, DataLoader
import matplotlib.pyplot as plt
min_x = -1.57
max_x = 1.57
batch_size = 16
epoch_num = 20


def value(x):
    return x[1]


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def f(self, x):
        return math.sin(x)

    def __getitem__(self, idx):
        input = np.random.uniform(min_x, max_x, 1).astype('float32')
        label = np.array(self.f(input)).astype('float32').reshape(-1)
        return input, label

    def __len__(self):
        return self.num_samples


class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = paddle.nn.Linear(1, 128, weight_attr=fluid.initializer.MSRAInitializer(uniform=False))
        self.fc2 = paddle.nn.Linear(128, 64, weight_attr=fluid.initializer.MSRAInitializer(uniform=False))
        self.fc3 = paddle.nn.Linear(64, 4, weight_attr=fluid.initializer.MSRAInitializer(uniform=False))
        self.fc4 = paddle.nn.Linear(4, 1, weight_attr=fluid.initializer.MSRAInitializer(uniform=False))
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        pred = self.fc1(inputs)
        pred = self.relu(pred)
        pred = self.fc2(pred)
        pred = self.relu(pred)
        pred = self.fc3(pred)
        pred = self.relu(pred)
        pred = self.fc4(pred)
        return pred


dataset = RandomDataset(200)
model = Regressor()
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=2)

sgd_optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

for e in range(epoch_num):
    for i, (data, label) in enumerate(loader()):
        out = model(data)
        loss = fluid.layers.square_error_cost(input=out, label=label)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()
        sgd_optimizer.minimize(avg_loss)
        model.clear_gradients()
        print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))

testset = RandomDataset(200)
testloader = DataLoader(testset,batch_size=200,shuffle=False,drop_last=True,num_workers=2)
for i, (data, label) in enumerate(testloader()):
    _x = data
    result = model(data)
    true_value = label
x = []
x_ = np.zeros(200)
y = np.zeros(200)
y_pred = np.zeros(200)
for i in range(200):
    x.append((i, _x[i]))
x.sort(key = value)
for i in range(200):
    x_[i] = x[i][1]
    y[i] = true_value[x[i][0]]
    y_pred[i] = result[x[i][0]]

plt.figure
plt.plot(x_, y)
plt.plot(x_, y_pred)
plt.savefig('./test.png')
