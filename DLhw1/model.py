import jittor as jt
from jittor import nn, Module, init


class Regressor(Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.linear1 = nn.Linear(1, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)
    def execute (self,x) :
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = nn.relu(self.linear3(x))
        y = self.linear4(x)
        return y