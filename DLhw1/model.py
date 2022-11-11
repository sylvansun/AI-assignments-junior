from jittor import nn, Module

class Regressor(Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.linear1 = nn.Linear(1, 16)
        self.linear2 = nn.Linear(16, 1)
    def execute (self,x) :
        x = nn.relu(self.linear1(x))
        y = self.linear2(x)
        return y