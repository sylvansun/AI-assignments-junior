from jittor.dataset.dataset import Dataset
import jittor as jt
class RandomData(Dataset):
    def __init__(self, 
                 f, 
                 batch_size = 50, 
                 sample_size = 800,
                 interval = [-5, 5]):
        super().__init__(batch_size = batch_size, shuffle = True, drop_last= True)
        self.set_attrs(total_len = sample_size)
        self.sample_size = sample_size
        self.interval = interval
        self.func = f
        self.data = self.generate_data()
    
    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]
    
    def generate_data(self):
        x = jt.rand(self.sample_size) * (self.interval[1] - self.interval[0]) + self.interval[0]
        y = self.func(x)
        return (x, y)
    