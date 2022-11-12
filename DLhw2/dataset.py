import os
import pickle
import numpy as np
from jittor.dataset import Dataset
import jittor as jt
from model import Classifier

class CIFAR10(Dataset):
    base_folder = 'cifar-10-batches-py'
    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']
    meta = {'filename': 'batches.meta','key': 'label_names'}

    def __init__(self, batch_size=64, root="cifar_data/", train=True, drop_last=False, shuffle=False):
        super(CIFAR10, self).__init__(batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
        self.root = root
        self.train = train 
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data, self.targets = [], []
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = jt.float32(self.data)
        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    
    train_data = CIFAR10(train=True, batch_size=64)
    test_data = CIFAR10(train=False)

    print(len(train_data),len(test_data))
    
    