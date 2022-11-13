import os
import pickle
import numpy as np
from jittor.dataset import Dataset
import jittor as jt

class CIFAR10(Dataset):
    base_folder = 'cifar-10-batches-py'
    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']
    meta = {'filename': 'batches.meta','key': 'label_names'}

    def __init__(self,
                 data_choice = "default",
                 train=True, 
                 batch_size=64,
                 drop_last=False,
                 shuffle=True,
                 root="cifar_data/"):
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
        self._load_meta()
        self.data_choice = data_choice
        self.data_dropped = []
        self.targets_dropped = []
        self.data_process()
            
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

    def show_num_classes(self):
        count = np.zeros(10).astype(np.int32)
        for index in range(len(self.targets)):
            target = self.targets[index]
            count[target] += 1
        print("This is a {} dataset".format(self.data_choice))
        print("number of data in each class: ", count)
        print("data shape", self.data.shape, "target shape", self.targets.shape)
        
    def data_process(self):
        if self.data_choice == "default":
            self.data = jt.float32(self.data)
            self.targets = np.array(self.targets).astype(np.int32)
            return
        elif self.data_choice == "drop" or self.data_choice == "upsample":
            self.targets = np.array(self.targets).astype(np.int32)
            for i in range(5):
                drop_mask = self.targets == i
                self.data_dropped.append(self.data[drop_mask])
                self.targets_dropped.append(self.targets[drop_mask])
            mask = self.targets >= 5
            self.targets = self.targets[mask]
            self.data = self.data[mask]
            
            if self.data_choice == "upsample":
                for i in range(5):
                    sample_idx = np.random.randint(low = 0, high = 10)
                    for j in range(10):
                        self.data = np.concatenate((self.data, self.data_dropped[i][sample_idx*500:(sample_idx+1)*500]), axis=0)
                        self.targets = np.concatenate((self.targets, self.targets_dropped[i][sample_idx*500:(sample_idx+1)*500]), axis=0)
                self.data = jt.float32(self.data)
            else:
                for i in range(5):
                    self.data = np.concatenate((self.data, self.data_dropped[i][:500]), axis=0)
                    self.targets = np.concatenate((self.targets, self.targets_dropped[i][:500]), axis=0)
                self.data = jt.float32(self.data)
        else:
            raise NotImplementedError("data choice not support")
            
            


        
if __name__ == "__main__":
    
    train_data = CIFAR10(train=True, batch_size=64)
    drop_data = CIFAR10(train=True, batch_size=64, data_choice="drop")
    upsample_data = CIFAR10(train=True, batch_size=64, data_choice="upsample")

    train_data.show_num_classes()
    drop_data.show_num_classes()
    upsample_data.show_num_classes()
    
    # a = np.arange(20)
    # np.random.shuffle(a)
    # print(a)
    # mask = a>=6
    # a=a[mask]
    # print(a)
    print("Done")
    