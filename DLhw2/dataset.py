from jittor.dataset.cifar import CIFAR10
from model import Classifier
import jittor as jt


class mCIFAR10(CIFAR10):
    def __init__(self, 
                 train=True):
        super(mCIFAR10, self).__init__(train=train, download=False)
        self.set_attrs(batch_size=4)



if __name__ == "__main__":
    
    train_data = mCIFAR10(train=True)
    test_data = mCIFAR10(train=False)
    model = Classifier()

    
    input = jt.randn(4, 3, 32, 32)
    print(len(train_data.data), len(test_data.data))
    for imgs, labels in train_data:
        print(imgs.shape, labels)
        break
    