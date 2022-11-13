import jittor as jt
import numpy as np


def cross_entropy_loss(output, target):
    target = target.reshape((-1, ))
    target_weight = jt.ones(target.shape[0], dtype='float32')
    
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight
    return loss.mean() / target_weight.mean()

def weighted_cross_entropy_loss(output, target, weight = 10):
    target = target.reshape((-1, ))
    target_weight = np.ones(target.shape[0], dtype='float32')
    target_weight[target.numpy() < 5] = weight
    target_weight = jt.array(target_weight)
    
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight
    return loss.mean() / target_weight.mean()

def make_loss():
    loss_choice = {
        "cross_entropy": cross_entropy_loss,
        "weighted_cross_entropy": weighted_cross_entropy_loss,
    }
    return loss_choice

if __name__ == "__main__":
    batch_size = 64
    pred = jt.array(np.random.randn(batch_size, 10))
    label = jt.array(np.random.randint(0, 10, size=batch_size))
    print(cross_entropy_loss(pred, label))
    print(weighted_cross_entropy_loss(pred, label, weight=10))