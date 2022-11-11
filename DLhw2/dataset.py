from jittor.dataset.cifar import CIFAR10

a = CIFAR10()
a.set_attrs(batch_size=1)
print(len(a.data))
print(len(a.targets))
for imgs, labels in a:
    print(imgs.shape, labels)
    break
