"""MNIST dataset generation and dataloader
"""

import torchvision
import numpy as np
from .utils_dataset import return_dataloaders, get_dataset_with_index, LoadToMemory

def get_mnist_dataset(mode):
    return get_dataset_with_index(torchvision.datasets.MNIST('./', train=mode, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.5,), (0.5,))
                               ])), 0 if mode else 60000)

def get_dataloaders(opt, mode):
    mode_ = 'train' if mode=='train' else opt.split_validation
    mnist_dataset = get_mnist_dataset(mode_!='test')
    
    valid_size = 0.1
    num_train = len(mnist_dataset)
    indices = list(range(len(mnist_dataset)))
    
    if mode_!='test':
        valid_size = 0.1
        num_train = len(indices)
        split = int(valid_size * num_train)
        # get a fixed random validation set for every run
        np.random.RandomState(0).shuffle(indices)
        indices = {'train':indices[split:], 'val':indices[:split]}[mode_]
        mnist_dataset.targets = mnist_dataset.targets[indices]
        mnist_dataset.data = mnist_dataset.data[indices]
        mnist_dataset.indices = indices
    return return_dataloaders(lambda: LoadToMemory(mnist_dataset), opt, split = mode)