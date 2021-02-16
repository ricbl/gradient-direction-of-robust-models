"""MNIST dataset generation and dataloader
"""

import torch
import torchvision
import numpy as np
from .utils_dataset import TransformsDataset

class ToBinary(object):
    def __init__(self, class_0):
        self.class_0 = class_0
    
    def __call__(self, label):
        return np.float32([not (label == self.class_0)])
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Subset(torch.utils.data.sampler.SubsetRandomSampler):
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
        
def get_dataloaders(opt, mode):
    mode_ = 'train' if mode=='train' else opt.split_validation
    mnist_dataset = torchvision.datasets.MNIST('./', train=mode_!='test', download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.5,), (0.5,))
                               ]))
    
    valid_size = 0.1
    num_train = len(mnist_dataset)
    indices = []
    
    # only using the two selected classes
    for i in range(len(mnist_dataset)):
        if mnist_dataset[i][1] in [opt.class_0_mnist, opt.class_1_mnist]:
            indices.append(i)
    
    if mode_!='test':
        valid_size = 0.1
        num_train = len(indices)
        split = int(valid_size * num_train)
        # get a fixed random validation set for every run
        np.random.RandomState(0).shuffle(indices)
        indices = {'train':indices[split:], 'val':indices[:split]}[mode_]
    
    mnist_dataset = TransformsDataset(mnist_dataset, ToBinary(opt.class_0_mnist), i=1)
    if mode=='train':
        sampler = torch.utils.data.sampler.SubsetRandomSampler
    else:
        sampler = Subset
    
    return torch.utils.data.DataLoader(mnist_dataset, 
                    batch_size=opt.batch_size, sampler=sampler(indices), num_workers=0)