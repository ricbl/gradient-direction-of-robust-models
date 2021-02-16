import torchvision
import numpy as np
from .utils_dataset import return_dataloaders, get_dataset_with_index, LoadToMemory

def get_cifar_dataset(mode):
    return get_dataset_with_index(torchvision.datasets.CIFAR10('./', train=mode, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])), 0 if mode else 50000)

def get_dataloaders(opt, mode):
    mode_ = 'train' if mode=='train' else opt.split_validation
    cifar_dataset = get_dataset_with_index(torchvision.datasets.CIFAR10('./', train=mode_!='test', download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])), 0 if mode_!='test' else 50000)
    
    num_train = len(cifar_dataset)
    indices = list(range(len(cifar_dataset)))
    
    if mode_!='test':
        valid_size = 0.1
        num_train = len(indices)
        split = int(valid_size * num_train)
        np.random.RandomState(0).shuffle(indices)
        indices = {'train':indices[split:], 'val':indices[:split]}[mode_]
        cifar_dataset.targets = np.array(cifar_dataset.targets)[indices]
        cifar_dataset.data = cifar_dataset.data[indices]
        cifar_dataset.indices = indices
    return return_dataloaders(lambda: LoadToMemory(cifar_dataset), opt, split = mode)
	