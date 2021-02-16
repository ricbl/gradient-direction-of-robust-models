""" Spheres dataset

dataset of vectors of two classes, one with l2 norm of 1 and another with 
norm of 1.3
"""

from torch.utils.data import Dataset
import numpy as np
import random
import torch

class Sphere(Dataset):
    def __init__(self, size, opt):
        super().__init__()
        self.size = size
        self.n_dimensions = opt.n_dimensions_sphere
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        x = np.random.normal(size=[self.n_dimensions])
        x = x/np.linalg.norm(x)
        y = random.getrandbits(1)
        if bool(y):
            x = 1.3*x
        return torch.tensor(x).float(), torch.tensor(y)[None].float()

def get_dataloaders(opt, mode):
    return torch.utils.data.DataLoader(Sphere( (opt.total_iters_per_epoch*opt.batch_size) if mode == 'train' else opt.validation_examples_spheres, opt), batch_size=opt.batch_size,
                                                 shuffle=True, num_workers=0, pin_memory=False, drop_last = False)