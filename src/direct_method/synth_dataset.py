""" Squares dataset

dataset of images of squares of two sizes with added smoothed noise to
the background
"""

import numpy as np
from torch.utils.data import Dataset
from skimage import filters
import h5py
import os
import torch
import sys
sys.modules['synth_dataset']=None
from .utils_dataset import LoadToMemory
from .utils_dataset import return_dataloaders, RegressionToClassification
NUM_WORKERS = 0

class SynthDataset(Dataset):

    def __init__(self, output_folder, mode='train', transform=None, anomaly = False, filter_disease = False):
        super().__init__()
        self.filter_disease = filter_disease
        self.output_folder = output_folder
        self.anomaly = anomaly
        self.mode = mode
        self.transform = transform
        self.load_cache()
        self.indices = np.arange(len(self.images))
        
    def load_cache(self):
        data = load_and_generate_data(output_folder = self.output_folder, mode = self.mode)
        imsize = 224
        images = np.reshape(data['features'][:], [-1, imsize, imsize])
        images = np.expand_dims(images, 1)
        labels = data['regression_target'][:]
        if self.filter_disease:
            if self.anomaly:
                indexes_to_use = np.where(labels<0.7)[0]
            else:
                indexes_to_use = np.where(labels>=0.7)[0]
            labels = labels[indexes_to_use]
            images = images[indexes_to_use]
        self.images = images
        self.n_images = len(self.images)
        self.targets = labels
        
    def __len__(self):
        return self.n_images

    def __getitem__(self, index):
        
        index = self.indices[index]
        x = self.images[index, ...]
        y = np.expand_dims(self.targets[index, ...],axis = 1)
        if self.transform is not None:
            x = self.transform(x)
        return torch.tensor(x), torch.tensor(y).float()

def load_and_generate_data(output_folder, mode = 'train'):
    np_random_state = np.random.get_state()
    np.random.seed({'train':7, 'val':8, 'test':9}[mode])
    h5_filename = 'synthetic_mode_'+mode+'.hdf5'
    h5_filepath = os.path.join(output_folder, h5_filename)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(h5_filepath):
        size_datasets = {'train':10000, 'val':200, 'test':1000}
        regression_target, features = prepare_data_squares_by_size(num_samples = size_datasets[mode])
        with h5py.File(h5_filepath, 'w') as hdf5_file:
            hdf5_file.create_dataset('features', 
                data=features, dtype=np.float32)
            hdf5_file.create_dataset('regression_target',
                data=regression_target, dtype=np.float32)
    np.random.set_state(np_random_state)
    return h5py.File(h5_filepath, 'r')

def prepare_data_squares_by_size(image_size = 224,
                    num_samples=10000):
    regression_target = np.around(np.random.choice([0.5,0.8],num_samples), decimals = 2)
    features = np.zeros([num_samples, image_size, image_size])
    for i in range(num_samples):
        features[i,:,:] = get_clean_square(regression_target[i], image_size)
        noise = np.random.normal(scale=1, 
            size=np.asarray([image_size, image_size]))
        smoothed_noise = filters.gaussian(noise, 2.5)
        smoothed_noise = smoothed_noise / np.std(smoothed_noise) * 0.5
        features[i,:,:] += smoothed_noise
    return regression_target, features.reshape([-1, num_samples])   

def get_clean_square(regression_target, image_size):
    half_image_size = int(image_size / 2)
    block_size = int((half_image_size*0.8)*regression_target)
    to_return = np.zeros([image_size, image_size])
    to_return -= 0.5
    to_return[half_image_size - block_size: half_image_size + block_size, 
        half_image_size - block_size: half_image_size + block_size] = 0.5
    return to_return

def get_dataloaders(opt, mode='train'):
    output_folder = opt.folder_dataset
    transform_classification = lambda x: RegressionToClassification(x)
    apply_all_dataset_modifiers = lambda x: LoadToMemory(x)
    
    instantiate_all_dataset = lambda: transform_classification(apply_all_dataset_modifiers(SynthDataset(output_folder, mode=mode)))
    return return_dataloaders(instantiate_all_dataset, opt, split = mode)