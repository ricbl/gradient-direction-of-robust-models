"""Auxiliary dataset functions
This module provides generic functions related to all datasets
"""

from torch.utils.data import Dataset
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import os
import h5py

import signal
import hashlib
import PIL
import shutil
import torch

NUM_WORKERS = 0

class MyKeyboardInterruptionException(Exception):
    "Keyboard Interrupt activate signal handler"
    pass
    
def interupt_handler(signum, frame):
    raise MyKeyboardInterruptionException

signal.signal(signal.SIGINT, interupt_handler)

#dataset wrapper to convert a regression COPD dataset to a classification dataset
class RegressionToClassification(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        example = self.original_dataset[index]
        return example[0], ((example[1]<0.7)*1).float()

#dataset wrapper to load a dataset to memory for faster batch loading
class LoadToMemory(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.list_elements = [original_dataset[0]]*len(original_dataset)
        indices_iterations = np.arange(len(original_dataset))
        
        for list_index, element_index in enumerate(indices_iterations): 
            self.list_elements[list_index] = original_dataset[element_index]

    def __len__(self):
        return len(self.list_elements)
    
    def __getitem__(self, index):
        return self.list_elements[index]

#dataset wrapper to apply transformations to a pytorch dataset. i defines the index of the element
# of the tuple returned by original_dataset to which the transformation should be applied
class TransformsDataset(Dataset):
    def __init__(self, original_dataset, transform, i=0):
        super().__init__()
        self.original_dataset = original_dataset
        self.transform = transform
        self.i = i
    
    def apply_transform_ith_element(self, batch, transform):        
        to_return = *batch[:self.i], transform(batch[self.i]), *batch[(self.i+1):]
        return to_return
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        return self.apply_transform_ith_element(self.original_dataset[index], self.transform)

# limit the epoch size to n_iter_per_epoch.
# only to be used with dataloaders that load samples randomly
class ChangeLoaderSize:
    def __init__(self, loader, n_iter_per_epoch):
        self.loader = loader
        self.__iter__()
        self.count_iterations = 0
        self.n_iter_per_epoch = n_iter_per_epoch
        
    def __iter__(self):
        self.iterLoader = iter(self.loader)
        return self
    
    def nextI(self, this_iter):
        return next(this_iter,None)
    
    def __next__(self):
        if self.count_iterations >= self.n_iter_per_epoch:
            self.count_iterations = 0
            raise StopIteration
        current_batch_loader = self.nextI(self.iterLoader)
        if current_batch_loader is None:
            self.iterLoader = iter(self.loader)
            current_batch_loader = self.nextI(self.iterLoader)
        self.count_iterations += 1
        return current_batch_loader
    next = __next__

#generic class to save a pytorch dataset to H5 files, and load them to memory if
# they have already been saved. It assumes that filename is a uniue identifier for the dataset content, 
# such that if filename exists, it will directly load the h5 file and not use original_dataset.
# it also saves a pickle file to store the original organization of the dataset, while the h5
# file store the data.
class H5Dataset(Dataset):
    def __init__(self, original_dataset, path = '.', filename = None):
        super().__init__()
        self.len_ = len(original_dataset)
        if filename is None:
            # if filename is not provided, try to get a hash key for the dataset to characterize its content
            # for several datasets, this will take a really long time, since it has to iterate through the full dataset.
            # it is better to provide your own unique name
            def hash_example(name_, structure, fixed_args):
                structure = np.array(structure)
                structure.flags.writeable = False
                fixed_args['sha1'].update(structure.data)
            sha1 = hashlib.sha1()
            for example in original_dataset:
                apply_function_to_nested_iterators(example, {'sha1': sha1}, hash_example)
            filename = str(sha1.hexdigest())
        filename = filename + self.get_extension()
        self.filepath_h5 = path + '/' + filename
        structure_file = path + '/' + filename + '_structure.pkl'
        if not os.path.exists(self.filepath_h5):
            try:
                with self.get_file_handle()(self.filepath_h5, 'w') as h5f:
                    structure = self.create_h5_structure(original_dataset[0], h5f, len(original_dataset))
                    for index, element in enumerate(original_dataset): 
                        self.pack_h5(element, index, h5f)
                with open(structure_file, 'wb') as output:
                    pickle.dump(structure, output, pickle.HIGHEST_PROTOCOL)
            except Exception as err:
                # if there is an error in the middle of writing, delete the generated files
                # to not have corrupted files
                if os.path.exists(self.filepath_h5):
                    if not os.path.isdir(self.filepath_h5):
                        os.remove(self.filepath_h5)
                    else:
                        shutil.rmtree(self.filepath_h5)
                if os.path.exists(structure_file):
                    if not os.path.isdir(structure_file):
                        os.remove(structure_file)
                    else:
                        shutil.rmtree(structure_file)
                raise Exception('Error while writing hash ' + filename + '. Deleting files ' + self.filepath_h5 + ' and ' + structure_file).with_traceback(err.__traceback__)
        elif not os.path.exists(structure_file):
            structure = self.create_h5_structure(original_dataset[0], h5f = None, n_images = len(original_dataset))
            with open(structure_file, 'wb') as output:
                pickle.dump(structure, output, pickle.HIGHEST_PROTOCOL)
        self.file = None
        with open(structure_file, 'rb') as input:
            self.structure = pickle.load(input)
    
    def get_file_handle(self):
        return h5py.File
        
    def get_extension(self):
        return '.h5'
        
    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.filepath_h5, 'r', swmr = True)
            assert(len(self.file['root']['_index_0'])==len(self))
        return self.unpack_h5(self.structure, index, self.file)
    
    def __len__(self):
        return self.len_
        
    @staticmethod
    def create_h5_structure(structure, h5f, n_images):
        def function_(name_, value, fixed_args):
            if fixed_args['h5f'] is not None:
                fixed_args['h5f'].create_dataset(name_, shape = [fixed_args['n_images']] + list(np.array(value).shape))#, chunks  = [1,]+list(np.array(value).shape))
            return None
        return apply_function_to_nested_iterators(structure, {'n_images':n_images, 'h5f': h5f}, function_)
    
    @staticmethod
    def pack_h5(structure, index, h5f):
        def function_(name_, value, fixed_args):
            fixed_args['h5f'][name_][fixed_args['index'],...] = value
            return None
        return apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f}, function_)
    
    @staticmethod
    def unpack_h5(structure, index, h5f):
        return apply_function_to_nested_iterators(structure, {'index':index, 'h5f': h5f}, lambda name_, value, fixed_args: fixed_args['h5f'][name_][fixed_args['index']])

#auxiliary function to iterate and apply functions to all elements of a variable composed
# of nested variable of these types: list, tuple, dict
# leafs have to be of kind: np.ndarray, int, float, bool, PIL.Image.Image
def apply_function_to_nested_iterators(structure, fixed_args, function_, name_ = "root"):
    if structure is None or isinstance(structure, (np.ndarray, int, float, bool, PIL.Image.Image)):
        return function_(name_, structure, fixed_args)
    elif isinstance(structure, list) or isinstance(structure, tuple):
        return [apply_function_to_nested_iterators(item, fixed_args, function_, name_ = name_ + "/" + '_index_' + str(index)) for index, item in enumerate(structure)]
    elif isinstance(structure, dict):
        return {key: apply_function_to_nested_iterators(item, fixed_args, function_, name_ = name_ + "/" + key) for key, item in structure.items()}
    else:
        raise ValueError('Unsuported type: ' + str(type(structure)))

#generic function to get dataloaders from datasets
def return_dataloaders(instantiate_dataset, opt, split):
    batch_size = opt.batch_size
    return torch.utils.data.DataLoader(dataset=instantiate_dataset(), batch_size=batch_size,
                        shuffle=(split=='train'), num_workers=NUM_WORKERS, pin_memory=False, drop_last = (split=='train'))
    