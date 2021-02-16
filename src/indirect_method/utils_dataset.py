"""Auxiliary dataset functions
This module provides generic functions related to all datasets
"""

from torch.utils.data import Dataset
import torch

NUM_WORKERS = 0

def get_dataset_with_index(dataset, offset):
    class DatasetWithIndex(type(dataset)):
        def __getitem__(self,index):
            return super().__getitem__(index), self.indices[index]+offset
    dataset.__class__ = DatasetWithIndex
    dataset.indices = range(len(dataset))
    return dataset

class ImageDataset(Dataset):
    def __init__(self, image_list):
        super().__init__()
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        return self.image_list[index]

#dataset wrapper to convert a regression COPD dataset to a classification dataset
class RegressionToClassification(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        example = self.original_dataset[index]
        return (example[0][0], ((example[0][1]<0.7)*1).long()), example[1]

#dataset wrapper to load a dataset to memory for faster batch loading
class LoadToMemory(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.list_items = [item for item in original_dataset]

    def __len__(self):
        return len(self.list_items)
    
    def __getitem__(self, index):
        return self.list_items[index]

# limit the epoch size to n_iter_per_epoch.
# only to be used with dataloaders that load samples randomly
class ChangeDatasetToIndexList(Dataset):
    def __init__(self, original_dataset, index_list):
        self.original_dataset = original_dataset
        self.index_list = index_list
    
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        return self.original_dataset[self.index_list[index]]

#generic function to get dataloaders from datasets
def return_dataloaders(instantiate_dataset, opt, split):
    batch_size = opt.batch_size_train if split=='train' else opt.batch_size_val
    if len(opt.index_produce_val_image)>0 and split!='train':
        instantiate_dataset_ = instantiate_dataset
        instantiate_dataset = lambda :ChangeDatasetToIndexList(instantiate_dataset_(), opt.index_produce_val_image)
    return torch.utils.data.DataLoader(dataset=instantiate_dataset(), batch_size=batch_size,
                        shuffle=(split=='train'), num_workers=NUM_WORKERS, pin_memory=False, drop_last = (split=='train'))
    