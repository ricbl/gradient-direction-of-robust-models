"""Auxiliary dataset functions
This module provides generic functions related to all datasets
"""

from torch.utils.data import Dataset
import numpy as np

#dataset wrapper to convert a regression COPD dataset to a classification dataset
class RegressionToClassification(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        example = self.original_dataset[index]
        return example[0], ((example[1]<0.7)*1).long()

#dataset wrapper to load a dataset to memory for faster batch loading
class LoadToMemory(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.list_images = [original_dataset[0][0]]*len(original_dataset)
        self.list_labels = [original_dataset[0][1]]*len(original_dataset)
        indices_iterations = np.arange(len(original_dataset))
        
        for list_index, element_index in enumerate(indices_iterations): 
            self.list_images[list_index] = original_dataset[element_index][0]
            self.list_labels[list_index] = original_dataset[element_index][1]

    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, index):
        return self.list_images[index], self.list_labels[index]
    