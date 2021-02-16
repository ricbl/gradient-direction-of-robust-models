"""File defining the classifier models used in the paper
"""

import torch
import torchvision
import numpy as np
import types

#class to add a preprocessing module to a model
class ClassifierWithPreprocessing(torch.nn.Module):
    def __init__(self, original_model, preprocessing_model):
        super().__init__()
        self.preprocessing_model = preprocessing_model
        self.original_model = original_model
    
    def forward(self, x):
        x = self.preprocessing_model(x)
        x = self.original_model(x)
        return x

#given the ground truth labels y_corr and model logits logits_pred, calculates
# which examples were correctly classified
def get_correct_examples(logits_pred, y_corr):
    pred = (np.array(logits_pred)>0.0)*1
    return pred == np.array(y_corr)

# normalizes a batch of tensors according to mean and std
class BatchNormalizeTensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        to_return = (tensor-self.mean)/self.std
        return to_return

# preprocess the inputs of a classifier to normailze them with ImageNet statistics
class ClassifierInputs(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
    
    def forward(self, x):
        return BatchNormalizeTensor(torch.FloatTensor([0.485, 0.456, 0.406]).cuda().view([1,3,1,1]), 
            torch.FloatTensor([0.229, 0.224, 0.225]).cuda().view([1,3,1,1]))((x).expand([-1,3,-1,-1]))

# function that provides with the classifier models used in the paper
def init_model(opt):
    if opt.dataset_to_use!='spheres':
        resnet_model = torchvision.models.resnet18
        net_d = resnet_model(pretrained = True) 
        net_d.fc = torch.nn.Linear(in_features = net_d.fc.in_features, out_features = 1)
        net_d = ClassifierWithPreprocessing(net_d, ClassifierInputs(opt))
    else:
        sphere_hidden_neurons_per_layer = opt.sphere_hidden_neurons_per_layer
        net_d = torch.nn.Sequential(torch.nn.Linear(opt.n_dimensions_sphere,sphere_hidden_neurons_per_layer), 
        torch.nn.ReLU(), torch.nn.BatchNorm1d(sphere_hidden_neurons_per_layer), torch.nn.Linear(sphere_hidden_neurons_per_layer,sphere_hidden_neurons_per_layer), 
        torch.nn.ReLU(), torch.nn.BatchNorm1d(sphere_hidden_neurons_per_layer), torch.nn.Linear(sphere_hidden_neurons_per_layer,1))
    if opt.load_checkpoint_d is not None:
        net_d.load_state_dict(torch.load(opt.load_checkpoint_d))
    # following Baumgartner et al. (2018), turning off batch normalization 
    # on the critic
    def train(self, mode = True):
        super(type(net_d), self).train(mode)
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d) or \
            isinstance(module, torch.nn.modules.BatchNorm2d) or \
            isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()
    net_d.train = types.MethodType(train, net_d)
    return net_d.cuda()