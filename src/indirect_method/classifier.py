"""File defining the classifier models used in the paper
"""
import math
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
    pred = np.argmax(np.array(logits_pred), axis = 1)
    return np.array(pred) == np.array(y_corr)

# preprocess the inputs of a classifier to normailze them with ImageNet statistics
class ClassifierInputs(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda().view([1,3,1,1])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda().view([1,3,1,1])
    
    def forward(self, x):
        return ((x/2+0.5).expand([-1,3,-1,-1]) - self.mean)/self.std

# function that provides with the classifier models used in the paper
def init_model(opt):
    resnet_model = {18: torchvision.models.resnet18, 34: torchvision.models.resnet34, 50: torchvision.models.resnet50, 101: torchvision.models.resnet101, 152: torchvision.models.resnet152}[opt.resnet_n_layers]
    net_d = resnet_model(pretrained = False if opt.dataset_to_use=='cifar' else True) 
    net_d.fc = torch.nn.Linear(in_features = net_d.fc.in_features, out_features = opt.n_classes)
    if opt.dataset_to_use=='cifar':
        net_d.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        net_d.maxpool = torch.nn.Sequential()
        for m in net_d.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a = math.sqrt(5))
    net_d = ClassifierWithPreprocessing(net_d, ClassifierInputs(opt))
    
    if opt.load_checkpoint_d is not None:
        net_d.load_state_dict(torch.load(opt.load_checkpoint_d))
    
    if opt.deactivate_bn:
        # turning off batch normalization on the classifier
        def train(self, mode = True):
            super(type(net_d), self).train(mode)
            for module in self.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d) or \
                isinstance(module, torch.nn.modules.BatchNorm2d) or \
                isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
        net_d.train = types.MethodType(train, net_d)
    return net_d.cuda()