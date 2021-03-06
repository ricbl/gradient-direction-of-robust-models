"""This module provides a UNet class to be used as the generator model in the algorithm
parts were taken from 
https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch/blob/master/src/models/mask_generators.py
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from. import classifier
ACTIVATION = nn.ReLU

#initialize weights of the unet just like in https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        pass
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        pass
        torch.nn.init.normal_(m.weight, mean=0.1, std=0.01)
        m.bias.data.fill_(0)

class InputsAdjusted(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model
    def forward(self,x1,x2,x3):
        return self.model(torch.cat([x1, x2,x3], dim = 1))

def init_model(opt):
    if not opt.vrgan_training and opt.load_checkpoint_g is None:
        return NullGenerator()
    if opt.dataset_to_use!='spheres':
        net_g = UNet(nf=64, unet_downsamplings = opt.unet_downsamplings)
    else:
        sphere_hidden_neurons_per_layer = opt.sphere_hidden_neurons_per_layer
        net_g = InputsAdjusted(torch.nn.Sequential(torch.nn.Linear(opt.n_dimensions_sphere+2,sphere_hidden_neurons_per_layer), 
        torch.nn.ReLU(), torch.nn.BatchNorm1d(sphere_hidden_neurons_per_layer), torch.nn.Linear(sphere_hidden_neurons_per_layer,sphere_hidden_neurons_per_layer), 
        torch.nn.ReLU(), torch.nn.BatchNorm1d(sphere_hidden_neurons_per_layer), torch.nn.Linear(sphere_hidden_neurons_per_layer,opt.n_dimensions_sphere)))
    if opt.load_checkpoint_g is not None:
        net_g.load_state_dict(torch.load(opt.load_checkpoint_g))
    else:
        net_g.apply(weights_init)
    return net_g.cuda()

class NullGenerator(nn.Module):
    def forward(self, x):
        assert(False)

class Identity(nn.Module):
    def forward(self, x):
        return x

def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

def conv2d_bn_block(in_channels, out_channels, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block conv-bn-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor = self.scale_factor, mode='bilinear', align_corners=True)
    
def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=0, momentum=0.01, activation=ACTIVATION, dimensions = 2):
    if use_upsample:
        if dimensions == 2:
            conv_layer = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        elif dimensions == 3:
            conv_layer = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        up = nn.Sequential(
            Upsample(scale_factor=2),
            conv_layer
        )
    else:
        if dimensions == 2:
            up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
        elif dimensions == 3:
            up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    if dimensions == 2:
        bn_layer = nn.BatchNorm2d(out_channels, momentum=momentum)
    elif dimensions == 3:
        bn_layer = nn.BatchNorm3d(out_channels, momentum=momentum)
    return nn.Sequential(
        up,
        bn_layer,
        activation(),
    )

def preprocess_pft_values_for_generator(x):
    return classifier.BatchNormalizeTensor(torch.FloatTensor([0.7]).cuda(), torch.FloatTensor([0.2]).cuda())(x)

class UpCatC0nv(torch.nn.Module):
    def __init__(self, up, conv):
        super(UpCatC0nv, self).__init__()
        self.up = up
        self.conv = conv
    
    def forward(self, x, x_past):
        return self.conv(crop_and_concat(self.up(x), x_past))

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, nf=16, unet_downsamplings = 4):
        super(UNet, self).__init__()
        self.downsamplings = unet_downsamplings
        self.dimensions = 2
        
        max_pool = nn.MaxPool2d(2) 
        conv_block = conv2d_bn_block
        deconv_block = deconv2d_bn_block
        
        act = torch.nn.ReLU
        self.down = torch.nn.ModuleList()
        self.down.append(nn.Sequential(
            conv_block(n_channels, nf, activation=act) ,
            conv_block(nf, nf, activation=act)
        ))
        self.up = torch.nn.ModuleList()
        for i in range(self.downsamplings):
            self.down.append(nn.Sequential(
                    max_pool,
                    conv_block((2**i)*nf, (2**(i+1))*nf, activation=act),
                    conv_block((2**(i+1))*nf, (2**(i+1))*nf, activation=act),
                ))
            
            if i==0:
                output_channels = n_classes
            else:
                output_channels = (2**i)*nf
            input_channels = (2**(i+1))*nf + (3 if (i==(self.downsamplings-1)) else 0)
            up = deconv_block(input_channels, (2**i)*nf, activation=act, dimensions = self.dimensions)
            out_act = Identity if i == 0 else act
            conv = nn.Sequential(
                conv_block((2**(i+1))*nf, (2**i)*nf, activation=act),
                conv_block((2**i)*nf, output_channels, activation=out_act),
            )
            self.up.append(UpCatC0nv(up,conv))

    def forward(self, x, desired_output, groundtruth_regression):
        desired_output = preprocess_pft_values_for_generator(desired_output)
        groundtruth_regression = preprocess_pft_values_for_generator(groundtruth_regression)
        
        xs = [x]
        
        for i in range(self.downsamplings+1):
            xs.append(self.down[i](xs[-1]))
        
        desired_output = desired_output.view([-1,1]+[1,]*(x.ndim-2)).expand([xs[-1].size(0),1] + [xs[-1].size(i) for i in range(2,x.ndim)])
        groundtruth_regression = groundtruth_regression.view([-1,1]+[1,]*(x.ndim-2)).expand([xs[-1].size(0),1] + [xs[-1].size(i) for i in range(2,x.ndim)])
        xs.append( torch.cat([xs[-1], desired_output, groundtruth_regression, (groundtruth_regression - desired_output)], dim = 1))
        
        for i in range(self.downsamplings):
            xs.append(self.up[self.downsamplings-i-1](xs[-1], xs[self.downsamplings-i]))
        
        return xs[-1]