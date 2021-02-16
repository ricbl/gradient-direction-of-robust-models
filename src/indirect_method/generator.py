"""
code to load the generator from the chosen cgan and 
calculate projections to its manifold
"""

import torch
from . import util_defense_GAN

# class used to do the iterative optimizations in the latent space z
# to find the point in the support of each of all possible classes that is the closest to
# a given input image
class IterModel(torch.nn.Module):
    def __init__(self, opt, net_g):
        super().__init__()
        self.this_g = net_g
        self.this_g.load_state_dict(torch.load(opt.load_checkpoint_g+'.pth' ),strict=True)
        self.this_g = torch.nn.DataParallel(self.this_g)
        self.this_g.eval()
        self.list_of_class = opt.list_of_class
        self.n_classes = opt.n_classes
        mse_loss = torch.nn.MSELoss(reduction ='none').cuda()
        self.get_z_free = lambda y, x, z_init: util_defense_GAN.get_z_sets(self.this_g, y,x,opt.n_classes, opt.get_z_init_lr, 0,mse_loss, z_init, rec_iter=opt.get_z_init_iter)
        self.get_z = lambda y, x, z_init: util_defense_GAN.get_z_sets(self.this_g, y,x, opt.n_classes, opt.get_z_lr, opt.get_z_penalty,mse_loss, z_init, rec_iter=opt.get_z_iter)
        
    def forward(self, x, y, output):
        best_distance_images = torch.zeros_like(x).unsqueeze(1).repeat(1,self.n_classes,1,1,1)
        
        # initialize from the 0 vector to not be too far from any optimal z
        z_init_0 = torch.zeros([x.size(0), 128]).cuda()
        
        #iterate through all destination classes
        for i in range(self.n_classes):
            class_destination = self.list_of_class[i]
            y_ = torch.ones_like(y)*i
            
            # performs 600 iteration with no penalty on the norm of z to get
            # to an image that is really close to x
            z_init = self.get_z_free(y_,x, z_init_0)
            output.log_fixed(self.this_g(z_init, y_).detach(), None,'xzinit'+str(i))
            
            # performs 150 iterations to penalize the norm of z and get a 
            # plausible image for the desired class
            z_star = self.get_z(y_,x,z_init)
            
            gen_delta = self.this_g(z_star, y_)
            output.log_fixed(gen_delta.detach(), None,str(i))
            best_distance_images[:,class_destination,...] = gen_delta.detach()
            
            #release memory
            gen_delta = None
            z_star = None
            y_ = None
        return best_distance_images.detach()

# imports the generator architecture from the chosen cgan
from .bg import BigGANmh as bgmodel

# The generator for mnist was trained using versions of the images padded to be 
# 32x32 (original size 28x28). This class adjusts the size of the generated 
# images by cropping them
class OutputsAdjustedMNIST(bgmodel.Generator):
    def forward(self,z,y):
        return super().forward(z,y)[:,:,2:-2,2:-2]

def init_model(opt):
    if opt.dataset_to_use=='mnist':
        gen_class = OutputsAdjustedMNIST
    else:
        gen_class =  bgmodel.Generator
    net_g = gen_class(resolution=opt.im_size_g,G_attn='0',n_classes = opt.n_classes,G_shared=False,G_lr= 0.0002,SN_eps= 1e-08,G_init= 'N02', n_channels = opt.n_channels).cuda()
    net_g = IterModel(opt, net_g)
    return net_g.cuda()
