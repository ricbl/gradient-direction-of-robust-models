"""Functions to apply adversarial attacks
"""
import torch
import numpy as np
from .advertorch import blackbox_attack
def clip(input_tensor, min_tensor, max_tensor):
    return torch.max(torch.min(input_tensor, max_tensor), min_tensor)
import advertorch

#template class serving as basis for PGD attacks using any norm
class PGDAttack(object):
    def __init__(self, opt, model, loss_fn, epsilon, k=40, alpha=0.02, 
        random_start=True, alpha_multiplier=1.):
        self.model = model
        self.k = k
        self.rand = random_start
        self.loss_fn =  loss_fn
        self.range_max = 1
        self.range_min = -1
        self.epsilon = epsilon
        self.alpha = alpha*alpha_multiplier
    
    def perturb(self, X_input, y):
        if self.rand:
            noise_addition = self.get_random_noise_for_initialization(X_input)
            noisy_X = torch.clamp(X_input + noise_addition , self.range_min, self.range_max).cuda()
        else:
            noisy_X = X_input
        X = noisy_X.clone().detach()
        for i in range(self.k):
            self.model.zero_grad()
            X_var = X.clone().detach()
            y_var = y
            X_var.requires_grad_(True)
            scores = self.model(X_var)
            
            loss = self.loss_fn(scores.squeeze(1), y_var.squeeze(1)).sum()
            grad = torch.autograd.grad(loss, X_var, 
                                   retain_graph=False, create_graph=False)[0].detach()
            grad = self.adjust_gradient(grad)
            X = X.detach() + self.alpha * grad
            self.model.zero_grad()
            X = self.limit_norm_of_produced_change(X, X_input)
            X = torch.clamp(X, self.range_min, self.range_max) # ensure valid pixel range
        return X

#class for PGD attack using the L infinity norm.
class LinfPGDAttack(PGDAttack):    
    def limit_norm_of_produced_change(self, X_mod, X_input):
        return clip(X_mod, X_input - self.epsilon, X_input + self.epsilon)
    
    def adjust_gradient(self, grad):
        return torch.sign(grad.detach())
    
    def get_random_noise_for_initialization(self, X_input):
        return torch.tensor(np.random.uniform(-self.epsilon, self.epsilon,X_input.shape).astype('float32')).cuda()

#function to get a random point inside a hypersphere, considering a uniform distribution
def uniform_sphere_sampling(batch_size, dimensions, epsilon):
    #sample from gaussian and normalize it to get a random angle
    x = np.random.normal(size=[batch_size, dimensions]).astype('float32')
    x = x/np.sqrt((x**2).sum(axis=1))[:,None]
    #sample radius from uniform distribution, and apply a root with order depending on the dimension of the data, 
    #to get a random radius with probability proportional to the volume from r to r+dr
    sampled_weird_radius = np.random.uniform(0,1, size=[batch_size])
    radius = sampled_weird_radius**(1./dimensions)*epsilon
    #multiply radius (norm) by the angle unit vector to get the sample
    return radius[:,None]*x

#class for PGD attack using the L2 norm
class L2PGDAttack(PGDAttack):
    def limit_norm_of_produced_change(self, X_mod, X_input):
        delta = X_mod - X_input
        delta_norm = torch.norm(delta.view([X_mod.size(0), -1]), dim = 1)
        #if norm of vector is bigger than allowed norm, set the norm to the allowed norm
        # if norm of vector is smaller than allowed norm, do not change norm
        multiplier = torch.min(delta_norm,torch.tensor(self.epsilon).cuda())
        X_mod = X_input + delta/delta_norm[(...,) + (None,)*(X_mod.ndim-1)]*multiplier[(...,) + (None,)*(X_mod.ndim-1)]
        return X_mod
    
    def adjust_gradient(self, grad):
        return grad/torch.norm(torch.abs(grad.view([grad.size(0),-1]))+1e-20, dim=1)[(...,)+(None,)*(grad.ndim-1)]
    
    def get_random_noise_for_initialization(self, X_input):
        n_dimensions = np.prod(X_input.size()[1:])
        noise_addition =  uniform_sphere_sampling(X_input.size(0), n_dimensions, self.epsilon)
        return torch.tensor(noise_addition.astype('float32')).cuda().view(X_input.size())

#function to call to get the adversarially attacked image
def get_attack(opt, images, labels, model, attack_to_use, loss_fn, epsilon = 0.03, k=None, alpha_multiplier=1.):
    original_model_mode = model.training
    model.eval()
    prev_grad_enabled = torch.is_grad_enabled()
    if opt.blackbox_attack:
        if opt.dataset_to_use=='spheres':
            #adjusting spheres dataset to image dimensionality, so that Square
            # Attack can work
            images = images.view([images.size(0), 1, 20, 25])
            pred_fn = lambda x: model(x.view([x.size(0), 500]))
        else:
            pred_fn = lambda x: model(x)
        def single_pred(x):
            #Square Attack requires a class by class output
            logits = pred_fn(x)
            return torch.cat((-logits,logits),1)
        attack= blackbox_attack.SquareAttack(single_pred, {'inf':'Linf', 'l2':'L2'}[attack_to_use], eps = epsilon, seed = None, loss = 'margin', n_queries = 5000, p_init=0.8)
        adversarial = attack.perturb(images, labels.squeeze())
        if opt.dataset_to_use=='spheres':
            adversarial = adversarial.view([adversarial.size(0), 500])
    else:
        torch.set_grad_enabled(True)
        if attack_to_use=='inf':
            attack_class = LinfPGDAttack
        elif attack_to_use=='l2':
            attack_class = L2PGDAttack
        elif attack_to_use=='cwl2':
            def single_pred(x):
                #Square Attack requires a class by class output
                logits = model(x)
                return torch.cat((-logits,logits),1)
            def cw_attack(opt, model, loss_fn, epsilon, k=40, alpha_multiplier=1):
                return advertorch.attacks.CarliniWagnerL2Attack(predict=single_pred, num_classes=2, learning_rate = 0.02*alpha_multiplier, clip_min = -1)
            attack_class = cw_attack
        if k is None:
            attack = attack_class(opt, model = model, loss_fn = loss_fn, epsilon = epsilon, alpha_multiplier=alpha_multiplier)
        else:
            attack = attack_class(opt, model = model, loss_fn = loss_fn, epsilon = epsilon, k=k, alpha_multiplier=alpha_multiplier)
        if  attack_to_use=='cwl2':
            adversarial = attack.perturb(images, labels.squeeze().long())
        else:
            adversarial = attack.perturb(images, labels)
    torch.set_grad_enabled(prev_grad_enabled)
    if original_model_mode:
        model.train()
    return adversarial