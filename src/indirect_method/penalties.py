"""Cosine similarity function
"""

import torch

def batch_dot(a,b):
    return torch.sum(a.view([a.size(0),-1])*b.view([b.size(0),-1]), dim=1)

def stabilize_variable(v):
    v_sign = torch.sign(v.detach())
    v_sign[v_sign==0] = 1
    return v+v_sign*1e-15

def get_cosine_similarity(a,b):
    a = stabilize_variable(a)
    b = stabilize_variable(b)
    a = a.view([a.size(0),-1])
    b = b.view([b.size(0),-1])
    
    return batch_dot(a,b)/torch.norm(torch.abs(a), dim=1)/torch.norm(torch.abs(b), dim = 1)
