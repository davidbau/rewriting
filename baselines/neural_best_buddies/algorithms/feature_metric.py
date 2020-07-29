import torch
import math
from torch.autograd import Variable
import numpy as np
from PIL import Image
import numpy as np

def error_map(A, B):
    error_map = (A-B).pow(2).sum(1, keepdim=True)/A.size(1)
    return error_map.pow(0.5)

def patch_distance(A, B, average=True):
    error = (A-B).pow(2).sum()
    if average==False:
        return error
    else:
        tensor_volume = A.size(0)*A.size(1)*A.size(2)*A.size(3)
        return error/tensor_volume

def normalize_per_pix(A):
    return A/A.pow(2).sum(1, keepdim=True).pow(0.5).expand_as(A)

def normalize_tensor(A):
    return A/np.power(A.pow(2).sum(), 0.5)

def stretch_tensor_0_to_1(F):
    assert(F.dim() == 4)
    max_val = F.max()
    min_val = F.min()
    if max_val != min_val:
        F_normalized = (F - min_val)/(max_val-min_val)
    else:
        F_normalized = F.fill_(0)
    return F_normalized

def response(F, style='l2'):
    if style=='max':
        [response, indices] = F.max(1,keepdim=True)
    elif style=='l2':
        response = F.pow(2).sum(1,keepdim=True).pow(0.5)
    else:
        raise ValueError("unknown response style: ", style)
    return response

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum.expand_as(F)/(F.size(2)*F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    epsilon = 10**-20
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True)/(F.size(2)*F.size(3)) + epsilon
    return F_variance.expand_as(F).pow(0.5)

def identity_map(size):
    idnty_map = torch.Tensor(size[0],2,size[2],size[3])
    idnty_map[0,0,:,:].copy_(torch.arange(0,size[2]).repeat(size[3],1).transpose(0,1))
    idnty_map[0,1,:,:].copy_(torch.arange(0,size[3]).repeat(size[2],1))
    return idnty_map

def gaussian(kernel_width, stdv = 1):
    w = identity_map([1,1,kernel_width,kernel_width]) - math.floor(kernel_width/2)
    kernel = torch.exp(-w.pow(2).sum(1,keepdim=True)/(2*stdv))
    return kernel
