####
# This port of styleganv2 is derived from and perfectly compatible with
# the pytorch port by https://github.com/rosinality/stylegan2-pytorch.
#
# In this reimplementation, all non-leaf modules are subclasses of
# nn.Sequential so that the network can be more easily split apart
# for surgery and direct rewriting.

import os

import torch
from torch.utils import model_zoo

from .models import SeqStyleGAN2
from collections import defaultdict

# TODO: change these paths to non-antonio paths, probably load from url if not exists
WEIGHT_URLS = 'http://wednesday.csail.mit.edu/placesgan/tracer/utils/stylegan2/weights/'
sizes = defaultdict(lambda: 256, faces=1024, car=512)

def load_state_dict(category):
    chkpt_name = f'stylegan2_{category}.pt'
    model_path = os.path.join('weights', chkpt_name)
    os.makedirs('weights', exist_ok=True)

    if not os.path.exists(model_path):
        url = WEIGHT_URLS + chkpt_name
        state_dict = model_zoo.load_url(url, model_dir='weights', progress=True)
        torch.save(state_dict, model_path)
    else:
        state_dict = torch.load(model_path)
    return state_dict
    
def load_seq_stylegan(category, truncation=1.0, **kwargs):  # mconv='seq'):
    ''' loads nn sequential version of stylegan2 and puts on gpu'''
    state_dict = load_state_dict(category)
    size = sizes[category]
    g = SeqStyleGAN2(size, style_dim=512, n_mlp=8, truncation=truncation, **kwargs)
    g.load_state_dict(state_dict['g_ema'],
            latent_avg=state_dict['latent_avg'])
    g.cuda()
    return g
