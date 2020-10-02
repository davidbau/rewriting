####
# This port of styleganv2 is derived from and perfectly compatible with
# the pytorch port by https://github.com/rosinality/stylegan2-pytorch.
#
# In this reimplementation, all non-leaf modules are subclasses of
# nn.Sequential so that the network can be more easily split apart
# for surgery and direct rewriting.

import os

import torch

from .models import SeqStyleGAN2
from collections import defaultdict

# TODO: change these paths to non-antonio paths, probably load from url if not exists
WEIGHT_URLS = 'http://rewriting.csail.mit.edu/data/models/'
sizes = defaultdict(lambda: 256, faces=1024, car=512)

FILENAMES = dict(
    bedroom='stylegan2_bedroom-6fa55a6e.pt',
    car='stylegan2_car-3659b4b6.pt',
    cat='stylegan2_cat-d8dc98b2.pt',
    church='stylegan2_church-e8ca9fd0.pt',
    faces='stylegan2_faces-2858cc2e.pt',
    horse='stylegan2_horse-499b5380.pt',
    kitchen='stylegan2_kitchen-b3a526e9.pt',
    places='stylegan2_places-a3b72d71.pt'
)

def load_state_dict(category):
    url = WEIGHT_URLS + FILENAMES[category]
    try:
        sd = torch.hub.load_state_dict_from_url(url)  # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url)  # pytorch 1.0
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
