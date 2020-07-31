import os
import json
import argparse
import shutil
from tqdm import tqdm

import torch

from utils.stylegan2 import load_seq_stylegan
from utils.pidfile import reserve_dir
from utils.imgsave import SaveImagePool
from utils import zdataset
from torchvision.transforms import ToPILImage
from rewrite import ganrewrite
from .load_mask import load_mask_info

N = 10000

parser = argparse.ArgumentParser('sample edited images')
parser.add_argument('--mask', type=str)
parser.add_argument('--full_rank', action='store_true')
parser.add_argument('--no_tight_paste', action='store_true')
parser.add_argument('--single_context', type=int, default=-1)
args = parser.parse_args()

exp_name = args.mask
if args.full_rank:
    exp_name = exp_name + '_full_rank'
if args.single_context != -1:
    exp_name = exp_name + f'_context{args.single_context}'
rd = reserve_dir(os.path.join('results/samples', exp_name))
shutil.copyfile('utils/lightbox.html', rd('+lightbox.html'))

mask, dataset, layernum = load_mask_info(args.mask)
model = load_seq_stylegan(dataset, mconv='seq', truncation=0.5)
model.eval()

zds = zdataset.z_dataset_for_model(model, size=1000)
writer = ganrewrite.SeqStyleGanRewriter

gw = writer(model,
            zds,
            layernum=layernum,
            cachedir='results/rewrite/%s/%s/layer%d' % ('stylegan', dataset, layernum),
            low_rank_insert=not args.full_rank,
            key_method='zca',
            tight_paste=not args.no_tight_paste)

with open(mask) as f:
    print('Loading mask', mask)
    gw.apply_edit(json.load(f), rank=1, single_key=args.single_context)

saver = SaveImagePool()
to_pil = ToPILImage()
with torch.no_grad():
    for imgnum in tqdm(range(N)):
        z = zdataset.z_sample_for_model(model, size=1, seed=imgnum).cuda()
        x_real = gw.sample_image_from_latent(z).detach().cpu()
        saver.add(to_pil(x_real[0] * 0.5 + 0.5), rd(f'{imgnum}.png'))
saver.join()
rd.done()
