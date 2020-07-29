import os, sys, shutil
import argparse
from tqdm import tqdm

import torch
from utils.stylegan2 import load_seq_stylegan
from utils.pidfile import reserve_dir
from utils import zdataset
from torchvision.transforms import ToPILImage
from utils.imgsave import SaveImagePool

parser = argparse.ArgumentParser('Sample clean images from a generator')
parser.add_argument('--dataset', help='dataset to sample from',
        choices=['faces', 'church', 'horse'])
parser.add_argument('--fid_samples', action='store_true', help='generate a different set of clean samples')
args = parser.parse_args()

N = 10000
offset = 1000007 if args.fid_samples else 0
name = f'{args.dataset}_clean'
if args.fid_samples: name += '_fid'
rd = reserve_dir(os.path.join('results/samples', name))
shutil.copyfile('utils/lightbox.html', rd('+lightbox.html'))

model = load_seq_stylegan(args.dataset, mconv='seq', truncation=0.5)
model.eval()

saver = SaveImagePool()
to_pil = ToPILImage()
with torch.no_grad():
    for imgnum in tqdm(range(N)):
        z = zdataset.z_sample_for_model(model, size=1, seed=imgnum+offset).cuda()
        x_real = model(z).detach().cpu()
        saver.add(to_pil(x_real[0] * 0.5 + 0.5), rd(f'{imgnum}.png'))
saver.join()
rd.done()
