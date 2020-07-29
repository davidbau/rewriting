import argparse
import torch, copy, os, json

from tqdm import tqdm

from netdissect import ganrewrite
from netdissect import proggan, setting, zdataset, show
import utils.stylegan2
from utils.stylegan2 import load_seq_stylegan
from torchvision.utils import save_image
from fid import compute_fid

parser = argparse.ArgumentParser()
parser.add_argument('--layernum', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--crop_size', type=int, required=True)
parser.add_argument('--nimgs', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=10)

args = parser.parse_args()

layernum = args.layernum
ganname = args.model
modelname = args.dataset
batch_size = args.batch_size
crop_size = args.crop_size
N = 10
def main():
    expdir = 'results/pgw/%s/%s/layer%d' % (ganname, modelname, layernum)

    if ganname == 'proggan':
        model = setting.load_proggan(modelname).cuda()
        zds = zdataset.z_dataset_for_model(model, size=1000)
        writer = ganrewrite.ProgressiveGanRewriter
    elif ganname == 'stylegan':
        model = load_seq_stylegan(modelname, mconv='seq')
        zds = zdataset.z_dataset_for_model(model, size=1000)
        writer = ganrewrite.SeqStyleGanRewriter

    model.eval()
    gw = writer(model, zds, layernum, cachedir=expdir)

    images = []
    with torch.no_grad():
        for _ in tqdm(range(N//batch_size + 1)):
            z = zdataset.z_sample_for_model(model, size=batch_size, seed=len(images)).cuda()
            samples = gw.sample_image_patch(z, crop_size)
            samples = [s.data.cpu() for s in samples]
            images.extend(samples)
        images = torch.stack(images[:N], dim=0)
    
    gt_fid = 0
    fake_fid = compute_fid(images, f'{modelname}_cropped_{images.size(2)}_{ganname}')
    save_image(images[:32] * 0.5 + 0.5, f'patches_{layernum}_{ganname}_{modelname}_{crop_size}.png')

    return fake_fid, gt_fid, images.size(2)

if __name__ == '__main__':
    _, _, size = main()
    result_name = f'{ganname}_{modelname}_{layernum}_{size}'
    with open('patch_fid.txt') as f:
        results = {k: (float(v1), float(v2)) for k, v1, v2 in [l.strip().split(" ") for l in f.readlines()]}
        if result_name in results: 
            print(result_name, 'exists')
            exit()

    N = args.nimgs
    print('working', result_name)
    fake_fid, gt_fid, size = main()
    print(f'fake {fake_fid} gt {gt_fid}')
    with open('patch_fid.txt', 'a+') as f:
        f.write(f'{ganname}_{modelname}_{layernum}_{size} {fake_fid} {gt_fid}\n')