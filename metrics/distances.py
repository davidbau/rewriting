import json, sys, argparse, os
from tqdm import tqdm

import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image

from metrics.load_mask import load_mask_info
from metrics.load_seg import load_seg_info_from_exp_name

sys.path.append('./metrics/PerceptualSimilarity')
import models

class PerceptualLoss(nn.Module):
    #TODO: from jacob's code, cite it
    def __init__(self, net='vgg', use_gpu=True, precision='float'):
        """ LPIPS loss with spatial weighting """
        super(PerceptualLoss, self).__init__()
        self.lpips = models.PerceptualLoss(model='net-lin',
                                            net=net,
                                            spatial=True,
                                            use_gpu=use_gpu)
        if use_gpu:
            self.lpips = nn.DataParallel(self.lpips).cuda()
        if precision == 'half':
            self.lpips.half()
        elif precision == 'float':
            self.lpips.float()
        elif precision == 'double':
            self.lpips.double()
        return

    def check_loss_input(self, im0, im1, w):
        """ im0 is out and im1 is target and w is mask"""
        assert list(im0.size())[2:] == list(im1.size())[2:], 'spatial dim mismatch'
        assert list(im0.size())[2:] == list(w.size())[2:], 'spatial dim mismatch'

        if im1.size(0) != 1:
            assert im0.size(0) == im1.size(0)

        if w is not None and w.size(0) != 1:
            assert im0.size(0) == w.size(0)
        return

    def forward(self, im0, im1, w=None):
        """ ims have dimension BCHW while mask is 1HW """
        loss = self.lpips(im0, im1)
        if w is not None:
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss

    def __call__(self, im0, im1, w=None):
        return self.forward(im0, im1, w)


class Dataset():
    def __init__(self, before_imgs, before_seg, after_imgs, srcc=2):
        self.before_seg = before_seg
        self.before_img = before_imgs
        self.after_img = after_imgs
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
        self.srcc = srcc

    def __getitem__(self, key):
        before_seg = torch.load(os.path.join(self.before_seg, f'{key}.pth'), map_location='cpu')[self.srcc]
        
        before_img = self.transform(Image.open(os.path.join(self.before_img, f'{key}.png')))
        after_img = self.transform(Image.open(os.path.join(self.after_img, f'{key}.png')))

        if torch.utils.data.get_worker_info() is None:
            return before_seg, before_img, after_img
        else:
            return before_seg.clone(), before_img.clone(), after_img.clone()

class Sampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        yield from self.indices


def compute_dl(before_imgs, before_seg, after_imgs, indices, src=[1708], srcc=2, batch_size=100):
    dataset = Dataset(before_imgs, before_seg, after_imgs, srcc)
    sampler = Sampler(indices)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, pin_memory=True, sampler=sampler, batch_size=batch_size)

    total = 0
    count = 0

    lpips_model = PerceptualLoss()

    for before_segs, before_imgs, after_imgs in tqdm(loader):
        before_segs = before_segs.cuda()
        before_imgs = before_imgs.cuda()
        after_imgs = after_imgs.cuda()
        masks = torch.ones_like(before_segs)

        #take union of masks
        for index in src:
            masks = masks * (before_segs != index).long() 

        if lpips:
            for before, after, mask in zip(before_imgs, after_imgs, masks):
                before, after, mask = before.unsqueeze(dim=0), after.unsqueeze(dim=0), mask.unsqueeze(dim=0).unsqueeze(dim=0)
                with torch.no_grad():
                    loss = lpips_model(before, after, mask).item()
                total += loss
                count += 1
        elif mask_lpips:
            for before, after, mask in zip(before_imgs, after_imgs, masks):
                before, after, mask = before.unsqueeze(dim=0), after.unsqueeze(dim=0), mask.unsqueeze(dim=0).unsqueeze(dim=0)
                with torch.no_grad():
                    loss = lpips_model(before, after, torch.ones_like(mask)).item()
                total += loss
                count += 1
        else:
            differences = (after_imgs - before_imgs).abs().sum(dim=1)
            total += (differences * masks).sum().item()
            count += masks.long().sum().item()

    print(total, count, total/count)
    return total, count


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='seg2')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--mask_lpips', action='store_true')
    parser.add_argument('--lpips', action='store_true')
    args = parser.parse_args()

    lpips = args.lpips
    mask_lpips = args.mask_lpips

    after_imgs = os.path.join('results/samples', args.exp_name)
    _, dataset, _ = load_mask_info(args.exp_name)
    before_imgs = os.path.join('results/samples', f'{dataset}_clean')
    before_seg = os.path.join('results/samples/seg', f'{dataset}_clean')
    _, srcc, _, src, _ = load_seg_info_from_exp_name(args.exp_name)
        
    total, count = compute_dl(before_imgs, before_seg, after_imgs, torch.arange(10000), src=src, srcc=srcc)

    print(f"after: {args.exp_name}")
    print(f"total={total} count={count}")
