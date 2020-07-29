import argparse, os
from tqdm.auto import tqdm

import torch

from metrics.load_seg import load_seg_info_from_exp_name
from metrics.load_mask import load_mask_info

class Dataset():
    def __init__(self, before, after, tgtc=0, srcc=2):
        self.before = before
        self.after = after
        self.tgtc = tgtc
        self.srcc = srcc

    def __getitem__(self, key):
        before_seg = torch.load(os.path.join(
            self.before, f'{key}.pth'), map_location='cpu')[self.srcc]
        after_seg = torch.load(os.path.join(
            self.after, f'{key}.pth'), map_location='cpu')[self.tgtc]
        if torch.utils.data.get_worker_info() is None:
            return before_seg, after_seg
        else:
            return before_seg.clone(), after_seg.clone()


class Sampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        yield from self.indices


def compute_dl(before, after, indices, tgt=[5], tgtc=0, src=[1708], srcc=2, batch_size=100):
    print(src, tgt)

    dataset = Dataset(before, after, tgtc, srcc)
    sampler = Sampler(indices)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, pin_memory=True, sampler=sampler, batch_size=batch_size)

    total = 0
    count = 0

    for before_segs, after_segs in tqdm(loader):
        before_segs = before_segs.cuda()
        after_segs = after_segs.cuda()
        before_mask = torch.zeros_like(before_segs)
        for srci in src:
            before_mask = before_mask + (before_segs == srci).long()
        mapped = after_segs[before_mask > 0]
        after_mask = torch.zeros_like(mapped)
        for tgti in tgt:
            after_mask = after_mask + (mapped == tgti).long()
        total += (after_mask > 0).sum().item()
        count += mapped.shape[0]
    
    print(total, count)
    return total, count


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()

    _, dataset, _ = load_mask_info(args.exp_name)
    before = os.path.join('results/samples/seg', f'{dataset}_clean')
    after = os.path.join('results/samples/seg', args.exp_name)
    _, srcc, tgtc, src, tgt = load_seg_info_from_exp_name(args.exp_name)

    total, count = compute_dl(before, after, torch.arange(10000), tgt, tgtc, src, srcc)

    print(f"before: {before}")
    print(f"after: {args.exp_name}")
    print(f"total={total} count={count}")
