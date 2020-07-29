import argparse, os
from tqdm.auto import tqdm

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from utils.workerpool import WorkerBase, WorkerPool
from utils.pidfile import reserve_dir

from metrics.load_seg import load_seg_info_from_exp_name, load_seg

N = 10000

def process(segmodel, img_path, result_path, batch_size=128, transform=None,
        device='cuda', **kwargs):
    rd = reserve_dir(result_path)
    saver = SaveSegPool()
    with torch.no_grad():
        for i in tqdm(range(N)):
            img = transform(Image.open(os.path.join(img_path, f'{i}.png')).convert('RGB'))
            segs = segmodel.segment_batch(img.to(device)[None], **kwargs).detach().cpu()
            saver.add([os.path.join(result_path, f'{i}.pth')], segs)
            del segs
    saver.join()
    rd.done()

class SaveSegWorker(WorkerBase):
    def work(self, paths, segs):
        for path, seg in zip(paths, segs):
            torch.save(seg, path)

class SaveSegPool(WorkerPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, worker=SaveSegWorker, **kwargs)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='seg')
    parser.add_argument('exp_name', type=str)
    args = parser.parse_args()
    
    segmodel_name, _, _, _, _= load_seg_info_from_exp_name(args.exp_name)
    
    segmodel = load_seg(segmodel_name)
    img_path = os.path.join('results/samples', args.exp_name)
    result_path = os.path.join('results/samples/seg', args.exp_name)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    process(segmodel, img_path, result_path, batch_size=8, transform=transform)
