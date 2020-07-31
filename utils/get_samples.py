import os
import random

import torch
from tqdm import tqdm
from PIL import Image

from utils.stylegan2 import load_seq_stylegan
from netdissect import setting, zdataset
from torchvision import transforms

paths = {
    'ffhq': '/data/vision/torralba/datasets/ffhq/images1024x1024/',
    'celeba-hq':
    '/data/vision/torralba/datasets/celeba-hq/celeba-hq/celeba-hq/celeba-1024/',
    'church':
    '/data/vision/torralba/datasets/LSUN/image/church_outdoor_train/',
    'kitchen': '/data/vision/torralba/datasets/LSUN/image/kitchen_train/',
    'places':
    '/data/vision/torralba/datasets/places/places365_standard/places365standard_easyformat/train',
    'imagenet': '/data/vision/torralba/datasets/imagenet_pytorch_old/train'
}

sizes = {
    'ffhq': 1024,
    'church': 256,
    'celeba-hq': 1024,
    'kitchen': 256,
    'imagenet': 128,
    'places': 128
}


def get_image_paths(root, N):
    ''' returns list of paths of up to N images under root '''
    if os.path.exists(root + '.txt'):
        with open(root + '.txt') as f:
            all_files = [
                os.path.join(root,
                             fi.strip().split('train/')[1])
                for fi in f.readlines()
            ]
        random.shuffle(all_files)
        return all_files[:N]
    else:
        all_files = []
        for dp, dn, fn in os.walk(os.path.expanduser(root)):
            for f in fn:
                if f.endswith(('.png', '.webp', 'jpg')):
                    all_files.append(os.path.join(dp, f))
                else:
                    print('skipped', f)
                if len(all_files) >= N:
                    return all_files
        return all_files


def get_transform(size):
    ''' returns transform from PIL image to pytorch tensor in [-1,1]'''
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_cropped_gt_samples(dataset, nimgs=50000, crop_sizes=[32]):
    ''' returns list of cropped real images, sampled uniformily over all crop sizes'''
    # TODO: clean up duplicate code with get_cropped_fake_samples
    size = sizes[dataset]
    transform = get_transform(size)
    all_images = get_image_paths(paths[dataset], nimgs)
    images = [[] for _ in range(len(crop_sizes))]

    for file_path in tqdm(all_images[:nimgs]):
        for i, crop_size in enumerate(crop_sizes):
            end = size - crop_size
            img = transform(Image.open(file_path).convert('RGB'))
            xi, yi = random.randint(0, end), random.randint(0, end)
            images[i].append(img[:, xi:xi + crop_size, yi:yi + crop_size])
    for i in range(len(crop_sizes)):
        images[i] = torch.stack(images[i])
    return images


def get_gt_samples(dataset, nimgs=50000):
    ''' returns torch tensor of gt images '''
    transform = get_transform(sizes[dataset])
    all_images = get_image_paths(paths[dataset], nimgs)
    images = []
    for file_path in tqdm(all_images[:nimgs]):
        images.append(transform(Image.open(file_path).convert('RGB')))
    images = torch.stack(images)
    return images


def load_model(model, name, truncated):
    if model == 'stylegan':
        if truncated:
            print('loading from truncated stylegan')
            g = load_seq_stylegan(name, truncation=0.5, mconv='seq')
        else:
            print('loading from stylegan w/o truncation')
            g = load_seq_stylegan(name, mconv='seq')
    elif model == 'proggan':
        print('loading from progressive gan')
        g = setting.load_proggan(name).cuda()
    else:
        raise NotImplementedError()
    return g


def get_samples(model, name, nimgs=50000, truncated=False):
    batch_size = 10
    g = load_model(model, name, truncated)
    g.eval()

    with torch.no_grad():
        samples = []
        for _ in tqdm(range(nimgs // batch_size + 1)):
            seed = len(samples) if samples is not None else 0
            z = zdataset.z_sample_for_model(g, size=batch_size,
                                            seed=seed).cuda()
            x_real = g(z)
            x_real = [x.detach().cpu() for x in x_real]
            samples.extend(x_real)
        samples = torch.stack(samples, dim=0)
        return samples


def seeded_cropped_sample(g,
                          gw,
                          imgnum,
                          crop_seed,
                          crop_size,
                          act=True,
                          size=None):

    with torch.no_grad():
        z = zdataset.z_sample_for_model(g, size=1, seed=imgnum).cuda()
        return gw.sample_image_patch(z,
                                     crop_size,
                                     seed=crop_seed,
                                     act=act,
                                     size=size)


def get_cropped_fake_samples(model,
                             dataset,
                             nimgs=50000,
                             crop_sizes=[32],
                             truncated=False,
                             seed=(None, None)):
    all_images = get_samples(model, dataset, nimgs, truncated)

    size = all_images.size(2)
    images = [[] for _ in range(len(crop_sizes))]

    for img in tqdm(all_images):
        for i, crop_size in enumerate(crop_sizes):
            end = size - crop_size
            xi, yi = random.randint(0, end), random.randint(0, end)
            images[i].append(img[:, xi:xi + crop_size, yi:yi + crop_size])

    for i in range(len(crop_sizes)):
        images[i] = torch.stack(images[i])

    return images
