import numpy as np
from PIL import Image
import cv2, os, shutil
from poisson_image_editing import poisson_edit
from tqdm import tqdm
import torch
from torch.nn import functional as F


def mask_edit(edited, target, mask):
    if len(mask.shape) == 2:
        mask = np.stack([mask for _ in range(3)], axis=2)
    elif len(mask.shape) != 3:
        raise NotImplementedError
    return edited * mask + (1 - mask) * target


def flow_edit(flow, src):
    dst = np.zeros(src.shape, dtype=np.uint8)
    cv2.remap(src=src,
              dst=dst,
              map1=flow[:, :, 1].astype(np.float32),
              map2=flow[:, :, 0].astype(np.float32),
              interpolation=cv2.INTER_NEAREST)
    return dst


def laplacian_blur(A, B, m, num_levels=6):
    if len(m.shape) == 2:
        m = np.stack([m for _ in range(3)], axis=2).astype(np.float32)
    elif len(m.shape) != 3:
        raise NotImplementedError
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA = [gpA[num_levels - 1]
           ]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_


def upsample(x, size):
    x_type = x.dtype
    return cv2.resize(x.astype('float32'),
                      dsize=(size, size),
                      interpolation=cv2.INTER_LINEAR).astype(x_type)


def generate_blurs(src_root, corr_root, target_img, mask_path, results_dir):
    size = 256
    for imgnum, file_path in tqdm(get_imgnums(src_root)):
        if os.path.exists(os.path.join(results_dir,
                                       f'naive/naive{imgnum}.png')):
            continue

        path = os.path.join(src_root, file_path)

        if os.path.exists(
                os.path.join(corr_root, str(imgnum), 'no_correspondence.txt')):
            print('copying', imgnum)
            shutil.copy(
                path, os.path.join(results_dir, f'warped/warped{imgnum}.png'))
            shutil.copy(
                path, os.path.join(results_dir,
                                   f'laplace/laplace{imgnum}.png'))
            shutil.copy(path,
                        os.path.join(results_dir, f'naive/naive{imgnum}.png'))
            shutil.copy(
                path, os.path.join(results_dir,
                                   f'poisson/poisson{imgnum}.png'))
        else:
            no_mustache = np.array(
                Image.open(path).convert("RGB").resize((size, size)))
            mustache = np.array(
                Image.open(target_img).convert("RGB").resize((size, size)))
            with np.load(mask_path) as data:
                mask = upsample(data['mask'].astype(np.uint8), size)
            try:
                corr = np.load(os.path.join(corr_root, str(imgnum),
                                            'BtoA.npy'))
                corr = upsample(corr, size=size)
            except Exception as e:
                print(imgnum, 'does not exist')
                continue
            for i in range(no_mustache.shape[0]):
                for j in range(no_mustache.shape[1]):
                    corr[i, j, 0] = i - corr[i, j, 0]
                    corr[i, j, 1] = j - corr[i, j, 1]
            edited = flow_edit(corr, mustache)
            Image.fromarray(edited).save(
                os.path.join(results_dir, f'warped/warped{imgnum}.png'))
            mask = flow_edit(corr, mask)
            laplace = laplacian_blur(edited, no_mustache,
                                     mask).astype(np.uint8)
            Image.fromarray(laplace).save(
                os.path.join(results_dir, f'laplace/laplace{imgnum}.png'))
            poisson = poisson_edit(edited, no_mustache, mask, (0, 0))
            Image.fromarray(poisson).save(
                os.path.join(results_dir, f'poisson/poisson{imgnum}.png'))
            naive = mask_edit(edited, no_mustache, mask)
            Image.fromarray(naive).save(
                os.path.join(results_dir, f'naive/naive{imgnum}.png'))


def check_missing(src_root, corr_root, *args, **kwargs):
    dne = []
    for imgnum, file_path in tqdm(get_imgnums(src_root)):
        if not os.path.exists(os.path.join(
                corr_root, str(imgnum), 'BtoA.npy')) and not os.path.exists(
                    os.path.join(corr_root, str(imgnum),
                                 'no_correspondence.txt')):
            dne.append(imgnum)
    return dne


def get_imgnums(root):
    base_name = os.path.basename(root)
    img_nums = sorted([
        int(f.strip().split(f'{base_name}_')[1].split('.')[0])
        for f in os.listdir(root)
    ])
    file_names = [f'{base_name}_{num}.png' for num in img_nums]
    return list(zip(img_nums, file_names))[:10000]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root')
    parser.add_argument('--corr_root')
    parser.add_argument('--target_img')
    parser.add_argument('--mask')
    parser.add_argument('--results_dir')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_dir, 'warped'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'naive'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'laplace'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'poisson'), exist_ok=True)

    generate_blurs(args.src_root, args.corr_root, args.target_img, args.mask,
                   args.results_dir)

    # src_root = '/data/vision/torralba/ganprojects/placesgan/tracer/utils/samples/clean'
    # corr_root = '/data/vision/torralba/ganprojects/placesgan/tracer/baselines/neural_best_buddies/results/'
    # target_img = ''
    # mask = ''
    # results_dir = ''
