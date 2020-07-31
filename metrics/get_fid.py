from get_samples import get_samples, get_cropped_fake_samples
from get_gt_lsun import get_gt_samples, get_cropped_gt_samples
from metrics import fid
import tensorflow as tf
import numpy as np
import os

N = 50000


def save_stats(imgs, dataset, model):
    if os.path.exists(f'{model}_{dataset}_stats.npz'):
        print(model, dataset, 'exists')
        return

    gpu_options = tf.GPUOptions(visible_device_list="")
    config = tf.ConfigProto(gpu_options=gpu_options)
    inception_path = fid.check_or_download_inception(None)
    fid.create_inception_graph(inception_path)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        m, s = fid.calculate_activation_statistics(imgs, sess)
        np.savez(os.path.join(f'{model}_{dataset}_stats.npz'), m=m, s=s)


def pt_to_np(imgs):
    '''normalizes pytorch image in [-1, 1] to [0, 255]'''
    return (imgs.permute(0, 2, 3, 1).mul_(0.5).add_(0.5).mul_(255)).clamp_(0, 255).numpy()


def get_dataset_statistics():
    generated = []
    gt = ['ffhq']

    for model, dataset in generated:
        imgs = pt_to_np(get_samples(model, dataset, N))
        save_stats(imgs, model, dataset)
        del imgs

    for dataset in gt:
        imgs = pt_to_np(get_gt_samples(dataset, N))
        save_stats(imgs, 'gt', dataset)
        del imgs

    for model, dataset in generated:
        for dataset_gt in gt:
            with np.load(f'{dataset}_{model}_stats.npz') as data:
                m1, s1 = data['m'], data['s']
            with np.load(f'{dataset}_gt_stats.npz') as data:
                m2, s2 = data['m'], data['s']
            print(model, dataset, dataset_gt, fid.calculate_frechet_distance(m1, s1, m2, s2))


def get_cropped_dataset_statistics():
    crop_sizes = [128]
    datasets = ['church']  # 'celeba-hq', 'ffhq']
    for dataset in datasets:
        samples = get_cropped_gt_samples(dataset, nimgs=N, crop_sizes=crop_sizes)
        for size, imgs in zip(crop_sizes, samples):
            imgs = pt_to_np(imgs)
            save_stats(imgs, f'cropped_{size}', dataset)


def get_truncation_samples():
    generated = ['church', 'kitchen']

    for dataset in generated:
        imgs = pt_to_np(get_samples('stylegan', dataset, N, truncated=True))
        save_stats(imgs, 'truncated', dataset)
        del imgs

    for dataset in generated:
        with np.load(f'fid_stats/{dataset}_gt_stats.npz') as data:
            m1, s1 = data['m'], data['s']
        with np.load(f'{dataset}_truncated_stats.npz') as data:
            m2, s2 = data['m'], data['s']
        print('stylegan', dataset, fid.calculate_frechet_distance(m1, s1, m2, s2))


def get_cropped_fake_statistics():
    crop_sizes = [8, 16, 32, 64, 128]
    datasets = ['celebhq']
    models = ['stylegan']

    for model in models:
        truncations = [False] if model == 'proggan' else [True, False]
        for truncated in truncations:
            for dataset in datasets:
                samples = get_cropped_fake_samples(model, dataset, nimgs=N, crop_sizes=crop_sizes, truncated=truncated)
                for size, imgs in zip(crop_sizes, samples):
                    imgs = pt_to_np(imgs)
                    if truncated:
                        save_stats(imgs, f'truncated_cropped_{size}_{model}', dataset)
                    else:
                        save_stats(imgs, f'cropped_{size}_{model}', dataset)


if __name__ == '__main__':
    # to compute fid with two random batches. around 0.4 for church 8 0.9 for church 16
    # dataset = 'church'
    # crop = 16

    # with np.load(f'{dataset}_cropped_{crop}_stats.npz') as data:
    #     m1, s1 = data['m'], data['s']
    # with np.load(f'1{dataset}_cropped_{crop}_stats.npz') as data:
    #     m2, s2 = data['m'], data['s']
    # print(dataset, crop, fid.calculate_frechet_distance(m1, s1, m2, s2))

    # get_cropped_dataset_statistics()
    # get_truncation_samples()
    get_cropped_fake_statistics()
