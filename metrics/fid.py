from __future__ import absolute_import, division, print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
from tqdm import tqdm
import warnings
import torch


def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


def calculate_activation_statistics(images,
                                    sess,
                                    batch_size=50,
                                    verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=200, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images:
        print(
            "warning: batch size is bigger than the data size. setting batch size to data size"
        )
        batch_size = n_images
    n_batches = n_images // batch_size
    pred_arr = np.empty((n_images, 2048))
    for i in tqdm(range(n_batches)):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches),
                  end="",
                  flush=True)
        start = i * batch_size

        if start + batch_size < n_images:
            end = start + batch_size
        else:
            end = n_images

        batch = images[start:end]
        pred = sess.run(inception_layer,
                        {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr


#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean


def pt_to_np(imgs):
    '''normalizes pytorch image in [-1, 1] to [0, 255]'''
    normalized = [((img / 2 + 0.5) * 255).clamp(0, 255) for img in imgs]
    return np.array([img.permute(1, 2, 0).numpy() for img in normalized])


def compute_fid_given_images(fake_images, real_images):
    '''requires that the image batches are numpy format, normalized to 0, 255'''
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if isinstance(fake_images, tuple):
            m1, s1 = fake_images
        else:
            m1, s1 = calculate_activation_statistics(fake_images, sess)
        if isinstance(real_images, tuple):
            m2, s2 = real_images
        else:
            m2, s2 = calculate_activation_statistics(real_images, sess)
    return calculate_frechet_distance(m1, s1, m2, s2)


def compute_fid_given_path(path):
    with np.load(path) as data:
        fake_imgs = data['fake']
        real_imgs = data['real']
    return compute_fid_given_images(fake_imgs, real_imgs)


def load_from_path(source):
    root = '/data/vision/torralba/ganprojects/placesgan/tracer/utils/fid_stats/'
    path = os.path.join(root, f'{source}_stats.npz')
    if os.path.exists(path):
        print('Loading statistics from ', path)
        with np.load(path) as data:
            return data['m'], data['s']
    else:
        print("Stats not found in path", path)
        exit()


def compute_fid(source1, source2):
    if isinstance(source1, str):
        source1 = load_from_path(source1)

    if isinstance(source1, torch.Tensor):
        source1 = pt_to_np(source1)

    if isinstance(source2, str):
        source2 = load_from_path(source2)

    if isinstance(source2, torch.Tensor):
        source2 = pt_to_np(source2)

    return compute_fid_given_images(source1, source2)


if __name__ == '__main__':
    import argparse
    from PIL import Image
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--target')
    args = parser.parse_args()


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    images1 = []
    for file_name in tqdm(os.listdir(args.source)):
        if file_name.lower().endswith(('.png', 'jpeg', '.jpg')):
            path = os.path.join(args.source, file_name)
            images1.append(transform(Image.open(path).convert('RGB')))
    images1 = torch.stack(images1)


    images2 = []
    for file_name in tqdm(os.listdir(args.source)):
        if file_name.lower().endswith(('.png', 'jpeg', '.jpg')):
            path = os.path.join(args.source, file_name)
            images2.append(transform(Image.open(path).convert('RGB')))
    images2 = torch.stack(images2)

    result = compute_fid(images1, images2)
    print(result)
    with open('fid_results.txt', 'a+') as f:
        f.write(args.source + args.target + ':\n')
        f.write(str(result) + '\n')
        



