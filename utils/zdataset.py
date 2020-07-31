import torch
import numpy
import itertools
from torch.utils.data import TensorDataset


def z_dataset_for_model(model, size=100, seed=1, indices=None):
    if indices is not None:
        indices = torch.as_tensor(indices, dtype=torch.int64, device='cpu')
        zs = z_sample_for_model(model, indices.max().item() + 1, seed)
        zs = zs[indices]
    else:
        zs = z_sample_for_model(model, size, seed)
    return TensorDataset(zs)


def z_sample_for_model(model, size=100, seed=1):
    # If the model is marked with an input shape, use it.
    if hasattr(model, 'input_shape'):
        sample = standard_z_sample(size, model.input_shape[1], seed=seed).view(
            (size,) + model.input_shape[1:])
        return sample
    # Examine first conv in model to determine input feature size.
    first_layer = [c for c in model.modules()
                   if isinstance(c, (torch.nn.Conv2d, torch.nn.ConvTranspose2d,
                                     torch.nn.Linear))][0]
    # 4d input if convolutional, 2d input if first layer is linear.
    if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        sample = standard_z_sample(
            size, first_layer.in_channels, seed=seed)[:, :, None, None]
    else:
        sample = standard_z_sample(
            size, first_layer.in_features, seed=seed)
    return sample


def standard_z_sample(size, depth, seed=1, device=None):
    '''
    Generate a standard set of random Z as a (size, z_dimension) tensor.
    With the same random seed, it always returns the same z (e.g.,
    the first one is always the same regardless of the size.)
    '''
    # Use numpy RandomState since it can be done deterministically
    # without affecting global state
    rng = numpy.random.RandomState(seed)
    result = torch.from_numpy(
        rng.standard_normal(size * depth)
        .reshape(size, depth)).float()
    if device is not None:
        result = result.to(device)
    return result


def standard_y_sample(size, num_classes, seed=1, device=None):
    '''
    Generate a standard set of random categorical as a (size,) tensor
of integers up to (num_classes-1).
    With the same random seed, it always returns the same y (e.g.,
    the first one is always the same regardless of the size.)
    '''
    # Use numpy RandomState since it can be done deterministically
    # without affecting global state
    rng = numpy.random.RandomState(seed)
    result = torch.from_numpy(
        rng.randint(num_classes, size=size)).long()
    if device is not None:
        result = result.to(device)
    return result


def training_loader(z_generator, batch_size, loader_size=10000):
    '''
    Returns an infinite generator that runs through randomized z
    batches, forever.
    '''
    g_epoch = 1
    while True:
        z_data = z_dataset_for_model(
            z_generator, size=epoch_size, seed=g_epoch + 1)
        dataloader = torch.utils.data.DataLoader(
            z_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True)
        for batch in dataloader:
            yield batch
        g_epoch += 1


def testing_loader(z_generator, batch_size, test_size=1000):
    '''
    Returns an a short iterator that returns a small set of test data.
    '''
    z_data = z_dataset_for_model(
        z_generator, size=test_size, seed=1)
    dataloader = torch.utils.data.DataLoader(
        z_data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True)
    return dataloader


def epoch_grouper(loader, epoch_size):
    '''
    To use with the infinite training loader: groups the training data
    batches into epochs of the given size.
    '''
    it = iter(loader)
    while True:
        chunk_it = itertools.islice(it, epoch_size)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
