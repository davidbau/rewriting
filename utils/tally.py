'''
Batchwise tally functions, analogous to tensor.topk, mean+variance,
bincount, covaraince, and sort (for quantiles), implemented in a way
that permits fast computation of statistics over large data sets that
do not fit in memory at once.

These functions are useful because, while many statistics are much
cheaper to compute on the GPU than on the CPU, they may require too
much memory to compute all at once.  Instead the statistics need
to be computed in a running fashion, one batch at a time, and
accumulated in a way that economizes GPU memory.

Use the tally functions by passing a batch computation function and
an underlying dataset.  A DataLoader will be created, and then the
function will be called to compute samples of data to tally.

Underlying running statistics algorithms are implemented in the
runningstats package.
'''
import torch
import numpy
import os
from . import runningstats, pbar
from .sampler import FixedSubsetSampler
from collections import defaultdict
import warnings


def tally_each(compute, dataset, sample_size=None, batch_size=10,
               summarize=None, cachefile=None, **kwargs):
    '''
    Calls compute on batches of data.
    '''
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return TensorDict(state=cached_state).data
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    for batch in pbar(loader):
        call_compute(compute, batch)
    if summarize is not None:
        result = summarize()
        save_cached_state(cachefile, TensorDict(data=result), args)
        return result


def tally_topk(compute, dataset, sample_size=None, batch_size=10, k=100,
               cachefile=None, **kwargs):
    '''
    Computes the topk statistics for a large data sample that can be
    computed from a dataset.  The compute function should return one
    batch of samples as a (sample, unit)-dimension tensor.

    k specifies the number of top samples to retain.
    Results are returned as a RunningTopK object.
    '''
    args = dict(sample_size=sample_size, k=k)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningTopK(state=cached_state)
    rtk = runningstats.RunningTopK(k=k)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    for batch in pbar(loader):
        sample = call_compute(compute, batch)
        rtk.add(sample)
    rtk.to_('cpu')
    save_cached_state(cachefile, rtk, args)
    return rtk


def gather_topk(compute, dataset, topk, k=None,
                cachefile=None, **kwargs):
    '''
    Gathers data from topk examples computed over a dataset.
    The compute function will be provided a batch of data together
    with a list of units [[(unit, rank), ...], [(unit, rank),...]]
    with one list of relevant units for each batch item.  It should
    return a generator that yields [(unit, rank), data]
    '''

    if k is None:
        k = topk.k
    args = dict(k=k, count=topk.count)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.GatherTensor(state=cached_state)
    gt = runningstats.GatherTensor(topk=topk, k=k)
    needed_images = defaultdict(list)
    for unit, imgnums in enumerate(topk.result()[1][:, :k]):
        for rank, imgnum in enumerate(imgnums.numpy()):
            needed_images[imgnum].append((unit, rank))
    needed_sample = FixedSubsetSampler(sorted(needed_images.keys()))
    loader = make_loader(dataset, sampler=needed_sample, **kwargs)
    for batchnum, batch in enumerate(pbar(loader)):
        assert isinstance(batch, list)
        batch.insert(0, [needed_images[
            needed_sample[index + batchnum * loader.batch_size]]
            for index in range(len(batch[0]))])
        sample = call_compute(compute, batch)
        for (unit, rank), data in sample:
            gt.add(unit, rank, data)
    save_cached_state(cachefile, gt, args)
    return gt


def tally_conditional_topk(compute, dataset, k=100,
                           batch_size=50, sample_size=None, cachefile=None, **kwargs):
    '''
    Computes conditional topk, e.g., top k examples for each class.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningConditionalTopK(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    ctk = runningstats.RunningConditionalTopK(k=k)
    for i, batch in enumerate(pbar(loader)):
        # Add the index to the batch
        first_index = i * loader.batch_size
        index_batch = torch.arange(first_index, first_index + len(batch[0]))
        batch.append(index_batch)
        sample_set = call_compute(compute, batch)
        for cond, data, index in sample_set:
            ctk.add(cond, data, index)
    # At the end, move all to the CPU
    ctk.to_('cpu')
    save_cached_state(cachefile, ctk, args)
    return ctk


def tally_quantile(compute, dataset, sample_size=None, batch_size=10,
                   r=4096, cachefile=None, **kwargs):
    '''
    Computes quantile sketch statistics for a large data sample that can
    be computed from a dataset.  The compute function should return one
    batch of samples as a (sample, unit)-dimension tensor.

    The underlying quantile sketch is an optimal KLL sorted sampler that
    retains at least r samples (where r is the specified resolution).
    '''

    args = dict(sample_size=sample_size, r=r)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningQuantile(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    rq = runningstats.RunningQuantile()
    for batch in pbar(loader):
        sample = call_compute(compute, batch)
        rq.add(sample)
    rq.to_('cpu')
    save_cached_state(cachefile, rq, args)
    return rq


def tally_topk_and_quantile(compute, dataset, sample_size=None,
                            batch_size=10, k=100, r=4096, cachefile=None, **kwargs):
    '''
    Computes both topk and quantile statistics in one pass over the
    data.  The compute function should return a pair (first for topk
    and second for quantile stats).
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size, k=k, r=r)
    cached_state = load_cached_state(cachefile, args)
    cs = CombinedState(state=cached_state,
                       rtk=runningstats.RunningTopK(k=k),
                       rq=runningstats.RunningQuantile(r=r))
    if cached_state is not None:
        return cs.rtx, cs.rq
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    for batch in pbar(loader):
        sample_tk, sample_q = call_compute(compute, batch)
        cs.rtk.add(sample_tk)
        cs.rq.add(sample_q)
    cs.rtk.to_('cpu')
    cs.rq.to_('cpu')
    save_cached_state(cachefile, cs, args)
    return cs.rtk, cs.rq


def tally_conditional_quantile(compute, dataset,
                               sample_size=None, batch_size=1, gpu_cache=64, r=1024,
                               cachefile=None, **kwargs):
    '''
    Computes conditional quantile sketches for a large data sample that
    can be computed from a dataset.  The compute function should return a
    sequence of sample batch tuples (condition, (sample, unit)-tensor),
    one for each condition relevant to the batch.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size, r=r)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningConditionalQuantile(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    cq = runningstats.RunningConditionalQuantile(r=r)
    most_common_conditions = set()
    for i, batch in enumerate(pbar(loader)):
        sample_set = call_compute(compute, batch)
        for cond, sample in sample_set:
            # Move uncommon conditional data to the cpu before collating.
            if cond not in most_common_conditions:
                sample = sample.cpu()
            cq.add(cond, sample)
        # Move uncommon conditions off the GPU.
        if i and not i & (i - 1):  # if i is a power of 2:
            common_conditions = set(cq.most_common_conditions(gpu_cache))
            cq.to_('cpu', [k for k in cq.keys()
                           if k not in common_conditions])
    # At the end, move all to the CPU
    cq.to_('cpu')
    save_cached_state(cachefile, cq, args)
    return cq


def conditional_samples(activations, segments):
    '''
    Helper function when defining generators for *_conditional tallies.
    Transforms a batch of activations and segmentations into a
    sequence of conditional statistics, i.e., activations that
    are at the same location as the segmentation label.
    Both activations nad segments should be 4d tensors with
    the same sample, y, and x dimensions.  Segments can be
    a multilabel segmentation.  The zero segmentation value is
    assumed to be unused.

    Returns a generator for a sequence of (condition, (sample, unit)-tensor)
    listing every condition present in the segments, along with the
    set of activations overlapping that condition.  The activation tensor
    is 2d in (sample, unit) order, where sample is the number of samples
    with for the condition.
    '''
    channels = activations.shape[1]
    activations_by_channel = activations.permute(0, 2, 3, 1).contiguous()
    segcounts = segments.view(-1).bincount()
    conditions = (segcounts[1:].nonzero() + 1)[:, 0]

    def sample_generator():
        # First yield the full set of activations, unconditioned
        yield (0, activations_by_channel.view(-1, channels))
        # Then a set of activations for each condition present in the image
        for condition in conditions:
            mask = (segments == condition).max(1)[0][..., None]
            mask = mask.expand(activations_by_channel.shape)
            yield (condition.item(),
                   activations_by_channel[mask].view(-1, channels))
    return sample_generator()


def tally_mean(compute, dataset, sample_size=None, batch_size=10,
               cachefile=None, **kwargs):
    '''
    Computes unitwise mean and variance stats for a large data sample that
    can be computed from a dataset.  The compute function should return one
    batch of samples as a (sample, unit)-dimension tensor.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningVariance(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    rv = runningstats.RunningVariance()
    for batch in pbar(loader):
        sample = call_compute(compute, batch)
        rv.add(sample)
    rv.to_('cpu')
    save_cached_state(cachefile, rv, args)
    return rv


def tally_conditional_mean(compute, dataset,
                           sample_size=None, batch_size=1, cachefile=None, **kwargs):
    '''
    Computes conditional mean and variance for a large data sample that
    can be computed from a dataset.  The compute function should return a
    sequence of sample batch tuples (condition, (sample, unit)-tensor),
    one for each condition relevant to the batch.
    '''

    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningConditionalVariance(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    cv = runningstats.RunningConditionalVariance()
    for i, batch in enumerate(pbar(loader)):
        sample_set = call_compute(compute, batch)
        for cond, sample in sample_set:
            # Move uncommon conditional data to the cpu before collating.
            cv.add(cond, sample)
    # At the end, move all to the CPU
    cv.to_('cpu')
    save_cached_state(cachefile, cv, args)
    return cv


def tally_bincount(compute, dataset, sample_size=None, batch_size=10,
                   multi_label_axis=None, cachefile=None, **kwargs):
    '''
    Computes bincount totals for a large data sample that can be
    computed from a dataset.  The compute function should return one
    batch of samples as a (sample, unit)-dimension tensor.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningBincount(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    rbc = runningstats.RunningBincount()
    for batch in pbar(loader):
        sample = call_compute(compute, batch)
        if multi_label_axis:
            multilabel = sample.shape[multi_label_axis]
            size = sample.numel() // multilabel
        else:
            size = None
        rbc.add(sample, size=size)
    rbc.to_('cpu')
    save_cached_state(cachefile, rbc, args)
    return rbc


def tally_cat(compute, dataset, sample_size=None, batch_size=10,
              cachefile=None, **kwargs):
    '''
    Computes a concatenated tensor for data batches that can be
    computed from a dataset.  The compute function should return
    a tensor that should be concatenated to the others along its
    first dimension.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return torch.from_numpy(cached_state['data'])
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    result = []
    for batch in pbar(loader):
        result.append(call_compute(compute, batch).cpu())
    data = torch.cat(result)
    save_cached_state(cachefile, SavedTensor(data), args)
    return data


def tally_cat_dict(compute, dataset, sample_size=None, batch_size=10,
                   cachefile=None, **kwargs):
    '''
    Computes a dict of concatenated tensors for data batches that can
    be computed from a dataset.  The compute function should return a
    dict of tensors tensor that can be concatenated to previous results
    with the same key along their first dimensions.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return {k: torch.from_numpy(v) for k, v in cached_state.items()}
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    result = defaultdict(list)
    for batch in pbar(loader):
        for k, v in call_compute(compute, batch).items():
            result[k].append(v.cpu())
    data = {k: torch.cat(v) for k, v in result.items()}
    data.update(args)

    class SavedDict:
        def __init__(self, data):
            self.data = data

        def state_dict(self):
            return data
    save_cached_state(cachefile, SavedDict(data), args)
    return data


def tally_covariance(compute, dataset, sample_size=None, batch_size=10,
                     cachefile=None, **kwargs):
    '''
    Computes covariance statistics for a large data sample that can
    be computed from a dataset.  The compute function should return one
    batch of samples as a (sample, unit)-dimension tensor.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningCovariance(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    rcov = runningstats.RunningCovariance()
    for batch in pbar(loader):
        sample = call_compute(compute, batch)
        rcov.add(sample)
    rcov.to_('cpu')
    save_cached_state(cachefile, rcov, args)
    return rcov


def tally_cross_covariance(compute, dataset, sample_size=None, batch_size=10,
                           cachefile=None, **kwargs):
    '''
    Computes cross covariance statistics for a large data sample that can
    be computed from a dataset.  The compute function should return one
    batch of samples as pair of [(sample, unitA), (sample, unitB)] tensors.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningCrossCovariance(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    rcov = runningstats.RunningCrossCovariance()
    for batch in pbar(loader):
        sampleA, sampleB = call_compute(compute, batch)
        rcov.add(sampleA, sampleB)
    rcov.to_('cpu')
    save_cached_state(cachefile, rcov, args)
    return rcov


def tally_second_moment(compute, dataset, sample_size=None, batch_size=10,
                        cachefile=None, **kwargs):
    '''
    Computes second_moment statistics for a large data sample that can
    be computed from a dataset.  The compute function should return one
    batch of samples as a (sample, unit)-dimension tensor.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningSecondMoment(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    r2mom = runningstats.RunningSecondMoment()
    for batch in pbar(loader):
        sample = call_compute(compute, batch)
        r2mom.add(sample)
    r2mom.to_('cpu')
    save_cached_state(cachefile, r2mom, args)
    return r2mom


def tally_all_intersection_and_union(compute, dataset, sample_size=None,
                                     batch_size=10, cachefile=None, **kwargs):
    '''
    Computes all-pairs intersection and union from a pair of
    streams of binary vector batches that come from the input.
    can be computed from a dataset.  The compute function should return
    two batches of samples as (sample, unita), (sample, unitb) tensors.
    '''
    # with torch.no_grad():
    args = dict(sample_size=sample_size)
    cached_state = load_cached_state(cachefile, args)
    if cached_state is not None:
        return runningstats.RunningAllIntersectionAndUnion(state=cached_state)
    loader = make_loader(dataset, sample_size, batch_size, **kwargs)
    riu = runningstats.RunningAllIntersectionAndUnion()
    for batch in pbar(loader):
        flat_acts, flat_segs = call_compute(compute, batch)
        riu.add(flat_acts, flat_segs)
    riu.to_('cpu')
    save_cached_state(cachefile, riu, args)
    return riu


def batch_bincount(data, num_labels):
    '''
    Computes elementwise bincount on a batch of data.  The input tensor is
    size (batch_size, ...), and the output is (batch_size, num_labels),
    where each row of the output is the bincount of the corresponding
    entry of the input.
    '''
    data = data.view(len(data), -1)
    strided = data + torch.arange(len(data), dtype=data.dtype,
                                  device=data.device)[:, None] * num_labels
    counts = torch.bincount(strided.view(-1), minlength=num_labels * len(data))
    return counts.view(len(data), num_labels)


def iou_from_conditional_indicator_mean(condmv):
    '''
    Given a RunningConditionalVariance containing mean values of
    indictors, estimates all-pairs IoU statistics for all units
    between the conditions and the indicators.
    The result is a tensor of dimension (units, conditions)
    containing IoU estimates for each combination.
    '''
    # Old code using python loops.
    # gt = {k: condi99.conditional(k).size() /
    #      condi99.conditional(0).size() for k in condi99.keys()}
    # act = condi99.conditional(0).mean()
    # isect = {k: condi99.conditional(k).mean() * gt[k] for k in condi99.keys()}
    # union = {k: act + gt[k] - isect[k] for k in condi99.keys()}
    # iou = {k: isect[k] / union[k] for k in condi99.keys()}

    # New code arranging with pytorch tensors
    uncond_size = condmv.conditional(0).size()
    units = condmv.conditional(0).mean().shape[0]
    conditions = max(condmv.keys()) + 1
    act = condmv.conditional(0).mean()
    gt = torch.zeros(conditions)
    isect = torch.zeros(conditions, units)
    for k in condmv.keys():
        gt[k] = condmv.conditional(k).size() / condmv.conditional(0).size()
        isect[k] = condmv.conditional(k).mean() * gt[k]
    union = act[None, :] + gt[:, None] - isect
    iou = isect / union
    return iou


def iou_from_conditional_quantile(condq, cutoff=0.95, min_batches=2):
    '''
    Given a RunningConditionalQuantile, estimates all-pairs
    IoU statistics for all units and conditions at the specified
    quantile cutoff.  Note that cutoff can be a list of cutoffs.
    The result is a tensor of dimension (units, conditions, cutoffs)
    containing IoU estimates for each combination.

    Conditions that are sampled in fewer than min_batches are given IoU 0.
    '''
    return intersection_from_conditional_quantile(condq,
                                                  statistic=intersection_over_union,
                                                  cutoff=cutoff, min_batches=min_batches)


def iqr_from_conditional_quantile(condq, cutoff=0.95, min_batches=2):
    '''
    Given a RunningConditionalQuantile, estimates all-pairs
    IQR statistics for all units and conditions at the specified
    quantile cutoff.  Similar to iou_from_conditional_quantile.
    '''
    return intersection_from_conditional_quantile(condq,
                                                  statistic=information_quality_ratio,
                                                  cutoff=cutoff, min_batches=min_batches)


def mi_from_conditional_quantile(condq, cutoff=0.95, min_batches=2):
    '''
    Given a RunningConditionalQuantile, estimates all-pairs
    mutual information for all units and conditions at the specified
    quantile cutoff.  Similar to iou_from_conditional_quantile.
    '''
    return intersection_from_conditional_quantile(condq,
                                                  statistic=mutual_information,
                                                  cutoff=cutoff, min_batches=min_batches)


def intersection_from_conditional_quantile(
        condq, statistic=lambda x: x[0, 0], cutoff=0.95, min_batches=2):
    '''
    There are a variety of ways of scoring the intersection between a
    prediction (a) and a true variable (b) that are all expressions of
    [[p(a&b), p(a&!b)], [p(!a&b), p(!a&!b)]].  This computes any of
    them by passing the above array to a 'statistic' function.
    By default it returns p(a&b).
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        cutoff = torch.tensor(cutoff)
    uncond_size = condq.conditional(0).size()
    units = condq.conditional(0).depth
    scores = torch.zeros((units, max(condq.keys()) + 1) + cutoff.shape)
    # math: actlevel = level such that p(x > level) = cutoff
    actlevel = condq.conditional(0).quantiles(cutoff)
    # use a progress bar if it's going to be more than a few seconds.
    prog = pbar if cutoff.numel() * units > 1e5 else lambda x: x
    for c in prog(sorted(condq.keys())):
        rq = condq.conditional(c)
        if c == 0 or rq.batchcount < min_batches:
            continue
        # math: condp = p(x > actlevel | cond)
        condp = rq.normalize(actlevel)
        truth = float(rq.size()) / uncond_size
        isect = truth * (1 - condp)
        pred = (1 - cutoff)
        union = pred + truth - isect
        # Compute relative mutual information directly.
        arr = torch.stack([
            isect,         pred - isect,
            truth - isect, 1 - union]).view((2, 2) + isect.shape)
        scores[:, c, ...] = statistic(arr)
    return scores


def intersection_over_union(arr):
    return arr[0, 0] / (1 - arr[1, 1])


def mutual_information(arr):
    total = 0
    for j in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            joint = arr[j, k]
            ind = arr[j, :].sum(dim=0) * arr[:, k].sum(dim=0)
            term = joint * (joint / ind).log()
            term[torch.isnan(term)] = 0
            total += term
    return total.clamp_(0)


def joint_entropy(arr):
    total = 0
    for j in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            joint = arr[j, k]
            term = joint * joint.log()
            term[torch.isnan(term)] = 0
            total += term
    return (-total).clamp_(0)


def information_quality_ratio(arr):
    iqr = mutual_information(arr) / joint_entropy(arr)
    iqr[torch.isnan(iqr)] = 0
    return iqr


def call_compute(compute, batch):
    '''Utility for passing a dataloader batch to a compute function.'''
    if isinstance(batch, list):
        return compute(*batch)
    elif isinstance(batch, dict):
        return compute(**batch)
    else:
        return compute(batch)


def make_loader(dataset, sample_size=None, batch_size=10, sampler=None,
                **kwargs):
    '''Utility for creating a dataloader on fixed sample subset.'''
    if isinstance(dataset, torch.Tensor):
        dataset = torch.utils.data.TensorDataset(dataset)
    if sampler is None:
        if sample_size is not None:
            if sample_size > len(dataset):
                pbar.print("Warning: sample size %d > dataset size %d" %
                           (sample_size, len(dataset)))
                sample_size = len(dataset)
            sampler = FixedSubsetSampler(list(range(sample_size)))
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        **kwargs)


def push_key_prefix(prefix, d):
    return {prefix + '.' + k: v for k, v in d.items()}


def pull_key_prefix(prefix, d):
    pd = prefix + '.'
    lpd = len(pd)
    return {k[lpd:]: v for k, v in d.items() if k.startswith(pd)}


class CombinedState(object):
    '''
    An object that can load and save state_dict made up of a
    hierarchy of objects that each have a state_dict.
    '''

    def __init__(self, state=None, **kwargs):
        self._objs = kwargs
        if state is not None:
            for prefix, obj in self._objs.items():
                objs.load_state_dict(pull_key_prefix(prefix, state))

    def __getattr__(self, k):
        if k in self._objs:
            return self._objs[k]
        raise AttributeError()

    def state_dict(self):
        result = {}
        for prefix, obj in self._objs.items():
            result.update(push_key_prefix(prefix, obj.state_dict()))
        return result


class SavedTensor:
    def __init__(self, data):
        self.data = data

    def state_dict(self):
        return dict(data=self.data.numpy())


class TensorDict:
    def __init__(self, data=None, state=None):
        if state is not None:
            self.data = {k: torch.from_numpy(v) for k, v in state.items()}
        else:
            self.data = data or {}

    def state_dict(self):
        return {k: v.detach().cpu().numpy() for k, v in self.data.items()}


def load_cached_state(cachefile, args):
    if cachefile is None:
        return None
    try:
        dat = numpy.load(cachefile, allow_pickle=True)
        for a, v in args.items():
            if a not in dat or dat[a] != v:
                pbar.print('%s %s changed from %s to %s' % (
                    cachefile, a, dat[a], v))
                return None
    except:
        return None
    else:
        pbar.descnext(None)
        pbar.print('Loading cached %s' % cachefile)
        return dat


def save_cached_state(cachefile, obj, args):
    if cachefile is None:
        return
    os.makedirs(os.path.dirname(cachefile), exist_ok=True)
    dat = obj.state_dict()
    for a, v in args.items():
        if a in dat:
            assert(dat[a] == v)
        dat[a] = v
    numpy.savez(cachefile, **dat)
