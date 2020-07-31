import argparse
import os
import json
import numpy
import PIL.Image
from . import pidfile, tally, nethook, zdataset
from . import upsample, imgviz, imgsave, proggan, segmenter


def main():
    parser = argparse.ArgumentParser(description='quickdissect')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--model', type=str, default='church')
    parser.add_argument('--layer', type=str, default='layer4')
    parser.add_argument('--seg', type=str, default='netpqc')
    parser.add_argument('--sample_size', type=int, default=1000)
    args = parser.parse_args()

    resfn = pidfile.exclusive_dirfn(
        args.outdir, args.model, args.layer, args.seg, str(args.sample_size))

    import torch
    torch.backends.cudnn.profile = True

    model = nethook.InstrumentedModel(
        proggan.load_pretrained(args.model)).cuda()
    model.retain_layer(args.layer)

    zds = zdataset.z_dataset_for_model(model, size=args.sample_size, seed=1)

    model(zds[0][0][None].cuda())
    sample_act = model.retained_layer(args.layer).cpu()
    upfn = upsample.upsampler((64, 64), sample_act.shape[2:])

    def flat_acts(zbatch):
        _ = model(zbatch.cuda())
        acts = upfn(model.retained_layer(args.layer))
        return acts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

    rq = tally.tally_quantile(flat_acts, zds, cachefile=resfn('rq.npz'))
    level_at_cutoff = rq.quantiles(0.99)[None, :, None, None].cuda()

    segmodel, seglabels = segmenter.load_segmenter(args.seg)

    def compute_cond_indicator(zbatch):
        image_batch = model(zbatch.cuda())
        seg = segmodel.segment_batch(image_batch, downsample=4)
        acts = upfn(model.retained_layer(args.layer))
        iacts = (acts > level_at_cutoff).float()
        return tally.conditional_samples(iacts, seg)

    cmv = tally.tally_conditional_quantile(compute_cond_indicator, zds,
                                           cachefile=resfn('cmv.npz'), pin_memory=True)

    iou_table = tally.iou_from_conditional_indicator_mean(cmv).permute(1, 0)
    numpy.save(resfn('iou.npy'), iou_table.numpy())

    unit_list = enumerate(zip(*iou_table.max(1)))
    unit_records = {
        'units': [{
            'unit': unit,
            'iou': iou.item(),
            'label': seglabels[segc],
            'cls': segc.item()
        } for unit, (iou, segc) in unit_list]
    }
    with open(resfn('labels.json'), 'w') as f:
        json.dump(unit_records, f)
    with open(resfn('seglabels.json'), 'w') as f:
        json.dump(seglabels, f)

    def compute_image_max(zbatch):
        image_batch = model(zbatch.cuda())
        return model.retained_layer(args.layer).max(3)[0].max(2)[0]

    topk = tally.tally_topk(compute_image_max, zds,
                            cachefile=resfn('topk.npz'))

    def compute_acts(zbatch):
        image_batch = model(zbatch.cuda())
        acts_batch = model.retained_layer(args.layer)
        return (acts_batch, image_batch)

    iv = imgviz.ImageVisualizer(128, quantiles=rq)
    unit_images = iv.masked_images_for_topk(compute_acts, zds, topk, k=5)
    imgsave.save_image_set(unit_images, resfn('imgs/unit_%d.png'))

    pidfile.mark_job_done(resfn.dir)


if __name__ == '__main__':
    main()


class DissectVis:
    '''
    Code to read out the dissection computed in the program above.
    '''

    def __init__(self, outdir='results', model='church', layers=None,
                 seg='netpqc', sample_size=1000):
        if not layers:
            layers = ['layer%d' % i for i in range(1, 15)]

        basedir = 'results/church'
        setting = 'netpqc/1000'
        labels = {}
        iou = {}
        images = {}
        for k in layers:
            dirname = os.path.join(outdir, model, k, seg, str(sample_size))
            with open(os.path.join(dirname, 'labels.json')) as f:
                labels[k] = json.load(f)['units']
            iou[k] = numpy.load(os.path.join(dirname, 'iou.npy'))
            images[k] = [None] * len(iou[k])
        with open(os.path.join(dirname, 'seglabels.json')) as f:
            self.seglabels = json.load(f)
        self.labels = labels
        self.ioutable = iou
        self.images = images
        self.basedir = os.path.join(outdir, model)
        self.setting = os.path.join(seg, str(sample_size))

    def label(self, layer, unit):
        return self.labels[layer][unit]['label']

    def iou(self, layer, unit):
        return self.labels[layer][unit]['iou']

    def top_units(self, layer, seglabel, k=20):
        return self.ioutable[layer][:, self.seglabels.index(seglabel)
                                    ].argsort()[::-1][:k].tolist()

    def image(self, layer, unit):
        result = self.images[layer][unit]
        # Lazy loading of images.
        if result is None:
            result = PIL.Image.open(os.path.join(
                self.basedir, layer,
                self.setting, 'imgs/unit_%d.png' % unit))
            result.load()
            self.images[layer][unit] = result
        return result
