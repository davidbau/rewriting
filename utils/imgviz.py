import PIL
import torch
from . import upsample, renormalize, tally
from matplotlib import cm


class ImageVisualizer:
    def __init__(self, size, image_size=None, data_size=None,
                 renormalizer=None, scale_offset=None, level=None, actrange=None,
                 source=None, convolutions=None, quantiles=None,
                 percent_level=None):
        '''
        An ImageVisualizer produces visualizations of unit activations
        as heatmaps or overlays on top of the original image.  The output
        visualization size is given by `size`.  This may be scaled
        from the original image size `image_size` (inferred from source
        if not specified) and activation `data_size` (inferred from
        convolutions if not specified).  Imagedata is renormalized as
        rendered RGB using renormalizer (inferred from source if not
        specified) and upsampling can be offset using scale_offset
        (inferred from convolutions if not specified).  Per-unit
        `quantiles` can be provided for computing heatmap ranges
        (default to 1% to 99% quantiles), and threshold levels
        (default to `percent_level`=0.95).
        '''
        if image_size is None and source is not None:
            image_size = upsample.image_size_from_source(source)
        if renormalizer is None and source is not None:
            renormalizer = renormalize.renormalizer(
                source=source, target='byte')
        if scale_offset is None and convolutions is not None:
            scale_offset = upsample.sequence_scale_offset(convolutions)
        if data_size is None and convolutions is not None:
            data_size = upsample.sequence_data_size(convolutions, image_size)
        if level is None and quantiles is not None:
            level = quantiles.quantiles([percent_level or 0.95])[:, 0]
        if actrange is None and quantiles is not None:
            actrange = quantiles.quantiles([0.01, 0.99])
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.image_size = image_size
        self.data_size = data_size
        self.renormalizer = renormalizer
        self.scale_offset = scale_offset
        self.percent_level = percent_level
        self.level = level
        self.actrange = actrange
        self.quantiles = quantiles
        self.upsampler = None
        if self.data_size is not None:
            self.upsampler = upsample.upsampler(size, data_size,
                                                image_size=self.image_size,
                                                scale_offset=scale_offset)

    def heatmap(self, activations, unit=None, mode='bilinear',
                amax=None, amin=None):
        '''
        Produces a heatmap from the given activations.  The unit,
        if specified, is an index into the activations tensor,
        and is also used to dereference quantile ranges.
        '''
        if amax is None or amin is None:
            amin, amax = self.range_for(activations, unit)
        if unit is None:
            a = activations
        else:
            a = activations[unit]
        upsampler = self.upsampler_for(a)
        a = upsampler(a[None, None, ...], mode=mode)[0, 0].cpu()
        return PIL.Image.fromarray(
            (cm.hot((a - amin) / (1e-10 + amax - amin)) * 255
             ).astype('uint8'))

    def image(self, imagedata):
        '''
        Converts the given tensor imagedata to a PIL image, scaling
        and renormalizing as needed.
        '''
        return PIL.Image.fromarray(self.pytorch_image(imagedata)
                                   .permute(1, 2, 0).byte().cpu().numpy())

    def masked_image(self, imagedata, activations=None, unit=None,
                     level=None, percent_level=None, **kwargs):
        '''
        Visualizes the given activations, thresholded at a specified level,
        overlaid on the given image, as a PIL image.
        '''
        result_image = self.pytorch_masked_image(imagedata,
                                                 activations=activations,
                                                 unit=unit, level=level, percent_level=percent_level,
                                                 **kwargs)
        return PIL.Image.fromarray(
            result_image.permute(1, 2, 0).cpu().numpy())

    def pytorch_masked_image(self, imagedata, activations=None, unit=None,
                             level=None, percent_level=None, thickness=1, mask=None,
                             border_color=None, outside_bright=0.5, inside_color=None):
        '''
        Visualizes the given activations, thresholded at a specified level,
        overlaid on the given image, as a pytorch byte tensor (channel first).
        '''
        scaled_image = self.pytorch_image(imagedata).float().cpu()
        if mask is None:
            mask = self.pytorch_mask(activations, unit, level=level,
                                     percent_level=percent_level).cpu()
        border = border_from_mask(mask, thickness)
        inside = (mask & (~border))
        outside = (~mask & (~border))
        inside, outside, border = [d.float() for d in [inside, outside, border]]
        if border_color is None:
            border_color = [255.0, 255.0, 0]  # yellow
        border_color = torch.tensor(border_color,
                                    dtype=border.dtype, device=border.device)[:, None, None]
        if inside_color is not None:
            inside_color = torch.tensor(inside_color,
                                        dtype=border.dtype, device=border.device)[:, None, None]
        result_image = (
            (scaled_image if inside_color is None
             else inside_color) * inside + border_color * border +
            outside_bright * scaled_image * outside).clamp(0, 255).byte()
        return result_image

    def masked_delta(self, imagedata, activations, unit=None,
                     above=None, below=None):
        '''
        Visualizes the given activations, thresholded at a specified level,
        overlaid on the given image, as a PIL image.
        '''
        result_image = self.pytorch_masked_delta(imagedata, activations,
                                                 unit=unit, above=above, below=below)
        return PIL.Image.fromarray(
            result_image.permute(1, 2, 0).cpu().numpy())

    def pytorch_masked_delta(self, imagedata, delta, unit=None,
                             above=None, below=None):
        '''
        Visualizes the given activations, thresholded at a specified level,
        with green for high numbrers and red for low numbers.
        '''
        scaled_image = self.pytorch_image(imagedata).float().cpu()
        amask, bmask, aborder, bborder = [torch.tensor(0) for _ in range(4)]
        if above is not None:
            amask = self.pytorch_mask(delta, unit, level=above).cpu()
            aborder = border_from_mask(amask)
        if below is not None:
            bmask = ~self.pytorch_mask(delta, unit, level=below).cpu()
            bborder = border_from_mask(bmask)
        inside = ((amask | bmask) & ~(aborder | bborder))
        outside = (~(amask | bmask) & ~(aborder | bborder))
        inside, outside, aborder, bborder = [d.float()
                                             for d in [inside, outside, aborder, bborder]]
        red, green = [torch.tensor(c, dtype=torch.float, device=aborder.device
                                   )[:, None, None] for c in [[255, 0, 0], [0, 255, 0]]]
        result_image = (
            scaled_image * inside + green * aborder + red * bborder +
            0.5 * scaled_image * outside).clamp(0, 255).byte()
        return result_image

    def pytorch_mask(self, activations, unit, level=None, percent_level=None):
        '''
        Computes a pytorch mask of the (upsampled) activations above a
        specified level.
        '''
        if unit is None:
            a = activations
        else:
            a = activations[unit]
        if level is None:
            level = self.level_for(activations, unit,
                                   percent_level=percent_level)
        upsampler = self.upsampler_for(a)
        return (upsampler(a[None, None, ...])[0, 0] > level)

    def pytorch_image(self, imagedata):
        '''
        Scales the given image to the visualized size.  returns as a
        pytorch byte tensor, in (rgb, y, x) channel order.
        '''
        if len(imagedata.shape) == 4:
            imagedata = imagedata[0]
        renormalizer = self.renormalizer_for(imagedata)
        return torch.nn.functional.interpolate(
            renormalizer(imagedata).float()[None, ...],
            size=self.size)[0]

    def upsampler_for(self, a):
        '''
        Returns an upsampler instance, defaulting to simple upscaling
        if a specific upsampler is not specified.
        '''
        if self.upsampler is not None:
            return self.upsampler
        return upsample.upsampler(self.size, a.shape,
                                  image_size=self.image_size,
                                  scale_offset=self.scale_offset,
                                  dtype=a.dtype, device=a.device)

    def range_for(self, activations, unit):
        '''
        Returns a range of activations, using quantiles (1% to 99%)
        if given, or just using the min and max of the data if
        quantiles are not given.
        '''
        if unit is not None and self.actrange is not None:
            if hasattr(unit, '__len__'):
                unit = unit[1]
            return tuple(i.item() for i in self.actrange[unit])
        return activations.min(), activations.max()

    def level_for(self, activations, unit, percent_level=None):
        '''
        Returns the cutoff level for a unit, using quantiles if given
        or just using stats over the specific instance if not.
        '''
        if unit is not None:
            if hasattr(unit, '__len__'):
                unit = unit[1]
            if percent_level is not None and self.quantiles is not None:
                return self.quantiles.quantiles(percent_level)[unit].item()
            if self.level is not None:
                return self.level[unit].item()
        s, _ = activations.view(-1).sort()
        if percent_level is None:
            percent_level = self.percent_level or 0.95
        return s[int(len(s) * percent_level)]

    def renormalizer_for(self, image):
        '''
        Returns the renormalizer to use for visualizing the tensor
        image data as RGB.
        '''
        if self.renormalizer is not None:
            return self.renormalizer
        return renormalize.renormalizer('zc', 'byte')

    def masked_image_grid_for_topk(
            self, compute, dataset, topk, k=None, **kwargs):
        def compute_viz(gather_indices, *data_batch):
            acts_batch = compute(*data_batch)
            if isinstance(acts_batch, tuple):
                acts_batch, image_batch = acts_batch
            else:
                image_batch = data_batch[0]
            for gather_for, acts, imgt in (
                    zip(gather_indices, acts_batch, image_batch)):
                for unit, rank in gather_for:
                    yield((unit, rank), self.pytorch_masked_image(
                        imgt, acts, unit).permute(1, 2, 0).cpu())
        gt = tally.gather_topk(compute_viz, dataset, topk=topk, k=k, **kwargs)
        return gt.result()

    def individual_masked_images_for_topk(
            self, compute, dataset, topk, k=None, **kwargs):
        # Example compute function:
        # def compute(image_batch):
        #   image_batch = image_batch.cuda()
        #   acts_batch = model.retained_layer(layername)
        gt = self.masked_image_grid_for_topk(
            compute, dataset, topk, k=k, **kwargs)
        return [[PIL.Image.fromarray(d.cpu().numpy()) for d in row]
                for row in gt]

    def masked_images_for_topk(
            self, compute, dataset, topk, k=None, gap=5, **kwargs):
        # Example compute function:
        # def compute(image_batch):
        #   image_batch = image_batch.cuda()
        #   acts_batch = model.retained_layer(layername)
        gt = self.masked_image_grid_for_topk(
            compute, dataset, topk, k=k, **kwargs)
        return [strip_image_from_grid_row(row, gap=gap) for row in gt]

    def masked_image_grid_for_row(self, compute, dataset, unit, indexes):
        results = []
        for rank in indexes:
            img_batch = dataset[rank][0][None, ...]
            acts_batch = compute(img_batch)
            results.append(self.pytorch_masked_image(
                img_batch[0], acts_batch[0], unit)
                .permute(1, 2, 0).cpu()[None, ...])
        return torch.cat(results)

    def masked_image_row(self, compute, dataset, unit, indexes, gap=5):
        row = self.masked_image_grid_for_row(
            compute, dataset, unit, indexes)
        return strip_image_from_grid_row(row, gap=gap)

    def masked_image_for_conditional_topk(self, compute, dataset,
                                          ctk, classnum, unit, k=10, gap=5):
        row = self.masked_image_grid_for_row(
            compute, dataset, unit,
            ctk.conditional(classnum).result()[1][unit][:k])
        return strip_image_from_grid_row(row, gap=gap)


def strip_image_from_grid_row(row, gap=5, bg=255):
    strip = torch.full(
        (row.shape[1],
         row.shape[0] * (row.shape[2] + gap) - gap,
         row.shape[3]), bg, dtype=row.dtype)
    for i, img in enumerate(row):
        strip[:,
              i * (row.shape[2] + gap): (i + 1) * (row.shape[2] + gap) - gap,
              :] = img
    return PIL.Image.fromarray(strip.numpy())


def border_from_mask(mask, thickness=1, outside=True):
    a = mask
    out = torch.zeros_like(a)
    for it in range(thickness):
        h = (a[:-1, :] != a[1:, :])
        v = (a[:, :-1] != a[:, 1:])
        d = (a[:-1, :-1] != a[1:, 1:])
        u = (a[1:, :-1] != a[:-1, 1:])
        out[:-1, :-1] |= d
        out[1:, 1:] |= d
        out[1:, :-1] |= u
        out[:-1, 1:] |= u
        out[:-1, :] |= h
        out[1:, :] |= h
        out[:, :-1] |= v
        out[:, 1:] |= v
        if it > 0:
            out |= a
        a = out
    if outside:
        out &= ~mask
    return out
