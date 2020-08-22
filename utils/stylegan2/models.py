####
# A sequential version of stylegan that is perfectly compatible with
# the pytorch port by https://github.com/rosinality/stylegan2-pytorch.
# (this port by David Bau, with low level operations frrom rosinality).
#
# In this implementation, all non-leaf modules are subclasses of
# nn.Sequential so that they can be more easily split apart for
# surgery.  Because stylegan has style, featuremap, RGB, and noise
# data that all flow through the network in parallel, these are
# gathered in a DataBag (dict subclass).
#
# An essential trick: to rewrite convolutional layers inside StyleGAN
# effectively, we need to express the rewrritten layer as a linear
# convolution followed by nonlinearities; this allows us to treat
# the learned convolution as a linear associative memory. (See
# the paper for details.) But stylegan code happens to combine
# the linear convolution with a style modulation, which breaks this
# form.  So in the code below, we write ModulatedConv2dSeq to
# explicitly separate the style modulation from the convolution
# steps, so we can rewrite the learned convolution directly.
# This separated form is slightly slower, so we only do it if
# constructing with the option mconv='seq'.

from . import op
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import math, torch, warnings
import numpy as np

class SeqStyleGAN2(nn.Sequential):
    '''
    This is a sequential version of StyleGanv2: all the steps are expressed
    as a composition of torch.nn.Sequential modules.

    Internally data flows as DataBags.  bag_input allows you to pass a
    DataBag directly as input, supplying noises explicitly.  bag_output
    provides access to DataBag directly as output, including not only
    the image but also the last featuremap, latents, and noise.
    '''
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        truncation=1.0,
        mconv=None,
        bag_input=False,
        bag_output=False
    ):
        self.size = size
        self.style_dim = style_dim
        self.mconv = mconv
        self.bag_input = bag_input
        self.bag_output = bag_output
        style_layers = [PixelNormL()]

        # Make the style subnetwork.
        for i in range(n_mlp):
            style_layers.append(EqualLinearL(
                style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
            ))
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_latent = self.log_size * 2 - 2

        in_channel = self.channels[4]

        # What to do about noise?
        noises = FixedNoiseBuffers(self.num_layers, 1, replace_input=False)
        # for layer_idx in range(self.num_layers):
        #     res = (layer_idx + 5) // 2
        #     shape = [1, 1, 2 ** res, 2 ** res]
        #     noises.register_buffer(
        #         f'noise_{layer_idx}', torch.randn(*shape))

        layers = []
        if not bag_input:
            layers.append(('bag_in', InputLatent()))

        layers.extend([
            ('style', nn.Sequential(*style_layers)),
            ('latents', AdjustLatent(self.n_latent, truncation)),
            ('noises', noises),
            ('input', ConstantInputF(self.channels[4])),
            ('layer2', nn.Sequential(OrderedDict([
                ('lat0', PickLatent(0)),
                ('conv', StyledConvSeq(self.channels[4], self.channels[4],
                    3, style_dim, blur_kernel=blur_kernel, mconv=mconv))
            ]))),
            ('to_rgb1', nn.Sequential(OrderedDict([
                ('lat1', PickLatent(1)),
                ('rgb', ToRGBF(self.channels[4], style_dim, upsample=False))
            ])))
        ])

        lat_i = 1
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            layers.extend([
                ('up_rgb%d' % (i-2), UpsampleO()),
                ('layer%d' % (lat_i+2), nn.Sequential(OrderedDict([
                    ('lat%d' % lat_i, PickLatent(lat_i)),
                    ('sconv', StyledConvSeq(in_channel, out_channel, 3,
                            style_dim, upsample=True, blur_kernel=blur_kernel,
                            mconv=mconv))
                ]))),
                ('layer%d' % (lat_i+3), nn.Sequential(OrderedDict([
                    ('lat%d' % (lat_i+1), PickLatent(lat_i+1)),
                    ('sconv', StyledConvSeq(out_channel, out_channel, 3,
                            style_dim, blur_kernel=blur_kernel, mconv=mconv)),
                ]))),
                ('to_rgb%d' % (i-1), nn.Sequential(OrderedDict([
                    ('lat%d' % (lat_i+2), PickLatent(lat_i+2)),
                    ('rgb', ToRGBF(out_channel, style_dim, skip=True,
                        upsample=False))
                ]))),
            ])
            in_channel = out_channel
            lat_i += 2

        if not bag_output:
            layers.append(('output', ReturnOutput()))

        super().__init__(OrderedDict(layers))

    def bag_from_z(self, z):
        return InputLatent()(z)

    def output_from_bag(self, bag):
        return ReturnOuptput()(bag)

    def load_state_dict(self, data, latent_avg=None, **kwargs):
        try:
            super().load_state_dict(data, **kwargs)
            return
        except:
            pass
        # If the state dict does not match, try converting other versions
        if len(data) < 10 and 'g_ema' in data and 'latent_avg' in data:
            latent_avg = data['latent_avg']
            data = data['g_ema']
        import re
        newdata = {}
        for k, v in data.items():
            # Convert from the nonsequential weights
            k = re.sub(r'^conv1\.conv\.', 'layer2.conv.mconv.', k)
            k = re.sub(r'^conv1\.', 'layer2.conv.', k)
            k = re.sub(
                r'^convs\.(\d+)\.conv',
                lambda x: f'layer{int(x.group(1))+3}.sconv.mconv',
                k)
            k = re.sub(
                r'^convs\.(\d+)\.',
                lambda x: f'layer{int(x.group(1))+3}.sconv.',
                k)
            k = re.sub(
                r'^to_rgb1\.(conv\.|bias$)',
                lambda x: f'to_rgb1.rgb.{x.group(1)}',
                k)
            k = re.sub(
                r'^to_rgbs\.(\d+)\.upsample\.',
                lambda x: f'up_rgb{int(x.group(1))+1}.',
                k)
            k = re.sub(
                r'^to_rgbs\.(\d+)\.',
                lambda x: f'to_rgb{int(x.group(1))+2}.rgb.',
                k)
            # Convert between sequential and faster nonsequential mconv
            if self.mconv == 'seq':
                k = re.sub(r'mconv\.weight$', 'mconv.dconv.weight', k)
            else:
                k = re.sub(r'mconv\.dconv\.weight$', 'mconv.weight', k)
            newdata[k] = v
        # optional fields just leave current state dict unchanged
        cur_state = self.state_dict()
        if latent_avg is not None:
            newdata['latents.latent_avg'] = latent_avg
        elif 'latents.latent_avg' not in newdata:
            if self.latents.truncation != 1.0:
                warnings.warn('Need to provide latent_avg to use truncation.')
            newdata['latents.latent_avg'] = cur_state['latents.latent_avg']
        for key in [k for k in cur_state.keys() if k.startswith('noises')]:
            if key not in newdata:
                newdata[key] = cur_state[key]
        super().load_state_dict(newdata, **kwargs)

class DataBag(dict):
    '''
    A databag is a dict that provides access to keys as attributes;
    We will use it to bundle different types of data as they
    pass through the sequential StyleGan.

    By convention, there will be:
       latent     - initial Z and then the W after the FC network.
       style      - component of latent after modulation
       fmap       - the featuremap for the layer
       output     - the accumulated rgb output so far.
    '''
    def __init__(self, rep=None, **kwargs):
        self.update(rep=rep, **kwargs)
    def __setattr__(self, name, value):
        super().__setattr__(name, value); super().__setitem__(name, value)
    def __delattr__(self, name):
        super().__delattr__(name); super().__delitem__(name)
    __setitem__, __delitem__ = __setattr__, __delattr__
    def update(self, rep=None, **f):
        d = dict() if rep is None else dict(rep)
        d.update(f)
        for k, v in d.items():
            setattr(self, k, v)
    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)

class StyledConvSeq(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        mconv=None,
    ):
        assert mconv in [None, 'seq', 'fast']
        MConv = ModulatedConv2dSeq if mconv == 'seq' else ModulatedConv2dF
        super().__init__(OrderedDict([
            ('mconv', MConv(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                upsample=upsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate)),
            ('noise', NoiseInjectionF()),
            ('activate', FusedLeakyReLUF(out_channel))
        ]))

class ModulatedConv2dSeq(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        blur_kernel=[1, 3, 3, 1]
    ):
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            blur_step = [('blur', BlurF(
                    blur_kernel, pad=(pad0, pad1), upsample_factor=factor))]
        else:
            blur_step = []
        super().__init__(OrderedDict([
            ('modulation', EqualLinearS(style_dim, in_channel, bias_init=1)),
            ('adain', ApplyStyle()),
            ('dconv', DemodulatedConv2dF(in_channel, out_channel, kernel_size,
                demodulate=demodulate, upsample=upsample)),
        ] + blur_step))

class DemodulatedConv2dF(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
            demodulate=True, upsample=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.demodulate = demodulate
        self.upsample = upsample
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, '
            f'{self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample})'
        )

    def forward(self, d):
        if self.upsample:
            weight = self.scale * self.weight.transpose(1, 2).squeeze(0)
            out = F.conv_transpose2d(d.fmap, weight, padding=0, stride=2)
        else:
            weight = self.scale * self.weight.squeeze(0)
            out = F.conv2d(d.fmap, weight, padding=self.padding)
        if self.demodulate:
            # Hack - we do some redundant computation here.
            # in the original code this is already computed
            # because the temp_weight is directly used for the computation.
            batch, in_channel, height, width = d.fmap.shape
            style = d.style.view(batch, 1, in_channel, 1, 1)
            temp_weight = self.scale * self.weight * style
            demod = torch.rsqrt(temp_weight.pow(2).sum([2, 3, 4]) + 1e-8)
            out = out * demod[:,:,None,None]
        return DataBag(d, fmap=out)

class NoiseBuffers(nn.Module):
    def __init__(self, replace_input=False):
        super().__init__()
        self.replace_input = replace_input
    def forward(self, d):
        for att in dir(self):
            if att.startswith('noise_'):
                if self.replace_input or att not in d:
                    d[att] = getattr(self, att)
        return d

class FixedNoiseBuffers(NoiseBuffers):
    def __init__(self, num_layers, seed, replace_input=False):
        super().__init__(replace_input=replace_input)
        self.num_layers = num_layers
        rng = np.random.RandomState(seed)
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            np_noise = rng.randn(*shape).astype('float32')
            self.register_buffer(
                f'noise_{layer_idx}', torch.from_numpy(np_noise))

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1),
                    upsample_factor=factor)
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={False})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channel, in_channel,
            self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel,
                self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel,
                self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight,
                    padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out

class ModulatedConv2dF(ModulatedConv2d):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
            demodulate=True, upsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__(in_channel, out_channel, kernel_size, style_dim,
            demodulate=demodulate, upsample=upsample, blur_kernel=blur_kernel)
    def forward(self, d):
        return DataBag(d, fmap=super().forward(d.fmap, d.style))

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)
    def forward(self, input):
        return op.upfirdn2d(input, self.kernel, up=self.factor, down=1,
                pad=self.pad)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k

class UpsampleF(Upsample):
    def __init__(self, kernel, factor=2):
        super().__init__(kernel, factor)
    def forward(self, d):
        return DataBag(d, fmap=super().forward(d.fmap))

class UpsampleO(Upsample):
    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__(kernel, factor)
    def forward(self, d):
        return DataBag(d, output=super().forward(d.output))

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = op.upfirdn2d(input, self.kernel, pad=self.pad)
        return out

class BlurF(Blur):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__(kernel, pad=pad, upsample_factor=upsample_factor)
    def forward(self, d):
        return DataBag(d, fmap=super().forward(d.fmap))

class EqualLinear(nn.Linear):
    def __init__(
        self, in_dim, out_dim, bias=True,
        bias_init=0, lr_mul=1, activation=None
    ):
        self.bias_init = bias_init
        self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        super().__init__(in_dim, out_dim, bias)
        self.activation = activation

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=(1.0 / self.lr_mul))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, self.bias_init)

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = op.fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class EqualLinearL(EqualLinear):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0,
            lr_mul=1, activation=None):
        super().__init__(in_dim, out_dim, bias=bias, bias_init=bias_init,
                lr_mul=lr_mul, activation=activation)
    def forward(self, d):
        return DataBag(d, latent=super().forward(d.latent))

class EqualLinearS(EqualLinear):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0,
            lr_mul=1, activation=None):
        super().__init__(in_dim, out_dim, bias=bias, bias_init=bias_init,
                lr_mul=lr_mul, activation=activation)
    def forward(self, d):
        return DataBag(d, style=super().forward(d.style))

class NoiseInjectionF(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    def forward(self, data):
        image, noise = data.fmap, data.get('noise', None)
        batch, _, height, width = image.shape
        if noise is None:
            noise = np.random.RandomState(0).randn(
                    batch, height * width).astype('float32')
            noise = torch.from_numpy(noise).cuda().view(batch, 1, height, width)
        return DataBag(data, fmap=image + self.weight * noise)

class ConstantInputF(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, data):
        batch = data.latent.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return DataBag(data, fmap=out)

class BagLatent(nn.Module):
    def __init__(self, n_latent, truncation=1.0):
        super().__init__()
        self.n_latent = n_latent
        self.truncation = truncation
        self.latent_avg = None
    def forward(self, latent):
        if self.truncation != 1.0 and self.latent_avg is not None:
            latent = self.latent_avg + (
                    self.truncation * (latent - self.latent_avg))
        return DataBag(latent=latent.unsqueeze(1).repeat(1, self.n_latent, 1))

class AdjustLatent(nn.Module):
    def __init__(self, n_latent, truncation=1.0):
        super().__init__()
        self.n_latent = n_latent
        self.truncation = truncation
        self.register_buffer('latent_avg', torch.tensor(0.0))
    def forward(self, d):
        if self.truncation != 1.0 and self.latent_avg.ndim > 0:
            latent = self.latent_avg + (
                    self.truncation * (d.latent - self.latent_avg))
        else:
            latent = d.latent
        return DataBag(d,
            latent=latent.unsqueeze(1).repeat(1, self.n_latent, 1))

class PickLatent(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def __repr__(self):
        return f'{self.__class__.__name__}({self.index})'
    def forward(self, d):
        # print('pick latent %d  fmap size %s   output size %s' % (
        #     self.index, d.fmap.shape,
        #     d.output.shape if 'output' in d else 'na'))
        return DataBag(d, style=d.latent[:,self.index])

class InputLatent(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z):
        return DataBag(latent=z)

class ReturnOutput(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, d):
        return d.output

class PixelNormL(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data):
        return DataBag(data, latent=data.latent * torch.rsqrt(torch.mean(
            data.latent ** 2, dim=1, keepdim=True) + 1e-8))

class ApplyStyle(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, d):
        return DataBag(d, fmap=d.style[:,:,None,None] * d.fmap)

class FusedLeakyReLUF(op.FusedLeakyReLU):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__(channel, negative_slope=negative_slope, scale=scale)
    def forward(self, d):
        return DataBag(d, fmap=super().forward(d.fmap))

class ToRGBF(nn.Module):
    def __init__(self, in_channel, style_dim,
            upsample=True, blur_kernel=[1, 3, 3, 1], skip=False):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim,
                demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.skip = skip

    def forward(self, data):
        input, style = data.fmap, data.style
        skip = data.output if self.skip else None
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            if not skip.shape[2:] == out.shape[2:]:
                # print('mismatch skip %s vs in %s vs out %s' % (
                #     skip.shape, input.shape, out.shape))
                if hasattr(self, 'upsample'):
                    skip = self.upsample(skip)
                else:
                    # print('Missing upsample!')
                    upsample = Upsample([1, 3, 3, 1]).cuda()
                    skip = upsample(skip)
            out = out + skip
        return DataBag(data, output=out)
