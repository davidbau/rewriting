from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np
import scipy.io
from PIL import Image
import inspect, re
import numpy as np
import os
import math
from PIL import Image
import torchvision.transforms as transforms
import collections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def read_image(path, witdh):
    I = Image.open(path).convert('RGB')
    transform = get_transform(witdh)
    return transform(I).unsqueeze(0)

def get_transform(witdh):
    transform_list = []
    osize = [witdh, witdh]
    transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]

    return transforms.Compose(transform_list)

def save_final_image(image, name, save_dir):
    im_numpy = tensor2im(image)
    save_image(im_numpy, os.path.join(save_dir, name + '.png'))

def save_map_image(map_values, name, save_dir, level=0, binary_color=False):
    if level == 0:
        map_values = map_values
    else:
        scale_factor = int(math.pow(2,level-1))
        map_values = upsample_map(map_values, scale_factor)
    if binary_color==True:
        map_image = binary2color_image(map_values)
    else:
        map_image = map2image(map_values)
    save_image(map_image, os.path.join(save_dir, name + '.png'))

def upsample_map(map_values, scale_factor, mode='nearest'):
    if scale_factor == 1:
        return map_values
    else:
        upsampler = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsampler(Variable(map_values)).data

def downsample_map(map_values, scale_factor):
    if scale_factor == 1:
        return map_values
    else:
        d = scale_factor
        downsampler = torch.nn.AvgPool2d((d, d), stride=(d, d))
        return downsampler(Variable(map_values)).data

def tensor2im(image_tensor, imtype=np.uint8, index=0):
    image_numpy = image_tensor[index].cpu().float().numpy()
    mean = np.zeros((1,1,3))
    mean[0,0,:] = [0.485, 0.456, 0.406]
    stdv = np.zeros((1,1,3))
    stdv[0,0,:] = [0.229, 0.224, 0.225]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * stdv + mean) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[2] == 1:
        image_numpy = np.tile(image_numpy, [1,1,3])
    return image_numpy.astype(imtype)

def feature2images(feature, size=[1,1], imtype=np.uint8):
    feature_np = feature.cpu().float().numpy()
    mosaic = np.zeros((size[0]*feature_np.shape[2], size[1]*feature_np.shape[3]))
    for i in range(size[0]):
       for j in range(size[1]):
           single_feature = feature_np[0,i*size[1]+j,:,:]
           stretched_feature = stretch_image(single_feature)
           mosaic[(i*feature_np.shape[2]):(i+1)*(feature_np.shape[2]),
               j*feature_np.shape[3]:(j+1)*(feature_np.shape[3])] = stretched_feature
    mosaic = np.transpose(np.tile(mosaic, [3,1,1]), (1,2,0))
    return mosaic.astype(np.uint8)

def grad2image(grad, imtype=np.uint8):
    grad_np = grad.cpu().float().numpy()
    image = np.zeros((grad.shape[2], grad.shape[3]))
    for i in range(grad_np.shape[1]):
           image = np.maximum(image, grad_np[0,i,:,:])
    return stretch_image(image).astype(imtype)

def batch2im(images_tensor, imtype=np.uint8):
    image_numpy = images_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def map2image(values_map, imtype=np.uint8):
    image_numpy = values_map[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = stretch_image(image_numpy)
    image_numpy = np.tile(image_numpy, [1,1,3])
    return image_numpy.astype(imtype)

def binary2color_image(binary_map, color1=[0,185,252], color2=[245,117,255], imtype=np.uint8):
    assert(binary_map.size(1)==1)
    binary_ref = binary_map[0].cpu().float().numpy()
    binary_ref = np.transpose(binary_ref, (1, 2, 0))
    binary_ref = np.tile(binary_ref, [1,1,3])
    color1_ref = np.tile(np.array(color1), [binary_map.size(2),binary_map.size(3),1])
    color2_ref = np.tile(np.array(color2), [binary_map.size(2),binary_map.size(3),1])
    color_map = binary_ref*color1_ref + (1-binary_ref)*color2_ref

    return color_map.astype(imtype)

def stretch_image(image):
    min_image = np.amin(image)
    max_image = np.amax(image)
    if max_image != min_image:
        return (image - min_image)/(max_image - min_image)*255.0
    else:
        return image

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_tensor_as_mat(tensor, path):
    tensor_numpy = tensor.cpu().numpy()
    print(path)
    scipy.io.savemat(path, mdict={'dna': tensor_numpy})

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_mask(path):
    image = Image.open(path)
    np_image = np.array(image)
    np_image = np_image[:,:,0]
    print(np_image.shape)
    return np.where(np_image>128, 1, 0)
