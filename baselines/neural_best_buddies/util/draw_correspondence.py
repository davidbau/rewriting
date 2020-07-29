import torch
import os
import math
from torch.autograd import Variable
import numpy as np
from PIL import Image
from util import util
import numpy as np

def color_map(i):
    colors = [
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [128,128,0],
        [0,128,128]
    ]

    if i < 5:
        return colors[i]
    else:
        return np.random.randint(0,256,3)

def draw_square(image, center, color, radius = 2):
    d = 2*radius + 1
    image_p = np.pad(image, ((radius,radius),(radius,radius),(0,0)),'constant')
    center_p = [center[0]+radius, center[1]+radius]
    image_p[center_p[0]-radius, (center_p[1]-radius):(center_p[1]-radius+d), :] = np.tile(color,[d,1])
    image_p[(center_p[0]-radius):(center_p[0]-radius+d), center_p[1]-radius, :] = np.tile(color,[d,1])
    image_p[center_p[0]+radius, (center_p[1]-radius):(center_p[1]-radius+d), :] = np.tile(color,[d,1])
    image_p[(center_p[0]-radius):(center_p[0]-radius+d), center_p[1]+radius, :] = np.tile(color,[d,1])

    return image_p[radius:image_p.shape[0]-radius, radius:image_p.shape[1]-radius, :]

def draw_dots(image, center, color):
    image[center[0], center[1], :] = color
    return image

def draw_circle(image, center, color,  radius = 4, border_color = [255,255,255]):
    image_p = np.pad(image, ((radius,radius),(radius,radius),(0,0)),'constant')
    center_p = [center[0]+radius, center[1]+radius]
    edge_d = math.floor((2*radius + 1)/6)
    image_p[center_p[0]-radius, (center_p[1]-edge_d):(center_p[1]+edge_d+1), :] = np.tile(border_color,[3,1])
    image_p[center_p[0]+radius, (center_p[1]-edge_d):(center_p[1]+edge_d+1), :] = np.tile(border_color,[3,1])
    for i in range(1,radius):
        image_p[center_p[0]+i, center_p[1]-radius+i-1, :] = border_color
        image_p[center_p[0]-i, center_p[1]-radius+i-1, :] = border_color
        image_p[center_p[0]+i, (center_p[1]-radius+i):(center_p[1]+radius-i+1), :] = np.tile(color, [2*(radius-i)+1,1])
        image_p[center_p[0]-i, (center_p[1]-radius+i):(center_p[1]+radius-i+1), :] = np.tile(color, [2*(radius-i)+1,1])
        image_p[center_p[0]+i, center_p[1]+radius+1-i, :] = border_color
        image_p[center_p[0]-i, center_p[1]+radius+1-i, :] = border_color

    image_p[center_p[0], center_p[1]-radius, :] = border_color
    image_p[center_p[0], (center_p[1]-radius+1):(center_p[1]+radius), :] = np.tile(color, [2*(radius-1)+1,1])
    image_p[center_p[0], center_p[1]+radius, :] = border_color

    return image_p[radius:image_p.shape[0]-radius, radius:image_p.shape[1]-radius, :]

def draw_points(self, A, points, radius, name, save_dir, unicolor = False, level = 0):
    A_marked = util.tensor2im(A)
    for i in range(len(points)):
        center = [points[i][0], points[i][1]]
        if unicolor == True:
            color = color_map(0)
        else:
            color = color_map(i)
        if level > 2 :
            A_marked = draw_square(A_marked, center, color, radius=radius)
        elif level == 2 or level == 1:
            A_marked = draw_circle(A_marked, center, color)
        else:
            A_marked = draw_dots(A_marked, center, color)

    util.save_image(A_marked, os.path.join(save_dir, name + '.png'))

def draw_correspondence(A, B, correspondence, radius, save_dir, level = 0, name=''):
    A_marked = util.tensor2im(A)
    B_marked = util.tensor2im(B)
    for i in range(len(correspondence[0])):
        color = color_map(i)
        center_1 = [correspondence[0][i][0], correspondence[0][i][1]]
        center_2 = [correspondence[1][i][0], correspondence[1][i][1]]
        if level < 3 :
            A_marked = draw_circle(A_marked, center_1, color)
            B_marked = draw_circle(B_marked, center_2, color)
        else:
            A_marked = draw_square(A_marked, [center_1[0]+radius, center_1[1]+radius], color, radius=radius)
            B_marked = draw_square(B_marked, [center_2[0]+radius, center_2[1]+radius], color, radius=radius)

    util.save_image(A_marked, os.path.join(save_dir, 'A_level_'+str(level)+name+'.png'))
    util.save_image(B_marked, os.path.join(save_dir, 'B_level_'+str(level)+name+'.png'))
