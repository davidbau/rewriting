#!/usr/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def read_points(filename):
    points = []
    for line in open(filename):
        items = line.split(', ')
        if len(items) > 1:
            points.append([int(items[1]), int(items[0])])
    return points


class MLS():
    def __init__(self, step=15, v_class=np.int32):
        self.step = step
        self.v_class = v_class  # The type of vxy, can be np.float32, np.int32
        self.type = 'rigid'

    def interp2(self, x, y, v, xi, yi):
        from scipy import interpolate
        v_t = np.transpose(v)
        f = interpolate.interp2d(x, y, v_t, kind='linear')
        df = f(xi, yi)
        return df

    def bilinear_interp(self, x, y, image):
        import math
        x_f = max(math.floor(x), 0)
        y_f = max(math.floor(y), 0)
        x_c = min(math.ceil(x), image.shape[0] - 1)
        y_c = min(math.ceil(y), image.shape[1] - 1)

        b = x - x_f
        a = y - y_f
        output = (1 - a) * (1 - b) * image[x_f, y_f] \
                 + a * (1 - b) * image[x_f, y_c] \
                 + b * (1 - a) * image[x_c, y_f] \
                 + a * b * image[x_c, y_c]
        return output

    def init_weights(self, points, gv, a=2):
        weights = np.zeros(shape=(gv.shape[0], points.shape[0]))
        for i in range(points.shape[0]):
            norms_2 = np.sum((np.tile(points[i], (gv.shape[0], 1)) - gv) ** 2, axis=1)
            weights[:, i] = np.divide(1, norms_2 ** a + 1e-8)
        return weights

    def get_precompute_A(self, points_start, point_hat, gv, weights):
        matrix_A = [{} for i in range(len(point_hat))]
        R1 = gv - points_start
        R2 = np.column_stack([R1[:, 1], -R1[:, 0]])

        for i in range(len(point_hat)):
            L1 = point_hat[i]
            L2 = np.column_stack([L1[:, 1], -L1[:, 0]])
            matrix_A[i]['a'] = weights[:, i] * np.sum(L1 * R1, axis=1)
            matrix_A[i]['b'] = weights[:, i] * np.sum(L1 * R2, axis=1)
            matrix_A[i]['c'] = weights[:, i] * np.sum(L2 * R1, axis=1)
            matrix_A[i]['d'] = weights[:, i] * np.sum(L2 * R2, axis=1)

        return matrix_A, R1

    def get_precompute_rigid(self, points, gv, weights):
        points_start = np.divide(np.dot(weights, points),
                                 np.column_stack((np.sum(weights, axis=1), np.sum(weights, axis=1))))
        point_hat = [np.tile(points[i], (points_start.shape[0], 1)) - points_start for i in range(points.shape[0])]
        matrix_A, v_point_start = self.get_precompute_A(points_start, point_hat, gv, weights)
        norm_of_v_point_star = np.sqrt(np.sum(v_point_start ** 2, axis=1))
        return {'A': matrix_A, 'normof_v_Pstar': norm_of_v_point_star}

    def get_precompute_points(self, points, gv):
        weights = self.init_weights(points, gv)
        mlsd = {}
        mlsd['p'] = points
        mlsd['v'] = gv
        mlsd['type'] = self.type
        mlsd['w'] = weights
        if self.type == 'rigid':
            mlsd['data'] = self.get_precompute_rigid(points, gv, weights)
        return mlsd

    def points_transform_rigid(self, mlsd, points):
        weights = mlsd['w']
        points_start = np.divide(np.dot(weights, points),
                                 np.column_stack((np.sum(weights, axis=1), np.sum(weights, axis=1))))
        fv2 = np.zeros_like(points_start)
        for i in range(points.shape[0]):
            points_hat = np.tile(points[i], (points_start.shape[0], 1)) - points_start
            fv2 = fv2 + np.column_stack((
                np.sum(points_hat * np.column_stack((mlsd['data']['A'][i]['a'], mlsd['data']['A'][i]['c'])), axis=1),
                np.sum(points_hat * np.column_stack((mlsd['data']['A'][i]['b'], mlsd['data']['A'][i]['d'])), axis=1)
            ))
        norm_of_fv2 = np.sqrt(np.sum(fv2 ** 2, axis=1))
        norm_factor = np.divide(mlsd['data']['normof_v_Pstar'], norm_of_fv2)
        fv = fv2 * np.column_stack((norm_factor, norm_factor)) + points_start
        return fv

    def mlsd_2d_transform(self, mlsd, points):
        if mlsd['type'] == 'rigid':
            fv = self.points_transform_rigid(mlsd, points)
        return fv

    def mlsd_2d_warp(self, image, mlsd, points_end, grid_x, grid_y):
        v = np.column_stack((grid_y.flatten(), grid_x.flatten()))
        sfv = self.mlsd_2d_transform(mlsd, points_end)
        dxy = v - sfv

        vx = np.reshape(dxy[:, 1], newshape=(grid_x.shape))
        vy = np.reshape(dxy[:, 0], newshape=(grid_x.shape))

        x, y = np.arange(image.shape[0], step=15), np.arange(image.shape[1], step=15)
        xi, yi = np.arange(image.shape[0], step=1), np.arange(image.shape[0], step=1)

        dxT = self.interp2(x, y, vx, xi, yi)
        dyT = self.interp2(x, y, vy, xi, yi)

        vxy = np.column_stack((-dxT.flatten(), -dyT.flatten()))
        vxy = self.v_class(np.reshape(vxy, newshape=(image.shape[0], image.shape[1], 2)))

        warped_image = np.zeros_like(image)
        warped_image.fill(255)

        for x in range(warped_image.shape[0]):
            for y in range(warped_image.shape[1]):
                source_x = x + dxT[x, y]
                source_y = y + dyT[x, y]
                if source_x < warped_image.shape[0] and source_y < warped_image.shape[1]:
                    warped_image[x, y] = self.bilinear_interp(source_x, source_y, image)
        return warped_image, vxy

    def warp_MLS(self, image, points_start, points_end):
        assert (points_start.shape[0] == points_end.shape[0])
        grid_x, grid_y = np.meshgrid(np.arange(image.shape[0], step=15), np.arange(image.shape[0], step=self.step))
        gv = np.column_stack((grid_y.flatten(), grid_x.flatten()))
        mlsd = self.get_precompute_points(points_start, gv)
        warpped_image, vxy = self.mlsd_2d_warp(image, mlsd, points_end, grid_x, grid_y)
        return warpped_image, vxy

    """
        Return an warped image and the vxy.

        Parameters
        ----------
        image : Image array with opencv
        points_start : Original point locations with shape (n, 2), dtype: (list || np.array)
        points_end : Target point locations with shape (n, 2), dtype: (list || np.array)

        Returns
        -------
        warp_image : Warped image array with opencv
        vxy: The offset of point with shape (w, h, 2)
            For the point (x, y) in original image, the warped point location is [x + vxy(x, y, 0), y + vxy[x, y, 1]]
        """
    def run_MLS(self, image, points_start, points_end, padding=30):
        if np.ndim(image) == 4:
            img = np.pad(image.squeeze(0), ((padding, padding), (padding, padding), (0, 0)), 'symmetric')
        elif np.ndim(image) == 3:
            img = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')
        elif np.ndim(image) == 2:
            img = np.pad(image, ((padding, padding), (padding, padding)), 'symmetric')
        points_start = np.array(points_start) + padding
        points_end = np.array(points_end) + padding
        warp_image, vxy = self.warp_MLS(img, points_start, points_end)
        warp_image = warp_image[padding:-padding, padding:-padding]
        vxy = vxy[padding:-padding, padding:-padding]
        return warp_image, vxy

    def run_MLS_in_folder(self, root_folder):
        img_A = Image.open('%s/original_A.png' % root_folder).convert('RGB')
        img_B = Image.open('%s/original_B.png' % root_folder).convert('RGB')

        points_A = read_points('%s/correspondence_A.txt' % root_folder)
        points_B = read_points('%s/correspondence_Bt.txt' % root_folder)
        points_middle = (np.array(points_A) + np.array(points_B)) / 2

        mls_util = MLS(v_class=np.int32)
        warp_AtoB, vxy_AtoB = mls_util.run_MLS(img_A, points_A, points_B)
        warp_AtoM, vxy_AtoM = mls_util.run_MLS(img_A, points_A, points_middle)
        warp_BtoA, vxy_BtoA = mls_util.run_MLS(img_B, points_B, points_A)
        warp_BtoM, vxy_BtoM = mls_util.run_MLS(img_B, points_B, points_middle)

        np.save('%s/AtoB.npy' % root_folder,vxy_AtoB)
        plt.imsave('%s/warp_AtoM.png' % root_folder, warp_AtoM)
        np.save('%s/BtoA.npy' % root_folder, vxy_BtoA)
        plt.imsave('%s/warp_BtoM.png' % root_folder, warp_BtoM)
