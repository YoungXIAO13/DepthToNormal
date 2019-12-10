import argparse
import os
import numpy as np
import pandas as pd
import cv2
from math import atan, tan, pi
from tqdm import tqdm
import python.lib.python.NormalEstimatorHough as Estimator
import itertools

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

# reference:
# https://elcharolin.wordpress.com/2017/09/06/transforming-a-depth-map-into-a-3d-point-cloud/
def create_point_cloud(img, fov_x, fov_y):
    H, W = img.shape
    point_cloud = []

    # change value to meters
    img = img.astype('float32')

    for i, j in itertools.product(range(H), range(W)):
        alpha = (pi - fov_x) / 2
        gamma = alpha + fov_x * float((W - j) / W)
        delta_x = img[i, j] / tan(gamma)

        alpha = (pi - fov_y) / 2
        gamma = alpha + fov_y * float((H - i) / H)
        delta_y = img[i, j] / tan(gamma)

        point_cloud.append([delta_x, delta_y, float(img[i, j])])

    return np.array(point_cloud)


# Main code

parser = argparse.ArgumentParser(description='Transform depth maps to point cloud given the camera intrinsics')

parser.add_argument('--dataset', type=str, choices=['KITTI', 'NYU'], help='the organization format of the dataset')
parser.add_argument('--input', type=str, default=None, help='path to folder saving all the depth map')

opt = parser.parse_args()

if opt.dataset == 'KITTI':

    worlds = sorted([name for name in os.listdir(opt.input) if os.path.isdir(os.path.join(opt.input, name))])
    for world in tqdm(worlds, desc='Generating Normal Map'):
        world_dir = os.path.join(opt.input, world)
        subsets = sorted(os.listdir(world_dir))
        for subset in tqdm(subsets, desc='world {}'.format(world)):
            sub_dir = os.path.join(world_dir, subset)
            files = sorted(os.listdir(sub_dir))
            for file in tqdm(files, desc='subset {}'.format(subset)):
                depth_path = os.path.join(sub_dir, file)

                # set camera intrinsics and image size
                H, W = 375, 1242
                fx, fy = 725, 725
                fov_x = 2 * atan(W / (2 * fx))
                fov_y = 2 * atan(H / (2 * fy))

                # generate point-to-point depth map in mm
                depth_path_point = depth_path.replace('depthgt', 'depth_point')
                depth_path_point = depth_path_point.replace('png', 'npy')

                if not os.path.exists(os.path.dirname(depth_path_point)):
                    os.makedirs(os.path.dirname(depth_path_point))

                depth = cv2.imread(depth_path, -1)
                pts = create_point_cloud(depth, fov_x, fov_y)
                point_depth = np.linalg.norm(pts, 2, -1).reshape(H, W)
                np.save(depth_path_point, point_depth)

                # skip if normal map is generated
                normal_img_path = depth_path.replace('depthgt', 'normal_map')
                if os.path.exists(normal_img_path):
                    continue

                # create the normal map and save it in .xyz
                normal_xyz_path = depth_path.replace('depthgt', 'normal_xyz')
                if not os.path.exists(os.path.dirname(normal_xyz_path)):
                    os.makedirs(os.path.dirname(normal_xyz_path))
                estimator = Estimator.NormalEstimatorHough()
                estimator.set_points(pts)
                estimator.set_K(50)
                estimator.estimate_normals()
                estimator.saveXYZ(normal_xyz_path)

                # read in .xyz and transform it to .png of uint-16
                if not os.path.exists(os.path.dirname(normal_img_path)):
                    os.makedirs(os.path.dirname(normal_img_path))
                xyz_normal = np.loadtxt(normal_xyz_path)
                normal = xyz_normal[:, 3:].reshape(H, W, 3)[:, :, ::-1]
                normal_uint = ((2 ** 16 - 1) * (normal + 1) / 2).astype('uint16')
                cv2.imwrite(normal_img_path, normal_uint)

elif opt.dataset == 'NYU':
    normal_dir = os.path.join(os.path.dirname(opt.input), 'normal_map')
    if not os.path.exists(normal_dir):
        os.makedirs(normal_dir)
    depth_point_dir = os.path.join(os.path.dirname(opt.input), 'depth_point')
    if not os.path.exists(depth_point_dir):
        os.makedirs(depth_point_dir)
    depths = np.load(opt.input)

    eigen_crop = [21, 461, 25, 617]
    depth_point = []

    for i in tqdm(range(depths.shape[0])):
        depth = depths[i, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]

        # set camera intrinsics and image size
        H, W = eigen_crop[1] - eigen_crop[0], eigen_crop[3] - eigen_crop[2]
        fx, fy = 582.62, 582.69
        fov_x = 2 * atan(W / (2 * fx))
        fov_y = 2 * atan(H / (2 * fy))

        pts = create_point_cloud(depth, fov_x, fov_y)
        depth_point.append(np.linalg.norm(pts, 2, -1).reshape(H, W))

        depth_point_path = os.path.join(depth_point_dir, '{:04d}.png'.format(i))
        plt.imsave(depth_point_path, depth_point[i])

        # create the normal map and save it in .xyz
        normal_xyz_path = os.path.join(normal_dir, '{:04d}.xyz'.format(i))
        estimator = Estimator.NormalEstimatorHough()
        estimator.set_points(pts)
        estimator.set_K(50)
        estimator.estimate_normals()
        estimator.saveXYZ(normal_xyz_path)

        # read in .xyz and transform it to .png of uint-16
        normal_img_path = os.path.join(normal_dir, '{:04d}.png'.format(i))
        xyz_normal = np.loadtxt(normal_xyz_path)
        normal = xyz_normal[:, 3:].reshape(H, W, 3)[:, :, ::-1]
        normal_uint = ((2 ** 16 - 1) * (normal + 1) / 2).astype('uint16')
        cv2.imwrite(normal_img_path, normal_uint)

    np.save(opt.input.replace('depth', 'depth_point'), depth_point)

else:
    raise NameError
