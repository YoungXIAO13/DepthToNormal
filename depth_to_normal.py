import argparse
import os
import numpy as np
import pandas as pd
import cv2
from math import atan, tan, pi
from tqdm import tqdm
import python.lib.python.NormalEstimatorHough as Estimator
import itertools


# reference:
# https://elcharolin.wordpress.com/2017/09/06/transforming-a-depth-map-into-a-3d-point-cloud/
def create_point_cloud(depth_path, fov_x, fov_y, ratio=50/65535.):
    img = cv2.imread(depth_path, -1)
    H, W = img.shape
    point_cloud = []

    # change value to meters
    img = img.astype('float64')
    img *= ratio

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

parser.add_argument('--depth_dir', type=str, default=None, help='path to folder saving all the depth map')

opt = parser.parse_args()

worlds = sorted([name for name in os.listdir(opt.depth_dir) if os.path.isdir(os.path.join(opt.depth_dir, name))])

for world in tqdm(worlds, desc='Generating Normal Map'):
    world_dir = os.path.join(opt.depth_dir, world)
    subsets = sorted(os.listdir(world_dir))
    for subset in tqdm(subsets, desc='world {}'.format(world)):
        sub_dir = os.path.join(world_dir, subset)
        files = sorted(os.listdir(sub_dir))
        for file in tqdm(files):
            depth_path = os.path.join(sub_dir, file)

            # set camera intrinsics and image size
            H, W = 375, 1242
            fx, fy = 725, 725
            fov_x = 2 * atan(W / (2 * fx))
            fov_y = 2 * atan(H / (2 * fy))
            pts = create_point_cloud(depth_path, fov_x, fov_y)

            # generate point-to-point depth map in mm
            point_depth = np.linalg.norm(pts, 2, -1).reshape(H, W) / 10
            depth_path_point = depth_path.replace('depthgt', 'depth_point')
            if not os.path.exists(os.path.dirname(depth_path_point)):
                os.makedirs(os.path.dirname(depth_path_point))
            cv2.imwrite(depth_path_point, point_depth)

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
            normal_img_path = depth_path.replace('depthgt', 'normal_map')
            if not os.path.exists(os.path.dirname(normal_img_path)):
                os.makedirs(os.path.dirname(normal_img_path))
            xyz_normal = np.loadtxt(normal_xyz_path)
            normal = xyz_normal[:, 3:].reshape(H, W, 3)
            normal_uint = ((2 ** 16 - 1) * (normal + 1) / 2).astype('uint16')
            cv2.imwrite(normal_img_path, normal_uint)
