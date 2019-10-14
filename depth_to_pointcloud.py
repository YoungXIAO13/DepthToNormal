import argparse
import os
import numpy as np
import pandas as pd
import cv2
from math import atan, tan, pi
from tqdm import tqdm
import itertools


img_ext = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']


def read_calib(calib_path, img_path):
    """
    Read in the camera intrinsics and return FOV
    :param calib_path: where save the matrix K in format (fx, fy, cx, cy)
    :param img_path: where save the depth image
    :return: fov_x, fov_y
    """
    df = pd.read_csv(calib_path)
    fx, fy = float(df.columns[0]), float(df.columns[1])
    img = cv2.imread(img_path, -1)
    H, W = img.shape
    fov_x = 2 * atan(W / (2 * fx))
    fov_y = 2 * atan(H / (2 * fy))
    return fov_x, fov_y


def zero_depth_mask(depth_path):
    img = cv2.imread(depth_path, -1)
    mask = (img == 0).astype('uint8')
    mask *= 255
    return mask


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

parser.add_argument('--depth', type=str, default=None, help='path to folder saving all the depth map')
parser.add_argument('--calib', type=str, default=None, help='path to folder saving all the calibration matrix')
parser.add_argument('--pc', type=str, default=None, help='path to save all the point cloud')
parser.add_argument('--mask', type=str, default=None, help='path to save all the masks for invalid depth')

opt = parser.parse_args()

# read all the file names
names = sorted([name.split('.')[0] for name in os.listdir(opt.depth) if name.split('.')[1] in img_ext])

for name in tqdm(names, desc='Generating Point Cloud'):
    depth_path = os.path.join(opt.depth, '{}.png'.format(name))
    calib_path = os.path.join(opt.calib, '{}.txt'.format(name))
    pc_path = os.path.join(opt.pc, '{}.xyz'.format(name))
    mask_path = os.path.join(opt.mask, '{}.png'.format(name))

    # create the zero depth mask and save it
    invalid_mask = zero_depth_mask(depth_path)
    cv2.imwrite(mask_path, invalid_mask)

    # create the point cloud and save it
    fov_x, fov_y = read_calib(calib_path, depth_path)
    point_cloud = create_point_cloud(depth_path, fov_x, fov_y)
    np.savetxt(pc_path, point_cloud, fmt='%.3f', newline="\r\n")

