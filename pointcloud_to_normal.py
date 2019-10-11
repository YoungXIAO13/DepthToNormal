# Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
# Copyright (c) 2016 Alexande Boulch and Renaud Marlet
#
# This program is free software; you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation;
# either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details. You should have received a copy of
# the GNU General Public License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
#
# PLEASE ACKNOWLEDGE THE AUTHORS AND PUBLICATION:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
# Computer Graphics Forum
#
# The full license can be retrieved at https://www.gnu.org/licenses/gpl-3.0.en.html


import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
import python.lib.python.NormalEstimatorHough as Estimator

# Main code

parser = argparse.ArgumentParser(description='Transform depth maps to point cloud given the camera intrinsics')

parser.add_argument('--pc', type=str, default=None, help='path to directory saving all the point clouds')
parser.add_argument('--depth_plane', type=str, default=None, help='path to directory saving the plane-to-plane depth maps')
parser.add_argument('--depth_point', type=str, default=None, help='path to directory saving the point-to-point depth maps')
parser.add_argument('--normal_xyz', type=str, default=None, help='path to save all the normals in .xyz file')
parser.add_argument('--normal_img', type=str, default=None, help='path to save all the normals in .png file')

opt = parser.parse_args()

# read all the file names
names = sorted([name.split('.')[0] for name in os.listdir(opt.pc) if name.split('.')[1] == 'xyz'])

for name in tqdm(names, desc='Generating normal maps and point-to-point depth maps'):
    normal_xyz_path = os.path.join(opt.normal_xyz, '{}.xyz'.format(name))
    normal_img_path = os.path.join(opt.normal_img, '{}.png'.format(name))
    pc_path = os.path.join(opt.pc, '{}.xyz'.format(name))
    depth_path_plane = os.path.join(opt.depth_plane, '{}.png'.format(name))
    depth_path_point = os.path.join(opt.depth_point, '{}.png'.format(name))

    # get the H, W of depth map
    depth = cv2.imread(depth_path_plane, -1)
    H, W = depth.shape

    # create point-to-point depth map and save in .png of uint-16
    pts = np.loadtxt(pc_path)
    assert pts.shape[-1] == 3, "The last channel of point cloud must be 3 representing X, Y, Z"
    point_depth = np.linalg.norm(pts, 2, -1).reshape(H, W)
    assert np.max(point_depth) <= 100, "depth value too large, considering change the normalization ration !"
    point_depth_uint = (point_depth * (2 ** 16 - 1) / 100).astype('uint16')
    cv2.imwrite(depth_path_point, point_depth_uint)

    # create the normal map and save it in .xyz
    estimator = Estimator.NormalEstimatorHough()
    estimator.set_points(pts)
    estimator.set_K(50)
    estimator.estimate_normals()
    estimator.saveXYZ(normal_xyz_path)

    # read in .xyz and transform it to .png of uint-16
    xyz_normal = np.loadtxt(normal_xyz_path)
    normal = xyz_normal[:, 3:].reshape(H, W, 3)
    normal_uint = ((2**16 - 1) * (normal + 1) / 2).astype('uint16')
    cv2.imwrite(normal_img_path, normal_uint)
