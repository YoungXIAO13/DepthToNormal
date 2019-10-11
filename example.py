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


import numpy as np
import python.lib.python.NormalEstimatorHough as Estimator

## function for cube generation
def cube(N):
    cube = np.random.rand(N,3)*2 -1
    for i in range(N):
        cube[i, np.random.randint(0,3)] = np.random.randint(0,2)*2 - 1
    return cube

print("Create a cube...", end="", flush=True)
pts = cube(100000)
print("Done")

print("Estimate normals...", end="", flush=True)
estimator = Estimator.NormalEstimatorHough()
estimator.set_points(pts)
estimator.set_K(50)
estimator.estimate_normals()
print("Done")

print("Saving results...", end="", flush=True)
estimator.saveXYZ("out.xyz")
print("Done")
