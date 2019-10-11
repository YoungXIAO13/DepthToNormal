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

# distutils: language = c++
# distutils: sources = normEstHough.cxx

cimport c_normEstHough
import numpy as np
cimport numpy as np
import cython

cdef class NormalEstimatorHough:
    cdef c_normEstHough.NormEstHough *thisptr

    def __cinit__(self):
        self.thisptr = new c_normEstHough.NormEstHough()

    def __dealloc__(self):
        del self.thisptr

    cpdef loadXYZ(self, filename):
        self.thisptr.loadXYZ(str.encode(filename))

    cpdef saveXYZ(self,filename):
        self.thisptr.saveXYZ(str.encode(filename))

    cpdef estimate_normals(self):
        self.thisptr.estimate_normals()

    cpdef size(self):
        return self.thisptr.size()

    cpdef size_normals(self):
        return self.thisptr.size_normals()

    cpdef get_T(self):
        return self.thisptr.get_T()
    cpdef set_T(self, T):
        self.thisptr.set_T(T)

    cpdef get_n_phi(self):
        return self.thisptr.get_n_phi()
    cpdef set_n_phi(self, nphi):
        self.thisptr.set_n_phi(nphi)

    cpdef get_n_rot(self):
        return self.thisptr.get_n_rot()
    cpdef set_n_rot(self, nrot):
        self.thisptr.set_n_rot(nrot)

    cpdef get_K(self):
        return self.thisptr.get_K()
    cpdef set_K(self, K):
        self.thisptr.set_K(K)

    cpdef get_density_sensitive(self):
        return self.thisptr.get_density_sensitive()
    cpdef set_density_sensitive(self, d_s):
        self.thisptr.set_density_sensitive(d_s)

    cpdef get_tol_angle_rad(self):
        return self.thisptr.get_tol_angle_rad()
    cpdef set_tol_angle_rad(self, alpha):
        self.thisptr.set_tol_angle_rad(alpha)

    cpdef get_K_density(self):
        return self.thisptr.get_K_density()
    cpdef set_K_density(self, K_d):
        self.thisptr.set_K_density(K_d)

    def get_points(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.get_points(<double *> d2.data, m,n)
        return d

    def get_normals(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.get_normals(<double *> d2.data, m,n)
        return d

    def set_points(self, points):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = points
        self.thisptr.set_points(<double *> d2.data, points.shape[0], points.shape[1])

    def set_normals(self, normals):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = normals
        self.thisptr.set_normals(<double *> d2.data, normals.shape[0], normals.shape[1])
