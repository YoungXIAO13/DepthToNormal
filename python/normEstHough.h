// Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
// Copyright (c) 2016 Alexande Boulch and Renaud Marlet
//
// This program is free software; you can redistribute it and/or modify it under the terms
// of the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details. You should have received a copy of
// the GNU General Public License along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
//
// PLEASE ACKNOWLEDGE THE AUTHORS AND PUBLICATION:
// "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
// by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
// Computer Graphics Forum
//
// The full license can be retrieved at https://www.gnu.org/licenses/gpl-3.0.en.html

#ifndef NORMALS_HOUGH_PYTHON_HEADER
#define NORMALS_HOUGH_PYTHON_HEADER

#include "Normals.h"

class NormEstHough:public Eigen_Normal_Estimator{

public:
    NormEstHough();
    ~NormEstHough();

    int size();
    int size_normals();

    void get_points(double* array, int m, int n);
    void get_normals(double* array, int m, int n);
    void set_points(double* array, int m, int n);
    void set_normals(double* array, int m, int n);


};


#endif
