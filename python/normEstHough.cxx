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

#include "normEstHough.h"

NormEstHough::NormEstHough(){
        n_planes=700;
        n_rot=5;
        n_phi=15;
        tol_angle_rad=0.79;
        neighborhood_size = 200;
        use_density = false;
        k_density = 5;
}


NormEstHough::~NormEstHough(){}

int NormEstHough::size(){
    return pts.rows();
}

int NormEstHough::size_normals(){
    return nls.rows();
}



void NormEstHough::get_points(double* array, int m, int n) {

    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = pts(i,j);
            index ++ ;
            }
        }
    return ;
}

void NormEstHough::get_normals(double* array, int m, int n) {

    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = nls(i,j);
            index ++ ;
            }
        }
    return ;
}
 
void NormEstHough::set_points(double* array, int m, int n){
    // resize the point cloud
    pts.resize(m,3);

    // fill the point cloud
    int i, j ;
    int index = 0 ;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            pts(i,j) = array[index];
            index ++ ;
        }
    }
    return ;
}

void NormEstHough::set_normals(double* array, int m, int n){
    // resize the point cloud
    nls.resize(m,3);

    // fill the point cloud
    int i, j ;
    int index = 0 ;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            nls(i,j) = array[index];
            index ++ ;
        }
    }
    return ;
}
