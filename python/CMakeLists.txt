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

# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
 ${CMAKE_SOURCE_DIR}/python/NormalEstimatorHough.pyx
 PROPERTIES CYTHON_IS_CXX TRUE )



# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( NormalEstimatorHough
   NormalEstimatorHough.pyx
   normEstHough.cxx
   )
target_link_libraries(NormalEstimatorHough ${EXTRA_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${PROJECT_LINK_LIB} )
