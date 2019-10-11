# Depth To Normal
Python code used to transform depth map to normal map using point cloud as intermediate representation


## Compilation

This repo uses the code from 
["Fast and robust normal estimation for point clouds with sharp features" by Alexandre Boulch and Renaud Marlet](https://github.com/aboulch/normals_Hough)
to estimate the normal map from point clouds.

**python wrapper**

Using the library with Python requires building the wrapper. 
This works fine with g++ 5.4.0 on Ubuntu 16.04 with Anaconda. 

```
cd python
python setup.py install --home="."
```

**python test**
```
python example.py
```
The script creates a point cloud on a cube and estimate the normals. 
It produces a ```.xyz``` containing both points and normals, it can be displayed using Meshlab. 

## Usage

Our usage is based on the dataset providing accurate RGB-D image pairs such as [iBims-1](https://www.bgu.tum.de/lmf/ibims1/)

First, use ```depth_to_pointcloud.py``` to generate point clouds from depth maps.

* the default format for depth map is in ```uint16```

* some pixels in depth map is labeled as 0 as a result of invalid depth estimation

* the generated point cloud is of format ```[x, y, z]``` in meters 

Then, use ```pointcloud_to_normal.py``` to generate normal maps from point clouds.

* here we provide also a function to generate point to point depth map from plane to plane depth map

* the normal_xyz output saves the point cloud with normal map in ```.xyz``` file of format ```[x, y, z, n_x, n_y, n_z]```

* the normal_img output saves the normal map in ```.png``` file with RGB values corresponding to normal vectors

## License

 See [here](https://github.com/aboulch/normals_Hough) for the original license.