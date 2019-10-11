from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


ext_modules = [Extension(
        "NormalEstimatorHough",
        sources=["NormalEstimatorHough.pyx", "normEstHough.cxx"],
        include_dirs=[numpy.get_include(), "../third_party_includes/"],
        language="c++",             # generate C++ code
        extra_compile_args = ["-fopenmp", "-std=c++11"],
        extra_link_args=['-lgomp']
  )]

setup(
    name = "Hough Normal Estimator",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
