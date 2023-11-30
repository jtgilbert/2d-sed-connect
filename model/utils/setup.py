from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("upstream_basin.pyx"),
    include_dirs=[numpy.get_include()]
)