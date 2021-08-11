import sys

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extra_compile_args = []
extra_link_args = []
# macOS Clang doesn't support OpenMP
if sys.platform != 'darwin':
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')


extensions = [Extension("coniferest.calc_mean_paths",
                        ["coniferest/calc_mean_paths.pyx"],
                        include_dirs=[np.get_include()],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args,
                        )]


setup(name='coniferest',
      description='Coniferous forests for better machine learning',
      version='0.0.1-alpha.0',
      author='Vladimir Korolev',
      author_email='balodja@gmail.com',
      packages=['coniferest'],
      ext_modules=cythonize(extensions),
      install_requires=['numpy', 'sklearn', 'matplotlib'],
      zip_safe=False)


# To build extensions in-place: python setup.py build_ext --inplace
