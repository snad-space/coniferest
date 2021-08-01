from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [Extension("coniferest.calc_mean_paths",
                        ["coniferest/calc_mean_paths.pyx"],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'],
                        )]


setup(name='coniferest',
      description='Coniferous forests for better machine learning',
      version='dev',
      author='Vladimir Korolev',
      author_email='balodja@gmail.com',
      packages=['coniferest'],
      ext_modules=cythonize(extensions),
      install_requires=['numpy', 'sklearn', 'matplotlib'],
      zip_safe=False)


# To build extensions in-place: python setup.py build_ext --inplace
