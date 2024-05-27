import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

extra_compile_args = []
extra_link_args = []
# macOS Clang doesn't support OpenMP, but we allow to force build with it anyway
if sys.platform != "darwin" or os.environ.get("CONIFEREST_FORCE_OPENMP_ON_MACOS", False):
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")


extensions = [
    Extension(
        "coniferest.calc_paths_sum",
        ["src/coniferest/calc_paths_sum.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]


setup(
    ext_modules=cythonize(extensions),
    cmdclass={"build_ext": build_ext},
)
