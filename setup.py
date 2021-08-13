import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
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


def get_readme():
    return (Path(__file__).parent / 'README.md').read_text(encoding='utf8')


setup(name='coniferest',
      version='0.0.2',
      description='Coniferous forests for better machine learning',
      long_description=get_readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/snad-space/coniferest',
      author='Vladimir Korolev, SNAD team',
      author_email='balodja@gmail.com',
      license='MIT',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
      ],
      packages=['coniferest', 'coniferest.sklearn'],
      package_data={
          '': ['*.pxd'],
      },
      ext_modules=cythonize(extensions),
      install_requires=['numpy', 'sklearn', 'matplotlib'],
      cmdclass = {
          'build_ext': build_ext
      },
      zip_safe=False)

