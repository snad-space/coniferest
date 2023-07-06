# coniferest

[![PyPI version](https://badge.fury.io/py/coniferest.svg)](https://pypi.org/project/coniferest/)
[![Documentation Status](https://readthedocs.org/projects/coniferest/badge/?version=latest)](https://coniferest.readthedocs.io/en/latest/?badge=latest)
![Test Workflow](https://github.com/snad-space/coniferest/actions/workflows/test.yml/badge.svg)
![Build and publish wheels](https://github.com/snad-space/coniferest/actions/workflows/wheels.yml/badge.svg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/snad-space/coniferest/master.svg)](https://results.pre-commit.ci/latest/github/snad-space/coniferest/master)


Package for active anomaly detection with isolation forests, made by [SNAD collaboration](https://snad.space/).

It includes:
* `IsolationForest` - reimplementation of scikit-learn's isolation forest with much better scoring performance due to the use of Cython and multi-threading (the latter is not currently available on macOS).
* `AADForest` - reimplementation of Active Anomaly detection algorithm with isolation forests from Shubhomoy Das' [`ad_examples` package](https://github.com/shubhomoydas/ad_examples) with better performance, much less code and more flexible dependencies.
* `PineForest` - our own active learning model based on the idea of tree filtering.

Install the package with `pip install coniferest`.

See the documentation for the [**Tutorial**](https://coniferest.readthedocs.io/en/latest/tutorial.html).


### Installation

The project is using [Cython](https://cython.org/) for performance and requires compilation.
However, binary wheels are available for Linux, macOS and Windows, so you can install the package with `pip install coniferest` on these platforms with no build-time dependencies.
Currently multithreading is not available in macOS ARM wheels, but you can install the package from the source to enable it, see instructions below.

If your specific platform is not supported, or you need a development version, you can install the package from the source.
To do so, clone the repository and run `pip install .` in the root directory.

Note, that we are using OpenMP for multi-threading, which is not available on macOS with the Apple LLVM Clang compiler.
You still can install the package with Apple LLVM, but it will be single-threaded.
Alternatively, you can install the package with Clang from Homebrew (`brew install llvm libomp`) or GCC (`brew install gcc`), which will enable multi-threading.
In this case you will need to set environment variables `CC=gcc-12` (or whatever version you have installed) or `CC=$(brew --preifx llvm)/bin/clang` and `CONIFEREST_FORCE_OPENMP_ON_MACOS=1`.


### Development

This project makes use of [pre-commit](https://pre-commit.com/) hooks, you can install them with `pre-commit install`.
[Pre-commit CI](https://results.pre-commit.ci/repo/github/390823585) is used for continuous integration of the hooks, they are applied to every pull request, and CI is responsible for auto-updating the hooks.
