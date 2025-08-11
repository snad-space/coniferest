<img src="./images/CF_logo_green_sign_universal.svg" width=256>

<a href="https://ascl.net/2507.009"><img src="https://img.shields.io/badge/ascl-2507.009-blue.svg?colorB=262255" alt="ascl:2507.009" /></a>
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

[![asciicast](https://asciinema.org/a/686647.svg)](https://asciinema.org/a/686647?autoplay=1)

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

You can install the package in editable mode with `pip install -e .[dev]` to install the development dependencies.

#### Linters and formatters

This project makes use of [pre-commit](https://pre-commit.com/) hooks, you can install them with `pre-commit install`.
[Pre-commit CI](https://results.pre-commit.ci/repo/github/390823585) is used for continuous integration of the hooks, they are applied to every pull request, and CI is responsible for auto-updating the hooks.

#### Testing and benchmarking

We use [tox](https://tox.wiki/en/latest/) to build and test the package in isolated environments with different Python versions.
To run tests locally, install tox with `pip install tox` and run `tox` in the root directory.
We configure `tox` to skip long tests.

The project uses [pytest](https://docs.pytest.org/) as a testing framework.
Tests are located in the `tests` directory, and can be run with `pytest tests` in the root directory.
By default, all tests are run, but you can select specific tests with `-k` option, e.g. `pytest tests -k test_onnx.test_onnx_aadforest`.
You can also deselect a specific group of tests with `-m` option, e.g. `pytest tests -m'not long'`, see `pyproject.toml` for the list of markers.

We use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) for benchmarking.
You can run benchmarks with `pytest tests --benchmark-enable -m benchmark` in the root directory.
Most of the benchmarks have `n_jobs` fixture set to 1 by default, you can change it with `--n_jobs` option.
You can adjust the minimum number of iterations with `--benchmark-min-rounds` and maximum execution time per benchmark with `--benchmark-max-time` (note that the latter can be exceeded if the minimum number of rounds is not reached).
See `pyproject.toml` for the default benchmarking options.
You can make a snapshot the current benchmark result with `--benchmark-save=NAME` or with `--benchmark-autosave`, and compare benchmarks with `pytest-benchmark compare` command.

## Citation

If you found this project useful for your research, please
cite [Kornilov, Korolev, Malanchev, et al., 2025](https://doi.org/10.1016/j.ascom.2025.100960)

```bibtex
@article{Kornilov2025,
	title = {Coniferest: A complete active anomaly detection framework},
	journal = {Astronomy and Computing},
	volume = {52},
	pages = {100960},
	year = {2025},
	issn = {2213-1337},
	doi = {10.1016/j.ascom.2025.100960},
	url = {https://www.sciencedirect.com/science/article/pii/S2213133725000332},
	author = {M.V. Kornilov and V.S. Korolev and K.L. Malanchev and A.D. Lavrukhina and E. Russeil and T.A. Semenikhin and E. Gangler and E.E.O. Ishida and M.V. Pruzhinskaya and A.A. Volnova and S. Sreejith},
}
```

Additionally, you may also cite [the ASCL record](https://ascl.net/2507.009) for the package
```bibtex
@software{2025ascl.soft07009K,
       author = {{Korolev}, Vladimir and {Kornolov}, Matwey and {Malanchev}, Konstantin and {SNAD Team}},
        title = "{Coniferest: Python package for active anomaly detection}",
 howpublished = {Astrophysics Source Code Library, record ascl:2507.009},
         year = 2025,
        month = jul,
          eid = {ascl:2507.009},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ascl.soft07009K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
