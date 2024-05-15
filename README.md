# coniferest

[![PyPI version](https://badge.fury.io/py/coniferest.svg)](https://pypi.org/project/coniferest/)
[![Documentation Status](https://readthedocs.org/projects/coniferest/badge/?version=latest)](https://coniferest.readthedocs.io/en/latest/?badge=latest)
![Test Workflow](https://github.com/snad-space/coniferest/actions/workflows/test.yml/badge.svg)
![Build and publish wheels](https://github.com/snad-space/coniferest/actions/workflows/wheels.yml/badge.svg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/snad-space/coniferest/master.svg)](https://results.pre-commit.ci/latest/github/snad-space/coniferest/master)

Package for active anomaly detection with isolation forests, made by [SNAD collaboration](https://snad.space/).

It includes:

* `IsolationForest` - reimplementation of scikit-learn's isolation forest with much better scoring performance due to
  the use of Cython and multi-threading (the latter is not currently available on macOS).
* `AADForest` - reimplementation of Active Anomaly detection algorithm with isolation forests from Shubhomoy
  Das' [`ad_examples` package](https://github.com/shubhomoydas/ad_examples) with better performance, much less code and
  more flexible dependencies.
* `PineForest` - our own active learning model based on the idea of tree filtering.

Install the package with `pip install coniferest`.

See the documentation for the [**Tutorial**](https://coniferest.readthedocs.io/en/latest/tutorial.html).

### Installation

The project is using [Rust programming language](https://rust-lang.org/) for performance and requires compilation.
However, binary wheels are available for Linux, macOS and Windows, so you can install the package
with `pip install coniferest[datasets]` on these platforms with no build-time dependencies.

If your specific platform is not supported, or you need a development version, you can install the package from the
source.
To do so, install Rust toolchain, with [`rustup`](https://rustup.rs) or your favourite package manager, clone the
repository and
run `pip install .[datasets]` in the root directory.
Add flag `-e` to the command to install the package in editable mode.

### Development

You can install the package in editable mode with `pip install -e .[datasets,dev]` to install the development
dependencies.

#### Linters and formatters

This project makes use of [pre-commit](https://pre-commit.com/) hooks, you can install them with `pre-commit install`.
[Pre-commit CI](https://results.pre-commit.ci/repo/github/390823585) is used for continuous integration of the hooks,
they are applied to every pull request, and CI is responsible for auto-updating the hooks.

#### Testing and benchmarking

We use [tox](https://tox.wiki/en/latest/) to build and test the package in isolated environments with different Python
versions.
To run tests locally, install tox with `pip install tox` and run `tox` in the root directory.
We configure `tox` to skip long tests.

The project uses [pytest](https://docs.pytest.org/) as a testing framework.
Tests are located in the `tests` directory, and can be run with `pytest tests` in the root directory.
By default, all tests are run, but you can select specific tests with `-k` option,
e.g. `pytest tests -k test_onnx.test_onnx_aadforest`.
You can also deselect a specific group of tests with `-m` option, e.g. `pytest tests -m'not long'`, see `pyproject.toml`
for the list of markers.

We use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) for local benchmarking
and [Codspeed](https://codspeed.io) for benchmarking in CI.
You can run benchmarks with `pytest tests --benchmark-enable -m benchmark` in the root directory.
You can adjust the minimum number of iterations with `--benchmark-min-rounds` and maximum execution time per benchmark
with `--benchmark-max-time` (note that the latter can be exceeded if the minimum number of rounds is not reached).
See `pyproject.toml` for the default benchmarking options.
You can make a snapshot the current benchmark result with `--benchmark-save=NAME` or with `--benchmark-autosave`, and
compare benchmarks with `pytest-benchmark compare` command.
