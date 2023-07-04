# coniferest

[![PyPI version](https://badge.fury.io/py/coniferest.svg)](https://pypi.org/project/coniferest/)
[![Documentation Status](https://readthedocs.org/projects/coniferest/badge/?version=latest)](https://coniferest.readthedocs.io/en/latest/?badge=latest)
![Test Workflow](https://github.com/snad-space/coniferest/actions/workflows/test.yml/badge.svg)
![Build and publish wheels](https://github.com/snad-space/coniferest/actions/workflows/wheels.yml/badge.svg)


Package for active anomaly detection with isolation forests, made by [SNAD collaboration](https://snad.space/).

It includes:
* `IsolationForest` - reimplementation of scikit-learn's isolation forest with much better scoring performance due to the use of Cython and multi-threading (the latter is not currently available on macOS).
* `AADForest` - reimplementation of Active Anomaly detection algorithm with isolation forests from Shubhomoy Das' [`ad_examples` package](https://github.com/shubhomoydas/ad_examples) with better performance, much less code and more flexible dependencies.
* `PineForest` - our own active learning model based on the idea of tree filtering.

Install the package with `pip install coniferest`.

See the documentation for the [**Tutorial**](https://coniferest.readthedocs.io/en/latest/tutorial.html).
