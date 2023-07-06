Isolation forest
================

Introduction
------------

Isolation forest is an unsupervised learning algorithm for anomaly detection that works on the principle of isolating anomalies (`Liu et al. 2008 <https://doi.org/10.1109/ICDM.2008.17>`_).
It is based on stochastic decision trees, where partitions are created by first randomly selecting a feature and then selecting a random split value between the maximum and minimum value of the selected feature.
The number of splits required to isolate a sample is then used as a measure of the sample's abnormality.
The idea is that anomalies are easier to isolate than normal data points, and therefore require fewer splits.
The algorithm is implemented in the :class:`IsolationForest <coniferest.isoforest.IsolationForest>` class.

Implementation
--------------

The :class:`IsolationForest <coniferest.isoforest.IsolationForest>` class is a reimplementation of scikit-learn's `IsolationForest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>`_, designed for enhanced performance.
It employs the low-level trees and tree builders from ``scikit-learn``, but the score evaluation has been re-implemented using Cython for superior efficiency.

While training is typically the most time-consuming aspect of most machine learning algorithms, for the Isolation Forest and active learning algorithms based upon it, scoring is the more demanding process.
This is because scoring, which is essential for anomaly detection, requires the use of the entire dataset, whereas training often only utilizes a small data subset.

Despite the fact that the ``.score_samples()`` method is not parallelized, thus creating a performance bottleneck in the ``scikit-learn`` implementation, we have addressed this in our own version.
We have parallelized the ``.score_samples()`` method by leveraging Cython's support of OpenMP.
However, please note that achieving parallelization of ``.score_samples()`` on macOS still poses significant challenges, as discussed `here <https://github.com/snad-space/coniferest/pull/15>`_.

See `API documentation <coniferest.isoforest.IsolationForest>` for usage details, and see examples below for basic usage and performance comparison with scikit-learn.

Examples
--------

~1 million of samples, two feature.
We go almost three times faster with ``coniferest`` when running in a single thread (Apple M1Pro, macOS, single-thread):

.. code-block:: python

    import coniferest.isoforest
    import sklearn.ensemble
    from coniferest.datasets import non_anomalous_outliers

    # ~1e6 data points, 2 features
    data, _ = non_anomalous_outliers(
        inliers=1 << 20,
        outliers=1 << 6,
    )

    # %%timeit
    skforest = sklearn.ensemble.IsolationForest(
        n_estimators=1024,
        max_samples=1024,
        random_state=0,
        n_jobs=1,
    )
    skforest.fit(data)
    skscores = skforest.score_samples(data)
    # CPU times: user 2min 47s, sys: 22.7 s, total: 3min 10s
    # Wall time: 3min 10s

    # %%time
    isoforest = coniferest.isoforest.IsolationForest(
        n_trees=1024,
        n_subsamples=1024,
        random_seed=0,
    )
    isoforest.fit(data)
    scores = isoforest.score_samples(data)
    # CPU times: user 1min 8s, sys: 310 ms, total: 1min 9s
    # Wall time: 1min 9s

~26M samples, 42 features dataset.
We go ~100 times faster with ``coniferest`` when doing multithreading (2x 16 core Xeon 4216 with hyperthreading, Linux, multithreading):

.. code-block:: python

    from pathlib import Path

    import coniferest.isoforest
    import numpy as np
    import sklearn.ensemble
    from coniferest.datasets import non_anomalous_outliers

    # Dataset from Zenodo: https://zenodo.org/record/6998913
    data = np.concatenate([
        np.memmap(
            f,
            dtype=np.float32,
            mode='r',
        ).reshape(-1, 42)
        for f in sorted(Path('features/').glob('feature_*.dat'))
    ])
    print(data.shape)
    # (26537671, 42)

    # %%time
    skforest = sklearn.ensemble.IsolationForest(
        n_estimators=1024,
        max_samples=1024,
        random_state=0,
        n_jobs=-1,
    )
    skforest.fit(data)
    skscores = skforest.score_samples(data)
    # CPU times: user 1h 37min 19s, sys: 1h 10min 34s, total: 2h 47min 54s
    # Wall time: 2h 8min 31s

    # %%time
    isoforest = coniferest.isoforest.IsolationForest(
        n_trees=1024,
        n_subsamples=1024,
        random_seed=0,
    )
    isoforest.fit(data)
    scores = isoforest.score_samples(data)
    # CPU times: user 1h 15min 59s, sys: 8.43 s, total: 1h 16min 8s
    # Wall time: 1min 18s
