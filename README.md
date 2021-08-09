# Coniferests

Trying to make a slightly better isolation forest for anomaly detection.
At the moment there are two forests (subclasses of the of the coniferest base class):
isolation forest and pine forest.

## Isolation forest
[This](https://github.com/snad-space/coniferest/blob/master/coniferest/isoforest.py)
is the reimplementation of scikit-learn's 
[isolation forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
The low-level trees and builders are those of original isoforest. What is basically
reimplemented is the score evaluation to provide better efficiency. Compare runs (4-cores Intel Core i5-6300U):
```python
import sklearn.ensemble
import coniferest.isoforest
from coniferest.datasets import MalanchevDataset

# 1e6 data points
dataset = MalanchevDataset(inliers=2**20, outliers=2**6)

# %%time
isoforest = coniferest.isoforest.IsolationForest(n_subsamples=1024)
isoforest.fit(dataset.data)
scores = isoforest.score_samples(dataset.data)
# CPU times: user 16.4 s, sys: 26.1 ms, total: 16.4 s
# Wall time: 5.03 s

# %%time
skforest = sklearn.ensemble.IsolationForest(max_samples=1024)
skforest.fit(dataset.data)
skscores = skforest.score_samples(dataset.data)
# CPU times: user 32.3 s, sys: 4.48 s, total: 36.8 s
# Wall time: 36.8 s
```
And that's not the largest speedup. The more data we analyze, the more cores we have, the more trees we build -- the larger will be the speedup.
At one setup (analyzing 30M objects with 100-dimensional features on 80-core computer) I have seeen a speedup rate from 24 hours to 1 minute.

The main object of optimization is score evaluation. So if you'd like to test it without using the isolation forest reimplementation, you may use
just the evaluator as follows:
```python
# %%time
from coniferest.sklearn.isoforest import IsolationForestEvaluator

isoforest = sklearn.ensemble.IsolationForest(max_samples=1024)
isoforest.fit(dataset.data)
evaluator = IsolationForestEvaluator(isoforest)
scores = evaluator.score_samples(dataset.data)
# CPU times: user 17.1 s, sys: 13.9 ms, total: 17.2 s
# Wall time: 6.32 s
```

## Pine forest
Pine forest is an attempt to make isolation forest capable of applying a bit of prior information. TBD: to be described.
