import sys
sys.path.append('..')

import numpy as np
import plotext as plt
from coniferest.datasets import MalanchevDataset, Label
from coniferest.pineforest import PineForest


def plotme(data, anomaly_index=None, title=None):
    plt.clp()
    plt.colorless()
    plt.plotsize(70, 20)
    plt.scatter(*dataset.data.T, marker='.', color='none')
    if anomaly_index is not None:
        plt.scatter(*dataset.data[anomaly_index, :].T, marker='*', color='none')

    if title is not None:
        plt.title(title)

    plt.show()


# Take the data
dataset = MalanchevDataset(inliers=100, outliers=10)

# Plot the data
plotme(dataset.data, title='Plain data')


# Plot the data with classic isolation forest anomaly detection (i.e.
# PineForest without priors)
pineforest = PineForest(n_subsamples=16)
pineforest.fit(dataset.data)
scores = pineforest.score_samples(dataset.data)
plotme(dataset.data, np.argsort(scores)[:10], title='PineForest without priors')


# Add some priors
priors = np.array([[0.0, 1.0],
                   [1.0, 1.0],
                   [1.0, 0.0]])

prior_labels = np.array([Label.R, Label.R, Label.A])


# Plot the detected anomaly data based on some known prior information
pineforest.fit_known(dataset.data, priors, prior_labels)
scores = pineforest.score_samples(dataset.data)
plotme(dataset.data, np.argsort(scores)[:10], title='PineForest with 3 priors')
