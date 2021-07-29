import numpy as np


class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


class MalanchevDataset(Dataset):
    def __init__(self, inliers=2**16, outliers=2**4, seed=0):
        self.inliers = inliers
        self.outliers = outliers
        
        rng = np.random.default_rng(0)
        self.rng = rng

        x = np.concatenate([self._generate_inliers(inliers, rng),
                            self._generate_outliers(outliers, rng, [1, 1]),
                            self._generate_outliers(outliers, rng, [0, 1]),
                            self._generate_outliers(outliers, rng, [1, 0])])

        x_labels = np.concatenate([np.ones(inliers + outliers), -np.ones(2 * outliers)])

        super(MalanchevDataset, self).__init__(data=x, labels=x_labels)
        
    @staticmethod
    def _generate_inliers(n, rng):
        return rng.uniform([0, 0], [0.5, 0.5], (n, 2))

    @staticmethod
    def _generate_outliers(n, rng, loc=None):
        loc = loc or [1, 1]
        return rng.normal(loc, 0.1, (n, 2))