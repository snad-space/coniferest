import numpy as np


class Dataset:
    def __init__(self, data, labels):
        """
        * Dataset is an o-by-f array, where o is objects and f is features.

        * Labels is a one-dimentional o array with three types of labels:
            * -1 for anomalies,
            * 0 for unknowns,
            * 1 for regular data.
        """
        self.data = data
        self.labels = labels


class MalanchevDataset(Dataset):
    def __init__(self, inliers=2**10, outliers=2**5, regions=None, seed=0):
        self.inliers = inliers
        self.outliers = outliers
        regions = regions or np.array([1, 1, -1])
        self.regions = regions
        
        rng = np.random.default_rng(0)
        self.rng = rng

        x = np.concatenate([self._generate_inliers(inliers, rng),
                            self._generate_outliers(outliers, rng, [1, 1]),
                            self._generate_outliers(outliers, rng, [0, 1]),
                            self._generate_outliers(outliers, rng, [1, 0])])

        x_labels = np.concatenate([np.ones(inliers),
                                   self.regions[0] * np.ones(outliers),
                                   self.regions[1] * np.ones(outliers),
                                   self.regions[2] * np.ones(outliers)])

        super(MalanchevDataset, self).__init__(data=x, labels=x_labels)
        
    @staticmethod
    def _generate_inliers(n, rng):
        return rng.uniform([0, 0], [0.5, 0.5], (n, 2))

    @staticmethod
    def _generate_outliers(n, rng, loc=None):
        loc = loc or [1, 1]
        return rng.normal(loc, 0.1, (n, 2))