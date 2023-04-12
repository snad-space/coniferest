import numpy as np
from coniferest.label import Label

from .ztf_m31 import ztf_m31

__all__ = ["ztf_m31",]


class Dataset:
    def __init__(self, data, labels):
        """
        Dataset is an o-by-f array, where o is objects and f is features.
        """
        self.data = data
        self.labels = labels


class MalanchevDataset(Dataset):
    def __init__(self, inliers=2**10, outliers=2**5, regions=None, seed=0):
        """
        A simple dataset for testing the anomaly detection algorithms. It
        constits of one portion of regular data of `inliers` capacity, and
        three portions of outlier data of `outliers` capacity each. Every
        outlier portion maybe selected either as regular or anomalous. Example:

        ```
                MalanchevDataset(inliers=100, outliers=10, regions=(R,R,A))
             ┌───────────────────────────────────────────────────────────────┐
         1.12┤  .           .                                        .       │
             │        .  .         .                        .  .    .  .     │
         0.88┤.   . .        .                                      .      . │
             │   .                                             .             │
             │                                                               │
         0.64┤                                                               │
             │                .     .                                        │
             │         ... ..  .... ... .....                                │
          0.4┤        ....  .. .. .. .    .                                  │
             │          . ...     ..   ... .                                 │
         0.17┤        .  .  ...  ..... .  ..                   *             │
             │         .    .... .  . .. . .                 *  **           │
             │         .   .      . . . ...                 *     *         *│
        -0.07┤                                                       *       │
             │                                                               │
        -0.31┤                                               *               │
             └┬──────────────┬───────────────┬───────────────┬──────────────┬┘
             -0.2           0.16            0.53            0.89          1.26

        ```

        Here we have a plot of 100 inliers, 20 regular outliers (all plotted as
        dots) and 10 anomalous outliers (plotted as stars).
        """
        self.inliers = inliers
        self.outliers = outliers
        regions = regions or np.array([Label.R, Label.R, Label.A])
        self.regions = regions

        rng = np.random.default_rng(seed)
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
