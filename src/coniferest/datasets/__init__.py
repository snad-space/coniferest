import numpy as np

from coniferest.label import Label

from .plasticc_gp import plasticc_gp
from .ztf_m31 import ztf_m31

__all__ = ["ztf_m31", "plasticc_gp", "single_outlier", "non_anomalous_outliers"]


class Dataset:
    def __init__(self, data, labels):
        """
        Dataset is an o-by-f array, where o is objects and f is features.
        """
        self.data = data
        self.labels = labels

    def to_data_metadata(self):
        return np.asarray(self.data), np.asarray(list(map(Label, self.labels)))


class SingleOutlierDataset(Dataset):
    def __init__(self, inliers=10_000, rng=0):
        rng = np.random.default_rng(rng)
        data_inliers = rng.normal(loc=0, scale=1, size=(inliers, 2))
        data_outlier = np.array([[1e6, -1e6]])
        data = np.vstack([data_inliers, data_outlier])
        labels = np.append(np.zeros(inliers), np.ones(1))
        super().__init__(data, labels)


def single_outlier(inliers=10_000, rng=0):
    return SingleOutlierDataset(inliers, rng).to_data_metadata()


class MalanchevDataset(Dataset):
    def __init__(self, inliers=1 << 10, outliers=1 << 5, regions=None, rng=0):
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
        if regions is None:
            regions = np.array([Label.R, Label.R, Label.A])
        self.regions = regions

        rng = np.random.default_rng(rng)
        self.rng = rng

        x = np.concatenate(
            [
                self._generate_inliers(inliers, rng),
                self._generate_outliers(outliers, rng, [1, 1]),
                self._generate_outliers(outliers, rng, [0, 1]),
                self._generate_outliers(outliers, rng, [1, 0]),
            ]
        )

        x_labels = np.concatenate(
            [
                np.ones(inliers),
                self.regions[0] * np.ones(outliers),
                self.regions[1] * np.ones(outliers),
                self.regions[2] * np.ones(outliers),
            ]
        )

        super(MalanchevDataset, self).__init__(data=x, labels=x_labels)

    @staticmethod
    def _generate_inliers(n, rng):
        return rng.uniform([0, 0], [0.5, 0.5], (n, 2))

    @staticmethod
    def _generate_outliers(n, rng, loc=None):
        loc = loc or [1, 1]
        return rng.normal(loc, 0.1, (n, 2))


def non_anomalous_outliers(inliers=1 << 10, outliers=1 << 5, regions=None, seed=0):
    return MalanchevDataset(inliers, outliers, regions, seed).to_data_metadata()


class DevNetDataset(Dataset):
    """Deviation Network paper datasets.

    This class constructor would download datasets from the Deviation Network GitHub:
    https://github.com/GuansongPang/deviation-network

    It requires pandas to be installed.

    Arguments
    ---------
    name : str
        Name of the dataset to download. See `.avialble_datasets`.

    Attributes
    ----------
    avialble_datasets : list[str]
        List of available datasets to download.
    """

    _dataset_filenames = {
        "donors": "KDD2014_donors_10feat_nomissing_normalised.csv",
        "census": "census-income-full-mixed-binarized.tar.xz",
        "fraud": "creditcardfraud_normalised.tar.xz",
        "celeba": "celeba_baldvsnonbald_normalised.csv",
        "backdoor": "UNSW_NB15_traintest_backdoor.tar.xz",
        "campaign": "bank-additional-full_normalised.csv",
        "thyroid": "annthyroid_21feat_normalised.csv",
    }
    _dataset_urls = {
        name: f"https://github.com/GuansongPang/deviation-network/raw/master/dataset/{filename}"
        for name, filename in _dataset_filenames.items()
    }
    avialble_datasets = list(_dataset_filenames.keys())

    def __init__(self, name: str):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required to load DevNet datasets, install it with `pip install pandas` or "
                "reinstall the package with `pip install coniferest[datasets]`"
            )

        if name not in self.avialble_datasets:
            raise ValueError(f"Dataset {name} is not available. Available datasets are: {self.avialble_datasets}")

        self.name = name
        
        df = pd.read_csv(self._dataset_urls[name])

        # Last column is for class, the rest are features
        data = df.iloc[:, :-1].to_numpy(dtype=float)

        # In the original data, the labels are 1 for anomalies and 0 for regular data
        # We need 1 for regular data and -1 for anomalies
        labels = 1 - 2 * df.iloc[:, -1].to_numpy(dtype=int)

        super().__init__(data, labels)


def dev_net_dataset(name: str):
    f"""Download and return metadata and data for the Deviation Network dataset.
   
    This class constructor would download datasets from the Deviation Network GitHub:
    https://github.com/GuansongPang/deviation-network

    It requires pandas to be installed.
    
    Avialable datasets are: {", ".join(DevNetDataset.avialble_datasets)}

    Arguments
    ---------
    name : str
        Name of the dataset to download. See `.avialble_datasets`. 
    
    Returns
    -------
    data : array-like, shape (n_samples, n_features)
        2-D array of data points
    labels : array-like, shape (n_samples,)
        1-D array of `Label` objects for each data point
    """
    return DevNetDataset(name).to_data_metadata()
