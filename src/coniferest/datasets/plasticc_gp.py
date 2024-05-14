from importlib.resources import open_binary

import pandas as pd

from coniferest import datasets


def plasticc_gp():
    """
    Load PLAsTiCC dataset of GP-approximated light curves.

    It is a subset of the PLAsTiCC dataset, light curves are approximated
    with vector Gaussian processes (Kornilov et al., 2023). Features
    are these GP approximations and GP parameters, all normalized.

    Adopted from Ishida et al. 2021:
    https://ui.adsabs.harvard.edu/abs/2021A%26A...650A.195I/abstract

    Returns
    -------
    data : 2-D numpy.ndarray of float32
        2-D array of features.
    metadata : 1-D numpy.ndarray of bool
        1-D array of anomaly labels, True for anomalies.
    """
    with open_binary(datasets, "plasticc.parquet") as fh:
        df = pd.read_parquet(fh)
        feature_columns = [name for name in df.columns if name.startswith("feature")]
        return df[feature_columns].to_numpy(), df["answer"].to_numpy(dtype=bool)
