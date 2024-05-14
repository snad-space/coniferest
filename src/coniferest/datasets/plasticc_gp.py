from importlib.resources import open_binary

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
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Pandas and pyarrow are required to load PLAsTiCC datasets, install them with `pip install pandas pyarrow` or reinstall the package with `pip install coniferest[datasets]`")

    with open_binary(datasets, "plasticc.parquet") as fh:
        try:
            df = pd.read_parquet(fh)
        except ImportError:
            raise ImportError("PyArrow is required to load PLAsTiCC datasets, install it with `pip install pyarrow` or reinstall the package with `pip install coniferest[datasets]`")
        feature_columns = [name for name in df.columns if name.startswith("feature")]
        return df[feature_columns].to_numpy(), df["answer"].to_numpy(dtype=bool)
