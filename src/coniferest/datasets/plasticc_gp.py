from __future__ import annotations  # "|" syntax is not available in Python 3.9


IMPORT_ERROR_MESSAGE = (
    "`datasets' library is required to load PLAsTiCC datasets, install it with `pip install datasets` "
    "or reinstall the package with `pip install coniferest[datasets]`"
)


def plasticc_gp(**kwargs):
    """
    Load PLAsTiCC dataset of GP-approximated light curves.

    It is a subset of the PLAsTiCC dataset, light curves are approximated
    with vector Gaussian processes (Kornilov et al., 2023). Features
    are these GP approximations and GP parameters, all normalized.

    The dataset is hosted on Hugging Face, `datasets' package is required
    to load it.

    Adopted from Ishida et al. 2021:
    https://ui.adsabs.harvard.edu/abs/2021A%26A...650A.195I/abstract

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments passed to `datasets.load_dataset`.

    Returns
    -------
    data : 2-D numpy.ndarray of float32
        2-D array of features.
    metadata : 1-D numpy.ndarray of bool
        1-D array of anomaly labels, True for anomalies.
    """

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(IMPORT_ERROR_MESSAGE)

    dataset = load_dataset("snad-space/plasticc-gp", **kwargs)
    df = dataset["train"].to_pandas()
    features_columns = [name for name in df.columns if name.startswith("feature")]
    data = df[features_columns].to_numpy()
    # The source file uses 1 for anomaly and 0 for nominals,
    # while we need Label.ANOMALY == -1 and Label.REGULAR == 1.
    metadata = 1 - 2 * df["answer"].to_numpy(dtype=int)
    return data, metadata
