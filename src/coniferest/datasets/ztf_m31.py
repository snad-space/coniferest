import numpy as np


IMPORT_ERROR_MESSAGE = (
    "`datasets' library is required to load ZTF M31 features dataset, install it with `pip install datasets` "
    "or reinstall the package with `pip install coniferest[datasets]`"
)


def ztf_m31(**kwargs):
    """
    Load ZTF DR3 M31 light curve feature dataset.

    The dataset is hosted on Hugging Face, `datasets' package is required
    to load it.

    Adopted from Malanchev et al. 2021:
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract
    https://zenodo.org/record/4318700

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments passed to `datasets.load_dataset`.

    Returns
    -------
    data : 2-D numpy.ndarray of float32
        2-D array of light curve features.
    metadata : 1-D numpy.ndarray of uint64
        ZTF DR object IDs (OIDs) of the objects.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(IMPORT_ERROR_MESSAGE)

    dataset = load_dataset("snad-space/ztf-dr3-m31-features", **kwargs)
    df = dataset["train"].to_pandas()
    metadata = df["oid"].to_numpy(dtype=np.uint64)
    data = df.drop(columns=["oid"]).to_numpy(dtype=np.float32)
    return data, metadata
