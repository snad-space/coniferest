from importlib.resources import open_binary

import numpy as np

from coniferest import datasets


def ztf_m31():
    """
    Load ZTF DR3 M31 light curve feature dataset.

    Adopted from Malanchev et al. 2021:
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract
    https://zenodo.org/record/4318700

    Returns
    -------
    data : 2-D numpy.ndarray of float32
        2-D array of light curve features.
    metadata : 1-D numpy.ndarray of uint64
        ZTF DR object IDs (OIDs) of the objects.
    """
    with open_binary(datasets, "ztf_m31.npz") as fh:
        data = np.load(fh)
        return data["feature"], data["oid"]
