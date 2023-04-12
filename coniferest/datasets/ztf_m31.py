import numpy as np
from pkgutil import get_data
from io import BytesIO

def ztf_m31():
    # FIXME: find a way to resolve filename only
    data = get_data("coniferest", "datasets/ztf_m31.npz")
    data = np.load(BytesIO(data))

    return data["feature"], data["oid"]
