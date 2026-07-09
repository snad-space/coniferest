import numpy as np

from ._core import average_path_length as _average_path_length  # noqa

__all__ = ["average_path_length"]


def average_path_length(n):
    """
    Average path length computation.

    The formula itself is implemented in the Rust extension and is shared
    with the tree builder.

    Parameters
    ----------
    n
        Either array of tree depths to computer average path length of
        or one tree depth scalar.

    Returns
    -------
    Average path length.
    """
    is_scalar = np.isscalar(n)
    n = np.asarray(n)
    result = _average_path_length(n)
    if is_scalar:
        return result.item()
    return result
