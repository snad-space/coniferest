import numpy as np


__all__ = ['average_path_length']


def _average_path_length(n):
    """
    Average path length formula.
    """
    # Thank you Matwey.
    return 2.0 * (np.log(n) + np.euler_gamma + 1 / (2 * n) - 1 / (12 * n ** 2) - 1.0)


def average_path_length(n):
    """
    Average path length computation.

    Parameters
    ----------
    n
        Either array of tree depths to computer average path length of
        or one tree depth scalar.

    Returns
    -------
    Average path length.
    """
    if np.isscalar(n):
        if n <= 1:
            apl = 0.0
        else:
            apl = _average_path_length(n)
    else:
        n = np.asarray(n)
        apl = np.zeros_like(n)
        apl[n > 1] = _average_path_length(n[n > 1])

    return apl
