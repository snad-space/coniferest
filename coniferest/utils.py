import numpy as np


def _average_path_length(n):
    return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n


def average_path_length(n):

    if np.isscalar(n):
        if n <= 1:
            apl = 0
        elif n == 2:
            apl = 1
        else:
            apl = _average_path_length(n)
    else:
        n = np.asarray(n)
        apl = np.zeros_like(n)
        apl[n > 1] = _average_path_length(n[n > 1])
        apl[n == 2] = 1

    return apl