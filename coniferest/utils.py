import numpy as np


def average_path_length(n):
    n = np.asarray(n)

    apl = 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n
    apl[n <= 1] = 0
    apl[n == 2] = 1

    return apl
