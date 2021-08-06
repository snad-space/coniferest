import numpy as np
from .utils import average_path_length
from .calc_mean_paths import calc_mean_paths  # noqa


class ForestEvaluator:
    selector_dtype = np.dtype([('feature', np.int32), ('left', np.int32), ('value', np.double), ('right', np.int32)])

    def __init__(self, samples, selectors, indices):
        """
        Base class for the forest evaluators. Does the trivial job:
        * runs calc_mean_paths written in cython,
        * helps to combine several trees' representations into two arrays.

        Parameters
        ----------
        samples
            Number of samples in every tree.

        selectors
            Array with all the nodes of all the trees.

        indices
            Indices of starting nodes of every tree.
        """
        self.samples = samples

        self.selectors = selectors
        self.indices = indices

    @classmethod
    def combine_selectors(cls, selectors_list):
        """
        Combine several node arrays into one array of nodes and one array of
        start indices.

        Parameters
        ----------
        selectors_list
            List of node arrays to combine.

        Returns
        -------
        Pair of two arrays: node array and array of starting indices.
        """
        lens = [len(sels) for sels in selectors_list]
        full_len = sum(lens)

        selectors = np.empty((full_len,), dtype=cls.selector_dtype)

        indices = np.empty((len(selectors_list) + 1,), dtype=np.int64)
        indices[0] = 0
        indices[1:] = np.add.accumulate(lens)

        for i in range(len(selectors_list)):
            selectors[indices[i]:indices[i + 1]] = selectors_list[i]

        return selectors, indices

    def score_samples(self, x):
        """
        Perform the computations.

        Parameters
        ----------
        x
            Features to calculate scores of.

        Returns
        -------
        Array of scores.
        """
        return -2 ** (- calc_mean_paths(self.selectors, self.indices, x) / average_path_length(self.samples))