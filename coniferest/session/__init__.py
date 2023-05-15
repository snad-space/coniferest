from typing import Callable, Dict

import numpy as np

from coniferest.coniferest import Coniferest
from coniferest.pineforest import PineForest

from .callback import prompt_decision_callback
from ..label import Label


class Session:
    """
    Active anomaly detection session

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features), dtype is number
        2-D array of data points
    metadata : array-like, shape (n_samples,), dtype is any
        1-D array of metadata for each data point
    decision_callback : callable, optional
        Function to be called when expert decision is required
        TODO: signature
    on_refit_callbacks : list of callable or callable, optional
        Functions to be called when model is refitted
        TODO: signature
    on_decision_callbacks : list of callable or callable, optional
        Functions to be called when expert decision is made
        TODO: signature
    known_labels : dict, optional
        Dictionary of known anomaly labels, keys are data/metadata indices,
        values are labels of type `Label`
    model : Coniferest, optional
        Anomaly detection model to use, default is `PineForest`

    Attributes
    ----------
    current : int
        Index of current anomaly candidate
    scores : array-like, shape (n_samples,)
        Current anomaly scores for all data points
    terminated : bool
        True if session is terminated
    known_labels : dict[int, Label]
        Current dictionary of known anomaly labels
    model : Coniferest
        Anomaly detection model used

    Examples
    --------

    >>> from coniferest.session import Session
    >>> data, metadata = ztf_dataset()
    >>> s = Session(data, metadata)
    >>> s.run()
    """
    @staticmethod
    def _validate_callbacks(callbacks):
        return all([isinstance(cb, Callable) for cb in callbacks])

    @staticmethod
    def _invoke_callbacks(callbacks, *args, **kwargs):
        for cb in callbacks:
            cb(*args, **kwargs)


    def __init__(self, data, metadata, decision_callback = prompt_decision_callback, *, on_refit_callbacks = [], on_decision_callbacks = [], known_labels: Dict[int, Label] = None, model: Coniferest = PineForest()):

        self._data     = np.atleast_2d(data)
        self._metadata = np.atleast_1d(metadata)

        if not isinstance(decision_callback, Callable):
            raise ValueError("decision_callback is not a callable")

        self._decision_cb = decision_callback

        if not isinstance(on_refit_callbacks, list):
            on_refit_callbacks = [on_refit_callbacks,]

        if not self._validate_callbacks(on_refit_callbacks):
            raise ValueError("on_refit_callbacks contains not callable object")

        self._on_refit_cb  = on_refit_callbacks

        if not isinstance(on_decision_callbacks, list):
            on_decision_callbacks = [on_decision_callbacks,]

        if not self._validate_callbacks(on_decision_callbacks):
            raise ValueError("on_decision_callbacks is not callable")

        self._on_decision_cb = on_decision_callbacks

        if known_labels is None:
            self._known_labels = {}
        else:
            self._known_labels = dict(known_labels)

        if not isinstance(model, Coniferest):
            raise ValueError("model is not a Coniferest object")

        self._model = model

        self._scores = None
        self._current = None
        self._terminated = False

    def run(self):
        """Evaluate interactive anomaly detection session"""

        if self._terminated:
            raise RuntimeError("Session is already terminated")

        self.model.fit(self._data)

        while not self._terminated:
            known_data = self._data[list(self._known_labels.keys())]
            known_labels = list(self._known_labels.values())
            self.model.fit_known(self._data, known_data, known_labels)

            self._invoke_callbacks(self._on_refit_cb, self)

            self._scores = self.model.score_samples(self._data)

            self._current = None
            for ind in np.argsort(self._scores):
                if ind not in self._known_labels:
                    self._current = ind
                    break

            if self._current is None:
                self.terminate()
                break

            decision = self._decision_cb(self._metadata[self._current], self._data[self._current], self)
            self._known_labels[self._current] = decision

            self._invoke_callbacks(self._on_decision_cb, self)

        return self

    def terminate(self):
        self._terminated = True

    @property
    def current(self):
        return self._current

    @property
    def scores(self):
        return self._scores

    @property
    def known_labels(self):
        return self._known_labels

    @property
    def model(self):
        return self._model

    @property
    def terminated(self):
        return self._terminated
