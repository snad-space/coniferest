from typing import Callable, Dict, Optional

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
        Function to be called when expert decision is required, it must return
        `Label` object with the decision and may terminate the session via
        `Session.terminate()`. Default is `prompt_decision_callback`
        Signature: '(metadata, data, session) -> Label', where metadata is
        metadata of the object to be labeled, data is data of the object to be
        labeled, session is this session instance.
    on_refit_callbacks : list of callable, or callable, or None, optional
        Functions to be called when model is refitted (before
        "decision_callback"), default is empty list. This function may call
        `Session.terminate()`.
        Signature: '(session) -> None', where session is this session instance.
    on_decision_callbacks : list of callable, or callable, or None, optional
        Functions to be called when expert decision is made (after
        "decision_callback"), default is empty list. This function may call
        `Session.terminate()`.
        Signature: '(metadata, data, session) -> None', where metadata is
        metadata of the object has just been labeled, data is data of this
        object, session is this session instance.
    known_labels : dict, optional
        Dictionary of known anomaly labels, keys are data/metadata indices,
        values are labels of type `Label`. Default is empty dictionary.
    model : Coniferest or None, optional
        Anomaly detection model to use, default is `PineForest()`.

    Attributes
    ----------
    current : int
        Index of the last anomaly candidate
    last_decision : Label or None
        Label of the last anomaly candidate or None if no decision was made
    scores : array-like, shape (n_samples,)
        Current anomaly scores for all data points
    terminated : bool
        True if session is terminated
    known_labels : dict[int, Label]
        Current dictionary of known anomaly labels
    known_anomalies : array-like
        Array of indices of known anomalies
    known_regulars : array-like
        Array of indices of known regular objects
    known_unknowns : array-like
        Array of indices of known objects marked with `Label::UNKNOWN`
    trajectory : array-like
        Array of indices of objects that were labeled during the session.
        If no `known_labels` were provided, it is equal to `list(known_labels.keys())`.
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
    def _prepare_callbacks(input_argument):
        if input_argument is None:
            callbacks = []
        elif isinstance(input_argument, list):
            callbacks = input_argument
        else:
            callbacks = [input_argument, ]

        if not all([isinstance(cb, Callable) for cb in callbacks]):
            raise ValueError("At least one of the callbacks is not callable")
        return callbacks

    @staticmethod
    def _invoke_callbacks(callbacks, *args, **kwargs):
        for cb in callbacks:
            cb(*args, **kwargs)


    def __init__(self, data, metadata, decision_callback = prompt_decision_callback, *, on_refit_callbacks = None, on_decision_callbacks = None, known_labels: Dict[int, Label] = None, model: Coniferest = None):

        self._data     = np.atleast_2d(data)
        self._metadata = np.atleast_1d(metadata)

        if not isinstance(decision_callback, Callable):
            raise ValueError("decision_callback is not a callable")
        self._decision_cb = decision_callback

        try:
            self._on_refit_cb = self._prepare_callbacks(on_refit_callbacks)
        except ValueError:
            raise ValueError("on_refit_callbacks contains not callable object")

        try:
            self._on_decision_cb = self._prepare_callbacks(on_decision_callbacks)
        except ValueError:
            raise ValueError("on_decision_callbacks contains not callable object")

        if known_labels is None:
            self._known_labels = {}
        else:
            self._known_labels = dict(known_labels)

        if not isinstance(model, Coniferest):
            raise ValueError("model is not a Coniferest object")

        if model is None:
            model = PineForest()
        self._model = model

        self._scores = None
        self._current = None
        self._terminated = False
        self._trajectory = []

    def run(self) -> 'Session':
        """Evaluate interactive anomaly detection session"""

        if self._terminated:
            raise RuntimeError("Session is already terminated")

        self.model.fit(self._data)

        while not self._terminated:
            known_data = self._data[list(self._known_labels.keys())]
            known_labels = np.fromiter(self._known_labels.values(), dtype=int, count=len(self._known_labels))
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
            self._trajectory.append(self._current)

            self._invoke_callbacks(self._on_decision_cb, self._metadata[self._current], self._data[self._current], self)

        return self

    def terminate(self) -> None:
        self._terminated = True

    @property
    def current(self) -> Optional[int]:
        """Index of the last anomaly candidate or None if no decision was made"""
        return np.array(self._current) if self._current is not None else None

    @property
    def last_decision(self) -> Optional[Label]:
        """Label of the last anomaly candidate or None if no decision was made"""
        return self._known_labels.get(self._current, None)

    @property
    def scores(self) -> Optional[np.ndarray]:
        """Current anomaly scores for all data points or None if session is not yet run"""
        return np.array(self._scores) if self._scores is not None else None

    @property
    def known_labels(self) -> Dict[int, Label]:
        """Current dictionary of labeled objects and their labels"""
        return self._known_labels

    @property
    def known_anomalies(self) -> np.ndarray:
        """Array of indices of known anomalies"""
        return np.array([idx for idx, label in self._known_labels.items() if label == Label.ANOMALY])

    @property
    def known_regulars(self) -> np.ndarray:
        """Array of indices of known regular objects"""
        return np.array([idx for idx, label in self._known_labels.items() if label == Label.REGULAR])

    @property
    def known_unknowns(self) -> np.ndarray:
        """Array of indices of known objects marked with `Label.UNKNOWN`"""
        return np.array([idx for idx, label in self._known_labels.items() if label == Label.UNKNOWN])

    @property
    def model(self) -> Coniferest:
        """Anomaly detection model used"""
        return self._model

    @property
    def terminated(self) -> bool:
        """True if session is terminated"""
        return self._terminated

    def trajectory(self) -> np.ndarray:
        """Array of indices of objects that were labeled during the session.

        If no `known_labels` were provided, it is equal to `list(known_labels.keys())`.
        """
        return np.array(self._trajectory)
