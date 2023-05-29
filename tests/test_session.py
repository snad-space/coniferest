import numpy as np
import pytest

from coniferest.aadforest import AADForest
from coniferest.datasets import ztf_m31, non_anomalous_outliers
from coniferest.label import Label
from coniferest.pineforest import PineForest
from coniferest.session import Session


@pytest.mark.e2e
@pytest.mark.long
@pytest.mark.regression
def test_e2e_ztf_m31():
    """Basically the same example as in the docs"""
    class Callback():
        """Say NO for first three objectsm then say YES and terminate"""
        counter = 0

        def decision(self, _metadata, _data, _session) -> Label:
            self.counter += 1
            if self.counter < 4:
                return Label.REGULAR
            return Label.ANOMALY

        def on_decision(self, _metadata, _data, session) -> None:
            if self.counter >= 4:
                session.terminate()

    callback = Callback()

    data, metadata = ztf_m31()
    model = AADForest(
        n_trees=1024,
        random_seed=0,
    )
    session = Session(
        data=data,
        metadata=metadata,
        model=model,
        decision_callback=callback.decision,
        on_decision_callbacks=[callback.on_decision],
    )
    session.run()

    assert len(session.known_labels) == 4

    oid = 695211200075348
    idx = np.where(metadata == oid)[0][0]
    assert idx in session.known_labels
    assert session.known_labels[idx] == Label.ANOMALY


@pytest.mark.e2e
@pytest.mark.regression
@pytest.mark.parametrize(
    "model,n_iter,last_idx",
    [
        (AADForest(n_trees=128, random_seed=0), 91, 1093),
        (PineForest(n_trees=128, n_spare_trees=512, random_seed=0), 34, 1109),
    ],
)
def test_non_anomalous_outliers(model, n_iter, last_idx):
    # Number of normal objects
    n_regular = 1 << 10
    # Number of outliers in each of three groups, only one of which is anomalous
    n_anomalies = 1 << 5
    data, metadata = non_anomalous_outliers(inliers=n_regular, outliers=n_anomalies)

    def terminate_after_all_detected(_metadata, _data, session):
        n_detected = sum(label == Label.ANOMALY for label in session.known_labels.values())
        if n_detected == n_anomalies:
            session.terminate()

    session = Session(
        data=data,
        metadata=metadata,
        model=model,
        decision_callback=lambda metadata, _data, _session: metadata,
        on_decision_callbacks=[terminate_after_all_detected],
    )
    session.run()

    assert len(session.known_labels) == n_iter
    assert session.current == last_idx
