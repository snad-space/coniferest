import pickle

from coniferest.datasets import single_outlier
from coniferest.isoforest import IsolationForest
from coniferest.session.callback import PickleSession


class _FakeSession:
    def __init__(self, model):
        self.model = model


def _get_fitted_model():
    data, _metadata = single_outlier()
    model = IsolationForest(n_trees=10, random_seed=0).fit(data)
    return model


def test_pickle_session_overwrite(tmp_path):
    model = _get_fitted_model()
    fake_session = _FakeSession(model)

    callback = PickleSession(directory=str(tmp_path), filename="session.pickle", every_n_decisions=2, overwrite=True)

    expected_path = tmp_path / "session.pickle"

    # counter=1, should not save
    callback(None, None, fake_session)
    assert not expected_path.exists()

    # counter=2, should save
    callback(None, None, fake_session)
    assert expected_path.exists()

    with open(expected_path, "rb") as f:
        loaded_session = pickle.load(f)
    assert loaded_session.model.n_trees == fake_session.model.n_trees

    # Third and fourth calls: still overwriting same file
    callback(None, None, fake_session)
    callback(None, None, fake_session)
    assert expected_path.exists()


def test_pickle_session_numbered_files(tmp_path):
    model = _get_fitted_model()
    fake_session = _FakeSession(model)

    callback = PickleSession(
        directory=str(tmp_path),
        filename="session.pickle",
        every_n_decisions=1,
        overwrite=False,
    )

    callback(None, None, fake_session)
    callback(None, None, fake_session)

    path_1 = tmp_path / "session_1.pickle"
    path_2 = tmp_path / "session_2.pickle"

    assert path_1.exists()
    assert path_2.exists()


def test_pickle_session_can_restore_full_session(tmp_path):
    model = _get_fitted_model()
    fake_session = _FakeSession(model)
    fake_session.some_extra_state = "example known labels or progress"

    callback = PickleSession(directory=str(tmp_path), every_n_decisions=1)
    callback(None, None, fake_session)

    expected_path = tmp_path / "session.pickle"
    with open(expected_path, "rb") as f:
        restored_session = pickle.load(f)

    assert restored_session.some_extra_state == "example known labels or progress"
