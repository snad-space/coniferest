``coniferest`` package
========================================

Installation
------------

**tl;dr:** ``python3 -mpip install coniferest``

You usually want to create virtual environment first, for example with ``venv``:

    python3 -mvenv venv
    source venv/bin/activate

We need up-to-date pip:

    python3 -mpip install -U pip

Finally, install ``coniferest``:

    python3 -mpip install coniferest

For any problems, please `file an issue on the GitHub <https://github.com/snad-space/coniferest/issues>`_.

Example: non-active anomaly detection
-------------------------------------

Let's generate a simple 2-D dataset with a single outlier as a last object, and run Isolation Forest model on it:

.. code-block:: python

        from coniferest.datasets import single_outlier
        from coniferest.isoforest import IsolationForest

        data, _metadata = single_outlier(10_000)
        model = IsolationForest(random_seed=0)
        model.fit(data)
        scores = model.score_samples(data)
        print("Index of the outlier:", scores.argmin())

Example: ZTF light curves of M31 field
--------------------------------------

Let's use built-in dataset of ZTF light curve features adopted from `Malanchev at al. (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract>`_:

.. code-block:: python

        from coniferest.datasets import ztf_m31

        data, metadata = ztf_m31()
        print(data.shape)

Here ``data`` is 2-D feature dataset (first axis is for objects, second is for features) and ``metadata`` is 1-D array of ZTF DR object IDs.
Next we need a active anomaly detection model to find outliers in this dataset.
Let's use ``AADForest`` model (see `Das et al., 2017 <https://arxiv.org/abs/1708.09441>`_ and `Ishida et al., 2021 <https://ui.adsabs.harvard.edu/abs/2021A%26A...650A.195I/abstract>`_ for details):

.. code-block:: python

        from coniferest.aadforest import AADForest

        model = AADForest(
            # Use 1024 trees, a trade-off between speed and accuracy
            n_trees=1024,
            # Fix random seed for reproducibility
            random_seed=0,
        )

Now we are ready to run active anomaly detection session:

.. code-block:: python

        from coniferest.session import Session
        from coniferest.session.callback import (
            TerminateAfter, viewer_decision_callback,
        )

        session = Session(
            data=data,
            metadata=metadata,
            model=model,
            # Prompt for a decision and open object's page on the SNAD Viewer
            decision_callback=viewer_decision_callback,
            on_decision_callbacks=[
                # Terminate session after 10 decisions
                TerminateAfter(10),
            ],
        )
        session.run()

This will prompt you to make a decision for an object with the highest outlier score and show you this object in the browser.
Each decision you make retrains the model and updates the outlier scores.
After 10 decisions the session will be terminated, but you can also stop it by pressing ``Ctrl+C``.

If you answer ``n`` for the first three objects, you should get a recurrent variable `ZTF DR 695211200075348 <https://ztf.snad.space/dr3/view/695211200075348>`_ / `M31N 2013-11b <https://www.astronomerstelegram.org/?read=5569>`_ / `MASTER OTJ004126.22+414350.0 <https://ui.adsabs.harvard.edu/abs/2016ATel.9470....1S/abstract>`_ as a fourth object. SNAD team reported this object as an anomaly in `Malanchev at al. (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract>`_, it is believed to be a recurrent Nova or `a long-period variable star <https://www.astronomerstelegram.org/?read=5640>`_.

After the session is finished you can explore ``session`` objects for the decisions you made and final state of the model:

.. code-block:: python

        from pprint import pprint

        print('Decisions:')
        pprint({metadata[idx]: label for idx, label in session.known_labels.items()})
        print('Final scores:')
        pprint({metadata[idx]: session.scores[idx] for idx in session.known_labels})

``coniferest`` provides a new active anomaly detection model developed by the SNAD team, ``PineForest``.
Try to replace the model with ``model = PineForest(256, n_spare_trees=768, random_seed=0)`` and run the session again.