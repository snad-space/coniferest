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
            n_trees=100,
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

After the session is finished you can explore ``session`` objects for the decisions you made and final state of the model:

.. code-block:: python

        from pprint import pprint

        print('Decisions:')
        pprint({metadata[idx]: label for idx, label in session.known_labels.items()})
        print('Final scores:')
        pprint({metadata[idx]: session.scores[idx] for idx in session.known_labels})