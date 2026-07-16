coniferest
==========

**coniferest** is a package for active anomaly detection with isolation forests, made by the `SNAD collaboration <https://snad.space/>`_.

It includes:

- :class:`IsolationForest <coniferest.isoforest.IsolationForest>` — a reimplementation of scikit-learn's isolation forest with much better scoring performance, thanks to the use of the Rust programming language and multi-threading.
- :class:`AADForest <coniferest.aadforest.AADForest>` — a reimplementation of the Active Anomaly Detection algorithm with isolation forests from Shubhomoy Das' `ad_examples <https://github.com/shubhomoydas/ad_examples>`_ package, with better performance, much less code and more flexible dependencies.
- :class:`PineForest <coniferest.pineforest.PineForest>` — our own active learning model based on the idea of tree filtering.

Installation
------------

.. code-block:: bash

    python3 -mpip install coniferest

Binary wheels are available for Linux, macOS and Windows, so the package can be installed from PyPI on these platforms with no build-time dependencies.
See the :doc:`tutorial` for a complete walk-through, including building from source.

Getting started
----------------

- :doc:`tutorial` — install the package and run your first isolation forest and active anomaly detection session.
- :doc:`notebooks` — worked examples and workshop tutorials as Jupyter notebooks.
- :doc:`isoforest` — details on the isolation forest implementation.
- :doc:`pariou` — mathematical background of anomaly detection feature signatures.
- :doc:`modules` — full API reference.

Citation
--------

If you found this project useful for your research, please cite `Kornilov, Korolev, Malanchev, et al., 2025 <https://doi.org/10.1016/j.ascom.2025.100960>`_:

.. code-block:: bibtex

    @article{Kornilov2025,
        title = {Coniferest: A complete active anomaly detection framework},
        journal = {Astronomy and Computing},
        volume = {52},
        pages = {100960},
        year = {2025},
        issn = {2213-1337},
        doi = {10.1016/j.ascom.2025.100960},
        url = {https://www.sciencedirect.com/science/article/pii/S2213133725000332},
        author = {M.V. Kornilov and V.S. Korolev and K.L. Malanchev and A.D. Lavrukhina and E. Russeil and T.A. Semenikhin and E. Gangler and E.E.O. Ishida and M.V. Pruzhinskaya and A.A. Volnova and S. Sreejith},
    }

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   tutorial.rst
   isoforest.rst
   pariou.rst
   notebooks.rst
   modules.rst

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
