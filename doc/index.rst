HiRank: High-dimensional Rank-based Outlier Detection
======================================================

.. image:: https://img.shields.io/pypi/v/hirank.svg
   :target: https://pypi.org/project/hirank/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/hirank.svg
   :target: https://pypi.org/project/hirank/
   :alt: Python Versions

.. image:: https://img.shields.io/github/license/TutteInstitute/hirank.svg
   :target: https://github.com/TutteInstitute/hirank/blob/main/LICENSE
   :alt: License

**HiRank** is a tightly-scoped outlier detection library implementing reverse k-NN 
density estimation with kernel smoothing, optimized for high-dimensional data using 
PyNNDescent for efficient approximate nearest neighbor search.

Quick Start
-----------

.. code-block:: python

    import numpy as np
    from hirank import RankOD

    # Generate sample data
    X = np.random.randn(1000, 50)

    # Create and fit detector
    detector = RankOD(n_neighbors=15, max_rank=100, kernel='harmonic')
    detector.fit(X)

    # Get outlier scores
    scores = detector.score_samples(X)

    # Predict outliers
    labels = detector.predict(X, contamination=0.1)

Installation
------------

From pypi (coming soon!)::

    pip install hirank

From source::

    git clone https://github.com/TutteInstitute/hirank.git
    cd hirank
    pip install -e .

With optional dependencies::

    pip install -e ".[dev,benchmarks,docs]"

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   installation
   mnist_ood_detection
   knnood

.. toctree::
   :maxdepth: 1
   :caption: API

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
