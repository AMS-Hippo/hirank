Installation
============

Requirements
------------

HiRank requires Python 3.10 or later and the following core dependencies:

* numpy >= 1.22.0
* scipy >= 1.9.0
* scikit-learn >= 1.1.0
* pynndescent >= 0.5.0
* numba >= 0.56.0
* joblib >= 1.1.0

Installing from PyPI
--------------------

The easiest way to install HiRank is via pip::

    pip install hirank

Installing from Source
----------------------

To install the latest development version::

    git clone https://github.com/TutteInstitute/hirank.git
    cd hirank
    pip install -e .

Optional Dependencies
---------------------

Development Tools
~~~~~~~~~~~~~~~~~

For development (testing, formatting, linting)::

    pip install -e ".[dev]"

This installs:

* pytest
* pytest-cov
* pytest-benchmark
* black
* ruff

Benchmarking
~~~~~~~~~~~~

For running benchmarks against other outlier detection methods::

    pip install -e ".[benchmarks]"

This installs:

* pyod
* matplotlib
* seaborn
* pandas
* tables

Documentation
~~~~~~~~~~~~~

For building documentation::

    pip install -e ".[docs]"

This installs:

* sphinx
* sphinx-rtd-theme
* numpydoc

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

To install all optional dependencies::

    pip install -e ".[dev,benchmarks,docs]"

Verifying Installation
----------------------

To verify that HiRank is installed correctly::

    python -c "import hirank; print(hirank.__version__)"

Or run the test suite::

    pytest
