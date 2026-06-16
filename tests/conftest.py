"""
Pytest configuration and fixtures for HiRank tests.
"""

import numpy as np
import pytest


@pytest.fixture
def random_data():
    """Generate random test data."""
    np.random.seed(42)
    return np.random.randn(100, 10)


@pytest.fixture
def high_dimensional_data():
    """Generate high-dimensional test data."""
    np.random.seed(42)
    return np.random.randn(200, 50)


@pytest.fixture
def outlier_data():
    """Generate data with clear outliers."""
    np.random.seed(42)
    X_normal = np.random.randn(95, 10)
    X_outliers = np.random.randn(5, 10) * 5 + 10
    return np.vstack([X_normal, X_outliers])
