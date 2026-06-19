"""
Basic tests for HiRank RankOD implementation.
"""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from hirank import RankOD


def test_import():
    """Test that RankOD can be imported."""
    from hirank import RankOD

    assert RankOD is not None



def test_init_custom_params():
    """Test RankOD initialization with custom parameters."""
    detector = RankOD(n_neighbors=20, max_rank=50, kernel="inverse_sqrt")
    assert detector.n_neighbors == 20
    assert detector.max_rank == 50
    assert detector.kernel == "inverse_sqrt"


def test_fit_basic():
    """Test basic fitting on random data."""
    X = np.random.randn(100, 10)
    detector = RankOD(n_neighbors=10, max_rank=50)
    detector.fit(X)

    assert hasattr(detector, "index_")
    assert hasattr(detector, "outlier_scores_")
    assert hasattr(detector, "n_features_in_")
    assert detector.n_features_in_ == 10


def test_score_samples():
    """Test score_samples method."""
    X = np.random.randn(100, 10)
    detector = RankOD(n_neighbors=10, max_rank=50)
    detector.fit(X)

    scores = detector.score_samples(X)
    assert len(scores) == 100
    assert scores.dtype == np.float64


def test_predict():
    """Test predict method."""
    X = np.random.randn(100, 10)
    detector = RankOD(n_neighbors=10, max_rank=50)
    detector.fit(X)

    labels = detector.predict(X, contamination=0.1)
    assert len(labels) == 100
    assert set(labels).issubset({-1, 1})
    # Approximately 10% should be outliers
    assert 5 <= np.sum(labels == -1) <= 15


def test_fit_predict():
    """Test fit_predict method."""
    X = np.random.randn(100, 10)
    detector = RankOD(n_neighbors=10, max_rank=50)

    labels = detector.fit_predict(X)
    assert len(labels) == 100
    assert set(labels).issubset({-1, 1})


def test_reverse_scores():
    """Test with reversed scores"""
    X_normal = np.random.randn(99, 10)
    X_outliers = np.random.randn(1, 10) * 5 + 10  # Far from normal data
    X = np.vstack([X_normal, X_outliers])

    reverse_detector = RankOD(n_neighbors=10, max_rank=50, reverse_scores=True)
    reverse_outliers = reverse_detector.fit_predict(X)
    reverse_scores = reverse_detector.score_samples(X)

    assert len(reverse_scores) == 100
    assert reverse_scores.dtype == np.float64
    assert reverse_scores[-1] == np.max(reverse_scores)
    assert reverse_outliers[-1] == -1


def test_different_kernels():
    """Test with different kernel functions."""
    X = np.random.randn(50, 5)

    for kernel in ["harmonic", "inverse_sqrt", "gaussian"]:
        detector = RankOD(n_neighbors=10, max_rank=30, kernel=kernel)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 50


def test_custom_kernel():
    """Test with custom kernel function."""
    X = np.random.randn(50, 5)

    def custom_kernel(ranks):
        return 1.0 / (ranks + 1.0)

    detector = RankOD(n_neighbors=10, max_rank=30, kernel=custom_kernel)
    detector.fit(X)
    scores = detector.score_samples(X)
    assert len(scores) == 50


def test_invalid_k():
    """Test that invalid n_neighbors raises error."""
    X = np.random.randn(50, 5)
    detector = RankOD(n_neighbors=60)  # n_neighbors >= n_samples

    with pytest.raises(
        ValueError, match="n_neighbors=60 must be less than n_samples=50"
    ):
        detector.fit(X)


def test_invalid_kernel():
    """Test that invalid kernel name raises error."""
    detector = RankOD(kernel="invalid")
    X = np.random.randn(50, 5)

    with pytest.raises(ValueError, match="Unknown kernel"):
        detector.fit(X)


def test_different_dimensions():
    """Test fitting on data with different dimensions."""
    for n_features in [5, 10, 20, 50]:
        X = np.random.randn(100, n_features)
        detector = RankOD(n_neighbors=10, max_rank=30)
        detector.fit(X)
        assert detector.n_features_in_ == n_features


def test_reproducibility():
    """Test that results are reproducible with random_state."""
    X = np.random.randn(100, 10)

    detector1 = RankOD(n_neighbors=10, max_rank=50, random_state=42)
    detector1.fit(X)
    scores1 = detector1.score_samples(X)

    detector2 = RankOD(n_neighbors=10, max_rank=50, random_state=42)
    detector2.fit(X)
    scores2 = detector2.score_samples(X)

    np.testing.assert_array_almost_equal(scores1, scores2)


def test_outlier_detection():
    """Test that clear outliers are detected."""
    np.random.seed(42)

    # Create data with clear outliers
    X_normal = np.random.randn(95, 10)
    X_outliers = np.random.randn(5, 10) * 5 + 10  # Far from normal data
    X = np.vstack([X_normal, X_outliers])

    detector = RankOD(n_neighbors=10, max_rank=50)
    detector.fit(X)
    scores = detector.score_samples(X)

    # Outliers should have lower scores
    outlier_scores = scores[:5]
    normal_scores = scores[5:]
    assert np.mean(outlier_scores) < np.mean(normal_scores)


def test_single_vector_scoring():
    """Test that score_samples and predict work with single vectors (1D arrays)."""
    np.random.seed(42)
    X_train = np.random.randn(50, 10)

    # Fit detector
    detector = RankOD(n_neighbors=5, max_rank=20, contamination=0.1)
    detector.fit(X_train)

    # Test 1: Single vector as 1D array
    single_vector = np.random.randn(10)
    assert single_vector.ndim == 1, "Test vector should be 1D"

    # score_samples should work with 1D array
    score = detector.score_samples(single_vector)
    assert score.shape == (1,), f"Expected shape (1,), got {score.shape}"
    assert isinstance(score[0], (float, np.floating)), "Score should be a float"

    # predict should work with 1D array
    pred = detector.predict(single_vector)
    assert pred.shape == (1,), f"Expected shape (1,), got {pred.shape}"
    assert pred[0] in [-1, 1], f"Prediction should be -1 or 1, got {pred[0]}"

    # Test 2: Single vector as 2D array (1, n_features)
    single_2d = np.random.randn(1, 10)
    assert single_2d.ndim == 2 and single_2d.shape[0] == 1

    score_2d = detector.score_samples(single_2d)
    assert score_2d.shape == (1,), f"Expected shape (1,), got {score_2d.shape}"

    pred_2d = detector.predict(single_2d)
    assert pred_2d.shape == (1,), f"Expected shape (1,), got {pred_2d.shape}"
    assert pred_2d[0] in [-1, 1]

    # Test 3: Multiple vectors still work
    multi_vectors = np.random.randn(3, 10)
    scores_multi = detector.score_samples(multi_vectors)
    assert scores_multi.shape == (3,), f"Expected shape (3,), got {scores_multi.shape}"

    preds_multi = detector.predict(multi_vectors)
    assert preds_multi.shape == (3,), f"Expected shape (3,), got {preds_multi.shape}"
    assert all(p in [-1, 1] for p in preds_multi)

    # Test 4: List input (should also work)
    list_vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    score_list = detector.score_samples(list_vector)
    assert score_list.shape == (1,), f"Expected shape (1,), got {score_list.shape}"

    pred_list = detector.predict(list_vector)
    assert pred_list.shape == (1,), f"Expected shape (1,), got {pred_list.shape}"
    assert pred_list[0] in [-1, 1]


def test_single_vector_wrong_dimensions():
    """Test that single vectors with wrong number of features raise appropriate errors."""
    np.random.seed(42)
    X_train = np.random.randn(50, 10)

    detector = RankOD(n_neighbors=5, max_rank=20)
    detector.fit(X_train)

    # Wrong number of features (should raise ValueError)
    wrong_vector = np.random.randn(1, 5)  # Only 5 features instead of 10

    with pytest.raises(ValueError, match=".*features.*"):
        detector.score_samples(wrong_vector)

    with pytest.raises(ValueError, match=".*features.*"):
        detector.predict(wrong_vector)


def test_precompute_neighbors_modes():
    """Test that both precompute modes (True/False) produce identical results for both training and test data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    X_test = np.random.randn(20, 10)

    # Test with precompute_neighbors=False (memory-efficient, default)
    detector_memory_efficient = RankOD(
        n_neighbors=10, max_rank=30, precompute_neighbors=False, random_state=42
    )
    detector_memory_efficient.fit(X_train)
    scores_memory_train = detector_memory_efficient.score_samples(X_train)
    scores_memory_test = detector_memory_efficient.score_samples(X_test)

    # Test with precompute_neighbors=True (speed-optimized)
    detector_speed_optimized = RankOD(
        n_neighbors=10, max_rank=30, precompute_neighbors=True, random_state=42
    )
    detector_speed_optimized.fit(X_train)
    scores_speed_train = detector_speed_optimized.score_samples(X_train)
    scores_speed_test = detector_speed_optimized.score_samples(X_test)

    # Both modes should produce identical results for training data
    np.testing.assert_array_almost_equal(
        scores_memory_train, scores_speed_train, decimal=10
    )

    # Both modes should produce identical results for test data (critical optimization path)
    np.testing.assert_array_almost_equal(
        scores_memory_test, scores_speed_test, decimal=10
    )

    # Check that pre-computation actually happened in speed-optimized mode only
    assert hasattr(detector_speed_optimized, "_training_neighbors_")
    assert hasattr(
        detector_speed_optimized, "_training_distances_"
    ), "Speed-optimized mode should store distances"
    assert not hasattr(detector_memory_efficient, "_training_neighbors_")
    assert not hasattr(detector_memory_efficient, "_training_distances_")

    # Verify shapes of precomputed arrays
    assert detector_speed_optimized._training_neighbors_.shape == (
        100,
        30,
    ), "Should store max_rank neighbors for each sample"
    assert detector_speed_optimized._training_distances_.shape == (
        100,
        30,
    ), "Should store max_rank distances for each sample"

    # Both should have training data stored
    assert hasattr(detector_memory_efficient, "_training_data_")
    assert hasattr(detector_speed_optimized, "_training_data_")
    assert hasattr(detector_speed_optimized, "_training_data_")


def test_check_estimator():
    check_estimator(
        RankOD(n_neighbors=5),  # So pynndescent works with as few as 10 samples
        expected_failed_checks={
            "check_estimators_pickle": "pynndescent does not pickle nicely",
            "check_parameters_default_constructible": "We use dicts as params, update to frozen dict when possible",
            "check_methods_sample_order_invariance": "Pynndescent instability causes problems with random points",
            "check_methods_subset_invariance": "Pynndescent instability causes problems with random points",
            "check_fit2d_predict1d": "We auto convert correctly shaped 1d arrays to 2d",
        },
    )
