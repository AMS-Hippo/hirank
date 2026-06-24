"""
Additional comprehensive tests for RankOD.
"""

import numpy as np
import pytest

from hirank import RankOD


class TestIsolatedOutliers:
    """Tests with isolated outliers that should be easily detected."""

    def test_isolated_outliers_high_accuracy(self):
        """Test that isolated outliers are detected with high accuracy."""
        np.random.seed(42)
        n_normal = 200
        n_outliers = 10
        n_features = 20

        # Normal cluster
        X_normal = np.random.randn(n_normal, n_features) * 0.5

        # Isolated outliers
        X_outliers = []
        for _i in range(n_outliers):
            direction = np.random.randn(n_features)
            direction = direction / np.linalg.norm(direction)
            outlier = direction * (8 + 4 * np.random.rand())
            X_outliers.append(outlier)

        X_outliers = np.array(X_outliers)
        X = np.vstack([X_normal, X_outliers])

        detector = RankOD(n_neighbors=15, max_rank=50, random_state=42)
        detector.fit(X)

        # Get scores
        scores = detector.score_samples(X)

        # Outliers should have higher scores than normal points
        assert scores[n_normal:].mean() < scores[:n_normal].mean()

        # Predict with true contamination rate
        contamination = n_outliers / len(X)
        predictions = detector.predict(X, contamination=contamination)

        # Should detect most outliers
        true_outliers_detected = np.sum(predictions[n_normal:] == -1)
        assert true_outliers_detected >= n_outliers * 0.8  # At least 80%

    def test_outliers_get_bottom_scores(self):
        """Test that outliers get the highest scores."""
        np.random.seed(42)
        n_normal = 100
        n_outliers = 5

        X_normal = np.random.randn(n_normal, 10) * 0.3
        X_outliers = []
        for _i in range(n_outliers):
            direction = np.random.randn(10)
            direction = direction / np.linalg.norm(direction)
            X_outliers.append(direction * 10)

        X_outliers = np.array(X_outliers)
        X = np.vstack([X_normal, X_outliers])

        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        scores = detector.fit(X).score_samples(X)

        # Top 5 scores should all be outliers
        top_5_indices = np.argsort(scores)[:5]
        assert all(idx >= n_normal for idx in top_5_indices)


class TestHighDimensional:
    """Tests for high-dimensional data."""

    def test_high_dimensional_data(self):
        """Test on data with many dimensions."""
        np.random.seed(42)
        n_samples = 150
        n_features = 100  # High dimensional
        n_outliers = 8

        X_normal = np.random.randn(n_samples - n_outliers, n_features) * 0.5
        X_outliers = []
        for _i in range(n_outliers):
            direction = np.random.randn(n_features)
            direction = direction / np.linalg.norm(direction)
            X_outliers.append(direction * 8)

        X = np.vstack([X_normal, np.array(X_outliers)])

        detector = RankOD(n_neighbors=10, max_rank=40, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)

        # Outliers should still have lower scores
        assert (
            scores[n_samples - n_outliers :].mean()
            < scores[: n_samples - n_outliers].mean()
        )

    def test_very_high_dimensional(self):
        """Test with very high dimensions (500D)."""
        np.random.seed(42)
        n_samples = 100
        n_features = 500

        X = np.random.randn(n_samples, n_features)

        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)

        assert len(scores) == n_samples
        assert scores.dtype == np.float64


class TestKernelFunctions:
    """Tests for different kernel functions."""

    @pytest.fixture
    def outlier_data(self):
        """Generate test data with outliers."""
        np.random.seed(42)
        n_normal = 100
        n_outliers = 5

        X_normal = np.random.randn(n_normal, 10) * 0.5
        X_outliers = []
        for _i in range(n_outliers):
            direction = np.random.randn(10)
            direction = direction / np.linalg.norm(direction)
            X_outliers.append(direction * 8)

        X = np.vstack([X_normal, np.array(X_outliers)])
        return X, n_normal

    def test_inverse_sqrt_kernel(self, outlier_data):
        """Test inverse sqrt kernel."""
        X, n_normal = outlier_data
        detector = RankOD(
            n_neighbors=10, max_rank=30, kernel="inverse_sqrt", random_state=42
        )
        scores = detector.fit(X).score_samples(X)
        assert scores[n_normal:].mean() < scores[:n_normal].mean()

    def test_linear_kernel(self, outlier_data):
        """Test Linear kernel."""
        X, n_normal = outlier_data
        detector = RankOD(
            n_neighbors=10,
            max_rank=30,
            kernel="linear",
            random_state=42,
        )
        scores = detector.fit(X).score_samples(X)
        assert scores[n_normal:].mean() < scores[:n_normal].mean()

    def test_custom_kernel(self, outlier_data):
        """Test custom kernel function."""
        X, n_normal = outlier_data

        def custom_kernel(ranks):
            return 1.0 / (ranks + 0.5)

        detector = RankOD(
            n_neighbors=10, max_rank=30, kernel=custom_kernel, random_state=42
        )
        scores = detector.fit(X).score_samples(X)
        assert scores[n_normal:].mean() < scores[:n_normal].mean()


class TestParameters:
    """Tests for different parameter settings."""

    def test_small_k(self):
        """Test with small k value."""
        X = np.random.randn(100, 10)
        detector = RankOD(n_neighbors=5, max_rank=20, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 100

    def test_large_k(self):
        """Test with large k value."""
        X = np.random.randn(200, 10)
        detector = RankOD(n_neighbors=50, max_rank=100, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 200

    def test_small_J(self):
        """Test with small J value."""
        X = np.random.randn(100, 10)
        detector = RankOD(n_neighbors=10, max_rank=20, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 100

    def test_large_J(self):
        """Test with large J value."""
        X = np.random.randn(100, 10)
        detector = RankOD(n_neighbors=10, max_rank=200, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 100

    def test_k_equals_J(self):
        """Test when k equals J."""
        X = np.random.randn(100, 10)
        detector = RankOD(n_neighbors=20, max_rank=20, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 100


class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimum_samples(self):
        """Test with minimum number of samples (just above k)."""
        X = np.random.randn(20, 5)
        detector = RankOD(n_neighbors=10, max_rank=15, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 20

    def test_score_new_samples(self):
        """Test scoring new samples not in training data."""
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)

        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X_train)

        scores = detector.score_samples(X_test)
        assert len(scores) == 20
        assert scores.dtype == np.float64

    def test_uniform_data(self):
        """Test on uniformly distributed data."""
        X = np.random.uniform(-1, 1, (100, 10))
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 100

    def test_single_cluster(self):
        """Test on data with single tight cluster."""
        X = np.random.randn(100, 10) * 0.1  # Very tight cluster
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        # All scores should be similar in a tight cluster
        assert scores.std() < scores.mean()


class TestContamination:
    """Tests for contamination parameter in predict."""

    def test_contamination_values(self):
        """Test predict with different contamination values."""
        X = np.random.randn(100, 10)
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)

        for contamination in [0.01, 0.05, 0.1, 0.2]:
            predictions = detector.predict(X, contamination=contamination)
            n_outliers = np.sum(predictions == -1)
            expected_outliers = int(len(X) * contamination)
            # Should be close to expected
            assert abs(n_outliers - expected_outliers) <= 2

    def test_contamination_threshold(self):
        """Test that higher contamination detects more outliers."""
        X = np.random.randn(100, 10)
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)

        pred_low = detector.predict(X, contamination=0.05)
        pred_high = detector.predict(X, contamination=0.15)

        assert np.sum(pred_low == -1) < np.sum(pred_high == -1)
