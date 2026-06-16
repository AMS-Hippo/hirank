"""
Advanced tests for RankOD algorithm behavior and edge cases.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from hirank import RankOD


class TestRankODAlgorithm:
    """Test core algorithm behavior."""

    def test_isolated_outliers_detection(self):
        """Test detection of clearly isolated outliers."""
        np.random.seed(42)
        
        # Create tight cluster + well-separated outliers
        X_normal = np.random.randn(100, 10) * 0.3  # Tighter cluster
        
        # Outliers: scattered far from cluster
        X_outliers = []
        for i in range(10):
            direction = np.random.randn(10)
            direction = direction / np.linalg.norm(direction)
            outlier = direction * (8 + 2 * np.random.rand())
            X_outliers.append(outlier)
        X_outliers = np.array(X_outliers)
        X = np.vstack([X_normal, X_outliers])
        
        detector = RankOD(n_neighbors=15, max_rank=50, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        # Outliers should have higher median score
        assert np.median(scores[100:]) > np.median(scores[:100])
        
        # At least 7/10 outliers should be in top 15 scores
        top_indices = np.argsort(scores)[-15:]
        n_true_outliers_in_top = np.sum(top_indices >= 100)
        assert n_true_outliers_in_top >= 7

    def test_multiple_clusters(self):
        """Test behavior with multiple clusters."""
        np.random.seed(42)
        
        X, y = make_blobs(n_samples=150, centers=3, n_features=5, 
                         cluster_std=0.5, random_state=42)
        
        # Add outliers between clusters
        X_outliers = np.random.uniform(-5, 5, size=(10, 5))
        X = np.vstack([X, X_outliers])
        
        detector = RankOD(n_neighbors=10, max_rank=50, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        # Outliers should have higher scores than cluster points
        assert scores[150:].mean() > scores[:150].mean()

    def test_increasing_outlier_distance(self):
        """Test that more distant outliers get higher scores."""
        np.random.seed(42)
        
        # Smaller dataset for clearer rank differences
        X_normal = np.random.randn(30, 3) * 0.3
        
        # Create outliers at increasing distances (closer to cluster)
        distances = [2, 3, 4, 5]
        all_scores = []
        
        for dist in distances:
            X_outlier = np.array([[dist, dist, dist]])
            X = np.vstack([X_normal, X_outlier])
            
            detector = RankOD(n_neighbors=10, max_rank=25, random_state=42)
            detector.fit(X)
            scores = detector.score_samples(X)
            all_scores.append(scores[-1])
        
        # Check trend: further outliers should generally have higher scores
        # Allow some noise but check overall trend
        assert all_scores[3] >= all_scores[0]  # Furthest > closest


class TestRankODKernels:
    """Test different kernel functions."""

    @pytest.mark.parametrize("kernel", ["harmonic", "inverse_sqrt", "gaussian"])
    def test_kernel_produces_valid_scores(self, kernel):
        """Test that each kernel produces valid outlier scores."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        
        detector = RankOD(n_neighbors=10, max_rank=30, kernel=kernel, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        # All scores should be finite and non-negative
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)

    def test_gaussian_kernel_with_sigma(self):
        """Test Gaussian kernel with custom sigma parameter."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        detector = RankOD(n_neighbors=10, max_rank=30, kernel="gaussian", 
                         kernel_params={"sigma": 2.0}, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        assert len(scores) == 50
        assert np.all(np.isfinite(scores))

    def test_custom_kernel(self):
        """Test with custom kernel function."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        # Define custom kernel: exponential decay
        def exp_kernel(ranks):
            return np.exp(-0.1 * ranks)
        
        detector = RankOD(n_neighbors=10, max_rank=30, kernel=exp_kernel, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        assert len(scores) == 50
        assert np.all(np.isfinite(scores))

    def test_kernel_comparison(self):
        """Test that different kernels produce different scores."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        scores_dict = {}
        for kernel in ["harmonic", "inverse_sqrt", "gaussian"]:
            detector = RankOD(n_neighbors=10, max_rank=30, kernel=kernel, random_state=42)
            detector.fit(X)
            scores_dict[kernel] = detector.score_samples(X)
        
        # Scores should be different for different kernels
        assert not np.allclose(scores_dict["harmonic"], scores_dict["inverse_sqrt"])
        assert not np.allclose(scores_dict["harmonic"], scores_dict["gaussian"])


class TestRankODParameters:
    """Test parameter variations."""

    def test_varying_k(self):
        """Test with different values of k."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        for k in [5, 10, 20, 30]:
            detector = RankOD(n_neighbors=k, max_rank=50, random_state=42)
            detector.fit(X)
            scores = detector.score_samples(X)
            assert len(scores) == 100

    def test_varying_J(self):
        """Test with different values of J."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        for J in [20, 50, 80]:
            detector = RankOD(n_neighbors=15, max_rank=J, random_state=42)
            detector.fit(X)
            scores = detector.score_samples(X)
            assert len(scores) == 100

    def test_k_J_relationship(self):
        """Test that J >= k produces different results."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        detector1 = RankOD(n_neighbors=15, max_rank=15, random_state=42)
        detector1.fit(X)
        scores1 = detector1.score_samples(X)
        
        detector2 = RankOD(n_neighbors=15, max_rank=50, random_state=42)
        detector2.fit(X)
        scores2 = detector2.score_samples(X)
        
        # Scores should differ when J changes
        assert not np.allclose(scores1, scores2)


class TestRankODEdgeCases:
    """Test edge cases and error handling."""

    def test_minimum_sample_size(self):
        """Test with minimum viable sample size."""
        np.random.seed(42)
        X = np.random.randn(20, 5)  # Small dataset
        
        detector = RankOD(n_neighbors=5, max_rank=10, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 20

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 100)  # 100 dimensions
        
        detector = RankOD(n_neighbors=15, max_rank=50, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 100

    def test_single_feature(self):
        """Test with single-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert len(scores) == 100

    def test_uniform_data(self):
        """Test with uniform/constant data."""
        X = np.ones((100, 5))  # All points identical
        
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        # All scores should be similar for identical points
        assert np.std(scores) < 1.0


class TestRankODScikitLearnCompatibility:
    """Test scikit-learn API compatibility."""

    def test_get_params(self):
        """Test get_params method."""
        detector = RankOD(n_neighbors=20, max_rank=60, kernel="inverse_sqrt")
        params = detector.get_params()
        
        assert params["n_neighbors"] == 20
        assert params["max_rank"] == 60
        assert params["kernel"] == "inverse_sqrt"

    def test_set_params(self):
        """Test set_params method."""
        detector = RankOD(n_neighbors=15, max_rank=50)
        detector.set_params(n_neighbors=20, kernel="gaussian")
        
        assert detector.n_neighbors == 20
        assert detector.kernel == "gaussian"

    def test_predict_contamination_levels(self):
        """Test predict with different contamination levels."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        detector = RankOD(n_neighbors=15, max_rank=50, random_state=42)
        detector.fit(X)
        
        for contamination in [0.05, 0.1, 0.2]:
            predictions = detector.predict(X, contamination=contamination)
            n_outliers = np.sum(predictions == -1)
            expected = int(100 * contamination)
            # Allow some tolerance due to rounding
            assert abs(n_outliers - expected) <= 2

    def test_fit_predict_equivalence(self):
        """Test that fit_predict gives same result as fit then predict."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        detector1 = RankOD(n_neighbors=15, max_rank=50, random_state=42)
        pred1 = detector1.fit_predict(X, contamination=0.1)
        
        detector2 = RankOD(n_neighbors=15, max_rank=50, random_state=42)
        detector2.fit(X)
        pred2 = detector2.predict(X, contamination=0.1)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestRankODNumericalStability:
    """Test numerical stability."""

    def test_extreme_values(self):
        """Test with extreme data values."""
        np.random.seed(42)
        X = np.random.randn(50, 5) * 1000  # Large values
        
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        assert np.all(np.isfinite(scores))

    def test_small_values(self):
        """Test with very small data values."""
        np.random.seed(42)
        X = np.random.randn(50, 5) * 1e-6  # Tiny values
        
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        assert np.all(np.isfinite(scores))

    def test_mixed_scale_features(self):
        """Test with features at different scales."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[:, 0] *= 1000  # First feature at large scale
        X[:, 1] *= 0.001  # Second feature at small scale
        
        detector = RankOD(n_neighbors=10, max_rank=30, random_state=42)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        assert np.all(np.isfinite(scores))
