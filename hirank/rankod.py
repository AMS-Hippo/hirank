"""
RankOD: Rank-based Outlier Detection using Reverse k-NN Density Estimation.

This module implements the core RankOD algorithm with scikit-learn compatible API.
"""

import warnings
from typing import Callable, Optional, Union

import numpy as np
from numba import njit
from pynndescent import NNDescent
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted


@njit(cache=True)
def harmonic_kernel(ranks: np.ndarray) -> np.ndarray:
    """
    Harmonic kernel function: k(r) = 1/r

    Parameters
    ----------
    ranks : np.ndarray
        Array of ranks (1-indexed)

    Returns
    -------
    np.ndarray
        Kernel values
    """
    return 1.0 / ranks


@njit(cache=True)
def inverse_sqrt_kernel(ranks: np.ndarray) -> np.ndarray:
    """
    Inverse square root kernel: k(r) = 1/sqrt(r)

    Parameters
    ----------
    ranks : np.ndarray
        Array of ranks (1-indexed)

    Returns
    -------
    np.ndarray
        Kernel values
    """
    return 1.0 / np.sqrt(ranks)


@njit(cache=True)
def gaussian_kernel(ranks: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian kernel: k(r) = exp(-r^2 / (2*sigma^2))

    Parameters
    ----------
    ranks : np.ndarray
        Array of ranks (1-indexed)
    sigma : float
        Bandwidth parameter

    Returns
    -------
    np.ndarray
        Kernel values
    """
    return np.exp(-(ranks**2) / (2.0 * sigma**2))


class RankOD(BaseEstimator, OutlierMixin):
    """
    Rank-based Outlier Detection using Reverse k-NN Density Estimation.

    RankOD detects outliers by estimating local density based on reverse k-nearest
    neighbor ranks. For each point, it computes the ranks at which the point appears
    in its neighbors' nearest neighbor lists, applies a kernel function to smooth
    these ranks, and converts the resulting density to an outlier score.

    The algorithm is particularly effective in high-dimensional spaces where
    traditional distance-based methods struggle.

    Parameters
    ----------
    n_neighbors : int, default=15
        Number of nearest neighbors to use for density estimation.

    max_rank : int, default=100
        Maximum rank to consider in reverse nearest neighbor search.
        Ranks beyond max_rank are capped at max_rank.

    kernel : {'harmonic', 'inverse_sqrt', 'gaussian'} or callable, default='harmonic'
        Kernel function to apply to ranks:
        - 'harmonic': k(r) = 1/r
        - 'inverse_sqrt': k(r) = 1/sqrt(r)
        - 'gaussian': k(r) = exp(-r^2 / (2*sigma^2))
        - callable: custom kernel function taking ranks array and returning weights

    kernel_params : dict, optional
        Additional parameters for the kernel function (e.g., sigma for Gaussian).

    metric : str, default='euclidean'
        Distance metric to use for nearest neighbor search.
        See pynndescent documentation for available metrics.

    metric_kwds : dict, optional
        Additional keyword arguments for the metric.

    contamination : float, default=0.1
        Expected proportion of outliers in the dataset.
        Used to set the threshold for binary classification in predict().
        Must be in the range (0, 0.5].

    precompute_neighbors : bool, default=False
        Whether to pre-compute and store max_rank nearest neighbors for all training points.
        - False (default): Memory-efficient mode. Queries index on-demand during scoring.
          Memory: O(1), Scoring speed: Moderate (requires additional queries)
        - True: Speed-optimized mode. Pre-computes and stores neighbor arrays.
          Memory: O(n_samples * max_rank), Scoring speed: Fast (array lookups)
        For large datasets, False is recommended to avoid memory issues.

    dtype : numpy.dtype, default=np.float64
        Data type for internal storage of training data.
        - np.float64 (default): Standard sklearn precision, ~8 bytes per value
        - np.float32: Half the memory usage, ~4 bytes per value, sufficient precision
          for most distance-based outlier detection tasks
        Note: PyNNDescent internally uses float32, so using np.float32 here
        avoids precision conversion and reduces memory footprint.

    n_jobs : int, default=-1
        Number of parallel jobs for nearest neighbor search.
        -1 uses all available cores.

    random_state : int, optional
        Random seed for reproducibility.

    verbose : bool, default=False
        Whether to print progress messages.

    Attributes
    ----------
    outlier_scores_ : np.ndarray of shape (n_samples,)
        Outlier scores for training samples, normalized to [0, 1] range.
        Higher values indicate outliers (0=most normal, 1=most anomalous).

    density_scores_ : np.ndarray of shape (n_samples,)
        Raw density scores (before normalization to outlier scores).

    max_density_ : float
        Maximum possible density score (n_neighbors * kernel(1)).
        Used for score normalization.

    min_density_ : float
        Minimum possible density score (n_neighbors * kernel(max_rank)).
        Used for score normalization.

    index_ : NNDescent
        Fitted nearest neighbor index.

    n_features_in_ : int
        Number of features in training data.

    Examples
    --------
    >>> from hirank import RankOD
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> detector = RankOD(n_neighbors=15, max_rank=100)
    >>> detector.fit(X)
    >>> outlier_scores = detector.score_samples(X)
    >>> predictions = detector.predict(X)  # -1 for outliers, 1 for inliers

    References
    ----------
    Based on reverse k-NN density estimation with kernel smoothing for
    high-dimensional outlier detection.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        max_rank: int = 100,
        contamination: float = 0.1,
        precompute_neighbors: bool = False,
        dtype = np.float64,
        kernel: Union[str, Callable] = "harmonic",
        kernel_params: Optional[dict] = None,
        metric: str = "euclidean",
        metric_kwds: Optional[dict] = None,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.max_rank = max_rank
        self.contamination = contamination
        self.precompute_neighbors = precompute_neighbors
        self.dtype = dtype  # Store as-is for sklearn compatibility
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.metric = metric
        self.metric_kwds = metric_kwds or {}
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _get_kernel_function(self) -> Callable:
        """Get the kernel function based on the kernel parameter."""
        if callable(self.kernel):
            return self.kernel
        elif self.kernel == "harmonic":
            return harmonic_kernel
        elif self.kernel == "inverse_sqrt":
            return inverse_sqrt_kernel
        elif self.kernel == "gaussian":
            sigma = self.kernel_params.get("sigma", 1.0)
            return lambda r: gaussian_kernel(r, sigma)
        else:
            raise ValueError(
                f"Unknown kernel: {self.kernel}. "
                f"Must be 'harmonic', 'inverse_sqrt', 'gaussian', or callable."
            )

    def fit(self, X, y=None):
        """
        Fit the RankOD detector on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, accept_sparse=False, dtype=self.dtype)
        n_samples, n_features = X.shape

        if self.n_neighbors >= n_samples:
            raise ValueError(f"n_neighbors={self.n_neighbors} must be less than n_samples={n_samples}")

        self.n_features_in_ = n_features

        # Build nearest neighbor index
        if self.verbose:
            print(f"Building nearest neighbor index with n_neighbors={self.n_neighbors}...")

        self.index_ = NNDescent(
            X,
            metric=self.metric,
            metric_kwds=self.metric_kwds,
            n_neighbors=max(self.n_neighbors + 1, self.max_rank + 1),  # +1 to exclude self
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Optionally pre-compute and store training data neighbors for reverse rank computation
        if self.precompute_neighbors:
            if self.verbose:
                print(f"Pre-computing {self.max_rank} nearest neighbors for {n_samples} training samples...")
            self._training_neighbors_, self._training_distances_ = self.index_.query(X, k=self.max_rank + 1)
            # Exclude self (first neighbor)
            self._training_neighbors_ = self._training_neighbors_[:, 1:]
            self._training_distances_ = self._training_distances_[:, 1:]
        
        # Store training data for on-demand queries (necessary even though PyNNDescent has _raw_data
        # because it may internally transform the data, causing different query results)
        self._training_data_ = X  # Already checked to be correct dtype by check_array
        # Store training data size for later reference
        self._n_training_samples_ = n_samples

        # Compute outlier scores for training data
        if self.verbose:
            print("Computing outlier scores...")

        self.outlier_scores_ = self._compute_scores(X, is_training=True)

        return self

    def _compute_scores(self, X: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Compute outlier scores for given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to score.
        is_training : bool, default=False
            Whether X is the training data (allows proper reverse rank computation).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Outlier scores (higher = more outlier).
        """
        n_samples = X.shape[0]
        kernel_func = self._get_kernel_function()

        # Get n_neighbors-nearest neighbors for each point from the training index
        knn_indices, knn_distances = self.index_.query(X, k=self.n_neighbors + 1)
        
        # Exclude self only if point is in training data
        if is_training:
            knn_indices = knn_indices[:, 1:]  # Skip first neighbor (self)
        else:
            knn_indices = knn_indices[:, :self.n_neighbors]  # Take first n_neighbors neighbors

        # Compute density scores
        density_scores = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            # For each of the n_neighbors neighbors, find where point i appears in their max_rank-NN list
            neighbor_indices = knn_indices[i]
            reverse_ranks = np.zeros(self.n_neighbors, dtype=np.float64)

            for j, neighbor_idx in enumerate(neighbor_indices):
                # Get the distance from this neighbor to point i
                dist_to_point = knn_distances[i, j if not is_training else j + 1]
                
                # Get this neighbor's max_rank nearest neighbors and distances
                if hasattr(self, '_training_neighbors_'):
                    # Use pre-computed data (fast path - no queries needed!)
                    neighbor_nn_indices = self._training_neighbors_[neighbor_idx]
                    neighbor_nn_dists = self._training_distances_[neighbor_idx]
                else:
                    # Query on-demand (memory-efficient path)
                    neighbor_nn_result, neighbor_nn_dists_result = self.index_.query(
                        self._training_data_[neighbor_idx:neighbor_idx+1], 
                        k=self.max_rank + 1
                    )
                    neighbor_nn_indices = neighbor_nn_result[0, 1:]  # Exclude self
                    neighbor_nn_dists = neighbor_nn_dists_result[0, 1:]  # Exclude self
                
                if is_training:
                    # For training data, search by index in neighbor's NN list
                    rank = np.where(neighbor_nn_indices == i)[0]
                    if len(rank) > 0:
                        reverse_ranks[j] = rank[0] + 1  # 1-indexed
                    else:
                        reverse_ranks[j] = self.max_rank  # Cap at max_rank
                else:
                    # For test data, find rank by distance comparison
                    # Count how many of the neighbor's NNs are closer than point i
                    rank = np.sum(neighbor_nn_dists < dist_to_point) + 1  # 1-indexed
                    reverse_ranks[j] = min(rank, self.max_rank)  # Cap at max_rank

            # Apply kernel and sum to get density
            kernel_values = kernel_func(reverse_ranks)
            density_scores[i] = np.sum(kernel_values)

        # Store density scores and compute density bounds
        if is_training:
            self.density_scores_ = density_scores
        
        # Calculate min and max density for normalization
        self.max_density_ = self.n_neighbors * kernel_func(np.array([1.0]))[0]
        self.min_density_ = self.n_neighbors * kernel_func(np.array([float(self.max_rank)]))[0]

        # Convert to outlier scores: higher density = lower outlier score
        # Normalize to [0, 1] range for interpretability
        outlier_scores = (self.max_density_ - density_scores) / (self.max_density_ - self.min_density_)

        return outlier_scores

    def score_samples(self, X):
        """
        Compute outlier scores for samples.

        Higher scores indicate more likely outliers.
        
        **Note on scoring new samples:**  
        For new test samples not in the training set, RankOD computes reverse ranks
        based on distance comparisons: for each test point's neighbors, the algorithm
        determines where the test point would rank among that neighbor's nearest neighbors
        by comparing distances. This provides consistent reverse k-NN scoring for both
        training and test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
            Samples to score. Can be a single sample (1D array) or multiple samples (2D array).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Outlier scores for samples, normalized to [0, 1] range.
            Higher scores indicate outliers (0=most normal, 1=most anomalous).
        """
        check_is_fitted(self, ["index_", "n_features_in_"])
        
        # Handle single sample (1D array)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X = check_array(X, accept_sparse=False, dtype=self.dtype)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but RankOD was fitted with {self.n_features_in_} features."
            )

        # Check if this is the training data (quick heuristic)
        if hasattr(self, 'outlier_scores_') and X.shape[0] == len(self.outlier_scores_):
            # Try to detect if this is training data by checking first few points
            test_indices, _ = self.index_.query(X[:min(5, len(X))], k=1)
            if np.all(test_indices[:, 0] == np.arange(min(5, len(X)))):
                # This appears to be training data, return cached scores
                return self.outlier_scores_
        
        # For new test data, compute using proper reverse k-NN with distance comparisons
        return self._compute_scores(X, is_training=False)

    def predict(self, X, contamination: Optional[float] = None):
        """
        Predict outliers in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
            Samples to predict. Can be a single sample (1D array) or multiple samples (2D array).

        contamination : float, optional
            Expected proportion of outliers in the dataset.
            Used to set the threshold for binary classification.
            If None, uses the contamination value set during initialization.
            Must be in the range (0, 0.5].

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels: -1 for outliers, 1 for inliers.
        """
        if contamination is None:
            contamination = self.contamination
        
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))

        # Higher scores are outliers
        predictions = np.ones(len(scores), dtype=int)
        predictions[scores >= threshold] = -1

        return predictions

    def fit_predict(self, X, y=None, contamination: Optional[float] = None):
        """
        Fit the detector and predict outliers on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for sklearn compatibility.

        contamination : float, optional
            Expected proportion of outliers in the dataset.
            If None, uses the contamination value set during initialization.
            Must be in the range (0, 0.5].

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels: -1 for outliers, 1 for inliers.
        """
        return self.fit(X, y).predict(X, contamination=contamination)
