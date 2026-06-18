"""
RankOD: Rank-based Outlier Detection using Reverse k-NN Density Estimation.

This module implements the core RankOD algorithm with scikit-learn compatible API.
"""

from collections.abc import Callable

import numpy as np
from numba import njit, prange
from numba.typed import Dict
from pynndescent import NNDescent
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted, validate_data


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


@njit  # TODO cache=True breaks something?
def build_row_map(neighbors):
    # This needs to be njitted to get a numba Typed dict
    row_map = {neighbor: row for row, neighbor in enumerate(neighbors)}
    return row_map


@njit(cache=True, parallel=True)
def compute_reverse_ranks(
    nn_indices: np.ndarray,
    nn_distances: np.ndarray,
    neighbor_nn_distances: np.ndarray,
    row_map: dict | None = Dict.empty(key_type=np.int64, value_type=np.int64),
):
    """
    Compute the reverse ranks (up to max rank) of each nearest neighbor.

    Parameters
    ----------
    nn_indices : np.ndarray
        n_samples x n_neighbors array of nearest neighbors in the training set
    nn_distances : np.ndarray
        Array of distances to the neighbor at the same index in the nn_indices array.
    neighbor_nn_distances : np.ndarray
        Array of distances to a training point nearest neighbors
    neighbor_index : dict
        Dictionary that maps a neighbor id to its row in neighbor_nn_indices and neighbor_nn_distances

    Returns
    -------
    np.ndarray
        Reverse ranks
    """
    reverse_ranks = np.empty_like(nn_indices)
    for i in prange(nn_indices.shape[0]):
        for j in prange(nn_indices.shape[1]):
            neighbor = nn_indices[i, j]
            dist = nn_distances[i, j]
            if row_map is None:
                neighbor_row = neighbor
            else:
                neighbor_row = row_map[neighbor]
            # Break ties in favour of lower rank
            # Also properly gets rank of training data
            rank = np.searchsorted(neighbor_nn_distances[neighbor_row], dist) + 1
            reverse_ranks[i, j] = rank
    return reverse_ranks


class RankOD(OutlierMixin, BaseEstimator):
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

    reverse_scores : bool, default=False
        Flag to reverse the scores so that higher scores are more likely to be outliers.

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
        reverse_scores: bool = False,
        precompute_neighbors: bool = False,
        dtype=np.float64,
        kernel: str | Callable = "harmonic",
        kernel_params: dict | None = {},
        metric: str = "euclidean",
        metric_kwds: dict | None = {},
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.max_rank = max_rank
        self.contamination = contamination
        self.reverse_scores = reverse_scores
        self.precompute_neighbors = precompute_neighbors
        self.dtype = dtype  # Store as-is for sklearn compatibility
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

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
        X = validate_data(self, X, accept_sparse=False, dtype=self.dtype, reset=True)
        n_samples, n_features = X.shape

        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must be less than n_samples={n_samples}"
            )

        # Build nearest neighbor index
        if self.verbose:
            print(
                f"Building nearest neighbor index with n_neighbors={self.n_neighbors}..."
            )

        self.index_ = NNDescent(
            X,
            metric=self.metric,
            metric_kwds=self.metric_kwds,
            n_neighbors=max(
                self.n_neighbors + 1, self.max_rank + 1
            ),  # +1 to exclude self
            # TODO by taking max are we always storing up to max rank neighbors in the index?
            # Does this defeat the precompute option
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Optionally pre-compute and store training data neighbors for reverse rank computation
        if self.precompute_neighbors:
            if self.verbose:
                print(
                    f"Pre-computing {self.max_rank} nearest neighbors for {n_samples} training samples..."
                )
            # TODO can we just read the n_neighbor graph here
            self._training_neighbors_, self._training_distances_ = self.index_.query(
                X, k=self.max_rank + 1
            )
            # Exclude self (first neighbor)
            self._training_neighbors_ = self._training_neighbors_[:, 1:]
            self._training_distances_ = self._training_distances_[:, 1:]

        # Store training data for on-demand queries (necessary even though PyNNDescent has _raw_data
        # because it may internally transform the data, causing different query results)
        self._training_data_ = X  # Already checked to be correct dtype by validate_data
        # Store training data size for later reference
        self._n_training_samples_ = n_samples

        # Compute outlier scores for training data
        if self.verbose:
            print("Computing outlier scores...")

        # sklearn default is inlier scores
        self.outlier_scores_ = self._compute_scores(X, is_training=True)
        # Save offset
        self.offset_ = self._compute_offset(self.outlier_scores_)

        return self

    def score_samples(self, X):
        """
        Compute outlier scores for samples.

        Smaller scores indicate more likely outliers.

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
            Outlier scores for samples. Higher scores indicate outliers (0=most anomalous, 1=most normal).
        """
        check_is_fitted(self, ["index_", "n_features_in_"])
        # Handle single sample (1D array)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, accept_sparse=False, dtype=self.dtype, reset=False)

        # TODO check this works, add flag to override and recompute
        # Check if this is the training data (quick heuristic)
        if hasattr(self, "outlier_scores_") and X.shape[0] == len(self.outlier_scores_):
            # Try to detect if this is training data by checking first few points
            test_indices, _ = self.index_.query(X[: min(5, len(X))], k=1)
            if np.all(test_indices[:, 0] == np.arange(min(5, len(X)))):
                # This appears to be training data, return cached scores
                return self.outlier_scores_

        # For new test data, compute using proper reverse k-NN with distance comparisons
        outlier_scores = self._compute_scores(X, is_training=False)
        return outlier_scores

    def decision_function(self, X, contamination: float | None = None):
        """
        Returns a shifted copy of the scores such that non-positive scores are outliers.
        """
        # TODO implement other options (possibly using a learned global threshold) to
        # rescale scores so negatives are outliers
        check_is_fitted(self, ["offset_", "index_", "n_features_in_"])
        scores = self.score_samples(X)
        if contamination is not None:
            offset = self._compute_offset(scores, contamination=contamination)
        else:
            offset = self.offset_
        translated_scores = scores - offset
        return translated_scores

    def predict(self, X, contamination: float | None = None):
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
        decision_scores = self.decision_function(X, contamination=contamination)
        predictions = np.full_like(decision_scores, 1, dtype="int")
        outliers = (decision_scores >= 0) if self.reverse_scores else (decision_scores <= 0)
        predictions[outliers] = -1
        return predictions

    def fit_predict(self, X, y=None):
        """
        Fit the detector and predict outliers on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels: -1 for outliers, 1 for inliers.
        """
        self.fit(X)
        prediction = np.full_like(self.outlier_scores_, 1, dtype="int")
        outliers = self.outlier_scores_ >= self.offset_ if self.reverse_scores else self.outlier_scores_ <= self.offset_
        prediction[outliers] = -1
        return prediction

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
            Outlier scores (lower = more outlier).
        """
        kernel_func = self._get_kernel_function()

        # TODO batch for large inputs
        # TODO better name for knn_indices
        # Get n_neighbors-nearest neighbors for each point from the training index
        knn_indices, knn_distances = self.index_.query(X, k=self.n_neighbors + 1)

        if is_training:
            knn_indices = knn_indices[:, 1:]  # Exclude self
            knn_distances = knn_distances[:, 1:]  # Exclude self
        else:
            knn_indices = knn_indices[
                :, : self.n_neighbors
            ]  # Take first n_neighbors neighbors
            knn_distances = knn_distances[
                :, : self.n_neighbors
            ]  # Take first n_neighbors neighbors

        # Compute the nearest neighbors of the ranked neighbors
        if hasattr(self, "_training_distances_"):
            neighbor_nn_distances = self._training_distances_
            row_map = None
        else:
            nns = np.unique(knn_indices.flatten())
            row_map = build_row_map(nns)
            _, neighbor_nn_distances = self.index_.query(
                self._training_data_[nns], k=self.max_rank + 1
            )
            neighbor_nn_distances = neighbor_nn_distances[:, 1:]  # Exclude self

        reverse_ranks = compute_reverse_ranks(
            knn_indices, knn_distances, neighbor_nn_distances, row_map=row_map
        )

        # TODO factor out and implement more methods
        kernel_values = kernel_func(reverse_ranks)
        raw_scores = np.sum(kernel_values, axis=1)

        # Normalize to [0, 1] range for interpretability
        if is_training:
            self.max_raw_score_ = self.n_neighbors * kernel_func(np.array([1.0]))[0]
            self.min_raw_score_ = (
                self.n_neighbors * kernel_func(np.array([float(self.max_rank)]))[0]
            )
        outlier_scores = (raw_scores - self.min_raw_score_) / (
            self.max_raw_score_ - self.min_raw_score_
        )

        if self.reverse_scores:
            outlier_scores = 1 - outlier_scores

        return outlier_scores

    def _compute_offset(self, scores, contamination: float | None = None) -> float:
        """Get the decision threshold"""
        if contamination is None:
            contamination = self.contamination
        percentile_threshold = (
            (1 - contamination) if self.reverse_scores else contamination
        )
        percentile_threshold *= 100
        offset = np.percentile(scores, percentile_threshold)
        return offset
