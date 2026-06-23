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


def row_normalize(X):
    """
    Row normalize 2d numpy or scipy arrays.
    """
    # Easier to transpose since stuff operates columnwise by default
    X = X.transpose()
    X = X / X.sum(axis=0)
    X = X.transpose()
    return X


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


@njit(cache=True)
def ecdf_scores(
    training_raw_scores,
    raw_scores,
):
    """
    Transform raw scores into ECDF scores. Do not insert the
    raw scores before computing the ECDF. The ECDF value of
    each raw score is the proportion of training raw scores
    that are smaller.

    Parameters
    ----------
    training_raw_scores: np.ndarray
        A 1d array of the raw scores from the training data.

    raw_scores: np.ndarray
        A 1d array of raw scores for which to compute the ecdfs.

    """
    sorted_scores = np.sort(training_raw_scores)
    insert_index = np.searchsorted(sorted_scores, raw_scores)
    ecdfs = insert_index / len(training_raw_scores)
    return ecdfs


@njit(cache=True)
def local_ecdf_scores(knn_indices, training_raw_scores, raw_scores):
    """
    Transform raw scores into local ECDF scores. Do not insert the
    raw scores before computing the ECDF.

    Parameters
    ----------
    knn_indices : np.ndarray
        Each row denotes the indices of the nearest neighbors of the
        score in the correspoing raw_scores array. Used to get the
        local scores from training_raw_scores.

    training_raw_scores : np.ndarray
        A 1d array of the raw scores from the training data.

    raw_scores : np.ndarray
        A 1d array of the raw scores from the predict data.
    """
    ecdfs = np.empty_like(raw_scores, dtype=np.float64)
    for i in prange(len(raw_scores)):
        raw_score = raw_scores[i : i + 1]  # Return as len 1 array
        local_raw_scores = training_raw_scores[knn_indices[i]]
        ecdfs[i] = ecdf_scores(local_raw_scores, raw_score)[0]
    return ecdfs


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

    contamination : float, default=0.1
        Expected proportion of outliers in the dataset.
        Used to set the threshold for binary classification in predict().

    score : str, default="rank"
        Which score to compute.

        - "rank": raw scores are the average kerneled reverse rank.
        - "sun": raw scores are the distance to the n_neighbors nearest
          neighbors after L2 normalization.

    calibration : str | None
        Method for calibrating raw scores.

        - "local": Returned scores are calibrated locally using the empirical
          cumulative distribution function of the nearest n_neighbors.
        - "global": Returned scores are calibrated using the empirical
          cumulative distribution function of all the training data.
        - "raw" or None: Return the raw scores.

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

    kernel : {'harmonic', 'inverse_sqrt', 'gaussian'} or callable, default='inverse_sqrt'
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
        Lower values indicate outliers.

    offset_ : float
        Learned threshold to set the specified proportion of training data
        (as defined by the contamination parameter) to be outliers. Calling
        predict on new data uses the learned threshold to make a decision.

    max_raw_score_ : float
        Maximum possible raw density score. Used for score normalization.

    min_raw_score_ : float
        Minimum possible raw density score. Used for score normalization.

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
        score: str = "rank",
        calibration: str | None = None,
        reverse_scores: bool = False,
        precompute_neighbors: bool = False,
        dtype=np.float64,
        kernel: str | Callable = "inverse_sqrt",
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
        self.score = score
        self.calibration = calibration
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
        if self.score == "sun":
            if self.metric != "euclidean":
                raise ValueError(
                    "Sun score requires euclidean metric. Try setting score to rank or metric to euclidean."
                )
            X = row_normalize(X)
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
            n_neighbors=max(self.n_neighbors, self.max_rank) + 1,  # +1 to exclude self
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        training_neighbors, training_distances = self.index_.neighbor_graph
        training_neighbors = training_neighbors[:, 1:]  # Exclude self
        training_distances = training_distances[:, 1:]  # Exclude self
        # Optionally store training data neighbors for reverse rank computation
        if self.precompute_neighbors:
            if self.verbose:
                print(
                    f"Pre-computing {self.max_rank} nearest neighbors for {n_samples} training samples..."
                )
            self._training_neighbors_ = training_neighbors
            self._training_distances_ = training_distances

        # Store training data for on-demand queries (necessary even though PyNNDescent has _raw_data
        # because it may internally transform the data, causing different query results)
        self._training_data_ = X  # Already checked to be correct dtype by validate_data
        # Store training data size for later reference
        self._n_training_samples_ = n_samples

        # Compute outlier scores for training data
        if self.verbose:
            print("Computing outlier scores...")

        # sklearn default is inlier scores
        self.outlier_scores_ = self._compute_scores(
            training_neighbors[:, : self.n_neighbors],
            training_distances[:, : self.n_neighbors],
            training_distances,
            is_training=True,
        )
        # Save offset
        self.offset_ = self._compute_offset(self.outlier_scores_)

        return self

    def score_samples(self, X):
        """
        Compute outlier scores for samples. Smaller scores indicate more likely outliers.

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
        if self.score == "sun":
            X = row_normalize(X)
        # Check if this is the training data (quick heuristic)
        if hasattr(self, "outlier_scores_") and X.shape[0] == len(self.outlier_scores_):
            # Try to detect if this is training data by checking first few points
            n_rows_to_check = min(5, X.shape[0], self._training_data_.shape[0])
            if X.shape[0] == self._training_data_.shape[0] and np.all(
                np.asarray(X[n_rows_to_check, :])
                == np.asarray(self._training_data_[n_rows_to_check, :])
            ):
                # This appears to be training data, return cached scores
                return self.outlier_scores_

        # For new test data, compute using proper reverse k-NN with distance comparisons
        knn_indices, knn_distances = self.index_.query(X, k=self.n_neighbors)
        outlier_scores = self._compute_scores(
            knn_indices, knn_distances, is_training=False
        )
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
        outliers = (
            (decision_scores >= 0) if self.reverse_scores else (decision_scores <= 0)
        )
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
        outliers = (
            self.outlier_scores_ >= self.offset_
            if self.reverse_scores
            else self.outlier_scores_ <= self.offset_
        )
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

    def _compute_scores(
        self,
        knn_indices: np.ndarray,
        knn_distances: np.ndarray,
        neighbor_nn_distances: np.ndarray | None = None,
        is_training: bool = False,
    ) -> np.ndarray:
        """
        Compute outlier scores for given data.

        Parameters
        ----------
        knn_indices: np.ndarray
            Each row is a list of indices of the nearest neighbors.

        knn_distances: np.ndarray
            Each row is a list of distances to the nearest neighbors.

        neighbor_nn_distances: np.ndarray | None = None
            Each row is a list of distances to the training points nearest neighbors.
            May not be readily available but can be computed on demand.

        is_training : bool, default=False
            Whether X is the training data (allows proper reverse rank computation).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Outlier scores (lower = more outlier).

        """
        # Get n_neighbors-nearest neighbors for each point from the training index
        knn_indices = knn_indices[
            :, : self.n_neighbors
        ]  # Take first n_neighbors neighbors
        knn_distances = knn_distances[
            :, : self.n_neighbors
        ]  # Take first n_neighbors neighbors

        # Compute Scores
        if self.score == "rank":
            raw_scores = self._compute_rank_scores(
                knn_indices, knn_distances, neighbor_nn_distances
            )
        elif self.score == "sun":
            raw_scores = self._compute_sun_scores(knn_indices, knn_distances)
        else:
            raise ValueError(f"Score must be one of 'rank' or 'sun'. Got {self.score}")

        # Calibrate Scores
        if self.calibration == "raw" or self.calibration is None:
            if is_training:
                self._max_raw_score_ = np.max(raw_scores)
                self._min_raw_score_ = np.min(raw_scores)
            outlier_scores = (raw_scores - self._min_raw_score_) / (
                self._max_raw_score_ - self._min_raw_score_
            )
        elif self.calibration == "local":
            if is_training:
                self._training_raw_scores_ = raw_scores
            outlier_scores = local_ecdf_scores(
                knn_indices, self._training_raw_scores_, raw_scores
            )
        elif self.calibration == "global":
            if is_training:
                self._training_raw_scores_ = raw_scores
            outlier_scores = ecdf_scores(self._training_raw_scores_, raw_scores)

        else:
            raise ValueError(
                f"Calibration must be 'local', 'global', 'raw' or None. Got {self.calibration}"
            )

        if self.reverse_scores:
            outlier_scores = 1 - outlier_scores

        return outlier_scores

    def _compute_sun_scores(
        self,
        knn_indices,
        knn_distances,
    ):
        scores = knn_distances[:, -1]  # just the distance to knn
        # euclidean distance between L2 normed vectors
        # are in [0,2], so divide by 2 to normalize.
        # and reverse so bigger distance is smaller score
        scores = 1 - (scores / 2)
        return scores

    def _compute_rank_scores(
        self,
        knn_indices,
        knn_distances,
        neighbor_nn_distances=None,
    ):
        kernel_func = self._get_kernel_function()
        # Compute the nearest neighbors of the ranked neighbors
        if neighbor_nn_distances is not None:
            row_map = None
        elif hasattr(self, "_training_distances_"):
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
        kernel_values = kernel_func(reverse_ranks)
        scores = np.mean(kernel_values, axis=1)
        return scores

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
