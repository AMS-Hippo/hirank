"""
KNNOD: a compact family of nearest-neighbour outlier detectors.

The class in this module deliberately mirrors :class:`hirank.rankod.RankOD`
without changing its implementation or execution path.  It provides four
mode-sized estimators: Sun et al.'s normalized KNN score, empirical-CDF
calibration, reverse-rank aggregation (including RBDA), and a global
threshold distance-fraction score.
"""

import inspect
import warnings
from collections.abc import Callable

import numpy as np
from pynndescent import NNDescent
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, validate_data

from hirank._knnod_numba import (
    distance_fraction_inlier_scores,
    local_ecdf_inlier_probabilities,
    rank_raw_scores_from_graph,
    rank_raw_scores_from_precomputed,
    reverse_ranks_from_graph,
    reverse_ranks_from_precomputed,
)


class _ExactNNIndex:
    """Small exact index exposing the subset of the NNDescent API we use."""

    def __init__(
        self,
        X: np.ndarray,
        n_neighbors: int,
        metric: str | Callable = "euclidean",
        metric_kwds: dict | None = None,
        n_jobs: int = -1,
    ):
        self._X = np.asarray(X)
        self.n_neighbors = min(int(n_neighbors), self._X.shape[0])
        self._index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="brute",
            metric=metric,
            metric_params=metric_kwds or None,
            n_jobs=n_jobs,
        ).fit(self._X)
        distances, indices = self._index.kneighbors(
            self._X,
            n_neighbors=self.n_neighbors,
            return_distance=True,
        )
        self.neighbor_graph = (
            indices.astype(np.int64),
            distances.astype(np.float64),
        )

    def query(self, X, k=10, epsilon=0.1):
        del epsilon
        k = min(int(k), self._X.shape[0])
        distances, indices = self._index.kneighbors(
            np.asarray(X), n_neighbors=k, return_distance=True
        )
        return indices.astype(np.int64), distances.astype(np.float64)


class KNNOD(OutlierMixin, BaseEstimator):
    """Nearest-neighbour outlier detection with mode-sized fit structures.

    ``KNNOD`` follows the public API and score convention of ``RankOD``:
    by default, larger scores indicate more inlying observations.  It is a
    separate class so that adding modes does not alter or slow any existing
    ``RankOD`` path.

    Parameters
    ----------
    n_neighbors : int, default=15
        Base neighbourhood size ``k``.

    max_rank : int, default=100
        Reverse-rank depth for ``mode="rank"``.  In approximate mode, ranks
        beyond the common fitted depth are represented by one overflow value,
        ``effective_max_rank_ + 1``.

    contamination : float, default=0.1
        Expected outlier proportion used to fit ``offset_``.

    reverse_scores : bool, default=False
        If True, reverse the public score orientation so larger values indicate
        more outlying observations.  The default matches scikit-learn and
        ``RankOD``: larger values indicate inliers.

    mode : {"rank", "sun", "ecdf", "distance_fraction"}, default="rank"
        Scoring family.

        - ``"rank"`` aggregates reverse ranks.  Its default method is RBDA.
        - ``"sun"`` applies L2 normalization and uses kth-neighbour distance.
        - ``"ecdf"`` calibrates a distance statistic by a global or cached
          local empirical upper-tail CDF.
        - ``"distance_fraction"`` fits a global threshold and counts the
          fraction of query-neighbour distances above it.

    method : str, callable or None, default=None
        Mode-specific method.  ``None`` selects ``"rbda"`` for rank,
        ``"kth"`` for Sun/ECDF, and ``"fraction"`` for distance fraction.

        Built-in rank methods are ``"rbda"``, ``"harmonic"``,
        ``"inverse_sqrt"``, and ``"gaussian"``.  A rank callable receives a
        one-dimensional rank vector and may return either a scalar aggregation
        or one value per rank.  It must be monotone; orientation is inferred by
        comparing all-rank-1 and all-maximum-rank inputs.

        An ECDF callable receives one row of sorted neighbour distances and
        must return a scalar for which larger values are more outlying.

        A distance-fraction callable receives ``(distances, gamma)`` and must
        return scalar outlier evidence in ``[0, 1]``.  Built-in strings use
        Numba-accelerated paths; callables use a Python row loop.

    method_params : dict or None, default=None
        Keyword arguments for a callable method, or ``sigma`` for the Gaussian
        rank method.

    calibration : {"global", "local"}, default="global"
        ECDF calibration scope.  Local calibration compares against cached fit
        scores of the query's neighbours; it does not virtually insert and
        rescore the query.

    calibration_neighbors : int, default=50
        Local ECDF neighbourhood size ``K``.

    distance_quantile : float, default=0.75
        Quantile used to fit ``gamma`` in distance-fraction mode.

    threshold_source : {"pooled", "kth"}, default="pooled"
        Population used to fit the distance-fraction threshold. ``"pooled"``
        uses all first-k non-self fit distances; ``"kth"`` uses one kth radius
        per fit observation.  They require the same fitted neighbour graph.

    exact : bool, default=False
        If False, use PyNNDescent.  If True, use exact brute-force neighbours.
        Exact rank mode stores all reference distances so RBDA ranks are ranks
        among all other fit points, as in the paper.  This is quadratic in fit
        memory and should be reserved for small datasets.

    rank_reference : {"graph", "query"}, default="graph"
        Source of the reference-neighbour distance rows used to compute reverse
        ranks in ``mode="rank"``.  ``"graph"`` searches the fitted
        ``NNDescent.neighbor_graph`` rows and is the fast default.  ``"query"``
        obtains those rows with ``index_.query`` and closely follows legacy
        ``RankOD``.  With ``precompute_neighbors=True``, the selected rows are
        queried once during ``fit`` and cached.

    rank_graph_multiplier : float, default=1.0
        Multiplier applied to the minimum PyNNDescent graph width in approximate
        rank mode.  Values above one retain a wider fitted graph, which may
        improve the recall of graph-referenced reverse ranks at extra fit-time
        and memory cost.  It is ignored outside approximate rank mode.

    include_ties : bool, default=False
        If True, include every kth-distance tie visible in the returned
        candidate row.  This opts into the slower variable-width neighbour
        path.  Approximate mode can include only ties returned by the ANN
        query/graph; exact mode sees the complete reference row.

    tie_buffer : int, default=0
        Extra ANN candidates requested for best-effort boundary-tie inclusion.
        Increase this only when ties are plausible.  Rank queries always retain
        the legacy ``k + 1`` search cushion independently of this setting.

    tie_tolerance : float, default=0.0
        Relative/absolute tolerance added to the kth distance when identifying
        visible ties.  Even at zero, one floating-point spacing is allowed.

    precompute_neighbors : bool, default=False
        For rank mode, store dense non-self reference-neighbour rows.  This
        mirrors ``RankOD``'s memory/speed option.  With
        ``rank_reference="graph"`` the rows come from the fitted graph; with
        ``rank_reference="query"`` they are obtained once through
        ``index_.query``.  When False, the selected reference source is used on
        demand.

    dtype : numpy dtype, default=np.float64
        Input storage dtype.

    metric : str or callable, default="euclidean"
        Distance metric.  Sun mode requires Euclidean distance after internal
        L2 normalization.

    metric_kwds : dict or None, default=None
        Metric keyword arguments.

    query_epsilon : float or None, default=None
        Approximation parameter passed to ``NNDescent.query``. ``None``
        leaves the argument unspecified and therefore uses the backend's own
        default, exactly as legacy ``RankOD`` does. For PyNNDescent 0.6 this is
        ``0.1``. Supply an explicit nonnegative float to trade query speed
        against ANN recall.

    n_jobs : int, default=-1
        Number of jobs used by the neighbour backend.

    random_state : int or None, default=None
        Random seed passed to PyNNDescent.

    verbose : bool, default=False
        Print fit-stage progress.

    Attributes
    ----------
    outlier_scores_ : ndarray of shape (n_fit_rows,)
        Fit scores in the public orientation.  By default, lower values are
        more outlying.

    offset_ : float
        Contamination-derived decision threshold.

    effective_n_neighbors_ : int
        Common fit-time ``k`` after any required reduction.

    effective_max_rank_ : int
        Common reverse-rank depth in rank mode.

    effective_query_epsilon_ : float or None
        Query epsilon actually selected by the neighbour backend. This is
        inferred from the backend signature when ``query_epsilon=None``.

    gamma_ : float
        Fitted global threshold in distance-fraction mode.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        max_rank: int = 100,
        contamination: float = 0.1,
        reverse_scores: bool = False,
        mode: str = "rank",
        method: str | Callable | None = None,
        method_params: dict | None = None,
        calibration: str = "global",
        calibration_neighbors: int = 50,
        distance_quantile: float = 0.75,
        threshold_source: str = "pooled",
        exact: bool = False,
        rank_reference: str = "graph",
        rank_graph_multiplier: float = 1.0,
        include_ties: bool = False,
        tie_buffer: int = 0,
        tie_tolerance: float = 0.0,
        precompute_neighbors: bool = False,
        dtype=np.float64,
        metric: str | Callable = "euclidean",
        metric_kwds: dict | None = None,
        query_epsilon: float | None = None,
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.max_rank = max_rank
        self.contamination = contamination
        self.reverse_scores = reverse_scores
        self.mode = mode
        self.method = method
        self.method_params = method_params
        self.calibration = calibration
        self.calibration_neighbors = calibration_neighbors
        self.distance_quantile = distance_quantile
        self.threshold_source = threshold_source
        self.exact = exact
        self.rank_reference = rank_reference
        self.rank_graph_multiplier = rank_graph_multiplier
        self.include_ties = include_ties
        self.tie_buffer = tie_buffer
        self.tie_tolerance = tie_tolerance
        self.precompute_neighbors = precompute_neighbors
        self.dtype = dtype
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.query_epsilon = query_epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the selected KNN outlier detector."""
        del y
        self._clear_fitted_state()
        X = validate_data(self, X, accept_sparse=False, dtype=self.dtype, reset=True)
        self._validate_parameters()
        self._reset_warning_tracking()

        self._n_original_samples_ = X.shape[0]
        self._fit_input_shape_ = X.shape
        X_index = self._prepare_fit_data(X)
        n_reference = X_index.shape[0]
        if n_reference < 2:
            raise ValueError("At least two usable fit rows are required; got 1 sample.")

        self._training_data_ = X_index
        self._n_training_samples_ = n_reference
        self._set_initial_effective_sizes(n_reference)

        if self.verbose:
            backend = "exact" if self.exact else "PyNNDescent"
            print(
                f"Building {backend} index for mode={self.mode_!r} "
                f"with n_neighbors={self.effective_n_neighbors_}..."
            )
        self._fit_index(X_index)

        if self.mode_ == "rank" and self.rank_reference_ == "query":
            # Legacy RankOD obtains both the fit-point neighbourhoods and the
            # reverse-rank reference rows through index_.query.  Keep that
            # path explicit so users can trade speed for closer compatibility.
            score_indices, score_distances, score_counts = (
                self._fit_rank_query_neighbor_arrays(
                    self.effective_n_neighbors_, context="fit n_neighbors"
                )
            )
            self._prepare_query_rank_reference_rows(
                score_indices, score_counts
            )
        else:
            self._reduce_sizes_from_graph()
            if self.mode_ == "rank" and self.precompute_neighbors:
                self._precompute_rank_rows()
            score_indices, score_distances, score_counts = self._fit_neighbor_arrays(
                self.effective_n_neighbors_, context="fit n_neighbors"
            )

        if self.verbose:
            print("Computing outlier scores...")

        if self.mode_ == "sun":
            valid_scores = self._sun_scores(score_distances, score_counts)
        elif self.mode_ == "ecdf":
            valid_scores = self._fit_ecdf_scores(
                score_indices, score_distances, score_counts
            )
        elif self.mode_ == "rank":
            valid_scores = self._rank_scores(
                score_indices, score_distances, score_counts
            )
        else:
            self._fit_distance_threshold(score_distances, score_counts)
            valid_scores = self._distance_fraction_scores(
                score_distances, score_counts
            )

        # Query-referenced rank rows are retained only when explicitly
        # requested through precompute_neighbors; otherwise the fit-time table
        # is transient, matching RankOD's on-demand memory behavior.
        if hasattr(self, "_fit_query_reference_distances_"):
            del self._fit_query_reference_distances_
        if hasattr(self, "_fit_query_reference_ids_"):
            del self._fit_query_reference_ids_

        scores = np.zeros(self._n_original_samples_, dtype=np.float64)
        scores[self.fit_indices_] = valid_scores
        if self.reverse_scores:
            scores = 1.0 - scores
        self.outlier_scores_ = scores
        self.offset_ = self._compute_offset(scores)
        return self

    def score_samples(self, X):
        """Compute scores; by default, smaller values indicate outliers."""
        check_is_fitted(self, ["index_", "n_features_in_", "outlier_scores_"])
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, accept_sparse=False, dtype=self.dtype, reset=False)

        if self._looks_like_fit_data(X):
            return self.outlier_scores_

        X_query, valid_mask = self._prepare_query_data(X)
        high_inlier_scores = np.zeros(X.shape[0], dtype=np.float64)
        if X_query.shape[0] > 0:
            query_base = self.effective_n_neighbors_
            if self.mode_ == "ecdf" and self.calibration == "local":
                query_base = max(query_base, self.effective_calibration_neighbors_)

            indices, distances, counts = self._query_neighbor_arrays(
                X_query, query_base, context="query neighbours"
            )
            use_matched_fit_scores = not self._uses_fixed_width_fast_path()
            if use_matched_fit_scores:
                matched_rows, matched_scores = self._matched_fit_scores(
                    X_query, indices, distances
                )
            if (
                self._uses_fixed_width_fast_path()
                and int(query_base) == int(self.effective_n_neighbors_)
            ):
                score_indices, score_distances, score_counts = (
                    indices,
                    distances,
                    counts,
                )
            else:
                (
                    score_indices,
                    score_distances,
                    score_counts,
                ) = self._truncate_neighbor_arrays(
                    indices,
                    distances,
                    counts,
                    self.effective_n_neighbors_,
                    context="query n_neighbors",
                )

            if self.mode_ == "sun":
                valid_scores = self._sun_scores(score_distances, score_counts)
            elif self.mode_ == "rank":
                valid_scores = self._rank_scores(
                    score_indices, score_distances, score_counts
                )
            elif self.mode_ == "distance_fraction":
                valid_scores = self._distance_fraction_scores(
                    score_distances, score_counts
                )
            else:
                raw = self._distance_raw_scores(score_distances, score_counts)
                if self.calibration == "global":
                    valid_scores = self._global_ecdf_probabilities(raw)
                else:
                    if (
                        self._uses_fixed_width_fast_path()
                        and int(query_base)
                        == int(self.effective_calibration_neighbors_)
                    ):
                        cal_indices, cal_counts = indices, counts
                    else:
                        cal_indices, _, cal_counts = self._truncate_neighbor_arrays(
                            indices,
                            distances,
                            counts,
                            self.effective_calibration_neighbors_,
                            context="query calibration_neighbors",
                        )
                    valid_scores = local_ecdf_inlier_probabilities(
                        raw,
                        cal_indices,
                        cal_counts,
                        self.training_raw_scores_,
                    )

            if use_matched_fit_scores and np.any(matched_rows):
                valid_scores[matched_rows] = matched_scores[matched_rows]
            high_inlier_scores[valid_mask] = valid_scores

        if self.reverse_scores:
            high_inlier_scores = 1.0 - high_inlier_scores
        return high_inlier_scores

    def decision_function(self, X, contamination: float | None = None):
        """Shift scores so negative values are outliers by default."""
        check_is_fitted(self, ["offset_", "index_", "n_features_in_"])
        scores = self.score_samples(X)
        offset = (
            self._compute_offset(scores, contamination)
            if contamination is not None
            else self.offset_
        )
        return scores - offset

    def predict(self, X, contamination: float | None = None):
        """Predict labels: ``-1`` for outliers and ``1`` for inliers."""
        decision_scores = self.decision_function(X, contamination=contamination)
        predictions = np.full_like(decision_scores, 1, dtype="int")
        outliers = (
            decision_scores > 0 if self.reverse_scores else decision_scores < 0
        )
        predictions[outliers] = -1
        return predictions

    def fit_predict(self, X, y=None):
        """Fit the detector and predict labels for the fit data."""
        self.fit(X, y=y)
        predictions = np.full_like(self.outlier_scores_, 1, dtype="int")
        outliers = (
            self.outlier_scores_ > self.offset_
            if self.reverse_scores
            else self.outlier_scores_ < self.offset_
        )
        predictions[outliers] = -1
        return predictions

    def training_score_samples(self):
        """Return cached leave-one-out fit scores."""
        check_is_fitted(self, "outlier_scores_")
        return self.outlier_scores_

    # ------------------------------------------------------------------
    # Validation and fit setup
    # ------------------------------------------------------------------
    def _validate_parameters(self):
        aliases = {
            "rank": "rank",
            "sun": "sun",
            "sun22": "sun",
            "ecdf": "ecdf",
            "distance_fraction": "distance_fraction",
            "distance-fraction": "distance_fraction",
        }
        mode_key = str(self.mode).lower()
        if mode_key not in aliases:
            raise ValueError(
                "mode must be 'rank', 'sun', 'ecdf', or 'distance_fraction'."
            )
        self.mode_ = aliases[mode_key]
        if int(self.n_neighbors) != self.n_neighbors or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if int(self.max_rank) != self.max_rank or self.max_rank < 1:
            raise ValueError("max_rank must be a positive integer.")
        if not 0.0 < float(self.contamination) <= 0.5:
            raise ValueError("contamination must lie in (0, 0.5].")
        if self.calibration not in {"global", "local"}:
            raise ValueError("calibration must be 'global' or 'local'.")
        if (
            int(self.calibration_neighbors) != self.calibration_neighbors
            or self.calibration_neighbors < 1
        ):
            raise ValueError("calibration_neighbors must be a positive integer.")
        if not 0.0 <= float(self.distance_quantile) <= 1.0:
            raise ValueError("distance_quantile must lie in [0, 1].")
        if self.threshold_source not in {"pooled", "kth"}:
            raise ValueError("threshold_source must be 'pooled' or 'kth'.")
        if not isinstance(self.exact, (bool, np.bool_)):
            raise ValueError("exact must be boolean.")
        rank_reference = str(self.rank_reference).lower()
        if rank_reference not in {"graph", "query"}:
            raise ValueError("rank_reference must be 'graph' or 'query'.")
        self.rank_reference_ = rank_reference
        if (
            not np.isfinite(float(self.rank_graph_multiplier))
            or float(self.rank_graph_multiplier) < 1.0
        ):
            raise ValueError("rank_graph_multiplier must be finite and at least 1.")
        if not isinstance(self.include_ties, (bool, np.bool_)):
            raise ValueError("include_ties must be boolean.")
        if int(self.tie_buffer) != self.tie_buffer or self.tie_buffer < 0:
            raise ValueError("tie_buffer must be a nonnegative integer.")
        if float(self.tie_tolerance) < 0.0:
            raise ValueError("tie_tolerance must be nonnegative.")
        if not isinstance(self.precompute_neighbors, (bool, np.bool_)):
            raise ValueError("precompute_neighbors must be boolean.")
        if self.metric_kwds is not None and not isinstance(self.metric_kwds, dict):
            raise ValueError("metric_kwds must be a dict or None.")
        if self.method_params is not None and not isinstance(self.method_params, dict):
            raise ValueError("method_params must be a dict or None.")
        if self.query_epsilon is not None:
            if (
                not np.isfinite(float(self.query_epsilon))
                or float(self.query_epsilon) < 0.0
            ):
                raise ValueError(
                    "query_epsilon must be None or a finite nonnegative float."
                )
        if self.mode_ == "sun":
            if self.metric != "euclidean" or self.metric_kwds not in (None, {}):
                raise ValueError(
                    "mode='sun' requires metric='euclidean' and no metric_kwds."
                )

        self.method_ = self._resolve_method()
        if not callable(self.method_):
            allowed = {
                "rank": {"rbda", "harmonic", "inverse_sqrt", "gaussian"},
                "sun": {"kth"},
                "ecdf": {"kth"},
                "distance_fraction": {"fraction"},
            }[self.mode_]
            if self.method_ not in allowed:
                choices = ", ".join(sorted(allowed))
                raise ValueError(
                    f"Unknown method {self.method_!r} for mode={self.mode_!r}; "
                    f"expected one of {choices}, or a callable where supported."
                )
        elif self.mode_ == "sun":
            raise ValueError("mode='sun' does not accept a callable method.")

    def _resolve_method(self):
        if self.method is not None:
            return self.method
        return {
            "rank": "rbda",
            "sun": "kth",
            "ecdf": "kth",
            "distance_fraction": "fraction",
        }[self.mode_]

    def _prepare_fit_data(self, X):
        if self.mode_ != "sun":
            self.fit_indices_ = np.arange(X.shape[0], dtype=int)
            self.zero_norm_indices_ = np.array([], dtype=int)
            return X

        norms = np.linalg.norm(X, axis=1)
        valid = norms > 0.0
        self.fit_indices_ = np.flatnonzero(valid)
        self.zero_norm_indices_ = np.flatnonzero(~valid)
        if self.zero_norm_indices_.size:
            warnings.warn(
                f"KNNOD removed {self.zero_norm_indices_.size} zero-norm fit row(s) "
                "from the Sun-mode neighbour graph; their fit scores are set to 0.",
                RuntimeWarning,
                stacklevel=3,
            )
        return (X[valid] / norms[valid, None]).astype(self.dtype, copy=False)

    def _prepare_query_data(self, X):
        if self.mode_ != "sun":
            return X, np.ones(X.shape[0], dtype=bool)

        norms = np.linalg.norm(X, axis=1)
        valid = norms > 0.0
        if np.any(~valid):
            self._warn_once(
                "sun_query_zero",
                "KNNOD received zero-norm Sun-mode query row(s); their scores "
                "are set to 0 in the default high-is-inlier orientation.",
            )
        normalized = (X[valid] / norms[valid, None]).astype(self.dtype, copy=False)
        return normalized, valid

    def _set_initial_effective_sizes(self, n_reference):
        self.effective_n_neighbors_ = min(int(self.n_neighbors), n_reference - 1)
        if self.effective_n_neighbors_ < int(self.n_neighbors):
            self._record_reduction(
                "n_neighbors", int(self.n_neighbors), self.effective_n_neighbors_
            )

        self.effective_calibration_neighbors_ = min(
            int(self.calibration_neighbors), n_reference - 1
        )
        if (
            self.mode_ == "ecdf"
            and self.calibration == "local"
            and self.effective_calibration_neighbors_
            < int(self.calibration_neighbors)
        ):
            self._record_reduction(
                "calibration_neighbors",
                int(self.calibration_neighbors),
                self.effective_calibration_neighbors_,
            )

        if self.mode_ == "rank":
            if self.exact:
                # Paper-exact RBDA ranks each point among every other fit row;
                # max_rank is an ANN truncation control and is intentionally
                # ignored when exact=True.
                self.effective_max_rank_ = n_reference - 1
            else:
                self.effective_max_rank_ = min(int(self.max_rank), n_reference - 1)
                if self.effective_max_rank_ < int(self.max_rank):
                    self._record_reduction(
                        "max_rank", int(self.max_rank), self.effective_max_rank_
                    )
        else:
            self.effective_max_rank_ = None

    def _uses_fixed_width_fast_path(self):
        """Use the RankOD-like fixed-width path for ordinary ANN scoring.

        The default approximate, no-tie configuration assumes the backend
        returns complete sorted rows, just as legacy ``RankOD`` does.  Exact or
        tie-aware configurations retain the careful variable-width machinery.
        """
        return not self.exact and not self.include_ties

    def _uses_boundary_ties(self):
        if not self.include_ties:
            return False
        if self.mode_ in {"rank", "distance_fraction"}:
            return True
        if self.mode_ == "ecdf":
            return self.calibration == "local" or callable(self.method_)
        return False

    def _rank_query_extra_candidates(self):
        """Return extra rank-query candidates beyond the requested ``k``.

        Legacy RankOD asks PyNNDescent for ``k + 1`` candidates even for new
        query points and then keeps the first ``k``.  Preserve that small ANN
        search cushion independently of tie expansion.  When ties are enabled,
        ``tie_buffer`` can request a larger cushion.
        """
        tie_extra = int(self.tie_buffer) if self._uses_boundary_ties() else 0
        return max(1, tie_extra)

    def _required_graph_neighbors(self):
        required = self.effective_n_neighbors_
        if self.mode_ == "rank":
            required = max(required, self.effective_max_rank_)
        if self.mode_ == "ecdf" and self.calibration == "local":
            required = max(required, self.effective_calibration_neighbors_)
        return int(required)

    def _fit_index(self, X):
        required = self._required_graph_neighbors()
        n_reference = X.shape[0]

        uses_ties = self._uses_boundary_ties()
        if self.mode_ == "rank":
            minimum_width = max(
                int(self.effective_max_rank_),
                int(self.effective_n_neighbors_)
                + (int(self.tie_buffer) if uses_ties else 0),
            )
            if self.exact:
                width_without_self = minimum_width
            else:
                width_without_self = int(
                    np.ceil(float(self.rank_graph_multiplier) * minimum_width)
                )
        elif self.mode_ == "ecdf" and self.calibration == "local":
            width_without_self = max(
                int(self.effective_n_neighbors_),
                int(self.effective_calibration_neighbors_),
            ) + (int(self.tie_buffer) if uses_ties else 0)
        else:
            width_without_self = int(required) + (
                int(self.tie_buffer) if uses_ties else 0
            )

        if self.exact:
            # Paper-exact ranks and exact tie expansion require complete rows.
            full_rows = self.mode_ == "rank" or uses_ties
            index_width = (
                n_reference
                if full_rows
                else min(n_reference, width_without_self + 1)
            )
            if full_rows and n_reference > 5000:
                warnings.warn(
                    "exact=True will materialize a quadratic exact-neighbour graph; "
                    "this may require substantial memory.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            self.index_ = _ExactNNIndex(
                X,
                n_neighbors=index_width,
                metric=self.metric,
                metric_kwds=self.metric_kwds,
                n_jobs=self.n_jobs,
            )
        else:
            index_width = min(n_reference, width_without_self + 1)
            self.index_ = NNDescent(
                X,
                metric=self.metric,
                metric_kwds=self.metric_kwds or {},
                n_neighbors=index_width,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
            )

        self.index_n_neighbors_ = int(index_width)
        self.effective_query_epsilon_ = self._backend_query_epsilon()
        graph_indices, graph_distances = self.index_.neighbor_graph
        # Keep the backend arrays by reference.  The fixed-width fast path
        # should not pay for full graph dtype conversions merely to score a
        # small number of neighbours.
        self._graph_indices_ = np.asarray(graph_indices)
        self._graph_distances_ = np.asarray(graph_distances)

        # Only graph-referenced, non-precomputed rank scoring needs to know
        # whether each stored row contains self.  Query-referenced RankOD-like
        # scoring should not scan the full graph for an unused attribute.
        if (
            self.mode_ == "rank"
            and self.rank_reference_ == "graph"
            and not self.precompute_neighbors
        ):
            rows = np.arange(self._graph_indices_.shape[0])[:, None]
            self._row_has_self_ = np.any(self._graph_indices_ == rows, axis=1)
        else:
            self._row_has_self_ = None

    def _query_index(self, X, k):
        """Query the backend with the same default convention as RankOD.

        Legacy ``RankOD`` does not pass ``epsilon`` to ``NNDescent.query``;
        PyNNDescent therefore chooses its own default. Keeping
        ``query_epsilon=None`` follows that behavior exactly rather than
        hard-coding a version-specific value. An explicit user value is
        forwarded unchanged.
        """
        if self.query_epsilon is None:
            return self.index_.query(X, k=int(k))
        return self.index_.query(
            X,
            k=int(k),
            epsilon=float(self.query_epsilon),
        )

    def _backend_query_epsilon(self):
        """Return the effective epsilon used for ordinary backend queries."""
        if self.query_epsilon is not None:
            return float(self.query_epsilon)

        try:
            parameter = inspect.signature(self.index_.query).parameters.get(
                "epsilon"
            )
            if parameter is None or parameter.default is inspect.Parameter.empty:
                return None
            return float(parameter.default)
        except (TypeError, ValueError, OverflowError):
            # Some API-compatible backends may not expose a Python signature.
            # Query behavior is still correct because _query_index simply
            # omits epsilon in this case.
            return None

    def _reduce_sizes_from_graph(self):
        if self._uses_fixed_width_fast_path():
            # Actual fixed-width consumers validate their own slices.  The one
            # exception is lazy graph-referenced rank scoring, which binary-
            # searches through J graph distances without first materializing a
            # J-wide table.
            if (
                self.mode_ == "rank"
                and self.rank_reference_ == "graph"
                and not self.precompute_neighbors
            ):
                self._fixed_fit_graph_rows(
                    self.effective_max_rank_,
                    context="rank-reference neighbour graph",
                )
            return

        valid_counts = np.empty(self._n_training_samples_, dtype=int)
        for i in range(self._n_training_samples_):
            indices = self._graph_indices_[i]
            distances = self._graph_distances_[i]
            valid_counts[i] = int(
                np.sum((indices >= 0) & (indices != i) & np.isfinite(distances))
            )

        common = int(np.min(valid_counts))
        if common < 1:
            raise ValueError(
                "The fitted neighbour graph has a row with no valid non-self neighbour."
            )

        if common < self.effective_n_neighbors_:
            old = self.effective_n_neighbors_
            self.effective_n_neighbors_ = common
            self._record_reduction("n_neighbors", old, common)

        if self.mode_ == "rank" and common < self.effective_max_rank_:
            old = self.effective_max_rank_
            self.effective_max_rank_ = common
            self._record_reduction("max_rank", old, common)

        if (
            self.mode_ == "ecdf"
            and self.calibration == "local"
            and common < self.effective_calibration_neighbors_
        ):
            old = self.effective_calibration_neighbors_
            self.effective_calibration_neighbors_ = common
            self._record_reduction("calibration_neighbors", old, common)

    def _precompute_rank_rows(self):
        width = int(self.effective_max_rank_)
        if self._uses_fixed_width_fast_path():
            neighbors, distances, _ = self._fixed_fit_graph_rows(
                width, context="precomputed rank-reference graph"
            )
            self._training_neighbors_ = neighbors
            self._training_distances_ = distances
            return

        neighbors = np.empty((self._n_training_samples_, width), dtype=np.int64)
        distances = np.empty((self._n_training_samples_, width), dtype=np.float64)
        for i in range(self._n_training_samples_):
            row_indices, row_distances = self._clean_fit_graph_row(i)
            neighbors[i] = row_indices[:width]
            distances[i] = row_distances[:width]
        self._training_neighbors_ = neighbors
        self._training_distances_ = distances

    def _fit_rank_query_neighbor_arrays(self, base_k, context):
        """Return legacy-style fit neighbours obtained through ``query``.

        The default fixed-width path deliberately mirrors ``RankOD``: request
        ``k + 1`` candidates and drop the first column as self.  Exact/tie-aware
        configurations retain index-based self removal and variable rows.
        """
        if self._uses_fixed_width_fast_path():
            request_k = min(self._n_training_samples_, int(base_k) + 1)
            raw_indices, raw_distances = self._query_index(
                self._training_data_, k=request_k
            )
            return self._fixed_width_rows(
                raw_indices,
                raw_distances,
                int(base_k),
                context=context,
                start=1,
            )

        uses_ties = self._uses_boundary_ties()
        if self.exact and uses_ties:
            request_k = self._n_training_samples_
        else:
            tie_extra = int(self.tie_buffer) if uses_ties else 0
            request_k = min(
                self._n_training_samples_, int(base_k) + 1 + tie_extra
            )

        raw_indices, raw_distances = self._query_index(
            self._training_data_, k=request_k
        )
        raw_indices = np.asarray(raw_indices, dtype=np.int64)
        raw_distances = np.asarray(raw_distances, dtype=np.float64)

        cleaned = []
        available_counts = np.empty(self._n_training_samples_, dtype=int)
        for i in range(self._n_training_samples_):
            valid = (
                (raw_indices[i] >= 0)
                & (raw_indices[i] != i)
                & np.isfinite(raw_distances[i])
            )
            row_indices = raw_indices[i, valid]
            row_distances = raw_distances[i, valid]
            cleaned.append((row_indices, row_distances))
            available_counts[i] = row_indices.size

        common = int(np.min(available_counts))
        if common < 1:
            raise ValueError(
                "The fitted rank query returned a row with no valid non-self neighbour."
            )
        if common < self.effective_n_neighbors_:
            old = self.effective_n_neighbors_
            self.effective_n_neighbors_ = common
            self._record_reduction("n_neighbors", old, common)

        rows = []
        max_count = 0
        target = int(self.effective_n_neighbors_)
        for row_indices, row_distances in cleaned:
            count = self._selected_count(row_distances, target)
            if count < target:
                self._record_reduction(context, target, count)
            rows.append((row_indices[:count], row_distances[:count]))
            max_count = max(max_count, count)
        return self._pack_neighbor_rows(rows, max_count)

    def _prepare_query_rank_reference_rows(self, score_indices, score_counts):
        """Query the fit-time reference rows needed for reverse ranks.

        With ``precompute_neighbors=True`` this mirrors RankOD's eager path and
        queries every fit row. Otherwise it mirrors RankOD's memory-efficient
        path and queries only the unique centres that actually occur in the
        fit points' k-neighbour lists.
        """
        if self.precompute_neighbors:
            reference_ids = np.arange(self._n_training_samples_, dtype=np.int64)
        elif self._uses_fixed_width_fast_path():
            reference_ids = np.unique(score_indices.ravel()).astype(np.int64)
        else:
            valid_centers = [
                score_indices[i, : int(score_counts[i])]
                for i in range(score_indices.shape[0])
            ]
            reference_ids = np.unique(np.concatenate(valid_centers)).astype(
                np.int64
            )

        neighbors, distances, _ = self._query_rank_reference_rows(
            reference_ids, allow_rank_reduction=True
        )
        if self.precompute_neighbors:
            self._training_neighbors_ = neighbors
            self._training_distances_ = distances
        else:
            self._fit_query_reference_ids_ = reference_ids
            self._fit_query_reference_distances_ = distances

    def _query_rank_reference_rows(self, reference_ids, allow_rank_reduction=False):
        """Query non-self reference rows used for reverse-rank lookup.

        The default approximate path requests ``J + 1`` and drops the first
        result exactly as ``RankOD`` does.  The careful path can reduce or pad
        incomplete rows for explicit exact/tie-aware configurations.
        """
        reference_ids = np.asarray(reference_ids, dtype=np.int64)
        if reference_ids.ndim != 1 or reference_ids.size == 0:
            raise ValueError("reference_ids must be a nonempty one-dimensional array.")

        requested_rank = int(self.effective_max_rank_)
        if self._uses_fixed_width_fast_path():
            request_k = min(self._n_training_samples_, requested_rank + 1)
            raw_indices, raw_distances = self._query_index(
                self._training_data_[reference_ids], k=request_k
            )
            return self._fixed_width_rows(
                raw_indices,
                raw_distances,
                requested_rank,
                context="query-derived rank-reference rows",
                start=1,
            )

        request_k = min(self._n_training_samples_, requested_rank + 1)
        raw_indices, raw_distances = self._query_index(
            self._training_data_[reference_ids], k=request_k
        )
        raw_indices = np.asarray(raw_indices, dtype=np.int64)
        raw_distances = np.asarray(raw_distances, dtype=np.float64)

        cleaned = []
        counts = np.empty(reference_ids.size, dtype=np.int64)
        for row, reference_id in enumerate(reference_ids):
            valid = (
                (raw_indices[row] >= 0)
                & (raw_indices[row] != int(reference_id))
                & np.isfinite(raw_distances[row])
            )
            row_indices = raw_indices[row, valid]
            row_distances = raw_distances[row, valid]
            cleaned.append((row_indices, row_distances))
            counts[row] = row_indices.size

        common = int(np.min(counts))
        if common < 1:
            raise ValueError(
                "A queried rank-reference row has no valid non-self neighbour."
            )
        if allow_rank_reduction and common < self.effective_max_rank_:
            old = self.effective_max_rank_
            self.effective_max_rank_ = common
            self._record_reduction("max_rank", old, common)

        width = int(self.effective_max_rank_)
        neighbors = np.full((reference_ids.size, width), -1, dtype=np.int64)
        distances = np.empty((reference_ids.size, width), dtype=np.float64)
        effective_counts = np.minimum(counts, width).astype(np.int64)

        for row, (row_indices, row_distances) in enumerate(cleaned):
            count = int(effective_counts[row])
            neighbors[row, :count] = row_indices[:count]
            distances[row, :count] = row_distances[:count]
            if count < width:
                self._warn_once(
                    "query_rank_reference_shortage",
                    "KNNOD received fewer query-derived rank-reference distances "
                    "than effective_max_rank_ for at least one row; distances "
                    "beyond the observed row are treated as rank overflow.",
                )
                neighbors[row, count:] = row_indices[count - 1]
                distances[row, count:] = row_distances[count - 1]

        return neighbors, distances, effective_counts

    def _rank_reference_inputs(self, indices, counts):
        """Return local centre ids and a dense query-derived distance table."""
        if hasattr(self, "_training_distances_"):
            return indices, self._training_distances_
        if hasattr(self, "_fit_query_reference_distances_"):
            reference_ids = self._fit_query_reference_ids_
            if self._uses_fixed_width_fast_path():
                local_indices = np.searchsorted(reference_ids, indices)
            else:
                local_indices = np.full_like(indices, -1)
                for i in range(indices.shape[0]):
                    count = int(counts[i])
                    local_indices[i, :count] = np.searchsorted(
                        reference_ids, indices[i, :count]
                    )
            return local_indices, self._fit_query_reference_distances_

        if self._uses_fixed_width_fast_path():
            reference_ids = np.unique(indices.ravel()).astype(np.int64)
        else:
            valid_centers = [
                indices[i, : int(counts[i])] for i in range(indices.shape[0])
            ]
            reference_ids = np.unique(np.concatenate(valid_centers)).astype(np.int64)
        _, reference_distances, _ = self._query_rank_reference_rows(
            reference_ids, allow_rank_reduction=False
        )

        if self._uses_fixed_width_fast_path():
            local_indices = np.searchsorted(reference_ids, indices)
        else:
            local_indices = np.full_like(indices, -1)
            for i in range(indices.shape[0]):
                count = int(counts[i])
                local_indices[i, :count] = np.searchsorted(
                    reference_ids, indices[i, :count]
                )
        return local_indices, reference_distances

    # ------------------------------------------------------------------
    # Neighbour selection and tie handling
    # ------------------------------------------------------------------
    def _fixed_width_rows(
        self, indices, distances, width, context, start=0
    ):
        """Return a validated fixed-width view and constant row counts.

        This is the ordinary approximate path.  It intentionally avoids
        per-row filtering, packing, and automatic reductions.  ``start=1``
        mirrors RankOD's convention that the first result of a fitted-row query
        is self.
        """
        indices = np.asarray(indices)
        distances = np.asarray(distances)
        width = int(width)
        start = int(start)
        stop = start + width

        if (
            indices.ndim != 2
            or distances.ndim != 2
            or indices.shape != distances.shape
            or indices.shape[1] < stop
        ):
            raise RuntimeError(
                f"KNNOD expected {width} fixed-width neighbours for {context}, "
                f"but the backend returned shapes {indices.shape} and "
                f"{distances.shape}. Increase query_epsilon, rebuild with a "
                "wider graph, or use exact=True for problematic data."
            )

        selected_indices = indices[:, start:stop]
        selected_distances = distances[:, start:stop]
        if np.any(selected_indices < 0) or not np.all(
            np.isfinite(selected_distances)
        ):
            raise RuntimeError(
                f"KNNOD received an incomplete ANN result for {context}. "
                "Increase query_epsilon, rebuild with a wider graph, or use "
                "exact=True rather than silently changing k or max_rank."
            )

        counts = np.full(selected_indices.shape[0], width, dtype=np.int64)
        return selected_indices, selected_distances, counts

    def _fixed_fit_graph_rows(self, width, context):
        """Take non-self fixed-width rows from the fitted neighbour graph."""
        return self._fixed_width_rows(
            self._graph_indices_,
            self._graph_distances_,
            width,
            context=context,
            start=1,
        )

    def _clean_fit_graph_row(self, i):
        indices = self._graph_indices_[i]
        distances = self._graph_distances_[i]
        valid = (indices >= 0) & (indices != i) & np.isfinite(distances)
        return indices[valid].astype(np.int64), distances[valid].astype(np.float64)

    def _tie_cutoff(self, distance):
        distance = float(distance)
        scale = max(1.0, abs(distance))
        jitter = max(
            float(self.tie_tolerance) * scale,
            float(np.spacing(np.float64(distance))),
        )
        return distance + jitter

    def _selected_count(self, distances, base_k, include_ties=None):
        available = int(distances.size)
        effective = min(int(base_k), available)
        if effective < 1:
            return 0
        count = effective
        if include_ties is None:
            include_ties = self._uses_boundary_ties()
        if include_ties and count < available:
            cutoff = self._tie_cutoff(distances[effective - 1])
            while count < available and distances[count] <= cutoff:
                count += 1
        return count

    def _fit_neighbor_arrays(self, base_k, context):
        if self._uses_fixed_width_fast_path():
            return self._fixed_fit_graph_rows(int(base_k), context=context)

        rows = []
        max_count = 0
        for i in range(self._n_training_samples_):
            indices, distances = self._clean_fit_graph_row(i)
            count = self._selected_count(distances, base_k)
            if count < int(base_k):
                self._record_reduction(context, int(base_k), count)
            if count < 1:
                raise ValueError("A fit row has no usable non-self neighbours.")
            rows.append((indices[:count], distances[:count]))
            max_count = max(max_count, count)
        return self._pack_neighbor_rows(rows, max_count)

    def _query_neighbor_arrays(self, X, base_k, context):
        if self._uses_fixed_width_fast_path():
            extra = 1 if self.mode_ == "rank" else 0
            request_k = min(
                self._n_training_samples_, int(base_k) + extra
            )
            indices, distances = self._query_index(X, k=request_k)
            return self._fixed_width_rows(
                indices,
                distances,
                int(base_k),
                context=context,
                start=0,
            )

        uses_ties = self._uses_boundary_ties()
        if self.exact and uses_ties:
            request_k = self._n_training_samples_
        else:
            if self.mode_ == "rank":
                extra = self._rank_query_extra_candidates()
            else:
                extra = int(self.tie_buffer) if uses_ties else 0
            request_k = min(self._n_training_samples_, int(base_k) + extra)

        indices, distances = self._query_index(X, k=request_k)
        indices = np.asarray(indices, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float64)

        rows = []
        max_count = 0
        for i in range(indices.shape[0]):
            valid = (indices[i] >= 0) & np.isfinite(distances[i])
            row_indices = indices[i, valid]
            row_distances = distances[i, valid]
            count = self._selected_count(row_distances, base_k)
            if count < int(base_k):
                self._record_reduction(context, int(base_k), count)
            if count < 1:
                raise ValueError("The neighbour query returned no valid candidate.")
            rows.append((row_indices[:count], row_distances[:count]))
            max_count = max(max_count, count)
        return self._pack_neighbor_rows(rows, max_count)

    @staticmethod
    def _pack_neighbor_rows(rows, width):
        n_rows = len(rows)
        indices = np.full((n_rows, width), -1, dtype=np.int64)
        distances = np.full((n_rows, width), np.inf, dtype=np.float64)
        counts = np.empty(n_rows, dtype=np.int64)
        for i, (row_indices, row_distances) in enumerate(rows):
            count = len(row_indices)
            counts[i] = count
            indices[i, :count] = row_indices
            distances[i, :count] = row_distances
        return indices, distances, counts

    def _truncate_neighbor_arrays(
        self, indices, distances, counts, base_k, context
    ):
        if self._uses_fixed_width_fast_path():
            del counts
            return self._fixed_width_rows(
                indices,
                distances,
                int(base_k),
                context=context,
                start=0,
            )

        rows = []
        max_count = 0
        for i in range(indices.shape[0]):
            available = int(counts[i])
            row_distances = distances[i, :available]
            count = self._selected_count(row_distances, base_k)
            if count < int(base_k):
                self._record_reduction(context, int(base_k), count)
            if count < 1:
                raise ValueError("A query row has no usable neighbours.")
            rows.append((indices[i, :count], distances[i, :count]))
            max_count = max(max_count, count)
        return self._pack_neighbor_rows(rows, max_count)

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------
    def _sun_scores(self, distances, counts):
        kth = np.empty(distances.shape[0], dtype=np.float64)
        for i in range(distances.shape[0]):
            k = min(self.effective_n_neighbors_, int(counts[i]))
            kth[i] = distances[i, k - 1]
        # Unit-vector Euclidean distances lie in [0, 2].  This affine form is
        # ranking-equivalent to the paper's negative-distance score.
        return 1.0 - np.clip(kth, 0.0, 2.0) / 2.0

    def _fit_ecdf_scores(self, score_indices, score_distances, score_counts):
        del score_indices
        self.training_raw_scores_ = self._distance_raw_scores(
            score_distances, score_counts
        )
        self.sorted_training_raw_scores_ = np.sort(self.training_raw_scores_)

        if self.calibration == "global":
            return self._global_ecdf_probabilities(self.training_raw_scores_)

        self._warn_once(
            "cached_local_ecdf",
            "KNNOD local ECDF uses cached fit statistics for calibration "
            "neighbours; it does not virtually insert and rescore each query.",
        )
        calibration_indices, _, calibration_counts = self._fit_neighbor_arrays(
            self.effective_calibration_neighbors_,
            context="fit calibration_neighbors",
        )
        return local_ecdf_inlier_probabilities(
            self.training_raw_scores_,
            calibration_indices,
            calibration_counts,
            self.training_raw_scores_,
        )

    def _distance_raw_scores(self, distances, counts):
        if self.method_ == "kth":
            raw = np.empty(distances.shape[0], dtype=np.float64)
            for i in range(distances.shape[0]):
                k = min(self.effective_n_neighbors_, int(counts[i]))
                raw[i] = distances[i, k - 1]
            return raw

        params = self.method_params or {}
        raw = np.empty(distances.shape[0], dtype=np.float64)
        for i in range(distances.shape[0]):
            value = self.method_(distances[i, : counts[i]], **params)
            value = np.asarray(value, dtype=np.float64)
            if value.size != 1 or not np.isfinite(value.ravel()[0]):
                raise ValueError(
                    "An ECDF callable must return one finite scalar per distance row."
                )
            raw[i] = float(value.ravel()[0])
        return raw

    def _global_ecdf_probabilities(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        sorted_raw = self.sorted_training_raw_scores_
        left = np.searchsorted(sorted_raw, raw, side="left")
        return (sorted_raw.size - left).astype(np.float64) / float(sorted_raw.size)

    def _rank_scores(self, indices, distances, counts):
        max_rank = int(self.effective_max_rank_)
        if callable(self.method_):
            ranks = self._reverse_rank_matrix(indices, distances, counts)
            params = self.method_params or {}
            scores = np.empty(indices.shape[0], dtype=np.float64)
            normalization_rank = max(max_rank, 2)
            for i in range(indices.shape[0]):
                row_ranks = np.asarray(ranks[i, : counts[i]], dtype=np.float64)
                observed = self._rank_callable_aggregate(row_ranks, params)
                best = self._rank_callable_aggregate(
                    np.ones_like(row_ranks), params
                )
                worst = self._rank_callable_aggregate(
                    np.full_like(row_ranks, float(normalization_rank)), params
                )
                denominator = best - worst
                if (
                    not np.isfinite(denominator)
                    or abs(denominator) <= np.finfo(float).eps
                ):
                    raise ValueError(
                        "The rank callable must be monotone with distinct values "
                        "on all-rank-1 and all-maximum-rank inputs."
                    )
                scores[i] = (observed - worst) / denominator
            return scores
        else:
            method_code = {
                "rbda": 0,
                "harmonic": 1,
                "inverse_sqrt": 2,
                "gaussian": 3,
            }[self.method_]
            sigma = float((self.method_params or {}).get("sigma", 1.0))
            if self.method_ == "gaussian" and sigma <= 0.0:
                raise ValueError("Gaussian rank method requires sigma > 0.")
            if self.rank_reference_ == "query" or self.precompute_neighbors:
                reference_indices, reference_distances = self._rank_reference_inputs(
                    indices, counts
                )
                raw = rank_raw_scores_from_precomputed(
                    reference_indices,
                    distances,
                    counts,
                    reference_distances,
                    max_rank,
                    method_code,
                    sigma,
                )
            else:
                raw = rank_raw_scores_from_graph(
                    indices,
                    distances,
                    counts,
                    self._graph_distances_,
                    self._row_has_self_.astype(np.bool_),
                    max_rank,
                    method_code,
                    sigma,
                )

        low_rank_value, fitted_depth_value = self._rank_normalization_endpoints()
        denominator = low_rank_value - fitted_depth_value
        if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
            raise ValueError(
                "The rank method must be monotone with distinct values at ranks "
                "1 and effective_max_rank_."
            )
        return (raw - fitted_depth_value) / denominator

    def _rank_callable_aggregate(self, ranks, params):
        result = np.asarray(self.method_(ranks, **params), dtype=np.float64)
        if result.ndim == 0 or result.size == 1:
            value = float(result.reshape(-1)[0])
        elif result.shape == ranks.shape:
            value = float(np.mean(result))
        else:
            raise ValueError(
                "A rank callable must return a scalar or one value per input rank."
            )
        if not np.isfinite(value):
            raise ValueError("A rank callable returned a non-finite value.")
        return value

    def _reverse_rank_matrix(self, indices, distances, counts):
        if self.rank_reference_ == "query" or self.precompute_neighbors:
            reference_indices, reference_distances = self._rank_reference_inputs(
                indices, counts
            )
            return reverse_ranks_from_precomputed(
                reference_indices,
                distances,
                counts,
                reference_distances,
                int(self.effective_max_rank_),
            )
        return reverse_ranks_from_graph(
            indices,
            distances,
            counts,
            self._graph_distances_,
            self._row_has_self_.astype(np.bool_),
            int(self.effective_max_rank_),
        )

    def _rank_normalization_endpoints(self):
        # RankOD normalizes against ranks 1 and max_rank.  Preserve that
        # convention so its kernels are direct special cases.  A two-point
        # fallback is needed only for the degenerate J=1 small-data case.
        upper = max(int(self.effective_max_rank_), 2)
        ranks = np.array([1.0, float(upper)], dtype=np.float64)
        if self.method_ == "rbda":
            return 1.0, float(upper)
        if self.method_ == "harmonic":
            return 1.0, 1.0 / float(upper)
        if self.method_ == "inverse_sqrt":
            return 1.0, 1.0 / np.sqrt(float(upper))
        sigma = float((self.method_params or {}).get("sigma", 1.0))
        if sigma <= 0.0:
            raise ValueError("Gaussian rank method requires sigma > 0.")
        return (
            float(np.exp(-1.0 / (2.0 * sigma * sigma))),
            float(np.exp(-(upper**2) / (2.0 * sigma * sigma))),
        )

    def _fit_distance_threshold(self, distances, counts):
        k = int(self.effective_n_neighbors_)
        if self.threshold_source == "kth":
            values = np.array(
                [distances[i, min(k, int(counts[i])) - 1] for i in range(len(counts))],
                dtype=np.float64,
            )
        else:
            # The threshold population is the first-k fit graph, not any extra
            # tie observations included in the eventual score denominator.
            values = np.concatenate(
                [distances[i, : min(k, int(counts[i]))] for i in range(len(counts))]
            )
        self.gamma_ = float(np.quantile(values, float(self.distance_quantile)))

    def _distance_fraction_scores(self, distances, counts):
        if self.method_ == "fraction":
            return distance_fraction_inlier_scores(
                distances, counts, float(self.gamma_)
            )

        params = self.method_params or {}
        scores = np.empty(distances.shape[0], dtype=np.float64)
        for i in range(distances.shape[0]):
            evidence = self.method_(
                distances[i, : counts[i]], float(self.gamma_), **params
            )
            evidence = np.asarray(evidence, dtype=np.float64)
            if evidence.size != 1:
                raise ValueError(
                    "A distance-fraction callable must return one scalar per row."
                )
            value = float(evidence.ravel()[0])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "A distance-fraction callable must return finite outlier "
                    "evidence in [0, 1]."
                )
            scores[i] = 1.0 - value
        return scores

    # ------------------------------------------------------------------
    # Shared API helpers
    # ------------------------------------------------------------------
    def _looks_like_fit_data(self, X):
        """Return True only for the fitted rows in their original order.

        ``RankOD`` uses a deliberately cheap first-row heuristic here.  That
        heuristic can return cached training scores for a same-shaped matrix
        whose later rows have changed.  ``KNNOD`` keeps the useful cached
        leave-one-out behavior but verifies the complete matrix instead.
        """
        if X.shape != self._fit_input_shape_:
            return False

        if self.mode_ != "sun":
            return np.array_equal(X, self._training_data_)

        norms = np.linalg.norm(X, axis=1)
        valid = norms > 0.0
        if not np.array_equal(np.flatnonzero(valid), self.fit_indices_):
            return False
        normalized = (X[valid] / norms[valid, None]).astype(self.dtype, copy=False)
        return np.array_equal(normalized, self._training_data_)

    def _matched_fit_scores(self, X, indices, distances):
        """Recognize fitted rows inside arbitrary query batches.

        Exact matches use their cached leave-one-out fit score, even when fit
        rows are shuffled or scored in subsets.  New query rows retain the
        ordinary novelty-score path.  The nearest-neighbour query has already
        been performed, so normal nonmatching rows incur only a zero-distance
        mask; full row comparisons are limited to possible exact matches.
        """
        matched = np.zeros(X.shape[0], dtype=bool)
        cached = np.zeros(X.shape[0], dtype=np.float64)
        if indices.shape[1] == 0:
            return matched, cached

        candidates = indices[:, 0].astype(int)
        possible = (
            (candidates >= 0)
            & np.isfinite(distances[:, 0])
            & (distances[:, 0] == 0.0)
        )
        rows = np.flatnonzero(possible)
        if rows.size == 0:
            return matched, cached

        candidate_rows = candidates[rows]
        equal = np.all(X[rows] == self._training_data_[candidate_rows], axis=1)
        matched_rows = rows[equal]
        if matched_rows.size == 0:
            return matched, cached

        matched[matched_rows] = True
        original_ids = self.fit_indices_[candidates[matched_rows]]
        values = self.outlier_scores_[original_ids]
        # valid_scores is still in the internal high-is-inlier orientation;
        # score_samples applies reverse_scores only after this replacement.
        if self.reverse_scores:
            values = 1.0 - values
        cached[matched_rows] = values
        return matched, cached

    def _compute_offset(self, scores, contamination: float | None = None):
        if contamination is None:
            contamination = self.contamination
        if not 0.0 < float(contamination) <= 0.5:
            raise ValueError("contamination must lie in (0, 0.5].")
        percentile = (
            1.0 - float(contamination)
            if self.reverse_scores
            else float(contamination)
        )
        return float(np.percentile(scores, 100.0 * percentile))

    def _clear_fitted_state(self):
        fitted_attributes = (
            "index_",
            "outlier_scores_",
            "offset_",
            "gamma_",
            "training_raw_scores_",
            "sorted_training_raw_scores_",
            "_training_neighbors_",
            "_training_distances_",
            "_fit_query_reference_ids_",
            "_fit_query_reference_distances_",
            "_training_data_",
            "_graph_indices_",
            "_graph_distances_",
            "_row_has_self_",
            "effective_query_epsilon_",
        )
        for name in fitted_attributes:
            if hasattr(self, name):
                delattr(self, name)

    def _reset_warning_tracking(self):
        self.knn_reductions_ = {}
        self._warned_messages_ = set()

    def _record_reduction(self, context, requested, effective):
        requested = int(requested)
        effective = int(effective)
        key = (str(context), requested, effective)
        record = self.knn_reductions_.setdefault(
            key,
            {
                "context": str(context),
                "requested": requested,
                "effective": effective,
                "count": 0,
            },
        )
        record["count"] += 1
        self._warn_once(
            ("reduction",) + key,
            f"KNNOD reduced {context} from {requested} to {effective} because "
            "the fitted/query neighbour structure contained fewer usable entries.",
        )

    def _warn_once(self, key, message):
        if key in self._warned_messages_:
            return
        self._warned_messages_.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def knn_reduction_summary(self):
        """Return recorded neighbourhood/rank-depth reductions."""
        check_is_fitted(self, "index_")
        return list(self.knn_reductions_.values())
