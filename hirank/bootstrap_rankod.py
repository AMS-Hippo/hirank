"""Bootstrap and projection ensemble wrappers for RankOD."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_is_fitted

from hirank.aggregate import (
    default_aggregate_function as default_aggregate_function,
    make_aggregator,
)
from hirank.rankod import RankOD


ProjectionKind = Literal[
    "subspace", "dense", "dense_gaussian", "sparse", "sparse_gaussian"
]
ProjectionSpec = tuple[Literal["subspace", "matrix"], np.ndarray | sparse.spmatrix]


def default_m_function(n: int) -> int:
    return n // 10


def default_projection_dim_function(n_features: int) -> int:
    return max(1, int(np.sqrt(n_features)))


def default_sparsity_function(n_features: int) -> int:
    return max(1, int(np.sqrt(n_features)))


def _as_2d_array(X, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array or a single 1D sample")
    return X


class _RankODEnsembleMixin:
    """Shared aggregation and prediction logic for RankOD ensembles."""

    def _fit_aggregator(self, score_columns):
        # The aggregator is fitted on training scores so later single-point calls
        # can reuse a learned score/rank scale instead of ranking within one row.
        training_score_matrix = np.column_stack(score_columns)
        self.aggregator_ = make_aggregator(
            aggregator=self.aggregator,
            aggregate_function=self.aggregate_function,
        )
        if hasattr(self.aggregator_, "fit_transform"):
            self.outlier_scores_ = self.aggregator_.fit_transform(
                training_score_matrix
            )
        else:
            self.aggregator_.fit(training_score_matrix)
            self.outlier_scores_ = self.aggregator_.transform(training_score_matrix)
        self.offset_ = self._compute_offset(self.outlier_scores_)
        return self

    def _check_ensemble_is_fitted(self):
        check_is_fitted(self, ["aggregator_", "offset_", "n_features_in_"])

    def _validate_input_features(self, X):
        X = _as_2d_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_} features"
            )
        return X

    def _aggregate_score_columns(self, score_columns):
        score_matrix = np.column_stack(score_columns)
        return self.aggregator_.transform(score_matrix)

    def decision_function(self, X, contamination: float | None = None):
        scores = self.score_samples(X)
        if contamination is not None:
            offset = self._compute_offset(scores, contamination=contamination)
        else:
            offset = self.offset_
        return scores - offset

    def predict(self, X, contamination: float | None = None):
        decision_scores = self.decision_function(X, contamination=contamination)
        predictions = np.full_like(decision_scores, 1, dtype="int")
        outliers = (
            (decision_scores >= 0) if self.reverse_scores else (decision_scores <= 0)
        )
        predictions[outliers] = -1
        return predictions

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)


class BootstrapRankOD(_RankODEnsembleMixin, RankOD):
    def __init__(
        self,
        n_bootstrap_sample: int,
        m_function: Callable[[int], int] = default_m_function,
        aggregate_function: Callable[[list[float]], float] | None = None,
        replace: bool = True,
        aggregator=None,
        **rankod_kwargs,
    ):
        super().__init__(**rankod_kwargs)
        self.n_bootstrap_sample = n_bootstrap_sample
        self.m_function = m_function
        self.aggregate_function = aggregate_function
        self.replace = replace
        self.aggregator = aggregator
        self.rankod_kwargs = rankod_kwargs
        self.detectors = [RankOD(**rankod_kwargs) for _ in range(n_bootstrap_sample)]

    def fit(self, X, y=None):
        if self.n_bootstrap_sample <= 0:
            raise ValueError("n_bootstrap_sample must be a positive integer")

        X = _as_2d_array(X)
        n = len(X)
        m = int(self.m_function(n))
        if m <= 0:
            raise ValueError("m_function must return a positive integer")
        if not self.replace and m > n:
            raise ValueError(
                "m_function cannot return more samples than the training set when "
                "replace=False"
            )

        self.n_features_in_ = X.shape[1]
        self.detectors = [
            RankOD(**self.rankod_kwargs) for _ in range(self.n_bootstrap_sample)
        ]
        y_array = None if y is None else np.asarray(y)

        for detector in self.detectors:
            sample = np.random.choice(n, m, replace=self.replace)
            sample_y = None if y_array is None else y_array[sample]
            detector.fit(X[sample], sample_y)

        training_score_columns = [
            detector.score_samples(X) for detector in self.detectors
        ]
        return self._fit_aggregator(training_score_columns)

    def score_samples(self, X):
        self._check_ensemble_is_fitted()
        X = self._validate_input_features(X)
        return self._aggregate_score_columns(
            [detector.score_samples(X) for detector in self.detectors]
        )


class ProjectionRankOD(_RankODEnsembleMixin, RankOD):
    """RankOD ensemble over random feature subspaces or linear projections.

    This wrapper keeps RankOD unchanged. Each ensemble member receives a transformed
    view of the same rows: either a coordinate subspace (feature bagging), a dense
    Gaussian projection, or a sparse Gaussian projection.

    projection_kind options are:
    - "subspace": sample coordinate columns without replacement.
    - "dense_gaussian" or "dense": sample a dense Gaussian matrix.
    - "sparse_gaussian" or "sparse": sample sparse Gaussian columns.
    """

    def __init__(
        self,
        n_projection_sample: int,
        projection_kind: ProjectionKind = "subspace",
        projection_dim_function: Callable[[int], int] = default_projection_dim_function,
        aggregate_function: Callable[[list[float]], float] | None = None,
        sparsity_function: Callable[[int], int] = default_sparsity_function,
        random_state: int | None = None,
        aggregator=None,
        **rankod_kwargs,
    ):
        super().__init__(**rankod_kwargs)
        self.n_projection_sample = n_projection_sample
        self.projection_kind = projection_kind
        self.projection_dim_function = projection_dim_function
        self.aggregate_function = aggregate_function
        self.sparsity_function = sparsity_function
        self.random_state = random_state
        self.aggregator = aggregator
        self.rankod_kwargs = rankod_kwargs
        self.detectors = []

    def fit(self, X, y=None):
        if self.n_projection_sample <= 0:
            raise ValueError("n_projection_sample must be a positive integer")

        X = _as_2d_array(X)
        n_features = X.shape[1]
        n_components = self._resolve_dimension(
            self.projection_dim_function, n_features, "projection_dim_function"
        )
        rng = np.random.default_rng(self.random_state)

        self.n_features_in_ = n_features
        self.projections_ = [
            self._sample_projection(n_features, n_components, rng)
            for _ in range(self.n_projection_sample)
        ]
        self.detectors = [self._make_detector(rng) for _ in self.projections_]
        projected_training_data = [
            self._project(X, projection) for projection in self.projections_
        ]

        for detector, projected_X in zip(self.detectors, projected_training_data):
            detector.fit(projected_X, y)

        training_score_columns = [
            detector.score_samples(projected_X)
            for detector, projected_X in zip(self.detectors, projected_training_data)
        ]
        return self._fit_aggregator(training_score_columns)

    def _make_detector(self, rng: np.random.Generator) -> RankOD:
        rankod_kwargs = dict(self.rankod_kwargs)
        if self.random_state is not None:
            rankod_kwargs["random_state"] = int(
                rng.integers(np.iinfo(np.int32).max)
            )
        return RankOD(**rankod_kwargs)

    def _resolve_dimension(self, dimension_function, n_features: int, name: str) -> int:
        dimension = int(dimension_function(n_features))
        if dimension <= 0:
            raise ValueError(f"{name} must return a positive integer")
        if dimension > n_features and self.projection_kind == "subspace":
            raise ValueError(
                f"{name} returned {dimension}, but subspace projections cannot "
                f"use more than n_features={n_features} columns"
            )
        return dimension

    def _sample_projection(
        self, n_features: int, n_components: int, rng: np.random.Generator
    ) -> ProjectionSpec:
        if self.projection_kind == "subspace":
            features = rng.choice(n_features, n_components, replace=False)
            return "subspace", np.sort(features)

        if self.projection_kind in {"dense", "dense_gaussian"}:
            scale = 1.0 / np.sqrt(n_components)
            matrix = rng.normal(0.0, scale, size=(n_features, n_components))
            return "matrix", matrix

        if self.projection_kind in {"sparse", "sparse_gaussian"}:
            n_nonzero = self._resolve_dimension(
                self.sparsity_function, n_features, "sparsity_function"
            )
            if n_nonzero > n_features:
                raise ValueError(
                    f"sparsity_function returned {n_nonzero}, but sparse projections "
                    f"cannot use more than n_features={n_features} rows per component"
                )
            rows = []
            cols = []
            data = []
            scale = 1.0 / np.sqrt(n_nonzero)
            for column in range(n_components):
                selected_rows = rng.choice(n_features, n_nonzero, replace=False)
                rows.extend(selected_rows)
                cols.extend([column] * n_nonzero)
                data.extend(rng.normal(0.0, scale, size=n_nonzero))
            matrix = sparse.csr_matrix(
                (data, (rows, cols)), shape=(n_features, n_components)
            )
            return "matrix", matrix

        raise ValueError(
            "projection_kind must be one of 'subspace', 'dense_gaussian', "
            "'dense', 'sparse_gaussian', or 'sparse'"
        )

    def _project(self, X, projection: ProjectionSpec):
        X = self._validate_input_features(X)
        kind, value = projection
        if kind == "subspace":
            return X[:, value]
        return X @ value

    def score_samples(self, X):
        self._check_ensemble_is_fitted()
        X = self._validate_input_features(X)
        return self._aggregate_score_columns(
            [
                detector.score_samples(self._project(X, projection))
                for detector, projection in zip(self.detectors, self.projections_)
            ]
        )
