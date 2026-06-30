"""Score aggregation helpers for RankOD ensemble wrappers.

Aggregators operate on matrices with shape ``(n_samples, n_detectors)``:
rows are points and columns are base-detector scores. The ensemble wrappers fit
an aggregator on the training score matrix at the end of ``fit``. That makes
rank-like normalization possible while still supporting later single-point calls
to ``score_samples`` and ``predict``.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


AggregatorName = Literal["mean", "ecdf", "ecdf_mean"]
ECDFSide = Literal["left", "right"]


def default_aggregate_function(values: list[float]) -> float:
    """Legacy row-wise mean aggregation function."""
    return sum(values) / len(values)


@runtime_checkable
class ScoreAggregator(Protocol):
    """Protocol for training-aware score aggregators."""

    def fit(self, score_matrix, y=None):
        """Learn aggregation state from a training score matrix."""

    def transform(self, score_matrix) -> np.ndarray:
        """Aggregate a score matrix into one score per row."""


def _validate_score_matrix(score_matrix, *, name: str = "score_matrix") -> np.ndarray:
    """Return a finite two-dimensional float score matrix."""
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if score_matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if score_matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if score_matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one detector column")
    if not np.all(np.isfinite(score_matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return score_matrix


def _check_transform_width(aggregator, score_matrix: np.ndarray) -> None:
    if not hasattr(aggregator, "n_detectors_in_"):
        raise ValueError(f"{aggregator.__class__.__name__} must be fitted first")
    if score_matrix.shape[1] != aggregator.n_detectors_in_:
        raise ValueError(
            "score_matrix has a different number of detector columns than the "
            "matrix used during fit"
        )


class MeanAggregator:
    """Average detector scores without normalization.

    This is the default and preserves the previous ensemble behavior when all
    detector scores already have comparable scales.
    """

    def fit(self, score_matrix, y=None):
        score_matrix = _validate_score_matrix(score_matrix)
        self.n_detectors_in_ = score_matrix.shape[1]
        return self

    def transform(self, score_matrix) -> np.ndarray:
        score_matrix = _validate_score_matrix(score_matrix)
        _check_transform_width(self, score_matrix)
        return np.mean(score_matrix, axis=1)

    def fit_transform(self, score_matrix, y=None) -> np.ndarray:
        return self.fit(score_matrix, y=y).transform(score_matrix)


class CallableAggregator:
    """Adapter for legacy row-wise aggregation functions.

    The callable receives a list containing one point's detector scores and must
    return one aggregate score.
    """

    def __init__(self, aggregate_function: Callable[[list[float]], float]):
        self.aggregate_function = aggregate_function

    def fit(self, score_matrix, y=None):
        score_matrix = _validate_score_matrix(score_matrix)
        self.n_detectors_in_ = score_matrix.shape[1]
        return self

    def transform(self, score_matrix) -> np.ndarray:
        score_matrix = _validate_score_matrix(score_matrix)
        _check_transform_width(self, score_matrix)
        return np.asarray(
            [self.aggregate_function(list(row)) for row in score_matrix],
            dtype=np.float64,
        )

    def fit_transform(self, score_matrix, y=None) -> np.ndarray:
        return self.fit(score_matrix, y=y).transform(score_matrix)


# Backwards-compatible alias for callers who prefer the older name.
FunctionAggregator = CallableAggregator


class ECDFMeanAggregator:
    """Average detector-wise ECDF-normalized scores.

    Each detector column is converted to an empirical CDF value using the
    training scores observed in ``fit``. This keeps the ordering convention of
    the underlying scores: if lower detector scores are more anomalous, lower
    aggregate scores are more anomalous; if higher detector scores are more
    anomalous, higher aggregate scores are more anomalous.
    """

    def __init__(self, side: ECDFSide = "left"):
        self.side = side

    def fit(self, score_matrix, y=None):
        if self.side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        score_matrix = _validate_score_matrix(score_matrix)
        self.sorted_training_scores_ = np.sort(score_matrix, axis=0)
        self.n_training_samples_, self.n_detectors_in_ = score_matrix.shape
        return self

    def transform(self, score_matrix) -> np.ndarray:
        score_matrix = _validate_score_matrix(score_matrix)
        _check_transform_width(self, score_matrix)
        ecdf_scores = np.empty_like(score_matrix, dtype=np.float64)

        for detector_idx in range(self.n_detectors_in_):
            training_scores = self.sorted_training_scores_[:, detector_idx]
            ecdf_scores[:, detector_idx] = np.searchsorted(
                training_scores,
                score_matrix[:, detector_idx],
                side=self.side,
            ) / self.n_training_samples_

        return np.mean(ecdf_scores, axis=1)

    def fit_transform(self, score_matrix, y=None) -> np.ndarray:
        return self.fit(score_matrix, y=y).transform(score_matrix)


AggregatorSpec = ScoreAggregator | Callable[[list[float]], float] | AggregatorName


def make_aggregator(
    aggregator: AggregatorSpec | type | None = None,
    aggregate_function: Callable[[list[float]], float] | None = None,
):
    """Return a fresh unfitted aggregator instance.

    ``aggregate_function`` is retained for backwards compatibility with the
    original row-wise callable API. New code should prefer ``aggregator``.
    """
    if aggregator is not None and aggregate_function is not None:
        raise ValueError("Pass either aggregator or aggregate_function, not both")

    if aggregator is None:
        if aggregate_function is None:
            return MeanAggregator()
        return CallableAggregator(aggregate_function)

    if isinstance(aggregator, str):
        if aggregator == "mean":
            return MeanAggregator()
        if aggregator in {"ecdf", "ecdf_mean"}:
            return ECDFMeanAggregator()
        raise ValueError("aggregator must be one of 'mean', 'ecdf', or 'ecdf_mean'")

    if isinstance(aggregator, type):
        aggregator = aggregator()
    elif callable(aggregator) and not hasattr(aggregator, "fit"):
        return CallableAggregator(aggregator)
    else:
        try:
            aggregator = deepcopy(aggregator)
        except Exception:
            pass

    candidate: Any = aggregator
    if not hasattr(candidate, "fit") or not hasattr(candidate, "transform"):
        raise TypeError(
            "aggregator must expose fit(score_matrix) and transform(score_matrix)"
        )
    return candidate


__all__ = [
    "CallableAggregator",
    "ECDFMeanAggregator",
    "FunctionAggregator",
    "MeanAggregator",
    "ScoreAggregator",
    "default_aggregate_function",
    "make_aggregator",
]
