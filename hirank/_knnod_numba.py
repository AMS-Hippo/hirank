"""Numba-accelerated kernels for :mod:`hirank.knnod`."""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _binary_reverse_rank_graph(
    graph_distances: np.ndarray,
    row_has_self: np.ndarray,
    center: int,
    distance: float,
    max_rank: int,
) -> int:
    """Return a minimum-tie reverse rank, censored at ``max_rank + 1``."""
    row = graph_distances[center]
    lo = 0
    hi = row.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if row[mid] < distance:
            lo = mid + 1
        else:
            hi = mid

    closer = lo
    if row_has_self[center] and distance > 0.0:
        closer -= 1
    if closer < 0:
        closer = 0

    rank = closer + 1
    if rank > max_rank:
        rank = max_rank + 1
    return rank


@njit(cache=True)
def _binary_reverse_rank_precomputed(
    training_distances: np.ndarray,
    center: int,
    distance: float,
    max_rank: int,
) -> int:
    """Reverse rank from a dense non-self distance table."""
    row = training_distances[center]
    lo = 0
    hi = row.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if row[mid] < distance:
            lo = mid + 1
        else:
            hi = mid

    rank = lo + 1
    if rank > max_rank:
        rank = max_rank + 1
    return rank


@njit(cache=True)
def _rank_value(rank: int, method_code: int, sigma: float) -> float:
    if method_code == 0:  # RBDA / identity
        return float(rank)
    if method_code == 1:  # harmonic
        return 1.0 / float(rank)
    if method_code == 2:  # inverse square root
        return 1.0 / np.sqrt(float(rank))
    # Gaussian
    rank_float = float(rank)
    return np.exp(-(rank_float * rank_float) / (2.0 * sigma * sigma))


@njit(cache=True, parallel=True)
def rank_raw_scores_from_graph(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_counts: np.ndarray,
    graph_distances: np.ndarray,
    row_has_self: np.ndarray,
    max_rank: int,
    method_code: int,
    sigma: float,
) -> np.ndarray:
    """Mean transformed reverse rank for each query row."""
    n_rows = neighbor_indices.shape[0]
    scores = np.empty(n_rows, dtype=np.float64)

    for i in prange(n_rows):
        count = neighbor_counts[i]
        total = 0.0
        for j in range(count):
            center = neighbor_indices[i, j]
            distance = neighbor_distances[i, j]
            rank = _binary_reverse_rank_graph(
                graph_distances, row_has_self, center, distance, max_rank
            )
            total += _rank_value(rank, method_code, sigma)
        scores[i] = total / float(count)

    return scores


@njit(cache=True, parallel=True)
def rank_raw_scores_from_precomputed(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_counts: np.ndarray,
    training_distances: np.ndarray,
    max_rank: int,
    method_code: int,
    sigma: float,
) -> np.ndarray:
    """Mean transformed reverse rank using cached non-self distance rows."""
    n_rows = neighbor_indices.shape[0]
    scores = np.empty(n_rows, dtype=np.float64)

    for i in prange(n_rows):
        count = neighbor_counts[i]
        total = 0.0
        for j in range(count):
            center = neighbor_indices[i, j]
            distance = neighbor_distances[i, j]
            rank = _binary_reverse_rank_precomputed(
                training_distances, center, distance, max_rank
            )
            total += _rank_value(rank, method_code, sigma)
        scores[i] = total / float(count)

    return scores


@njit(cache=True, parallel=True)
def reverse_ranks_from_graph(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_counts: np.ndarray,
    graph_distances: np.ndarray,
    row_has_self: np.ndarray,
    max_rank: int,
) -> np.ndarray:
    """Return a padded matrix of reverse ranks for callable rank methods."""
    ranks = np.ones(neighbor_indices.shape, dtype=np.float64)
    for i in prange(neighbor_indices.shape[0]):
        for j in range(neighbor_counts[i]):
            ranks[i, j] = _binary_reverse_rank_graph(
                graph_distances,
                row_has_self,
                neighbor_indices[i, j],
                neighbor_distances[i, j],
                max_rank,
            )
    return ranks


@njit(cache=True, parallel=True)
def reverse_ranks_from_precomputed(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_counts: np.ndarray,
    training_distances: np.ndarray,
    max_rank: int,
) -> np.ndarray:
    """Return reverse ranks using cached non-self distance rows."""
    ranks = np.ones(neighbor_indices.shape, dtype=np.float64)
    for i in prange(neighbor_indices.shape[0]):
        for j in range(neighbor_counts[i]):
            ranks[i, j] = _binary_reverse_rank_precomputed(
                training_distances,
                neighbor_indices[i, j],
                neighbor_distances[i, j],
                max_rank,
            )
    return ranks


@njit(cache=True, parallel=True)
def distance_fraction_inlier_scores(
    distances: np.ndarray,
    counts: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return one minus the fraction of distances above ``threshold``."""
    scores = np.empty(distances.shape[0], dtype=np.float64)
    for i in prange(distances.shape[0]):
        count = counts[i]
        above = 0
        for j in range(count):
            if distances[i, j] > threshold:
                above += 1
        scores[i] = 1.0 - float(above) / float(count)
    return scores


@njit(cache=True, parallel=True)
def local_ecdf_inlier_probabilities(
    raw_scores: np.ndarray,
    calibration_indices: np.ndarray,
    calibration_counts: np.ndarray,
    training_raw_scores: np.ndarray,
) -> np.ndarray:
    """Cached local upper-tail ECDF probabilities."""
    probabilities = np.empty(raw_scores.shape[0], dtype=np.float64)
    for i in prange(raw_scores.shape[0]):
        count = calibration_counts[i]
        n_tail = 0
        for j in range(count):
            fit_id = calibration_indices[i, j]
            if training_raw_scores[fit_id] >= raw_scores[i]:
                n_tail += 1
        probabilities[i] = float(n_tail) / float(count)
    return probabilities
