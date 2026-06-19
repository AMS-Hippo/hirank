"""Tests for the additive KNNOD estimator."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from hirank import KNNOD as TopLevelKNNOD
from hirank.knnod import KNNOD
from hirank.rankod import RankOD


def _fit_neighbor_distances(X, k):
    distances = pairwise_distances(X, X)
    rows = []
    for i in range(X.shape[0]):
        row = np.delete(distances[i], i)
        rows.append(np.sort(row)[:k])
    return np.asarray(rows)


def _rbda_training_scores(X, k):
    distances = pairwise_distances(X, X)
    n = X.shape[0]
    max_rank = n - 1
    scores = np.empty(n, dtype=float)

    for point in range(n):
        order = np.argsort(distances[point], kind="mergesort")
        neighbors = order[order != point][:k]
        ranks = []
        for neighbor in neighbors:
            distance = distances[neighbor, point]
            comparison = np.delete(distances[neighbor], neighbor)
            rank = 1 + np.sum(comparison < distance)
            ranks.append(rank)
        mean_rank = np.mean(ranks)
        scores[point] = (mean_rank - max_rank) / (1.0 - max_rank)
    return scores


def test_rank_mode_defaults_to_rbda_and_matches_exact_oracle():
    X = np.array([[0.0], [0.7], [2.0], [4.5], [8.0], [13.0]])
    detector = KNNOD(
        mode="rank",
        n_neighbors=2,
        max_rank=2,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)

    assert detector.method_ == "rbda"
    assert detector.effective_max_rank_ == X.shape[0] - 1
    np.testing.assert_allclose(
        detector.outlier_scores_, _rbda_training_scores(X, k=2), atol=1e-12
    )




def test_exact_rbda_includes_all_exact_boundary_ties():
    X = np.array([[0.0], [1.0], [3.0], [3.0], [9.0]])
    k = 2
    distances = pairwise_distances(X, X)
    max_rank = X.shape[0] - 1
    expected = np.empty(X.shape[0], dtype=float)

    for point in range(X.shape[0]):
        ids = np.array([i for i in range(X.shape[0]) if i != point])
        order = np.argsort(distances[point, ids], kind="mergesort")
        ids = ids[order]
        row_distances = distances[point, ids]
        cutoff = np.nextafter(row_distances[k - 1], np.inf)
        neighbors = ids[row_distances <= cutoff]
        ranks = []
        for neighbor in neighbors:
            comparison = np.delete(distances[neighbor], neighbor)
            ranks.append(1 + np.sum(comparison < distances[neighbor, point]))
        mean_rank = np.mean(ranks)
        expected[point] = (mean_rank - max_rank) / (1.0 - max_rank)

    detector = KNNOD(
        mode="rank",
        method="rbda",
        n_neighbors=k,
        max_rank=2,
        exact=True,
        include_ties=True,
        n_jobs=1,
    ).fit(X)
    np.testing.assert_allclose(detector.outlier_scores_, expected)


def test_rank_identity_callable_is_rbda():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(24, 4))
    Q = rng.normal(size=(5, 4))

    rbda = KNNOD(
        mode="rank",
        method="rbda",
        n_neighbors=4,
        max_rank=15,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)
    identity = KNNOD(
        mode="rank",
        method=lambda ranks: ranks,
        n_neighbors=4,
        max_rank=15,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)

    np.testing.assert_allclose(rbda.outlier_scores_, identity.outlier_scores_)
    np.testing.assert_allclose(rbda.score_samples(Q), identity.score_samples(Q))




def test_rank_callable_may_return_a_scalar_aggregation():
    rng = np.random.default_rng(44)
    X = rng.normal(size=(20, 3))
    Q = rng.normal(size=(4, 3))

    detector = KNNOD(
        mode="rank",
        method=lambda ranks: np.median(ranks),
        n_neighbors=4,
        max_rank=12,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)
    scores = detector.score_samples(Q)
    assert scores.shape == (Q.shape[0],)
    assert np.all(np.isfinite(scores))


def test_rankod_is_rank_mode_special_case_for_harmonic_and_rbda():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 5))
    Q = rng.normal(size=(4, 5))

    old_harmonic = RankOD(
        n_neighbors=4,
        max_rank=12,
        kernel="harmonic",
        precompute_neighbors=True,
        random_state=1,
        n_jobs=1,
    ).fit(X)
    new_harmonic = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=4,
        max_rank=12,
        include_ties=False,
        rank_reference="query",
        precompute_neighbors=True,
        random_state=1,
        n_jobs=1,
    ).fit(X)

    np.testing.assert_allclose(
        old_harmonic.outlier_scores_, new_harmonic.outlier_scores_, atol=1e-12
    )
    np.testing.assert_allclose(
        old_harmonic.score_samples(Q), new_harmonic.score_samples(Q), atol=1e-12
    )

    old_rbda = RankOD(
        n_neighbors=4,
        max_rank=12,
        kernel=lambda ranks: ranks,
        precompute_neighbors=True,
        random_state=1,
        n_jobs=1,
    ).fit(X)
    new_rbda = KNNOD(
        mode="rank",
        method="rbda",
        n_neighbors=4,
        max_rank=12,
        include_ties=False,
        rank_reference="query",
        precompute_neighbors=True,
        random_state=1,
        n_jobs=1,
    ).fit(X)

    np.testing.assert_allclose(
        old_rbda.outlier_scores_, new_rbda.outlier_scores_, atol=1e-12
    )
    np.testing.assert_allclose(
        old_rbda.score_samples(Q), new_rbda.score_samples(Q), atol=1e-12
    )


@pytest.mark.parametrize("rank_reference", ["graph", "query"])
def test_precompute_rank_rows_does_not_change_scores(rank_reference):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(25, 3))
    Q = rng.normal(size=(6, 3))

    lazy = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=4,
        max_rank=10,
        exact=True,
        rank_reference=rank_reference,
        precompute_neighbors=False,
        n_jobs=1,
    ).fit(X)
    cached = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=4,
        max_rank=10,
        exact=True,
        rank_reference=rank_reference,
        precompute_neighbors=True,
        n_jobs=1,
    ).fit(X)

    np.testing.assert_allclose(lazy.outlier_scores_, cached.outlier_scores_)
    np.testing.assert_allclose(lazy.score_samples(Q), cached.score_samples(Q))
    assert hasattr(cached, "_training_neighbors_")
    assert hasattr(cached, "_training_distances_")
    assert not hasattr(lazy, "_training_neighbors_")


def test_best_effort_ties_include_visible_kth_ties():
    X = np.array([[0.0], [1.0], [-1.0], [3.0], [6.0]])
    detector = KNNOD(
        mode="distance_fraction",
        n_neighbors=2,
        exact=True,
        include_ties=True,
        n_jobs=1,
    ).fit(X)

    indices, distances, counts = detector._query_neighbor_arrays(
        np.array([[0.25]]), 2, context="test"
    )
    # Distances are 0.25, 0.75, and 1.25, so no kth tie here.
    assert counts[0] == 2

    _, _, tied_counts = detector._query_neighbor_arrays(
        np.array([[0.0]]), 2, context="test"
    )
    # The exact reference point at 0 is first; +1 and -1 are tied at the
    # second-neighbour boundary and are both retained.
    assert tied_counts[0] == 3
    assert np.all(indices[:, 0] >= 0)
    assert np.all(np.isfinite(distances[:, 0]))


def test_global_ecdf_matches_direct_upper_tail_probability():
    X = np.array([[0.0], [0.4], [1.2], [2.5], [5.0], [9.0]])
    Q = np.array([[0.2], [3.2], [12.0]])
    k = 2

    detector = KNNOD(
        mode="ecdf",
        calibration="global",
        n_neighbors=k,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)

    training_raw = _fit_neighbor_distances(X, k)[:, -1]
    query_distances = np.sort(pairwise_distances(Q, X), axis=1)[:, :k]
    query_raw = query_distances[:, -1]
    expected = np.array([np.mean(training_raw >= value) for value in query_raw])

    np.testing.assert_allclose(detector.training_raw_scores_, training_raw)
    np.testing.assert_allclose(detector.score_samples(Q), expected)


def test_cached_local_ecdf_matches_its_documented_definition():
    X = np.array([[0.0], [0.5], [1.1], [2.2], [4.0], [7.0]])
    Q = np.array([[1.5], [8.0]])
    k = 2
    calibration_k = 3

    with pytest.warns(RuntimeWarning, match="cached fit statistics"):
        detector = KNNOD(
            mode="ecdf",
            calibration="local",
            calibration_neighbors=calibration_k,
            n_neighbors=k,
            exact=True,
            include_ties=False,
            n_jobs=1,
        ).fit(X)

    training_raw = _fit_neighbor_distances(X, k)[:, -1]
    query_all = pairwise_distances(Q, X)
    query_order = np.argsort(query_all, axis=1, kind="mergesort")
    expected = []
    for i in range(Q.shape[0]):
        raw = np.sort(query_all[i])[:k][-1]
        calibration_ids = query_order[i, :calibration_k]
        expected.append(np.mean(training_raw[calibration_ids] >= raw))

    np.testing.assert_allclose(detector.score_samples(Q), expected)


def test_ecdf_callable_uses_larger_is_more_outlying_convention():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(18, 2))
    Q = rng.normal(size=(3, 2))

    detector = KNNOD(
        mode="ecdf",
        method=lambda distances: np.mean(distances),
        n_neighbors=3,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)

    fit_rows = _fit_neighbor_distances(X, 3)
    training_raw = np.mean(fit_rows, axis=1)
    query_rows = np.sort(pairwise_distances(Q, X), axis=1)[:, :3]
    query_raw = np.mean(query_rows, axis=1)
    expected = np.array([np.mean(training_raw >= value) for value in query_raw])
    np.testing.assert_allclose(detector.score_samples(Q), expected)


def test_sun_mode_normalizes_and_removes_zero_rows():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [-2.0, 1.0],
        ]
    )
    with pytest.warns(RuntimeWarning, match="zero-norm fit row"):
        detector = KNNOD(
            mode="Sun", n_neighbors=2, exact=True, n_jobs=1
        ).fit(X)

    norms = np.linalg.norm(detector._training_data_, axis=1)
    np.testing.assert_allclose(norms, 1.0)
    assert detector.zero_norm_indices_.tolist() == [0]
    assert detector.outlier_scores_[0] == 0.0

    q = np.array([[0.2, 0.8]])
    np.testing.assert_allclose(
        detector.score_samples(q), detector.score_samples(10 * q)
    )
    with pytest.warns(RuntimeWarning, match="zero-norm Sun-mode query"):
        assert detector.score_samples(np.zeros((1, 2)))[0] == 0.0


def test_distance_fraction_threshold_sources_and_callable():
    X = np.array([[0.0], [0.5], [1.5], [3.0], [6.0], [10.0]])
    Q = np.array([[2.0], [12.0]])
    k = 2
    quantile = 0.6
    fit_rows = _fit_neighbor_distances(X, k)

    pooled = KNNOD(
        mode="distance_fraction",
        n_neighbors=k,
        distance_quantile=quantile,
        threshold_source="pooled",
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)
    kth = KNNOD(
        mode="distance_fraction",
        n_neighbors=k,
        distance_quantile=quantile,
        threshold_source="kth",
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)

    assert pooled.gamma_ == pytest.approx(np.quantile(fit_rows.ravel(), quantile))
    assert kth.gamma_ == pytest.approx(np.quantile(fit_rows[:, -1], quantile))

    query_rows = np.sort(pairwise_distances(Q, X), axis=1)[:, :k]
    expected = 1.0 - np.mean(query_rows > pooled.gamma_, axis=1)
    np.testing.assert_allclose(pooled.score_samples(Q), expected)

    custom = KNNOD(
        mode="distance_fraction",
        method=lambda distances, gamma: np.mean(distances > gamma),
        n_neighbors=k,
        distance_quantile=quantile,
        threshold_source="pooled",
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)
    np.testing.assert_allclose(custom.score_samples(Q), pooled.score_samples(Q))


def test_reductions_are_common_and_recorded():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    with pytest.warns(RuntimeWarning, match="reduced max_rank"):
        detector = KNNOD(
            mode="rank",
            n_neighbors=3,
            max_rank=50,
            exact=False,
            include_ties=False,
            n_jobs=1,
        ).fit(X)

    assert detector.effective_max_rank_ == X.shape[0] - 1
    summary = detector.knn_reduction_summary()
    assert any(row["context"] == "max_rank" for row in summary)


def test_reverse_scores_and_sklearn_style_api():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(20, 3))
    Q = rng.normal(size=(4, 3))

    normal = KNNOD(
        mode="ecdf", n_neighbors=3, exact=True, reverse_scores=False, n_jobs=1
    ).fit(X)
    reversed_detector = KNNOD(
        mode="ecdf", n_neighbors=3, exact=True, reverse_scores=True, n_jobs=1
    ).fit(X)

    np.testing.assert_allclose(
        reversed_detector.score_samples(Q), 1.0 - normal.score_samples(Q)
    )
    assert set(normal.predict(Q)).issubset({-1, 1})
    assert normal.decision_function(Q).shape == (Q.shape[0],)


def test_top_level_lazy_export_and_mode_specific_graph_widths():
    assert TopLevelKNNOD is KNNOD
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))

    sun = KNNOD(
        mode="Sun",
        n_neighbors=5,
        max_rank=60,
        exact=False,
        random_state=42,
        n_jobs=1,
    ).fit(X)
    rank = KNNOD(
        mode="rank",
        n_neighbors=5,
        max_rank=60,
        include_ties=False,
        exact=False,
        random_state=42,
        n_jobs=1,
    ).fit(X)

    wider_rank = KNNOD(
        mode="rank",
        n_neighbors=5,
        max_rank=20,
        rank_graph_multiplier=2.0,
        include_ties=False,
        exact=False,
        random_state=42,
        n_jobs=1,
    ).fit(X)

    assert sun.index_n_neighbors_ == 6
    assert rank.index_n_neighbors_ == 61
    assert wider_rank.index_n_neighbors_ == 41


def test_rank_graph_and_query_references_agree_with_exact_backend():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(32, 5))
    Q = rng.normal(size=(7, 5))

    for method in ("harmonic", "rbda"):
        graph = KNNOD(
            mode="rank",
            method=method,
            n_neighbors=5,
            max_rank=14,
            exact=True,
            rank_reference="graph",
            include_ties=False,
            n_jobs=1,
        ).fit(X)
        query = KNNOD(
            mode="rank",
            method=method,
            n_neighbors=5,
            max_rank=14,
            exact=True,
            rank_reference="query",
            include_ties=False,
            n_jobs=1,
        ).fit(X)

        np.testing.assert_allclose(graph.outlier_scores_, query.outlier_scores_)
        np.testing.assert_allclose(graph.score_samples(Q), query.score_samples(Q))


def test_rank_queries_request_one_extra_candidate():
    rng = np.random.default_rng(92)
    X = rng.normal(size=(28, 4))
    Q = rng.normal(size=(3, 4))
    k = 5
    max_rank = 12

    graph = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=k,
        max_rank=max_rank,
        exact=True,
        rank_reference="graph",
        include_ties=False,
        n_jobs=1,
    ).fit(X)
    graph_calls = []
    graph_query = graph.index_.query

    def record_graph_query(data, k=10, **kwargs):
        graph_calls.append(int(k))
        return graph_query(data, k=k, **kwargs)

    graph.index_.query = record_graph_query
    graph.score_samples(Q)
    assert graph_calls[0] == k + 1

    query = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=k,
        max_rank=max_rank,
        exact=True,
        rank_reference="query",
        include_ties=False,
        precompute_neighbors=False,
        n_jobs=1,
    ).fit(X)
    query_calls = []
    original_query = query.index_.query

    def record_query_reference(data, k=10, **kwargs):
        query_calls.append(int(k))
        return original_query(data, k=k, **kwargs)

    query.index_.query = record_query_reference
    query.score_samples(Q)
    assert query_calls[0] == k + 1
    assert query.effective_max_rank_ + 1 in query_calls[1:]


def test_query_epsilon_defaults_to_backend_and_can_be_overridden():
    rng = np.random.default_rng(93)
    X = rng.normal(size=(24, 4))
    Q = rng.normal(size=(3, 4))

    default = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=4,
        max_rank=10,
        exact=True,
        rank_reference="graph",
        include_ties=False,
        n_jobs=1,
    ).fit(X)
    assert default.query_epsilon is None
    assert default.effective_query_epsilon_ == pytest.approx(0.1)

    default_calls = []
    default_query = default.index_.query

    def record_default(data, k=10, **kwargs):
        default_calls.append(dict(kwargs))
        return default_query(data, k=k, **kwargs)

    default.index_.query = record_default
    default.score_samples(Q)
    assert default_calls
    assert all("epsilon" not in kwargs for kwargs in default_calls)

    explicit = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=4,
        max_rank=10,
        exact=True,
        rank_reference="graph",
        include_ties=False,
        query_epsilon=0.0,
        n_jobs=1,
    ).fit(X)
    assert explicit.effective_query_epsilon_ == pytest.approx(0.0)

    explicit_calls = []
    explicit_query = explicit.index_.query

    def record_explicit(data, k=10, **kwargs):
        explicit_calls.append(dict(kwargs))
        return explicit_query(data, k=k, **kwargs)

    explicit.index_.query = record_explicit
    explicit.score_samples(Q)
    assert explicit_calls
    assert all(kwargs.get("epsilon") == 0.0 for kwargs in explicit_calls)


def test_lazy_query_reference_fit_queries_only_used_centres(monkeypatch):
    rng = np.random.default_rng(94)
    X = rng.normal(size=(26, 4))
    k = 4
    seen_reference_ids = []
    original = KNNOD._query_rank_reference_rows

    def record_reference_ids(self, reference_ids, allow_rank_reduction=False):
        seen_reference_ids.append(np.asarray(reference_ids, dtype=np.int64).copy())
        return original(
            self,
            reference_ids,
            allow_rank_reduction=allow_rank_reduction,
        )

    monkeypatch.setattr(
        KNNOD, "_query_rank_reference_rows", record_reference_ids
    )
    KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=k,
        max_rank=12,
        exact=True,
        rank_reference="query",
        include_ties=False,
        precompute_neighbors=False,
        n_jobs=1,
    ).fit(X)

    distances = pairwise_distances(X, X)
    expected = []
    for i in range(X.shape[0]):
        order = np.argsort(distances[i], kind="mergesort")
        expected.extend(order[order != i][:k])
    expected = np.unique(np.asarray(expected, dtype=np.int64))

    assert len(seen_reference_ids) == 1
    np.testing.assert_array_equal(seen_reference_ids[0], expected)


def test_query_reference_uses_rankod_ann_query_defaults(monkeypatch):
    import hirank.knnod as knnod_module
    import hirank.rankod as rankod_module

    class RecordingNNDescent:
        instances = []

        def __init__(
            self,
            X,
            metric="euclidean",
            metric_kwds=None,
            n_neighbors=15,
            n_jobs=-1,
            random_state=None,
            verbose=False,
        ):
            del n_jobs, random_state, verbose
            self.X = np.asarray(X)
            self.metric = metric
            self.metric_kwds = metric_kwds or {}
            self.n_neighbors = int(n_neighbors)
            self.calls = []
            distances = pairwise_distances(
                self.X, self.X, metric=self.metric, **self.metric_kwds
            )
            indices = np.argsort(distances, axis=1, kind="mergesort")
            rows = np.arange(self.X.shape[0])[:, None]
            self.neighbor_graph = (
                indices[:, : self.n_neighbors].astype(np.int64),
                distances[rows, indices][:, : self.n_neighbors].astype(float),
            )
            self.__class__.instances.append(self)

        def query(self, X, k=10, epsilon=0.1):
            X = np.asarray(X)
            self.calls.append((X.shape[0], int(k), float(epsilon)))
            distances = pairwise_distances(
                X, self.X, metric=self.metric, **self.metric_kwds
            )
            indices = np.argsort(distances, axis=1, kind="mergesort")[:, :k]
            rows = np.arange(X.shape[0])[:, None]
            return (
                indices.astype(np.int64),
                distances[rows, indices].astype(float),
            )

    RecordingNNDescent.instances.clear()
    monkeypatch.setattr(rankod_module, "NNDescent", RecordingNNDescent)
    monkeypatch.setattr(knnod_module, "NNDescent", RecordingNNDescent)

    rng = np.random.default_rng(95)
    X = rng.normal(size=(30, 5))
    Q = rng.normal(size=(4, 5))
    k = 4
    max_rank = 12

    old = RankOD(
        n_neighbors=k,
        max_rank=max_rank,
        kernel="harmonic",
        precompute_neighbors=False,
        random_state=1,
        n_jobs=1,
    ).fit(X)
    new = KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=k,
        max_rank=max_rank,
        rank_reference="query",
        include_ties=False,
        tie_buffer=0,
        precompute_neighbors=False,
        query_epsilon=None,
        random_state=1,
        n_jobs=1,
    ).fit(X)

    old_scores = old.score_samples(Q)
    new_scores = new.score_samples(Q)
    np.testing.assert_allclose(old.outlier_scores_, new.outlier_scores_)
    np.testing.assert_allclose(old_scores, new_scores)

    old_index, new_index = RecordingNNDescent.instances
    assert old_index.n_neighbors == new_index.n_neighbors == max_rank + 1
    assert [call[1:] for call in old_index.calls] == [
        call[1:] for call in new_index.calls
    ]
    assert all(epsilon == pytest.approx(0.1) for _, _, epsilon in old_index.calls)
    assert new.effective_query_epsilon_ == pytest.approx(0.1)


def test_rank_reference_and_graph_multiplier_validation():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="rank_reference"):
        KNNOD(mode="rank", rank_reference="other", exact=True).fit(X)
    with pytest.raises(ValueError, match="rank_graph_multiplier"):
        KNNOD(mode="rank", rank_graph_multiplier=0.5, exact=True).fit(X)
    with pytest.raises(ValueError, match="query_epsilon"):
        KNNOD(mode="rank", query_epsilon=-0.1, exact=True).fit(X)
    with pytest.raises(ValueError, match="query_epsilon"):
        KNNOD(mode="rank", query_epsilon=np.nan, exact=True).fit(X)


def test_fit_rows_are_order_and_subset_invariant():
    rng = np.random.default_rng(72)
    X = rng.normal(size=(24, 5))
    detector = KNNOD(
        mode="rank",
        method="rbda",
        n_neighbors=4,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)

    baseline = detector.score_samples(X)
    permutation = rng.permutation(X.shape[0])
    np.testing.assert_allclose(
        detector.score_samples(X[permutation]), baseline[permutation]
    )
    subset = np.array([2, 7, 11, 19])
    np.testing.assert_allclose(detector.score_samples(X[subset]), baseline[subset])


def test_fit_data_cache_requires_a_full_matrix_match():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(18, 4))
    detector = KNNOD(
        mode="ecdf",
        n_neighbors=3,
        exact=True,
        include_ties=False,
        n_jobs=1,
    ).fit(X)

    assert detector._looks_like_fit_data(X.copy())

    modified = X.copy()
    modified[-1] += 100.0
    # The first five rows and the shape still match; the complete matrix does not.
    assert np.array_equal(modified[:5], X[:5])
    assert not detector._looks_like_fit_data(modified)
    assert detector.score_samples(modified).shape == (X.shape[0],)

def test_rankod_source_remains_byte_for_byte_unchanged():
    path = Path(__file__).parents[1] / "hirank" / "rankod.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "6602b00bc02d8c4b510fb7a4579a80df4e855402d7b2cb85f598683c42f434af"
