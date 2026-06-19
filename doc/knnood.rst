KNNOD modes
===========

``KNNOD`` is an additive estimator. It follows ``RankOD``'s fit, scoring,
decision, and prediction API, but builds only the neighbour information needed
by the selected mode. Scores are high-is-inlier by default.

Quick examples
--------------

Approximate RBDA is the default::

    from hirank import KNNOD

    detector = KNNOD(
        mode="rank",
        method="rbda",
        n_neighbors=15,
        max_rank=100,
        exact=False,
    ).fit(X)

Sun et al.'s normalized-feature kth-distance score::

    detector = KNNOD(
        mode="sun",
        n_neighbors=20,
    ).fit(embeddings)

A global empirical upper-tail probability based on the kth radius::

    detector = KNNOD(
        mode="ecdf",
        method="kth",
        calibration="global",
        n_neighbors=20,
    ).fit(X)

A cached local ECDF::

    detector = KNNOD(
        mode="ecdf",
        method="kth",
        calibration="local",
        n_neighbors=20,
        calibration_neighbors=50,
    ).fit(X)

The local variant compares the query's raw score with cached leave-one-out fit
scores from its calibration neighbourhood. It does not perform virtual query
insertion; ``fit`` emits a warning documenting this approximation.

A global distance-fraction threshold::

    detector = KNNOD(
        mode="distance_fraction",
        method="fraction",
        n_neighbors=20,
        distance_quantile=0.75,
        threshold_source="pooled",
    ).fit(X)

``threshold_source="pooled"`` fits the threshold from the first ``k``
non-self distances for every fit row. ``threshold_source="kth"`` fits it from
one leave-one-out kth radius per fit row. Both options use the same fitted
neighbour graph; the pooled option processes more scalar distances.

Exact and approximate modes
---------------------------

``exact=False`` uses PyNNDescent and is the default. ``exact=True`` uses
brute-force neighbours. Exact RBDA stores complete reference-distance rows and
is quadratic in fit memory. Other tie-complete exact modes may also construct
quadratic neighbour data.

Approximate rank mode stores a common reverse-rank depth
``effective_max_rank_``. All ranks beyond that depth share the overflow value
``effective_max_rank_ + 1``. If PyNNDescent supplies fewer valid neighbours in
any fit row, the common depth is reduced and a warning is emitted.

When ``include_ties=True``, exact modes that use a variable neighbourhood see
the complete reference row. Approximate mode includes only boundary ties
visible among the returned candidates. ``tie_buffer`` controls how many extra
ANN candidates are requested, and ``tie_tolerance`` controls the small distance
tolerance used to identify a visible tie. The estimator does not claim to find
ties that ANN search did not return.


Rank reference source and graph width
-------------------------------------

Approximate rank mode provides two explicit reverse-rank reference paths::

    # Fast path: binary-search the fitted NNDescent graph rows.
    KNNOD(
        mode="rank",
        method="rbda",
        rank_reference="graph",
    )

    # Legacy-compatible path: query reference rows on demand.
    KNNOD(
        mode="rank",
        method="rbda",
        rank_reference="query",
    )

``rank_reference="graph"`` is the default and avoids the second ANN query that
legacy ``RankOD`` performs during each scoring call.  ``"query"`` asks the
index for the unique query neighbours' reference rows and is therefore slower,
but it is the appropriate comparison path when checking numerical agreement
with ``RankOD``.  With ``precompute_neighbors=True``, either source is resolved
once during ``fit`` and cached.

By default, ``query_epsilon=None`` leaves PyNNDescent's ``epsilon`` argument
unspecified. This deliberately matches legacy ``RankOD``, which also relies on
the backend default rather than forcing a value. Users can request a different
accuracy/speed trade-off explicitly, for example ``query_epsilon=0.0`` or
``query_epsilon=0.2``. The inferred backend value is recorded as
``effective_query_epsilon_`` after fitting.

Rank queries request at least ``k + 1`` candidates and keep the first ``k``
(or any visible boundary ties).  This matches the small ANN search cushion in
legacy ``RankOD`` even when tie expansion is disabled.

``rank_graph_multiplier`` controls an intermediate accuracy/cost option for the
graph path.  For example, ``rank_graph_multiplier=2`` asks PyNNDescent to retain
approximately twice the minimum rank graph width::

    KNNOD(
        mode="rank",
        method="rbda",
        rank_reference="graph",
        rank_graph_multiplier=2.0,
        max_rank=100,
    )

A wider approximate graph may improve the recall of the first ``max_rank``
reference neighbours, but it increases index construction time and memory.
The multiplier is ignored outside approximate rank mode.

Callable conventions
--------------------

Built-in strings use Numba/vectorized paths. Callables are flexible Python
fallbacks:

* Rank callable: ``method(ranks, **method_params)``. Return either a scalar
  aggregation or one value per rank. The callable must be monotone and have
  distinct values at rank 1 and the fitted rank depth. ``KNNOD`` infers whether
  it is increasing (RBDA-like) or decreasing (inverse-kernel-like) from those
  endpoints and orients the result as high-is-inlier.
* ECDF callable: ``method(distances, **method_params)``. Return one finite raw
  scalar for which larger means more outlying. ECDF calibration converts it to
  a high-is-inlier empirical upper-tail probability.
* Distance-fraction callable:
  ``method(distances, gamma, **method_params)``. Return one finite scalar in
  ``[0, 1]`` measuring outlier evidence; ``KNNOD`` returns one minus that value.

Relationship to RankOD
----------------------

The existing ``RankOD`` is the fixed-size, approximate rank mode with a
pointwise rank kernel. Its closest ``KNNOD`` configurations are::

    # RankOD(kernel="harmonic", ...)
    KNNOD(
        mode="rank",
        method="harmonic",
        n_neighbors=k,
        max_rank=J,
        exact=False,
        rank_reference="query",
        query_epsilon=None,
        include_ties=False,
        precompute_neighbors=precompute,
        reverse_scores=reverse,
    )

    # RankOD(kernel="inverse_sqrt", ...)
    KNNOD(mode="rank", method="inverse_sqrt", ...)

    # RankOD(kernel="gaussian", kernel_params={"sigma": sigma}, ...)
    KNNOD(
        mode="rank",
        method="gaussian",
        method_params={"sigma": sigma},
        ...,
    )

A ``RankOD`` custom kernel maps to a rank callable. In particular, the identity
callable ``lambda ranks: ranks`` is the RBDA aggregation; ``KNNOD`` provides the
Numba-accelerated string ``method="rbda"`` for it.

``rank_reference="query"`` is the closest execution-path match to legacy
``RankOD``.  ``rank_reference="graph"`` uses the same score formula but searches
fitted graph rows instead of issuing a second ANN query; it can therefore be
substantially faster and need not be numerically identical under approximate
search. With ``query_epsilon=None``, both classes make their ANN queries using
the same backend-default epsilon. Separate ANN fits may still differ slightly
even with the same random seed. ``KNNOD`` also uses robust self-index removal,
one explicit ``J + 1`` overflow bucket, and common neighbourhood reductions.
The existing ``RankOD`` implementation and hot path are unchanged.

API
---

.. autoclass:: hirank.KNNOD
   :members:
   :undoc-members:
   :show-inheritance:
