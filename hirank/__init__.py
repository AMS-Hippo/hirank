"""
HiRank: High-dimensional rank-based outlier detection.

A tightly-scoped outlier detection library implementing reverse k-NN density
estimation with kernel smoothing, optimized for high-dimensional data using
PyNNDescent for efficient approximate nearest neighbor search.
"""

__version__ = "0.1.2"

from hirank.aggregate import (
    CallableAggregator,
    ECDFMeanAggregator,
    FunctionAggregator,
    MeanAggregator,
    ScoreAggregator,
)
from hirank.bootstrap_rankod import BootstrapRankOD, ProjectionRankOD
from hirank.rankod import RankOD

__all__ = [
    "RankOD",
    "BootstrapRankOD",
    "ProjectionRankOD",
    "ScoreAggregator",
    "CallableAggregator",
    "FunctionAggregator",
    "ECDFMeanAggregator",
    "MeanAggregator",
]
