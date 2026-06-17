"""
HiRank: High-dimensional rank-based outlier detection.

A tightly-scoped outlier detection library implementing reverse k-NN density
estimation with kernel smoothing, optimized for high-dimensional data using
PyNNDescent for efficient approximate nearest neighbor search.
"""

__version__ = "0.1.1"

from hirank.rankod import RankOD

__all__ = ["RankOD"]
