"""Bootstrap wrapper for RankOD."""

from collections.abc import Callable

import numpy as np

from hirank.rankod import RankOD


def default_m_function(n: int) -> int:
    return n // 10


def default_aggregate_function(values: list[float]) -> float:
    return sum(values) / len(values)


class BootstrapRankOD(RankOD):
    def __init__(
        self,
        n_bootstrap_sample: int,
        m_function: Callable[[int], int] = default_m_function,
        aggregate_function: Callable[[list[float]], float] = default_aggregate_function,
        **rankod_kwargs,
    ):
        super().__init__(**rankod_kwargs)
        self.n_bootstrap_sample = n_bootstrap_sample
        self.m_function = m_function
        self.aggregate_function = aggregate_function
        self.rankod_kwargs = rankod_kwargs
        self.detectors = [RankOD(**rankod_kwargs) for _ in range(n_bootstrap_sample)]

    def fit(self, X, y=None):
        X = np.asarray(X)
        n = len(X)
        m = self.m_function(n)

        for detector in self.detectors:
            sample = np.random.choice(n, m, replace=True)
            detector.fit(X[sample], y)

        return self

    def _aggregate(self, outputs):
        return np.array(
            [self.aggregate_function(list(values)) for values in zip(*outputs)]
        )

    def score_samples(self, X):
        return self._aggregate(
            [detector.score_samples(X) for detector in self.detectors]
        )

    def decision_function(self, X, contamination: float | None = None):
        return self._aggregate(
            [
                detector.decision_function(X, contamination=contamination)
                for detector in self.detectors
            ]
        )

    def predict(self, X, contamination: float | None = None):
        return self._aggregate(
            [
                detector.predict(X, contamination=contamination)
                for detector in self.detectors
            ]
        )

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)
