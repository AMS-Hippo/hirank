# HiRank

**High-dimensional Rank-based Outlier Detection**

HiRank is a tightly-scoped outlier detection library implementing reverse k-NN density estimation with kernel smoothing, optimized for high-dimensional data using PyNNDescent for efficient approximate nearest neighbor search.

`RankOD` is a sci-kit learn compatible outlier class that can be a drop in replacement for 
[`LocalOutlierFactor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) or [`IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

## Installation

### From PyPI (when released)

```bash
pip install hirank
```

### From source

```bash
git clone https://github.com/TutteInstitute/hirank.git
cd hirank
pip install -e .
```

### With optional dependencies

```bash
# For development
pip install -e ".[dev]"

# For benchmarking
pip install -e ".[benchmarks]"

# For documentation
pip install -e ".[docs]"

# All extras
pip install -e ".[dev,benchmarks,docs]"
```

## Quick Start

```python
import numpy as np
from hirank import RankOD

# Generate sample data
X = np.random.randn(1000, 50)  # 1000 samples, 50 dimensions

# Create and fit detector (sklearn-standard pattern)
detector = RankOD(
    n_neighbors=15,
    max_rank=100,
    contamination=0.1,  # Expected 10% outliers
    kernel='harmonic'
)
detector.fit(X)

# Get outlier scores normalized to [0, 1] range (higher = more outlier)
scores = detector.score_samples(X)

# Predict outliers using contamination from initialization
labels = detector.predict(X)  # Uses contamination=0.1 from __init__
# labels: -1 for outliers, 1 for inliers

# Or override contamination at predict time for flexibility
labels_conservative = detector.predict(X, contamination=0.05)
```

## Algorithm

RankOD makes available two modes for evaluating outliers using k-nearest-neighbor queries, and three
options to calibrate the computed scores.

### `mode = 'rank'`
The rank mode (which is used by default) is a tweaked version of the RBDA algorithm from the paper

```
Huang, H.; Mehrotra, Kishan; and Mohan, Chilukuri K., "Rank-Based Outlier Detection" (2011).
Electrical Engineering and Computer Science - Technical Reports. 47.
https://surface.syr.edu/eecs_techreports/47
```

1. For each point, compute its `n_neighbors` nearest neighbors
2. For each neighbor, find the rank at which the point would appear in that neighbor's `max_rank`-nearest neighbor list
3. Apply a kernel function to smooth these ranks (default: inverse square root `k(r) = 1/sqrt(r)`)
4. Average the kernel values to estimate local density

### `mode = 'sun'`
The sun mode uses the the distance to the kth nearest neighbor in a normalized space.
```
Yiyou Sun, Yifei Ming, Xiaojin Zhu, Yixuan Li.
"Out-of-distribution Detection with Deep Nearest Neighbors" (2022)
International Conference on Machine Learning, PMLR 162.
https://proceedings.mlr.press/v162/sun22d/sun22d.pdf
```

*The sun mode is only available with `metric = 'euclidean'`*.

1. L2 normalize the data.
2. The score of each point is the euclidean distance to it's `n_neighbors` nearest neighbor. 

### `calibration = 'global'`
Compare the raw score of each point to the raw scores of the whole training set.
Scores are normalized to that the distribution of the training set scores are
uniform between 0 and 1.

### `calibration = 'local'`
Compare the raw score of each point to the scores of its `n_neighbors` nearest neighbors.

### `calibration = 'raw'`
Return raw (normalized) scores that are between 0 and 1.


## Performance and Scalability

**Memory Optimization Options:**

RankOD provides two parameters to control memory usage:

1. **Data Precision (`dtype`)**: Choose between float64 (default) and float32

```python
# Standard precision (sklearn-compatible, default)
detector = RankOD(dtype=np.float64)
# Memory: 8 bytes per value

# Memory-efficient precision (50% memory savings)
detector = RankOD(dtype=np.float32)
# Memory: 4 bytes per value
# Note: PyNNDescent uses float32 internally, so this avoids conversion overhead
```

2. **Neighbor Pre-computation (`precompute_neighbors`)**: Trade memory for speed

```python
# Speed-optimized mode (for smaller datasets or with sufficient memory
detector = RankOD(n_neighbors=15, max_rank=100, precompute_neighbors=True)

# Memory-efficient mode (only for large datasets)
detector = RankOD(n_neighbors=15, max_rank=100, precompute_neighbors=False)
```

## Development

```bash
# Clone repository
git clone https://github.com/TutteInstitute/hirank.git
cd hirank

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmarks
pytest tests/test_benchmarks.py -m benchmark

# Format code
black hirank tests benchmarks
ruff check hirank tests benchmarks
```

## License

HiRank is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Built on [PyNNDescent](https://github.com/lmcinnes/pynndescent) for efficient nearest neighbor search
- Part of the [Tutte Institute](https://www.tutteinstitute.com/) ecosystem
- Consider citing the Huang et al. or Sun et al. paper mentioned above.
