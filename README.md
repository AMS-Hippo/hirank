# HiRank

**High-dimensional Rank-based Outlier Detection**

HiRank is a tightly-scoped outlier detection library implementing reverse k-NN density estimation with kernel smoothing, optimized for high-dimensional data using PyNNDescent for efficient approximate nearest neighbor search.

## Features

- 🚀 **Fast**: Leverages PyNNDescent for approximate nearest neighbor search in high dimensions
- 🎯 **Focused**: Single algorithm (RankOD) optimized for high-dimensional outlier detection
- 🔬 **Scikit-learn Compatible**: Follows sklearn API conventions (`fit`, `predict`, `score_samples`)
- ⚡ **Numba-accelerated**: Critical loops optimized with Numba JIT compilation
- 🛠️ **Flexible**: Supports multiple kernel functions (harmonic, inverse sqrt, Gaussian, custom)

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

RankOD uses **Reverse k-NN Density Estimation**:

1. For each point, compute its k nearest neighbors
2. For each neighbor, find the rank at which the point appears in that neighbor's J-nearest neighbor list
3. Apply a kernel function to smooth these ranks (default: harmonic kernel `k(r) = 1/r`)
4. Sum the kernel values to estimate local density
5. Normalize density to [0, 1] outlier score: `score = (max_density - density) / (max_density - min_density)`

**Key Parameters:**
- `n_neighbors=15`: Number of nearest neighbors for density estimation
- `max_rank=100`: Maximum rank to consider (ranks beyond max_rank are capped)
- `kernel='harmonic'`: Kernel function (`'harmonic'`, `'inverse_sqrt'`, `'gaussian'`, or custom callable)
- `precompute_neighbors=False`: Memory/speed tradeoff (see Performance section below)
- `dtype=np.float64`: Data precision (float64 for sklearn compatibility, float32 for memory savings)

## Performance and Scalability

**Memory Optimization Options:**

RankOD provides two parameters to control memory usage:

1. **Data Precision (`dtype`)**: Choose between float64 (default) and float32

```python
# Standard precision (sklearn-compatible, default)
detector = RankOD(n_neighbors=15, max_rank=100, dtype=np.float64)
# Memory: 8 bytes per value

# Memory-efficient precision (50% memory savings)
detector = RankOD(n_neighbors=15, max_rank=100, dtype=np.float32)
# Memory: 4 bytes per value
# Note: PyNNDescent uses float32 internally, so this avoids conversion overhead
```

2. **Neighbor Pre-computation (`precompute_neighbors`)**: Trade memory for speed

```python
# Memory-efficient mode (default, recommended for large datasets)
detector = RankOD(n_neighbors=15, max_rank=100, precompute_neighbors=False)
# Memory: O(n_samples) | Scoring: Moderate (on-demand queries)

# Speed-optimized mode (for smaller datasets or when memory is plentiful)
detector = RankOD(n_neighbors=15, max_rank=100, precompute_neighbors=True)
# Memory: O(n_samples × max_rank) | Scoring: Fast (array lookups)
```

**Memory Example** (1M samples, 50 features, `max_rank=100`):
- Base (float64): ~400MB
- With float32: ~200MB (50% savings)
- With precompute: +800MB for neighbor indices
- Combined (float32 + precompute): ~1GB total

## Why HiRank?

Traditional distance-based outlier detection methods struggle in high dimensions due to the "curse of dimensionality". RankOD addresses this by:

- **Using ranks instead of distances**: Ranks are more stable in high dimensions
- **Reverse k-NN**: Captures how often a point appears in others' neighborhoods
- **Kernel smoothing**: Provides robustness to rank variations
- **Approximate NN search**: PyNNDescent enables efficient computation even for large datasets

## Documentation

Full documentation available at: https://hirank.readthedocs.io

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

## Citation

If you use HiRank in your research, please cite:

```bibtex
@software{hirank2024,
  title={HiRank: High-dimensional Rank-based Outlier Detection},
  author={Healy, John},
  year={2024},
  url={https://github.com/TutteInstitute/hirank}
}
```

## License

HiRank is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Built on [PyNNDescent](https://github.com/lmcinnes/pynndescent) for efficient nearest neighbor search
- Inspired by rank-based outlier detection research
- Part of the [Tutte Institute](https://www.tutteinstitute.com/) ecosystem
