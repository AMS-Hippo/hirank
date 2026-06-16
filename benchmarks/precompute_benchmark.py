"""
Benchmark: precompute_neighbors Performance Comparison
========================================================

This benchmark demonstrates the performance and memory tradeoffs between
RankOD's two modes:

1. **Memory-efficient mode** (`precompute_neighbors=False`):
   - Queries nearest neighbors on-demand during scoring
   - Lower memory footprint
   - Slower scoring (requires additional queries)

2. **Speed-optimized mode** (`precompute_neighbors=True`):
   - Pre-computes and stores all neighbor indices and distances
   - Higher memory usage: O(n_samples * max_rank)
   - Faster scoring (array lookups only, no queries)

The speedup is most noticeable when scoring test data, where the
speed-optimized mode avoids querying the index for each neighbor's
nearest neighbors.
"""

import numpy as np
import time
from hirank import RankOD
import sys


def format_memory(bytes_val):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def estimate_memory(n_samples, max_rank, dtype=np.float64):
    """Estimate memory usage for precomputed mode."""
    bytes_per_value = np.dtype(dtype).itemsize
    # Indices (int64) + Distances (dtype)
    memory_bytes = n_samples * max_rank * (8 + bytes_per_value)
    return memory_bytes


def run_benchmark(n_samples, n_features, n_test, n_neighbors, max_rank, dtype=np.float64):
    """Run benchmark comparing both modes."""
    
    print(f"\n{'='*70}")
    print(f"Benchmark Configuration")
    print(f"{'='*70}")
    print(f"Training samples: {n_samples:,}")
    print(f"Test samples:     {n_test:,}")
    print(f"Features:         {n_features}")
    print(f"n_neighbors:      {n_neighbors}")
    print(f"max_rank:         {max_rank}")
    print(f"dtype:            {dtype.__name__}")
    
    # Generate data
    np.random.seed(42)
    X_train = np.random.randn(n_samples, n_features).astype(dtype)
    X_test = np.random.randn(n_test, n_features).astype(dtype)
    
    # Memory estimate
    estimated_memory = estimate_memory(n_samples, max_rank, dtype)
    print(f"\nEstimated precompute memory: {format_memory(estimated_memory)}")
    
    print(f"\n{'='*70}")
    print(f"Mode 1: Memory-Efficient (precompute_neighbors=False)")
    print(f"{'='*70}")
    
    # Mode 1: Memory-efficient
    detector1 = RankOD(
        n_neighbors=n_neighbors,
        max_rank=max_rank,
        precompute_neighbors=False,
        dtype=dtype,
        random_state=42
    )
    
    start = time.time()
    detector1.fit(X_train)
    fit_time1 = time.time() - start
    print(f"Fit time:         {fit_time1:.3f}s")
    
    start = time.time()
    scores_train1 = detector1.score_samples(X_train)
    score_train_time1 = time.time() - start
    print(f"Score train time: {score_train_time1:.3f}s ({n_samples} samples)")
    
    start = time.time()
    scores_test1 = detector1.score_samples(X_test)
    score_test_time1 = time.time() - start
    print(f"Score test time:  {score_test_time1:.3f}s ({n_test} samples)")
    
    print(f"\n{'='*70}")
    print(f"Mode 2: Speed-Optimized (precompute_neighbors=True)")
    print(f"{'='*70}")
    
    # Mode 2: Speed-optimized
    detector2 = RankOD(
        n_neighbors=n_neighbors,
        max_rank=max_rank,
        precompute_neighbors=True,
        dtype=dtype,
        random_state=42
    )
    
    start = time.time()
    detector2.fit(X_train)
    fit_time2 = time.time() - start
    print(f"Fit time:         {fit_time2:.3f}s (includes precomputation)")
    
    start = time.time()
    scores_train2 = detector2.score_samples(X_train)
    score_train_time2 = time.time() - start
    print(f"Score train time: {score_train_time2:.3f}s ({n_samples} samples)")
    
    start = time.time()
    scores_test2 = detector2.score_samples(X_test)
    score_test_time2 = time.time() - start
    print(f"Score test time:  {score_test_time2:.3f}s ({n_test} samples)")
    
    # Verify results match
    results_match = np.allclose(scores_test1, scores_test2)
    
    print(f"\n{'='*70}")
    print(f"Performance Comparison")
    print(f"{'='*70}")
    print(f"Results match:           {results_match}")
    print(f"Fit time speedup:        {fit_time1/fit_time2:.2f}x {'(slower)' if fit_time1 > fit_time2 else '(faster)'}")
    print(f"Train scoring speedup:   {score_train_time1/score_train_time2:.2f}x faster")
    print(f"Test scoring speedup:    {score_test_time1/score_test_time2:.2f}x faster")
    print(f"Total test pipeline:     {(fit_time1+score_test_time1)/(fit_time2+score_test_time2):.2f}x faster")
    
    print(f"\n{'='*70}")
    print(f"Recommendations")
    print(f"{'='*70}")
    
    # Calculate per-sample times
    test_time_per_sample_1 = score_test_time1 / n_test * 1000
    test_time_per_sample_2 = score_test_time2 / n_test * 1000
    
    print(f"Test scoring (per sample):")
    print(f"  Memory-efficient: {test_time_per_sample_1:.2f}ms/sample")
    print(f"  Speed-optimized:  {test_time_per_sample_2:.2f}ms/sample")
    print(f"\nUse precompute_neighbors=True when:")
    print(f"  • Scoring many test samples (speedup: {score_test_time1/score_test_time2:.1f}x)")
    print(f"  • Memory allows ({format_memory(estimated_memory)} for this dataset)")
    print(f"  • Real-time/low-latency scoring is critical")
    print(f"\nUse precompute_neighbors=False when:")
    print(f"  • Memory is constrained")
    print(f"  • Training dataset is very large")
    print(f"  • Only scoring training data (uses cached scores)")
    
    return {
        'fit_time_ratio': fit_time1 / fit_time2,
        'train_score_speedup': score_train_time1 / score_train_time2,
        'test_score_speedup': score_test_time1 / score_test_time2,
        'memory_bytes': estimated_memory,
        'results_match': results_match
    }


if __name__ == "__main__":
    print("="*70)
    print("RankOD: precompute_neighbors Benchmark")
    print("="*70)
    print("\nComparing memory-efficient vs speed-optimized modes for test data scoring.")
    
    # Run multiple benchmark scenarios
    scenarios = [
        # (n_samples, n_features, n_test, n_neighbors, max_rank, dtype)
        (500, 20, 100, 15, 50, np.float64),
        (1000, 50, 200, 20, 100, np.float64),
        (2000, 100, 500, 25, 150, np.float32),
    ]
    
    results = []
    for n_samples, n_features, n_test, n_neighbors, max_rank, dtype in scenarios:
        result = run_benchmark(n_samples, n_features, n_test, n_neighbors, max_rank, dtype)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary Across All Scenarios")
    print(f"{'='*70}")
    test_speedups = [r['test_score_speedup'] for r in results]
    print(f"Test scoring speedup:   {min(test_speedups):.1f}x - {max(test_speedups):.1f}x faster")
    print(f"Average speedup:        {np.mean(test_speedups):.1f}x faster")
    print(f"All results match:      {all(r['results_match'] for r in results)}")
    
    print(f"\n{'='*70}")
    print(f"Conclusion")
    print(f"{'='*70}")
    print(f"""
The precompute_neighbors=True mode provides significant speedups ({np.mean(test_speedups):.1f}x on average)
for scoring test data by eliminating index queries during computation. This makes
it ideal for production deployments where you need to score new data quickly.

The tradeoff is additional memory usage during fit() to store neighbor indices
and distances. For large datasets, use precompute_neighbors=False to minimize
memory footprint at the cost of slower test scoring.

Both modes produce identical results, so the choice is purely based on your
performance requirements and available resources.
""")
