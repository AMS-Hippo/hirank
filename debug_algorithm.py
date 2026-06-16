"""
Debug script to understand why outliers aren't being detected properly.
"""

import numpy as np
from hirank import RankOD

# Set random seed
np.random.seed(42)

# Create simple 2D data for easy visualization
print("Creating simple 2D test data...")
X_normal = np.random.randn(50, 2) * 0.5  # Tight cluster
X_outliers = np.array([[5, 5], [-5, -5], [5, -5]])  # Clear outliers
X = np.vstack([X_normal, X_outliers])

print(f"Normal points: {len(X_normal)} (indices 0-49)")
print(f"Outliers: {len(X_outliers)} (indices 50-52)")
print(f"Outlier positions: {X_outliers}")

# Create detector with verbose output
detector = RankOD(n_neighbors=10, max_rank=30, kernel="harmonic", verbose=True, random_state=42)
detector.fit(X)

# Check specific points
print("\n" + "=" * 70)
print("ANALYZING SPECIFIC POINTS")
print("=" * 70)

# Analyze a normal point (index 0)
normal_idx = 0
print(f"\nNormal point {normal_idx}: {X[normal_idx]}")
print(f"  Density score: {detector.density_scores_[normal_idx]:.4f}")
print(f"  Outlier score: {detector.outlier_scores_[normal_idx]:.4f}")

# Get its k-NN
knn_indices, _ = detector.index_.query(X[[normal_idx]], k=detector.n_neighbors + 1)
knn_indices = knn_indices[0, 1:]  # Exclude self
print(f"  K-NN indices: {knn_indices[:5]}...")

# Check reverse ranks
all_neighbors, _ = detector.index_.query(X, k=detector.max_rank + 1)
reverse_ranks = []
for neighbor_idx in knn_indices[:5]:
    neighbor_nn = all_neighbors[neighbor_idx, 1:]
    rank = np.where(neighbor_nn == normal_idx)[0]
    if len(rank) > 0:
        reverse_ranks.append(rank[0] + 1)
    else:
        reverse_ranks.append(detector.max_rank)
print(f"  Sample reverse ranks (first 5 neighbors): {reverse_ranks}")

# Analyze an outlier (index 50)
outlier_idx = 50
print(f"\nOutlier point {outlier_idx}: {X[outlier_idx]}")
print(f"  Density score: {detector.density_scores_[outlier_idx]:.4f}")
print(f"  Outlier score: {detector.outlier_scores_[outlier_idx]:.4f}")

# Get its k-NN
knn_indices, knn_distances = detector.index_.query(X[[outlier_idx]], k=detector.n_neighbors + 1)
knn_indices = knn_indices[0, 1:]
knn_distances = knn_distances[0, 1:]
print(f"  K-NN indices: {knn_indices[:5]}...")
print(f"  K-NN distances: {knn_distances[:5]}")

# Check reverse ranks
reverse_ranks = []
for neighbor_idx in knn_indices[:5]:
    neighbor_nn = all_neighbors[neighbor_idx, 1:]
    rank = np.where(neighbor_nn == outlier_idx)[0]
    if len(rank) > 0:
        reverse_ranks.append(rank[0] + 1)
    else:
        reverse_ranks.append(detector.max_rank)
print(f"  Sample reverse ranks (first 5 neighbors): {reverse_ranks}")

# Overall statistics
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)
print(f"Max possible density: {detector.max_density_:.4f}")
print(f"\nNormal points (0-49):")
print(f"  Mean density: {detector.density_scores_[:50].mean():.4f}")
print(f"  Mean outlier score: {detector.outlier_scores_[:50].mean():.4f}")
print(f"\nOutlier points (50-52):")
print(f"  Mean density: {detector.density_scores_[50:].mean():.4f}")
print(f"  Mean outlier score: {detector.outlier_scores_[50:].mean():.4f}")

# Test prediction
predictions = detector.predict(X, contamination=0.1)
print(f"\nPredicted outliers (contamination=0.1): {np.where(predictions == -1)[0]}")
print(f"True outliers: [50, 51, 52]")
