"""
Basic example demonstrating HiRank RankOD usage.
"""

import numpy as np
from hirank import RankOD

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data: 200 normal points in a tight cluster + 10 isolated outliers
print("Generating data...")
n_normal = 200
n_outliers = 10
n_features = 20

# Normal points in a tight cluster
X_normal = np.random.randn(n_normal, n_features) * 0.5

# Outliers: scattered far from the cluster in random directions (truly isolated)
X_outliers = []
for i in range(n_outliers):
    direction = np.random.randn(n_features)
    direction = direction / np.linalg.norm(direction)
    outlier = direction * (8 + 4 * np.random.rand())  # Distance 8-12 from origin
    X_outliers.append(outlier)

X_outliers = np.array(X_outliers)
X = np.vstack([X_normal, X_outliers])

print(f"Data shape: {X.shape}")
print(f"Normal samples: {n_normal}")
print(f"Outlier samples: {n_outliers}")

# Create RankOD detector with contamination set at initialization
# (sklearn-standard pattern)
contamination = n_outliers / len(X)  # True contamination rate
print(f"\nCreating RankOD detector (contamination={contamination:.3f})...")
detector = RankOD(
    n_neighbors=15,        # Number of neighbors for density estimation
    max_rank=100,          # Maximum rank to consider
    contamination=contamination,  # Expected proportion of outliers
    kernel='harmonic',     # Kernel function (1/r)
    precompute_neighbors=False,  # Memory-efficient mode (default)
    random_state=42
)
# Note: Set precompute_neighbors=True for faster scoring on small datasets

# Fit the detector
print("Fitting detector...")
detector.fit(X)

# Get outlier scores
print("\nComputing outlier scores...")
scores = detector.score_samples(X)

print(f"Mean score for normal points: {scores[:n_normal].mean():.4f}")
print(f"Mean score for outlier points: {scores[n_normal:].mean():.4f}")

# Show top outliers detected
print("\nTop 15 highest scores:")
top_indices = np.argsort(scores)[-15:][::-1]
for i, idx in enumerate(top_indices, 1):
    label = "OUTLIER" if idx >= n_normal else "normal"
    print(f"{i:2d}. Index {idx:3d} ({label:7s}): {scores[idx]:.4f}")

# Predict outliers using contamination from initialization
print(f"\nPredicting outliers...")
predictions = detector.predict(X)  # Uses contamination from __init__

# Count predictions
n_detected = np.sum(predictions == -1)
n_inliers = np.sum(predictions == 1)

print(f"Detected outliers: {n_detected}")
print(f"Detected inliers: {n_inliers}")

# Check accuracy on known outliers
true_outliers_detected = np.sum(predictions[n_normal:] == -1)
false_positives = np.sum(predictions[:n_normal] == -1)
print(f"\nAccuracy:")
print(f"  True outliers detected: {true_outliers_detected}/{n_outliers} ({100*true_outliers_detected/n_outliers:.1f}%)")
print(f"  False positives: {false_positives}/{n_normal} ({100*false_positives/n_normal:.1f}%)")

# Demonstrate flexible override of contamination at predict time
print(f"\n--- Flexible contamination override ---")
print("Trying different contamination levels:")
for cont in [0.05, 0.10, 0.15]:
    preds = detector.predict(X, contamination=cont)
    n_out = np.sum(preds == -1)
    true_out = np.sum(preds[n_normal:] == -1)
    print(f"  contamination={cont:.2f}: {n_out} outliers detected ({true_out}/{n_outliers} true outliers)")

