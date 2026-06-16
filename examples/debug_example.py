"""
Debug example to test outlier detection with isolated outliers.
"""

import numpy as np
from hirank import RankOD

# Set random seed
np.random.seed(42)

# Generate data with isolated outliers (not clustered)
print("Generating data with isolated outliers...")
n_normal = 200
n_outliers = 5
n_features = 10

# Normal points in a tight cluster around origin
X_normal = np.random.randn(n_normal, n_features) * 0.5

# Outliers: scattered far from the cluster in different directions
X_outliers = []
for i in range(n_outliers):
    # Create outlier in a random direction, far from origin
    direction = np.random.randn(n_features)
    direction = direction / np.linalg.norm(direction)  # Normalize
    outlier = direction * (10 + 5 * np.random.rand())  # Distance 10-15 from origin
    X_outliers.append(outlier)

X_outliers = np.array(X_outliers)
X = np.vstack([X_normal, X_outliers])

print(f"Data shape: {X.shape}")
print(f"Normal samples: {n_normal}, Outlier samples: {n_outliers}")
print(f"Normal data mean norm: {np.mean(np.linalg.norm(X_normal, axis=1)):.2f}")
print(f"Outlier data mean norm: {np.mean(np.linalg.norm(X_outliers, axis=1)):.2f}")

# Create and fit detector
print("\nCreating RankOD detector...")
detector = RankOD(n_neighbors=15, max_rank=50, kernel='harmonic', random_state=42, verbose=True)
detector.fit(X)

# Get scores
scores = detector.score_samples(X)
normal_scores = scores[:n_normal]
outlier_scores = scores[n_normal:]

print(f"\nNormal points - mean score: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
print(f"Outlier points - mean score: {outlier_scores.mean():.4f}, std: {outlier_scores.std():.4f}")

# Show individual outlier scores
print("\nIndividual outlier scores:")
for i, score in enumerate(outlier_scores):
    print(f"  Outlier {i}: {score:.4f}")

# Check ranks for one outlier
print(f"\nTop 10 highest scores (should include outliers):")
top_indices = np.argsort(scores)[-10:][::-1]
for idx in top_indices:
    label = "OUTLIER" if idx >= n_normal else "normal"
    print(f"  Index {idx:3d} ({label}): {scores[idx]:.4f}")

# Predict with different contamination rates
for contamination in [0.025, 0.05, 0.1]:
    predictions = detector.predict(X, contamination=contamination)
    n_detected = np.sum(predictions == -1)
    true_outliers_detected = np.sum(predictions[n_normal:] == -1)
    false_positives = np.sum(predictions[:n_normal] == -1)
    
    print(f"\nContamination={contamination:.3f}:")
    print(f"  Detected: {n_detected}, True outliers: {true_outliers_detected}/{n_outliers}, False positives: {false_positives}")
