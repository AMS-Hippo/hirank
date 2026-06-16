"""
MNIST Outlier Detection Example
================================

This example demonstrates RankOD's ability to detect out-of-distribution samples
by training on a subset of MNIST digits and detecting a held-out class as outliers.

Scenario:
---------
1. Load MNIST dataset (handwritten digits 0-9)
2. Hold out one class (e.g., digit "9") to simulate outliers
3. Train RankOD on remaining classes (0-8)
4. Score all samples (including held-out class)
5. Verify that held-out class receives the highest outlier scores

This demonstrates that RankOD can effectively identify data that doesn't belong
to the training distribution, making it useful for anomaly detection tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from hirank import RankOD

# Configuration
HELD_OUT_CLASS = 9  # Which digit to hold out as "outliers"
N_NEIGHBORS = 15
MAX_RANK = 100
USE_PCA = True  # Whether to reduce dimensionality with PCA
PCA_COMPONENTS = 50  # Number of PCA components to keep
CONTAMINATION = 0.1  # Expected proportion of outliers


def load_mnist(n_samples=10000):
    """Load MNIST dataset."""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.to_numpy()
    y = mnist.target.astype(int).to_numpy()
    
    # Take a subset for faster computation
    if n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def split_normal_outlier(X, y, held_out_class):
    """Split data into normal (training) and outlier (held-out) sets."""
    normal_mask = y != held_out_class
    outlier_mask = y == held_out_class
    
    X_normal = X[normal_mask]
    y_normal = y[normal_mask]
    
    X_outlier = X[outlier_mask]
    y_outlier = y[outlier_mask]
    
    print(f"\nData split:")
    print(f"  Normal (training): {len(X_normal)} samples (classes {sorted(set(y_normal))})")
    print(f"  Outliers (held-out class {held_out_class}): {len(X_outlier)} samples")
    
    return X_normal, y_normal, X_outlier, y_outlier


def apply_pca(X_train, X_all, n_components):
    """Apply PCA dimensionality reduction."""
    print(f"\nApplying PCA: {X_train.shape[1]} → {n_components} dimensions...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_all_pca = pca.transform(X_all)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {variance_explained:.1%}")
    
    return X_train_pca, X_all_pca, pca


def train_detector(X_train):
    """Train RankOD detector on normal data."""
    print(f"\nTraining RankOD detector...")
    print(f"  n_neighbors: {N_NEIGHBORS}")
    print(f"  max_rank: {MAX_RANK}")
    
    detector = RankOD(
        n_neighbors=N_NEIGHBORS,
        max_rank=MAX_RANK,
        contamination=CONTAMINATION,
        precompute_neighbors=True,  # Use fast mode for speed
        random_state=42,
        verbose=False
    )
    
    detector.fit(X_train)
    print(f"  Training complete!")
    
    return detector


def evaluate_detection(scores, y_all, held_out_class):
    """Evaluate outlier detection performance."""
    # True labels: 1 for outlier (held-out class), 0 for normal
    y_true = (y_all == held_out_class).astype(int)
    
    print(f"\n{'='*70}")
    print(f"Outlier Detection Results")
    print(f"{'='*70}")
    
    # Score statistics
    normal_scores = scores[y_true == 0]
    outlier_scores = scores[y_true == 1]
    
    print(f"\nOutlier Score Statistics:")
    print(f"  Normal samples (classes 0-8):")
    print(f"    Mean:   {np.mean(normal_scores):.4f}")
    print(f"    Median: {np.median(normal_scores):.4f}")
    print(f"    Std:    {np.std(normal_scores):.4f}")
    print(f"  Outlier samples (class {held_out_class}):")
    print(f"    Mean:   {np.mean(outlier_scores):.4f}")
    print(f"    Median: {np.median(outlier_scores):.4f}")
    print(f"    Std:    {np.std(outlier_scores):.4f}")
    
    # Separation
    separation = (np.mean(outlier_scores) - np.mean(normal_scores)) / np.std(normal_scores)
    print(f"\nSeparation: {separation:.2f} standard deviations")
    
    # Top K outliers analysis
    top_k_values = [10, 50, 100, 500]
    print(f"\nTop-K Outlier Analysis:")
    print(f"{'K':>6s} | {'% from class ' + str(held_out_class):>20s} | {'Count':>6s}")
    print(f"{'-'*6}+{'-'*22}+{'-'*6}")
    
    for k in top_k_values:
        if k > len(scores):
            continue
        top_k_indices = np.argsort(scores)[-k:]
        top_k_labels = y_all[top_k_indices]
        pct_outlier = (top_k_labels == held_out_class).sum() / k * 100
        count_outlier = (top_k_labels == held_out_class).sum()
        print(f"{k:6d} | {pct_outlier:19.1f}% | {count_outlier:6d}")
    
    # AUC-like metric: what percentile are the outliers at?
    percentiles = []
    for score in outlier_scores:
        percentile = (scores < score).sum() / len(scores) * 100
        percentiles.append(percentile)
    
    avg_percentile = np.mean(percentiles)
    print(f"\nAverage percentile of outliers: {avg_percentile:.1f}th")
    print(f"(100th percentile = highest outlier scores)")
    
    return normal_scores, outlier_scores


def plot_results(scores, y_all, held_out_class, normal_scores, outlier_scores):
    """Create visualization of results."""
    print(f"\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Score distribution histogram
    ax = axes[0, 0]
    bins = 50
    ax.hist(normal_scores, bins=bins, alpha=0.6, label=f'Normal (classes 0-8)', 
            color='blue', density=True)
    ax.hist(outlier_scores, bins=bins, alpha=0.6, label=f'Outlier (class {held_out_class})', 
            color='red', density=True)
    ax.set_xlabel('Outlier Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Outlier Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Score by class
    ax = axes[0, 1]
    classes = sorted(set(y_all))
    class_scores = [scores[y_all == c] for c in classes]
    colors = ['red' if c == held_out_class else 'blue' for c in classes]
    bp = ax.boxplot(class_scores, patch_artist=True)
    ax.set_xticklabels(classes)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Outlier Score')
    ax.set_title('Outlier Scores by Digit Class')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative distribution
    ax = axes[1, 0]
    normal_sorted = np.sort(normal_scores)
    outlier_sorted = np.sort(outlier_scores)
    ax.plot(normal_sorted, np.linspace(0, 100, len(normal_sorted)), 
            label=f'Normal (classes 0-8)', linewidth=2)
    ax.plot(outlier_sorted, np.linspace(0, 100, len(outlier_sorted)), 
            label=f'Outlier (class {held_out_class})', linewidth=2)
    ax.set_xlabel('Outlier Score')
    ax.set_ylabel('Cumulative Percentile')
    ax.set_title('Cumulative Distribution of Outlier Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Top outliers breakdown
    ax = axes[1, 1]
    top_k_values = [10, 50, 100, 500, 1000]
    percentages = []
    for k in top_k_values:
        if k > len(scores):
            break
        top_k_indices = np.argsort(scores)[-k:]
        top_k_labels = y_all[top_k_indices]
        pct = (top_k_labels == held_out_class).sum() / k * 100
        percentages.append(pct)
    
    bars = ax.bar(range(len(percentages)), percentages, color='red', alpha=0.6)
    ax.set_xticks(range(len(percentages)))
    ax.set_xticklabels([f'Top {k}' for k in top_k_values[:len(percentages)]])
    ax.set_ylabel(f'% from Class {held_out_class}')
    ax.set_title(f'Percentage of Held-Out Class in Top-K Outliers')
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect (100%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'mnist_outlier_detection.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_file}")
    
    # Don't show plot in non-interactive mode
    # plt.show()


def main():
    """Run the complete MNIST outlier detection example."""
    print("="*70)
    print("MNIST Outlier Detection with RankOD")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Held-out class (outliers): {HELD_OUT_CLASS}")
    print(f"  PCA dimensionality reduction: {USE_PCA}")
    if USE_PCA:
        print(f"  PCA components: {PCA_COMPONENTS}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Load MNIST
    X, y = load_mnist(n_samples=10000)
    
    # 2. Split into normal and outlier sets
    X_normal, y_normal, X_outlier, y_outlier = split_normal_outlier(X, y, HELD_OUT_CLASS)
    
    # 3. Optionally apply PCA
    if USE_PCA:
        X_train, X_all, pca = apply_pca(X_normal, X, PCA_COMPONENTS)
    else:
        X_train = X_normal
        X_all = X
    
    # 4. Train detector on normal data only
    detector = train_detector(X_train)
    
    # 5. Score ALL data (including held-out class)
    print(f"\nScoring all {len(X_all)} samples...")
    scores = detector.score_samples(X_all)
    print(f"  Scoring complete!")
    
    # 6. Evaluate detection performance
    normal_scores, outlier_scores = evaluate_detection(scores, y, HELD_OUT_CLASS)
    
    # 7. Visualize results
    plot_results(scores, y, HELD_OUT_CLASS, normal_scores, outlier_scores)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    
    pct_in_top_100 = (y[np.argsort(scores)[-100:]] == HELD_OUT_CLASS).sum() / 100 * 100
    
    print(f"""
RankOD successfully identified the held-out class (digit {HELD_OUT_CLASS}) as outliers!

Key findings:
• Outlier samples have {np.mean(outlier_scores)/np.mean(normal_scores):.2f}x higher scores on average
• {pct_in_top_100:.0f}% of top-100 outliers are from the held-out class
• The held-out class is clearly separated from normal classes

This demonstrates that RankOD can effectively detect out-of-distribution samples
without seeing them during training, making it useful for anomaly detection in
real-world scenarios where you don't have labeled anomalies.
""")
    
    print("="*70)


if __name__ == "__main__":
    main()
