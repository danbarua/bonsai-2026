"""
Stage 0: raw-pixel nearest-centroid baseline for MNIST.

This is the essential control experiment before evaluating any oscillator-
based encoding: it establishes what accuracy is achievable with (a) no
learned classifier, just centroid matching, and (b) no encoding at all,
just raw pixel intensities. Any oscillator-based nearest-centroid readout
needs to beat this to demonstrate the dynamics are adding value over doing
nothing.

Usage: python stage0_raw_pixel_baseline.py [data_dir]
"""
import sys
import numpy as np
from mnist_loader import load_mnist


def build_centroids(X: np.ndarray, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Per-class mean pixel vector. X: (N, D) float, y: (N,) int."""
    D = X.shape[1]
    centroids = np.zeros((num_classes, D), dtype=np.float64)
    for c in range(num_classes):
        mask = y == c
        centroids[c] = X[mask].mean(axis=0)
    return centroids


def classify_nearest_centroid(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid classification via Euclidean distance. X: (N, D), centroids: (C, D)."""
    # (N, C) distance matrix via broadcasting
    dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(dists, axis=1)


def run_stage0(data_dir: str = ".") -> float:
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    # Flatten to (N, 784) and normalize to [0, 1]
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float64) / 255.0

    print(f"Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")

    centroids = build_centroids(X_train_flat, y_train)
    predictions = classify_nearest_centroid(X_test_flat, centroids)

    accuracy = np.mean(predictions == y_test)
    print(f"Stage 0 (raw-pixel nearest-centroid) accuracy: {accuracy:.4f}")

    # Per-class breakdown, since centroid-matching often does much better on
    # some digits (e.g. 0, 1) than others (e.g. 8, 9) -- worth knowing which.
    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_test == c
        class_acc = np.mean(predictions[mask] == c)
        print(f"  digit {c}: {class_acc:.4f} (n={mask.sum()})")

    return accuracy


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    run_stage0(data_dir)
