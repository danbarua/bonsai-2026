"""
Missing control for Stage 0.5: raw pixel intensity (no cos/sin encoding at
all) + the SAME trained classifier (LogisticRegression), at full scale.

This isolates the two things Stage 0.5 changed at once relative to Stage 0
(encoding AND classifier). Comparing this script's result against:
  - Stage 0   (raw pixels,  untrained centroid): isolates classifier effect alone
  - Stage 0.5 (cos/sin,     trained  classifier): the original combined result
  - This script (raw pixels, trained classifier): the missing third corner

If this beats or matches Stage 0.5's 0.8605, the cos/sin encoding added
nothing (or cost something) even in the well-powered full-scale regime --
the improvement from Stage 0 was really "trained classifier > untrained
centroid," not "cos/sin encoding > raw intensity." If cos/sin genuinely
beats this, that's real evidence the encoding specifically helps a trained
model.

Usage: python stage0_75_raw_pixel_trained_baseline.py [data_dir] [max_train_samples]
"""
import sys
import time
import numpy as np
from mnist_loader import load_mnist


def run_stage0_75(data_dir: str = ".", max_train_samples: int = None) -> float:
    from sklearn.linear_model import LogisticRegression

    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float64) / 255.0

    if max_train_samples is not None and max_train_samples < len(X_train_flat):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_flat), size=max_train_samples, replace=False)
        X_train_flat = X_train_flat[idx]
        y_train = y_train[idx]

    print(f"Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")
    print(f"Feature dim: {X_train_flat.shape[1]} (raw pixels, no encoding)")

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    t0 = time.time()
    clf.fit(X_train_flat, y_train)
    fit_time = time.time() - t0

    predictions = clf.predict(X_test_flat)
    accuracy = np.mean(predictions == y_test)
    print(f"Stage 0.75 (raw pixels + LogisticRegression) accuracy: {accuracy:.4f}")
    print(f"Fit time: {fit_time:.1f}s")

    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_test == c
        class_acc = np.mean(predictions[mask] == c)
        print(f"  digit {c}: {class_acc:.4f} (n={mask.sum()})")

    return accuracy


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    max_train = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_stage0_75(data_dir, max_train)
