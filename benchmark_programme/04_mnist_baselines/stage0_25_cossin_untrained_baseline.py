"""
Final missing cell in the 2x2 grid: cos/sin phase encoding + UNTRAINED
nearest-centroid (as opposed to Stage 0.5's trained LogisticRegression).

This is the closest existing analogue to Bonsai's own actual methodology --
an untrained, few-shot-style readout -- applied to the encoding that turned
out to lose to raw pixels under a trained classifier. Prediction, going in:
this should do WORSE than Stage 0's raw+untrained baseline (0.8203), and
plausibly by a larger margin than the trained-classifier comparison showed
(6.6pp) -- an untrained geometric method has no way to compensate for the
aliasing collision the way a regularized trained classifier partially can.

Usage: python stage0_25_cossin_untrained_baseline.py [data_dir]
"""
import sys
import numpy as np
from mnist_loader import load_mnist
from stage0_5_direct_encoding_baseline import direct_encode
from stage0_raw_pixel_baseline import build_centroids, classify_nearest_centroid


def run_stage0_25(data_dir: str = ".") -> float:
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float64) / 255.0

    print(f"Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")

    X_train_enc = direct_encode(X_train_flat)
    X_test_enc = direct_encode(X_test_flat)
    print(f"Encoded feature dim: {X_train_enc.shape[1]}")

    centroids = build_centroids(X_train_enc, y_train)
    predictions = classify_nearest_centroid(X_test_enc, centroids)

    accuracy = np.mean(predictions == y_test)
    print(f"Stage 0.25 (cos/sin encoding + untrained nearest-centroid) accuracy: {accuracy:.4f}")

    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_test == c
        class_acc = np.mean(predictions[mask] == c)
        print(f"  digit {c}: {class_acc:.4f} (n={mask.sum()})")

    return accuracy


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    run_stage0_25(data_dir)
