"""
Stage 0.5: direct cos/sin encoding + a trained classifier, pure NumPy/sklearn
(no torch), reproducing the "direct_encode" ablation from the phase-encoder
notebooks -- this pairs a simple, non-bio-inspired encoding with a TRAINED
classifier, to match that methodology directly (as opposed to Stage 0's
untrained nearest-centroid, which is the harder, more Bonsai-like test).

Encoding: pixel intensity [0,1] -> phase [0,2*pi] -> [cos(phase), sin(phase)],
concatenated to a 2*784-dim feature vector. Identical in spirit to
direct_encode() in the uploaded notebooks, just NumPy instead of torch.

Classifier: sklearn LogisticRegression (multinomial), chosen over a from-
scratch NumPy MLP for a first pass since it's simpler, deterministic given a
solver, and a perfectly good "some trained classifier" control -- the point
here isn't to match the notebook's exact 3-layer-MLP architecture, it's to
have *some* trained baseline, matched fairly against Stage 0's untrained
one, and against any later oscillator-based encoding under the same
protocol. Swap in a NumPy MLP later if a closer architectural match matters.

Usage: python stage0_5_direct_encoding_baseline.py [data_dir] [max_train_samples]
"""
import sys
import time
import numpy as np
from mnist_loader import load_mnist


def direct_encode(X_flat_01: np.ndarray) -> np.ndarray:
    """X_flat_01: (N, 784) float in [0,1]. Returns (N, 1568) [cos, sin] features."""
    phase = X_flat_01 * 2 * np.pi
    return np.concatenate([np.cos(phase), np.sin(phase)], axis=1)


def run_stage0_5(data_dir: str = ".", max_train_samples: int = None) -> float:
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

    X_train_enc = direct_encode(X_train_flat)
    X_test_enc = direct_encode(X_test_flat)
    print(f"Encoded feature dim: {X_train_enc.shape[1]}")

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    t0 = time.time()
    clf.fit(X_train_enc, y_train)
    fit_time = time.time() - t0

    predictions = clf.predict(X_test_enc)
    accuracy = np.mean(predictions == y_test)
    print(f"Stage 0.5 (direct cos/sin encoding + LogisticRegression) accuracy: {accuracy:.4f}")
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
    run_stage0_5(data_dir, max_train)
