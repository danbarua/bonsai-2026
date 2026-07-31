"""
Classifier sweep: is NearestCentroid actually a defensible choice, or just
the first untrained thing that worked? Compares several untrained/few-shot-
friendly classifiers against the same encodings, using few_shot_harness.py
unchanged (classifier_factory was already swappable -- this just exercises
that properly instead of asserting one choice).

Classifiers compared:
- NearestCentroid (sklearn): Euclidean distance to the class mean. What
  we've been using throughout.
- KNeighborsClassifier(k=1), (k=3): less brittle than a single centroid --
  doesn't collapse each class to one point, so it can capture
  within-class structure a centroid can't.
- StandardizedNearestCentroid (custom, below): nearest-centroid, but on
  per-feature standardized (variance-normalized) distance rather than raw
  Euclidean. This is NOT full Mahalanobis distance -- a true per-class
  Mahalanobis distance needs to invert a covariance matrix, and with only
  5-10 examples in 784+ dimensions that matrix is singular (more dimensions
  than samples). This uses a diagonal approximation (per-feature variance
  only, computed across the whole few-shot training set, with a floor to
  avoid dividing by ~0 for near-constant features like background pixels)
  instead -- numerically stable at these sample sizes, at the cost of not
  capturing feature correlations the way true Mahalanobis would.
"""
import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from few_shot_harness import evaluate_few_shot, print_few_shot_results


class StandardizedNearestCentroid:
    """Nearest-centroid on per-feature standardized distance (diagonal
    approximation to Mahalanobis -- see module docstring for why not full
    Mahalanobis at these sample sizes)."""

    def __init__(self, var_floor: float = 1e-3):
        self.var_floor = var_floor
        self.centroids_ = None
        self.classes_ = None
        self.scale_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Global (not per-class) feature variance -- per-class variance with
        # only 5-10 examples per class would itself be a very noisy estimate.
        self.scale_ = np.sqrt(np.maximum(X.var(axis=0), self.var_floor))
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        X_scaled = X / self.scale_
        centroids_scaled = self.centroids_ / self.scale_
        dists = np.linalg.norm(
            X_scaled[:, np.newaxis, :] - centroids_scaled[np.newaxis, :, :], axis=2
        )
        return self.classes_[np.argmin(dists, axis=1)]


def raw_encode(X):
    return X


def cossin_encode(X):
    phase = X * 2 * np.pi
    return np.concatenate([np.cos(phase), np.sin(phase)], axis=1)


def run_classifier_sweep(X_train, y_train, X_test, y_test,
                          sample_sizes=(5, 10, 50), n_trials=10):
    classifiers = {
        "NearestCentroid": NearestCentroid,
        "KNN (k=1)": lambda: KNeighborsClassifier(n_neighbors=1),
        "KNN (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
        "StandardizedNearestCentroid": StandardizedNearestCentroid,
    }
    encodings = {
        "raw pixels": raw_encode,
        "cos/sin": cossin_encode,
    }

    all_results = {}
    for enc_name, enc_fn in encodings.items():
        for clf_name, clf_factory in classifiers.items():
            label = f"{enc_name} + {clf_name}"
            results = evaluate_few_shot(
                enc_fn, clf_factory, X_train, y_train, X_test, y_test,
                sample_sizes=sample_sizes, n_trials=n_trials
            )
            all_results[label] = results
            print_few_shot_results(results, label)
            print()
    return all_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from mnist_loader import load_mnist

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float64) / 255.0

    run_classifier_sweep(X_train_flat, y_train, X_test_flat, y_test)
