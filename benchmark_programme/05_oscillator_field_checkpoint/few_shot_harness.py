"""
Few-shot evaluation harness: encoding-agnostic, classifier-agnostic.

This is the evaluation regime that actually matters for Bonsai's own
methodology -- 3-5 examples per class, not full-60k training sets. Full-
scale comparisons (Stages 0/0.25/0.5/0.75) answer a different question
("does this work with lots of labeled data") and can give a qualitatively
different, even opposite, answer than the few-shot regime (see
MNIST_BASELINES.md: a 5k-sample matched comparison and the full-60k
comparison disagreed on which encoding was better, because LogisticRegression
was underdetermined at 5k samples).

Design:
- `encode_fn(X) -> X_encoded`: a pure function, X is (N, 784) raw pixel
  values in [0,1], returns (N, D) encoded features. This is the thing that
  varies across raw pixels / cos-sin encoding / any future oscillator-based
  encoding -- swap it in without touching the harness.
- `classifier_factory() -> classifier`: a zero-arg callable returning a
  FRESH classifier instance with sklearn's .fit(X, y) / .predict(X) API.
  Covers untrained nearest-centroid (sklearn.neighbors.NearestCentroid,
  confirmed equivalent to the hand-rolled Euclidean-distance version used
  in Stages 0/0.25) and trained classifiers (e.g. LogisticRegression)
  identically -- "classifier-agnostic" just means "anything with fit/predict".
- Stratified sampling: each trial draws exactly n_per_class examples per
  class (not n_per_class total), since class-imbalanced few-shot draws would
  confound the comparison.
- Multiple trials per sample size, different random draw each time, mean +/-
  std reported -- given how much this whole investigation has shown small-
  sample results can vary run to run, a single-draw few-shot number isn't
  trustworthy on its own.
"""
import numpy as np
from typing import Callable


def stratified_few_shot_sample(X: np.ndarray, y: np.ndarray, n_per_class: int,
                                seed: int, num_classes: int = 10):
    """Draw exactly n_per_class examples per class. Returns (X_sub, y_sub)."""
    rng = np.random.default_rng(seed)
    indices = []
    for c in range(num_classes):
        class_indices = np.where(y == c)[0]
        if len(class_indices) < n_per_class:
            raise ValueError(f"Class {c} has only {len(class_indices)} examples, "
                              f"need {n_per_class}")
        chosen = rng.choice(class_indices, size=n_per_class, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def evaluate_few_shot(
    encode_fn: Callable[[np.ndarray], np.ndarray],
    classifier_factory: Callable[[], object],
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    sample_sizes: list = (5, 10, 50),
    n_trials: int = 10,
    num_classes: int = 10,
    base_seed: int = 42,
    encode_once: bool = True,
) -> dict:
    """
    Sweep few-shot training-set sizes, multiple random draws each, report
    mean +/- std test accuracy per size.

    X_train/X_test: (N, D) raw pixel arrays, already flattened and normalized
    to [0,1] -- encode_fn is applied inside this function, not before calling it.

    encode_once: if True (default), encode the full test set once outside the
    trial loop (test set doesn't change between trials, only the training
    subsample does) -- a real speedup when encode_fn is expensive, with no
    effect on results since it's a pure function of X_test alone.

    Returns: {n_per_class: {"mean": float, "std": float, "trials": [float, ...]}}
    """
    X_test_encoded = encode_fn(X_test) if encode_once else None

    results = {}
    for n_per_class in sample_sizes:
        trial_accuracies = []
        for trial in range(n_trials):
            seed = base_seed + trial * 1000 + n_per_class
            X_sub, y_sub = stratified_few_shot_sample(
                X_train, y_train, n_per_class, seed=seed, num_classes=num_classes
            )
            X_sub_encoded = encode_fn(X_sub)
            X_test_enc = X_test_encoded if encode_once else encode_fn(X_test)

            clf = classifier_factory()
            clf.fit(X_sub_encoded, y_sub)
            predictions = clf.predict(X_test_enc)
            accuracy = np.mean(predictions == y_test)
            trial_accuracies.append(accuracy)

        results[n_per_class] = {
            "mean": float(np.mean(trial_accuracies)),
            "std": float(np.std(trial_accuracies)),
            "trials": trial_accuracies,
        }
    return results


def print_few_shot_results(results: dict, label: str = ""):
    """Pretty-print a results dict from evaluate_few_shot."""
    if label:
        print(f"=== {label} ===")
    for n_per_class, stats in sorted(results.items()):
        trials_str = ", ".join(f"{t:.4f}" for t in stats["trials"])
        print(f"  n={n_per_class:>3} examples/class: "
              f"{stats['mean']:.4f} +/- {stats['std']:.4f}  (trials: {trials_str})")


if __name__ == "__main__":
    # Quick self-test / demo against real data, if mnist_loader is importable
    # and data is present, comparing raw pixels vs cos/sin encoding under an
    # untrained nearest-centroid, at the sample sizes that actually matter
    # for Bonsai's methodology.
    import sys
    sys.path.insert(0, ".")
    from mnist_loader import load_mnist
    from sklearn.neighbors import NearestCentroid

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float64) / 255.0

    def raw_encode(X):
        return X

    def cossin_encode(X):
        phase = X * 2 * np.pi
        return np.concatenate([np.cos(phase), np.sin(phase)], axis=1)

    print("Few-shot sweep: raw pixels, untrained nearest-centroid")
    raw_results = evaluate_few_shot(
        raw_encode, NearestCentroid, X_train_flat, y_train, X_test_flat, y_test,
        sample_sizes=[5, 10, 50], n_trials=10
    )
    print_few_shot_results(raw_results, "raw pixels + NearestCentroid")

    print()
    print("Few-shot sweep: cos/sin encoding, untrained nearest-centroid")
    cossin_results = evaluate_few_shot(
        cossin_encode, NearestCentroid, X_train_flat, y_train, X_test_flat, y_test,
        sample_sizes=[5, 10, 50], n_trials=10
    )
    print_few_shot_results(cossin_results, "cos/sin encoding + NearestCentroid")
