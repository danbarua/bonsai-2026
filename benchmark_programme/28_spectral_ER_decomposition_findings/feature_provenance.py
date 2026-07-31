"""
Provenance metadata system for cached feature artifacts. Built after two
separate normalization-state bugs (notMNIST's mystery 60D artifacts;
MNIST's already-normalized hybrid_20feat files being re-normalized) both
produced catastrophic, silently-wrong results that were only caught by
suspicious downstream accuracy numbers. This makes the failure mode
structurally harder to hit: every saved feature artifact carries explicit
metadata, and loading/combining two artifacts checks compatibility before
use rather than after a confusing result appears.
"""
import pickle
import time


PIPELINE_VERSION = "capacity-experiment-III-2026-07-28"


def save_features(path, X, y, dataset, feature_type, normalization_state,
                   topology_threshold, source_files, extra=None):
    assert normalization_state in ("raw", "zscore"), \
        f"normalization_state must be 'raw' or 'zscore', got {normalization_state!r}"
    metadata = {
        "dataset": dataset,
        "feature_type": feature_type,
        "normalization_state": normalization_state,
        "topology_threshold": topology_threshold,
        "pipeline_version": PIPELINE_VERSION,
        "source_files": list(source_files),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        metadata.update(extra)
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y, "metadata": metadata}, f)


def load_features(path, expect_normalization_state=None, expect_dataset=None,
                   expect_feature_type=None):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "metadata" not in d:
        raise ValueError(
            f"{path} has no provenance metadata -- refusing to load without "
            f"explicit normalization_state confirmation. This is exactly the "
            f"failure mode that caused the notMNIST and MNIST bugs; do not "
            f"bypass this check."
        )
    meta = d["metadata"]
    if expect_normalization_state is not None and meta["normalization_state"] != expect_normalization_state:
        raise ValueError(
            f"{path}: expected normalization_state={expect_normalization_state!r}, "
            f"found {meta['normalization_state']!r}. Refusing to load."
        )
    if expect_dataset is not None and meta["dataset"] != expect_dataset:
        raise ValueError(f"{path}: expected dataset={expect_dataset!r}, found {meta['dataset']!r}")
    if expect_feature_type is not None and meta["feature_type"] != expect_feature_type:
        raise ValueError(f"{path}: expected feature_type={expect_feature_type!r}, found {meta['feature_type']!r}")
    return d["X"], d["y"], meta


def check_combinable(*metas):
    """Reject combining artifacts with incompatible normalization states
    (e.g. raw + zscore) or from different datasets -- the specific
    combination that caused both prior bugs."""
    states = {m["normalization_state"] for m in metas}
    if len(states) > 1:
        raise ValueError(f"Refusing to combine artifacts with mixed normalization states: {states}")
    datasets = {m["dataset"] for m in metas}
    if len(datasets) > 1:
        raise ValueError(f"Refusing to combine artifacts from different datasets: {datasets}")
    return True
