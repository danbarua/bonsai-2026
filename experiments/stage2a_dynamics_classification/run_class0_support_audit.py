"""
Class-0-support audit, per external review's explicit request: quantify
how much information the class-0-derived 505-node active support
(`active_indices`) removes, before asking what graph evolution
subsequently does with what remains.

Part 1 (this script, local, free): retained ink fraction per image
(sum of pixel intensity inside the support / sum over all 784 pixels),
its distribution by class, and a heatmap of class-mean ink lying
outside the support. Uses the already-cached raw pixel data for the
full 60,000-image training set -- no new encoding, no new GPU time.

Part 2 (this script, local, cheap): builds the two baseline feature
sets external review asked for -- (a) raw pixels restricted to the
505-node support (a column slice of already-cached raw_feat, free),
and (b) the full 784-pixel locally-encoded state, gauge-featurized
WITHOUT restriction (a new encode pass, ~100s CPU, reusing
`_local_converged_phases` unchanged -- the expensive part was already
being computed and discarded by `encode_and_restrict`; this just keeps
it). Saves both feature sets for a later classifier CV+fit (via
`cuml.accel`, on GPU, since these are new, un-cached conditions) in
`run_class0_support_audit_classify.py`.
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.dynamics.learned_topology_construction import _local_converged_phases
import stage2a_topologies as topo
from stage2a_paths import scratch_root, train_scratch_dir, test_scratch_dir

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
AUDIT_DIR = os.path.join(scratch_root(), "class0_support_audit")


def reference_node_features_full(theta_784, ref_idx_784):
    """Identical gauge construction to stage2a_core.reference_node_features,
    applied to the FULL 784-pixel state (no restriction) -- same formula,
    same constant-column-drop, just a different index set."""
    shifted = theta_784 - theta_784[ref_idx_784]
    cos_part = np.cos(shifted)
    sin_part = np.sin(shifted)
    assert abs(cos_part[ref_idx_784] - 1.0) < 1e-12
    assert abs(sin_part[ref_idx_784] - 0.0) < 1e-12
    cos_part = np.delete(cos_part, ref_idx_784)
    sin_part = np.delete(sin_part, ref_idx_784)
    return np.concatenate([cos_part, sin_part])


def encode_full_784(image_01, seed=0):
    """Returns the FULL 784-dim converged phase field, unrestricted --
    the intermediate result stage2a_core.encode_and_restrict already
    computes internally and discards after slicing to active_indices."""
    return _local_converged_phases(image_01, seed=seed).flatten()


def _encode_one_image_full784(args):
    """Module-level (not nested) worker, per this project's own
    multiprocessing convention -- a nested closure cannot be pickled for
    a worker pool."""
    idx, image_flat, ref_idx_784 = args
    theta_784 = encode_full_784(image_flat.reshape(28, 28), seed=0)
    return idx, reference_node_features_full(theta_784, ref_idx_784)


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)

    print("Loading cached raw pixel data (train + test)...")
    with open(os.path.join(train_scratch_dir(), "stage3_encode_local.pkl"), "rb") as f:
        train_encode = pickle.load(f)
    with open(os.path.join(test_scratch_dir(), "stage4_encode_local.pkl"), "rb") as f:
        test_encode = pickle.load(f)

    active_indices = train_encode["active_indices"]
    ref_idx = train_encode["ref_idx"]
    ref_idx_784 = int(active_indices[ref_idx])
    y_train = train_encode["labels"]
    y_test = test_encode["labels"]
    raw_train = train_encode["raw_feat"]  # (60000, 784), already 0-1 normalized
    raw_test = test_encode["raw_feat"]
    print(f"n_train={len(y_train)}, n_test={len(y_test)}, n_active={len(active_indices)}, "
          f"ref_idx_784={ref_idx_784}")

    # ---- Part 1: retained ink fraction per class ----
    print("\nComputing retained ink fraction per image (train set)...")
    total_ink = raw_train.sum(axis=1)
    active_ink = raw_train[:, active_indices].sum(axis=1)
    # Avoid division by zero for any (pathological, shouldn't occur) blank image.
    retained_frac = np.divide(active_ink, total_ink, out=np.zeros_like(active_ink), where=total_ink > 0)

    print("Retained ink fraction distribution by class:")
    per_class_stats = {}
    for c in range(10):
        mask = y_train == c
        vals = retained_frac[mask]
        per_class_stats[c] = {
            "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
            "median": float(np.median(vals)), "n": int(mask.sum()),
        }
        print(f"  class {c}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"median={np.median(vals):.4f}, min={vals.min():.4f}, max={vals.max():.4f}, n={mask.sum()}")
    overall = {"mean": float(retained_frac.mean()), "std": float(retained_frac.std())}
    print(f"  OVERALL: mean={overall['mean']:.4f}, std={overall['std']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    box_data = [retained_frac[y_train == c] for c in range(10)]
    ax.boxplot(box_data, tick_labels=[str(c) for c in range(10)], showfliers=False)
    ax.set_xlabel("class")
    ax.set_ylabel("retained ink fraction\n(sum over 505-node support / sum over all 784 pixels)")
    ax.set_title("How much ink survives the class-0-derived support projection, by class\n"
                  "(full 60,000-image official training set)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "retained_ink_fraction_by_class.png")
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved {out1}")

    # ---- Heatmap: class-mean ink lying OUTSIDE the support ----
    print("\nBuilding class-mean-ink-outside-support heatmap...")
    outside_mask = np.ones(784, dtype=bool)
    outside_mask[active_indices] = False

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    for c, ax in zip(range(10), axes.flat):
        class_mean_image = raw_train[y_train == c].mean(axis=0).copy()
        display = class_mean_image.copy()
        display[active_indices] = np.nan  # mask IN-support pixels, show only outside-support ink
        im = ax.imshow(display.reshape(28, 28), cmap="hot", vmin=0, vmax=raw_train.mean(axis=0).max())
        ax.set_title(f"class {c}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Class-mean ink intensity OUTSIDE the 505-node active support\n"
                 "(masked/white = inside support; color = mean ink at excluded pixels)", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.6, label="mean pixel intensity", pad=0.02)
    out2 = os.path.join(RESULTS_DIR, "ink_outside_support_by_class.png")
    plt.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved {out2}")

    # ---- Part 2a: raw pixels restricted to the 505-node support (free -- a column slice) ----
    print("\nBuilding raw-pixels-restricted-to-support feature set (train + test)...")
    X_train_raw505 = raw_train[:, active_indices].astype(np.float32)
    X_test_raw505 = raw_test[:, active_indices].astype(np.float32)
    np.save(os.path.join(AUDIT_DIR, "X_train_raw505.npy"), X_train_raw505)
    np.save(os.path.join(AUDIT_DIR, "X_test_raw505.npy"), X_test_raw505)
    print(f"  X_train_raw505={X_train_raw505.shape}, X_test_raw505={X_test_raw505.shape}")

    # ---- Part 2b: full 784-pixel locally-encoded state, gauge-featurized, unrestricted ----
    print("\nEncoding full 784-pixel gauge-featurized state (train, ~70s)...")
    import time
    import multiprocessing as mp

    def encode_full_batch(raw_feat, label):
        n = raw_feat.shape[0]
        t0 = time.time()
        n_workers = max(1, mp.cpu_count() - 1)
        work_items = [(i, raw_feat[i], ref_idx_784) for i in range(n)]
        with mp.Pool(n_workers) as pool:
            results = pool.map(_encode_one_image_full784, work_items)
        results.sort(key=lambda r: r[0])
        feat = np.stack([r[1] for r in results]).astype(np.float32)
        print(f"  {label}: {n} images in {time.time()-t0:.1f}s, feat shape={feat.shape}")
        return feat

    X_train_encoded784 = encode_full_batch(raw_train, "train")
    X_test_encoded784 = encode_full_batch(raw_test, "test")
    np.save(os.path.join(AUDIT_DIR, "X_train_encoded784.npy"), X_train_encoded784)
    np.save(os.path.join(AUDIT_DIR, "X_test_encoded784.npy"), X_test_encoded784)

    np.save(os.path.join(AUDIT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(AUDIT_DIR, "y_test.npy"), y_test)

    with open(os.path.join(RESULTS_DIR, "class0_support_audit_stats.pkl"), "wb") as f:
        pickle.dump({"per_class_retained_ink": per_class_stats, "overall_retained_ink": overall,
                     "ref_idx_784": ref_idx_784}, f)

    print(f"\nAll audit feature sets saved to {AUDIT_DIR}")
    print("Next: run_class0_support_audit_classify.py (GPU, cuml.accel) for the CV+fit+evaluate step.")


if __name__ == "__main__":
    main()
