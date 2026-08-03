"""
Shared feasibility-ladder pipeline code, reused across stages 1-3
rather than duplicated per stage (this project's own reuse discipline --
CLAUDE.md, DESIGN.md's staged-ladder section). Subsampling, per-image
processing (encode/restrict/evolve/gauge), the go/no-go mechanical
checks, and the CV-based classifier-condition fitting all live here
once.
"""
import multiprocessing as mp
import time

import numpy as np

import stage2a_core as s2a
import stage2a_classifier as s2a_clf
from stage2a_classifier import NonConvergenceError

N_CLASSES = 10


def subsample_stratified(X_train, y_train, seed, n_per_class):
    """Deterministic, class-stratified subsample of the official KMNIST
    training set. Returns (images_01, labels, selected_idx) --
    selected_idx are each image's immutable index into the FULL official
    training set (0..59999), needed downstream for the encoder-seed
    robustness check (DESIGN.md: 'seed equal to the image's immutable
    dataset index'), not just its position within this subsample."""
    rng = np.random.default_rng(seed)
    selected_idx = []
    for c in range(N_CLASSES):
        class_idx = np.where(y_train == c)[0]
        chosen = rng.choice(class_idx, size=n_per_class, replace=False)
        selected_idx.extend(chosen.tolist())
    selected_idx = np.array(selected_idx)
    images_01 = X_train[selected_idx].astype(np.float64) / 255.0
    labels = y_train[selected_idx]
    return images_01, labels, selected_idx


def _process_one_image(args):
    idx, image_01, active_indices_tuple, W_T, ref_idx, encoder_seed = args
    active_indices = np.array(active_indices_tuple)

    theta0 = s2a.encode_and_restrict(image_01, active_indices, seed=encoder_seed)
    R_pre = s2a.order_parameter(theta0)
    feat_pre = s2a.reference_node_features(theta0, ref_idx)

    thetaT, diag = s2a.evolve_on_graph(theta0, W_T)
    if thetaT is None:
        return {
            "idx": idx, "solver_failed": True, "solver_diag": diag,
            "R_pre": R_pre, "R_post": None,
            "feat_pre": feat_pre, "feat_post": None,
            "raw_feat": image_01.flatten(),
        }

    R_post = s2a.order_parameter(thetaT)
    feat_post = s2a.reference_node_features(thetaT, ref_idx)
    return {
        "idx": idx, "solver_failed": False, "solver_diag": diag,
        "R_pre": R_pre, "R_post": R_post,
        "feat_pre": feat_pre, "feat_post": feat_post,
        "raw_feat": image_01.flatten(),
    }


def run_pipeline(images_01, labels, encoder_seeds=None):
    """encoder_seeds: None (default) uses the shared, locked primary
    seed (s2a.ENCODER_SEED = 0) for every image. Pass an array of
    per-image seeds (e.g. each image's immutable dataset index) for the
    encoder-RNG robustness check -- DESIGN.md's locked alternative
    condition, never the primary one."""
    active_indices, W_T, ink_mask_active, nodes_T = s2a.load_T()
    ref_idx = nodes_T["median"]
    assert ref_idx == 363, f"expected T's median node at index 363, got {ref_idx}"

    if encoder_seeds is None:
        encoder_seeds = [s2a.ENCODER_SEED] * len(images_01)

    n_workers = max(1, mp.cpu_count() - 1)
    work_items = [(i, images_01[i], tuple(active_indices), W_T, ref_idx, encoder_seeds[i])
                  for i in range(len(images_01))]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_one_image, work_items)
    elapsed = time.time() - t0

    results.sort(key=lambda r: r["idx"])
    return results, elapsed, active_indices, nodes_T


def check_go_no_go(results):
    """DESIGN.md's locked go/no-go criteria, mechanics only."""
    report = {}

    n_solver_failed = sum(1 for r in results if r["solver_failed"])
    report["n_images"] = len(results)
    report["n_solver_failed"] = n_solver_failed
    report["solver_failure_rate"] = n_solver_failed / len(results)
    report["solver_failure_rate_ok"] = report["solver_failure_rate"] <= 0.001

    non_finite_count = 0
    for r in results:
        if not np.all(np.isfinite(r["raw_feat"])):
            non_finite_count += 1
        if not np.all(np.isfinite(r["feat_pre"])):
            non_finite_count += 1
        if not r["solver_failed"] and not np.all(np.isfinite(r["feat_post"])):
            non_finite_count += 1
    report["n_non_finite_feature_vectors"] = non_finite_count
    report["non_finite_ok"] = non_finite_count == 0

    R_pre_vals = np.array([r["R_pre"] for r in results])
    R_post_vals = np.array([r["R_post"] for r in results if r["R_post"] is not None])
    report["R_pre_summary"] = {
        "min": float(R_pre_vals.min()), "max": float(R_pre_vals.max()),
        "mean": float(R_pre_vals.mean()), "median": float(np.median(R_pre_vals)),
        "n_below_0.01": int(np.sum(R_pre_vals < 0.01)), "n_above_0.99": int(np.sum(R_pre_vals > 0.99)),
    }
    report["R_post_summary"] = {
        "min": float(R_post_vals.min()), "max": float(R_post_vals.max()),
        "mean": float(R_post_vals.mean()), "median": float(np.median(R_post_vals)),
        "n_below_0.01": int(np.sum(R_post_vals < 0.01)), "n_above_0.99": int(np.sum(R_post_vals > 0.99)),
    } if len(R_post_vals) else None
    report["R_pre_values"] = R_pre_vals.tolist()
    report["R_post_values"] = R_post_vals.tolist()
    report["R_near_limits_flag"] = bool(
        report["R_pre_summary"]["n_below_0.01"] or report["R_pre_summary"]["n_above_0.99"] or
        (report["R_post_summary"] and (report["R_post_summary"]["n_below_0.01"]
                                        or report["R_post_summary"]["n_above_0.99"])))

    return report


def _process_one_image_multi_topology(args):
    """Stage 3's multi-topology variant: encode ONCE per image (shared
    across all topologies, since only the evolution step depends on
    which graph is used -- DESIGN.md), then evolve on every topology in
    `topologies_dict`. Returns raw/pre-evolution once, plus a per-topology
    dict of evolved results."""
    idx, image_01, active_indices_tuple, topologies_items, ref_idx, encoder_seed = args
    active_indices = np.array(active_indices_tuple)

    theta0 = s2a.encode_and_restrict(image_01, active_indices, seed=encoder_seed)
    R_pre = s2a.order_parameter(theta0)
    feat_pre = s2a.reference_node_features(theta0, ref_idx)

    evolved = {}
    for name, W in topologies_items:
        thetaT, diag = s2a.evolve_on_graph(theta0, W)
        if thetaT is None:
            evolved[name] = {"solver_failed": True, "solver_diag": diag,
                              "R_post": None, "feat_post": None}
        else:
            evolved[name] = {"solver_failed": False, "solver_diag": diag,
                              "R_post": s2a.order_parameter(thetaT),
                              "feat_post": s2a.reference_node_features(thetaT, ref_idx)}

    return {
        "idx": idx, "R_pre": R_pre, "feat_pre": feat_pre,
        "raw_feat": image_01.flatten(), "evolved": evolved,
    }


def _process_one_image_encode_only(args):
    """Stage 3's GPU-split variant: encode/restrict/gauge the pre-evolution
    features on CPU only, deferring evolution itself to a separate,
    GPU-side batched step (stage 3's 60,000-image scale makes per-image
    CPU evolution the dominant cost -- see stage2a_pipeline_jax.py and
    evolve_on_graph_jax.py). Calls the identical encode_and_restrict /
    order_parameter / reference_node_features functions used everywhere
    else in this pipeline -- not a separate reimplementation, just the
    first half of _process_one_image_multi_topology without the evolve
    loop. Returns theta0 itself (needed by the caller to build the batch
    uploaded to the GPU session)."""
    idx, image_01, active_indices_tuple, ref_idx, encoder_seed = args
    active_indices = np.array(active_indices_tuple)

    theta0 = s2a.encode_and_restrict(image_01, active_indices, seed=encoder_seed)
    R_pre = s2a.order_parameter(theta0)
    feat_pre = s2a.reference_node_features(theta0, ref_idx)

    return {
        "idx": idx, "theta0": theta0, "R_pre": R_pre, "feat_pre": feat_pre,
        "raw_feat": image_01.flatten(),
    }


def run_encode_only_multi_topology(images_01, ref_idx, active_indices, encoder_seeds=None):
    """CPU-only encode step for all images, no evolution. Returns
    (results, elapsed) where each result has idx/theta0/R_pre/feat_pre/raw_feat,
    sorted by idx. The caller stacks theta0 into a batch for GPU upload."""
    n_images = len(images_01)
    if encoder_seeds is None:
        encoder_seeds = [s2a.ENCODER_SEED] * n_images

    n_workers = max(1, mp.cpu_count() - 1)
    work_items = [(i, images_01[i], tuple(active_indices), ref_idx, encoder_seeds[i])
                  for i in range(n_images)]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_one_image_encode_only, work_items)
    elapsed = time.time() - t0

    results.sort(key=lambda r: r["idx"])
    return results, elapsed


def run_pipeline_multi_topology(images_01, labels, topologies, ref_idx, active_indices,
                                 encoder_seeds=None):
    """Stage 3: encode once per image, evolve on every topology in
    `topologies` (dict: name -> W). `active_indices`/`ref_idx` passed in
    directly (from stage2a_topologies.build_all_topologies()) rather than
    reloaded via s2a.load_T(), since stage 3 needs the SAME active
    support/nodes_T used to build the topologies themselves."""
    if encoder_seeds is None:
        encoder_seeds = [s2a.ENCODER_SEED] * len(images_01)

    topologies_items = list(topologies.items())
    n_workers = max(1, mp.cpu_count() - 1)
    work_items = [(i, images_01[i], tuple(active_indices), topologies_items, ref_idx, encoder_seeds[i])
                  for i in range(len(images_01))]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_one_image_multi_topology, work_items)
    elapsed = time.time() - t0

    results.sort(key=lambda r: r["idx"])
    return results, elapsed


def check_go_no_go_multi_topology(results, topology_names):
    """Multi-topology analog of check_go_no_go: solver failures and
    R(theta) are now per-topology, everything else (raw/pre-evolution
    finiteness) stays shared."""
    report = {"n_images": len(results)}

    non_finite_count = 0
    for r in results:
        if not np.all(np.isfinite(r["raw_feat"])):
            non_finite_count += 1
        if not np.all(np.isfinite(r["feat_pre"])):
            non_finite_count += 1
    report["n_non_finite_shared_feature_vectors"] = non_finite_count

    R_pre_vals = np.array([r["R_pre"] for r in results])
    report["R_pre_summary"] = {
        "min": float(R_pre_vals.min()), "max": float(R_pre_vals.max()),
        "mean": float(R_pre_vals.mean()), "median": float(np.median(R_pre_vals)),
        "n_below_0.01": int(np.sum(R_pre_vals < 0.01)), "n_above_0.99": int(np.sum(R_pre_vals > 0.99)),
    }

    per_topology = {}
    for name in topology_names:
        n_failed = sum(1 for r in results if r["evolved"][name]["solver_failed"])
        non_finite_evolved = sum(
            1 for r in results
            if not r["evolved"][name]["solver_failed"]
            and not np.all(np.isfinite(r["evolved"][name]["feat_post"])))
        R_post_vals = np.array([r["evolved"][name]["R_post"] for r in results
                                 if r["evolved"][name]["R_post"] is not None])
        per_topology[name] = {
            "n_solver_failed": n_failed,
            "solver_failure_rate": n_failed / len(results),
            "solver_failure_rate_ok": (n_failed / len(results)) <= 0.001,
            "n_non_finite_feature_vectors": non_finite_evolved,
            "non_finite_ok": non_finite_evolved == 0,
            "R_post_summary": {
                "min": float(R_post_vals.min()), "max": float(R_post_vals.max()),
                "mean": float(R_post_vals.mean()), "median": float(np.median(R_post_vals)),
                "n_below_0.01": int(np.sum(R_post_vals < 0.01)),
                "n_above_0.99": int(np.sum(R_post_vals > 0.99)),
            } if len(R_post_vals) else None,
        }
    report["per_topology"] = per_topology
    report["non_finite_ok"] = (report["n_non_finite_shared_feature_vectors"] == 0
                                and all(v["non_finite_ok"] for v in per_topology.values()))
    report["solver_failure_rate_ok"] = all(v["solver_failure_rate_ok"] for v in per_topology.values())
    return report


def run_classifier_conditions_multi_topology(results, labels, topology_names, label_prefix=""):
    """Stage 3's 6-condition version: raw pixels, encoded pre-evolution
    (both shared across topologies), plus one evolved condition per
    topology in topology_names. Same locked CV procedure per condition,
    each selecting its own C independently."""
    raw_X = np.stack([r["raw_feat"] for r in results])
    pre_X = np.stack([r["feat_pre"] for r in results])

    conditions_to_fit = [("raw_pixels", raw_X, labels), ("encoded_pre_evolution", pre_X, labels)]
    for name in topology_names:
        valid_mask = np.array([not r["evolved"][name]["solver_failed"] for r in results])
        evolved_X = np.stack([r["evolved"][name]["feat_post"] for r in results
                               if not r["evolved"][name]["solver_failed"]])
        evolved_y = labels[valid_mask]
        conditions_to_fit.append((f"evolved_{name}", evolved_X, evolved_y))

    conditions_out = {}
    for label, X, y in conditions_to_fit:
        full_label = f"{label_prefix}{label}"
        print(f"\nFitting condition: {full_label} (n={len(y)}, dim={X.shape[1]})")
        try:
            best_C, mean_val_loss, _non_convergence_log = s2a_clf.select_C_via_cv(X, y, full_label)
            conditions_out[label] = {
                "converged": True,
                "selected_C": best_C,
                "mean_val_loss_per_C": mean_val_loss,
                "mean_val_loss_at_selected_C": mean_val_loss[best_C],
            }
            print(f"  Converged in every fold/C. Selected C={best_C}, "
                  f"mean_val_loss_per_C={mean_val_loss}")
        except NonConvergenceError as e:
            conditions_out[label] = {"converged": False, "error": str(e)}
            print(f"  NON-CONVERGENCE (stage halts per locked gate): {e}")

    return conditions_out


def run_classifier_conditions(results, labels, label_prefix=""):
    """CV-selects C for all 3 conditions via the real, locked selection
    procedure (stage2a_classifier.select_C_via_cv) -- every (fold, C)
    combination is actually fit and checked for convergence. No
    held-out test evaluation here: the feasibility ladder never touches
    the official test set (or, for stage 1/2, any test set at all).
    label_prefix distinguishes conditions across ladder stages/robustness
    checks in logged output (e.g. 'stage2_seed0_')."""
    raw_X = np.stack([r["raw_feat"] for r in results])
    pre_X = np.stack([r["feat_pre"] for r in results])

    valid_mask = np.array([not r["solver_failed"] for r in results])
    evolved_X = np.stack([r["feat_post"] for r in results if not r["solver_failed"]])
    evolved_y = labels[valid_mask]

    conditions_out = {}
    for label, X, y in [
        ("raw_pixels", raw_X, labels),
        ("encoded_pre_evolution", pre_X, labels),
        ("evolved_T", evolved_X, evolved_y),
    ]:
        full_label = f"{label_prefix}{label}"
        print(f"\nFitting condition: {full_label} (n={len(y)}, dim={X.shape[1]})")
        try:
            best_C, mean_val_loss, _non_convergence_log = s2a_clf.select_C_via_cv(X, y, full_label)
            conditions_out[label] = {
                "converged": True,
                "selected_C": best_C,
                "mean_val_loss_per_C": mean_val_loss,
                "mean_val_loss_at_selected_C": mean_val_loss[best_C],
            }
            print(f"  Converged in every fold/C. Selected C={best_C}, "
                  f"mean_val_loss_per_C={mean_val_loss}")
        except NonConvergenceError as e:
            conditions_out[label] = {"converged": False, "error": str(e)}
            print(f"  NON-CONVERGENCE (stage halts per locked gate): {e}")

    return conditions_out
