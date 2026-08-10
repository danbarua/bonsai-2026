"""Pure calculations and guards for the frozen Stage 2B amendment audit.

This module deliberately does not load data or talk to GCS. The audit driver
owns artifact transport and calls these functions only after validated,
official-indexed inputs have been assembled.
"""
from itertools import combinations
import os

import numpy as np

from stage2b_conditions import ALL_CONDITIONS, EVOLVED_GRAPHS, PRE_EVOLUTION

AUDIT_STEPS = (150, 1200)
N_OFFICIAL_TRAIN = 60_000
GAUGE_NODE = 363
RESOLUTION_D = 1.151190e-12
CONTRAST_THRESHOLD = 4.604761e-10
PAIR_THRESHOLD = 9.209522e-10
ALPHA_EPS = (1e-10, 1e-11, 1e-12, 1e-13)
SENSITIVITY_STEPS = (75, 150, 300, 600, 1200)


class AuditInputError(ValueError):
    """An input violates a frozen audit contract."""


def assert_official_indices(indices, *, name="indices"):
    values = np.asarray(indices)
    if values.ndim != 1 or values.size != N_OFFICIAL_TRAIN:
        raise AuditInputError(
            f"{name} must contain all {N_OFFICIAL_TRAIN} official indices, "
            f"got shape {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise AuditInputError(f"{name} must be integer-valued")
    expected = np.arange(N_OFFICIAL_TRAIN, dtype=values.dtype)
    if not np.array_equal(np.sort(values), expected):
        raise AuditInputError(f"{name} is not a permutation of official indices")
    return values.astype(np.int64, copy=False)


def align_by_official_index(values, source_indices, target_indices):
    """Reorder rows by official index, never by positional prefix."""
    source = np.asarray(source_indices)
    target = np.asarray(target_indices)
    if source.ndim != 1 or target.ndim != 1:
        raise AuditInputError("source_indices and target_indices must be 1-D")
    if np.unique(source).size != source.size or np.unique(target).size != target.size:
        raise AuditInputError("official indices must be unique")
    positions = {int(index): row for row, index in enumerate(source)}
    try:
        rows = np.asarray([positions[int(index)] for index in target], dtype=np.int64)
    except KeyError as exc:
        raise AuditInputError(f"target index {exc.args[0]} is absent from source") from None
    return np.asarray(values)[rows]


def assert_same_audit_inputs(left, right):
    """Check that two encoded artifacts differ only in encoder steps."""
    left, right = dict(left), dict(right)
    differing = {key for key in set(left) | set(right)
                 if left.get(key) != right.get(key)}
    if differing - {"encoder_steps"}:
        raise AuditInputError(
            "150/1200 artifacts differ outside encoder_steps: "
            + ", ".join(sorted(differing - {"encoder_steps"})))
    if left.get("encoder_steps") == right.get("encoder_steps"):
        raise AuditInputError("audit artifacts must use distinct encoder steps")


def wrapped_phase_difference(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    try:
        np.broadcast_shapes(left.shape, right.shape)
    except ValueError as exc:
        raise AuditInputError(
            f"phase shapes are not broadcast-compatible: {left.shape} vs {right.shape}") from exc
    return (left - right + np.pi) % (2.0 * np.pi) - np.pi


def gauge_phases(theta, *, reference_column=GAUGE_NODE, active_indices=None):
    """Apply the production reference-node gauge to full or active phases."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2:
        raise AuditInputError("theta must be a 2-D per-image phase array")
    ref = int(reference_column)
    if active_indices is not None and theta.shape[1] != len(active_indices):
        active = np.asarray(active_indices)
        matches = np.flatnonzero(active == ref)
        if matches.size != 1:
            raise AuditInputError(f"reference node {ref} is absent from active indices")
        ref = int(matches[0])
    if ref < 0 or ref >= theta.shape[1]:
        raise AuditInputError(f"reference column {reference_column} is out of range")
    return wrapped_phase_difference(theta, theta[:, ref:ref + 1])


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise AuditInputError("distribution values must be finite and 1-D")
    return {"median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def feature_distances(theta_left, theta_right, *, active_indices=None,
                      reference_column=GAUGE_NODE, features_left=None,
                      features_right=None):
    """Compute the three frozen per-image feature-distance distributions."""
    left = gauge_phases(theta_left, reference_column=reference_column,
                        active_indices=active_indices)
    right = gauge_phases(theta_right, reference_column=reference_column,
                         active_indices=active_indices)
    phase_delta = wrapped_phase_difference(left, right)
    if features_left is None or features_right is None:
        raise AuditInputError("cos/sin features are required for Euclidean distance")
    fleft, fright = np.asarray(features_left), np.asarray(features_right)
    if fleft.shape != fright.shape or fleft.ndim != 2:
        raise AuditInputError("cos/sin feature arrays must have matching 2-D shapes")
    return {
        "phase_max": summarize_distribution(np.max(np.abs(phase_delta), axis=1)),
        "phase_rms": summarize_distribution(
            np.sqrt(np.mean(np.square(phase_delta), axis=1))),
        "cos_sin_euclidean": summarize_distribution(
            np.linalg.norm(fleft - fright, axis=1)),
    }


def audit_deltas(mse_by_condition):
    """Return evolved-minus-pre per-image MSE contrasts."""
    pre = np.asarray(mse_by_condition[PRE_EVOLUTION], dtype=np.float64)
    if pre.ndim != 1 or not np.all(np.isfinite(pre)):
        raise AuditInputError("pre-evolution MSE must be finite and 1-D")
    result = {}
    for graph in EVOLVED_GRAPHS:
        value = np.asarray(mse_by_condition[graph], dtype=np.float64)
        if value.shape != pre.shape or not np.all(np.isfinite(value)):
            raise AuditInputError(f"invalid MSE vector for {graph}")
        result[graph] = value - pre
    return result


def assert_oof_matches_fold_aggregates(oof_result, stored_cv, *, atol=1e-12):
    """Cross-check independent per-image OOF MSEs against stored fold means."""
    clipped = np.asarray(oof_result["oof_clipped_mse"], dtype=np.float64)
    folds = np.asarray(oof_result["fold_index"])
    expected = np.asarray(stored_cv["fold_clipped_val_mse"], dtype=np.float64)
    if clipped.ndim != 2 or folds.shape != (clipped.shape[0],):
        raise AuditInputError("invalid OOF result shape")
    observed = np.stack([
        [float(np.mean(clipped[folds == fold, alpha]))
         for alpha in range(clipped.shape[1])]
        for fold in range(expected.shape[0])
    ])
    if observed.shape != expected.shape or not np.allclose(
            observed, expected, rtol=0.0, atol=atol):
        raise AuditInputError(
            "independent OOF per-image means do not reproduce stored fold aggregates")
    return {"max_abs_diff": float(np.max(np.abs(observed - expected))),
            "atol": float(atol), "passed": True}


def compute_oof_alpha_regimes(features_by_budget, target, stratification,
                              production_alphas, ridge_module):
    """Compute both frozen alpha regimes using the production ridge module.

    ``ridge_module`` is injected so this audit layer cannot accidentally
    reimplement the SVD, scaler, fold splitter, clipping, or alpha rule.
    """
    ridge_module.assert_frozen_grid(ridge_module.ALPHA_GRID, where="audit")
    production_alphas = {str(condition): float(alpha)
                         for condition, alpha in production_alphas.items()}
    result = {"fixed": {}, "reselected": {}, "cross_checks": {}}
    for steps, features in features_by_budget.items():
        step_key = str(steps)
        result["fixed"][step_key] = {}
        result["reselected"][step_key] = {}
        result["cross_checks"][step_key] = {}
        for condition, X in features.items():
            cv = ridge_module.cross_validate_alpha(X, target, stratification)
            oof_all = ridge_module.oof_per_image_mse(X, target, stratification)
            result["reselected"][step_key][condition] = {
                "mse": oof_all["oof_clipped_mse"][:, int(cv["alpha_index"])],
                "alpha": float(cv["alpha"]),
            }
            result["fixed"][step_key][condition] = {
                "mse": ridge_module.oof_clipped_mse_at_alpha(
                    X, target, stratification, production_alphas[str(condition)]),
                "alpha": production_alphas[str(condition)],
            }
            result["cross_checks"][step_key][condition] = \
                assert_oof_matches_fold_aggregates(oof_all, cv)
    return result


def _sign_reversal(left, right):
    return bool((left < 0 < right) or (right < 0 < left))


def trigger_verdict(mse_by_budget, *, pair_threshold=PAIR_THRESHOLD,
                    contrast_threshold=CONTRAST_THRESHOLD):
    """Evaluate all three frozen triggers for one alpha regime."""
    if set(mse_by_budget) != set(AUDIT_STEPS):
        raise AuditInputError(f"expected budgets {AUDIT_STEPS}")
    deltas = {steps: audit_deltas(mse_by_budget[steps]) for steps in AUDIT_STEPS}
    primary = {steps: float(np.mean(deltas[steps]["T"])) for steps in AUDIT_STEPS}
    sign_reversals = {graph: _sign_reversal(
        float(np.mean(deltas[150][graph])), float(np.mean(deltas[1200][graph])))
        for graph in EVOLVED_GRAPHS}
    means = {steps: {condition: float(np.mean(values))
                     for condition, values in mse_by_budget[steps].items()}
             for steps in AUDIT_STEPS}
    pair_results = {}
    for left, right in combinations(EVOLVED_GRAPHS, 2):
        before = means[150][left] - means[150][right]
        after = means[1200][left] - means[1200][right]
        change = after - before
        pair_results[f"{left}__vs__{right}"] = {
            "before": before, "after": after, "change": change,
            "order_reversed": bool(before * after < 0),
            "resolved": bool(before * after < 0 and abs(change) > pair_threshold),
        }
    return {
        "primary_contrast": primary,
        "primary_sign_reversal": _sign_reversal(primary[150], primary[1200]),
        "graph_sign_reversals": sign_reversals,
        "pairwise": pair_results,
        "triggered": bool(_sign_reversal(primary[150], primary[1200])
                           or any(sign_reversals.values())
                           or any(item["resolved"] for item in pair_results.values())),
        "thresholds": {"contrast": contrast_threshold, "pair_ordering": pair_threshold},
        "sign_convention": "Delta_g = MSE_evolved_g - MSE_pre_evolution",
    }


def build_stress_indices(official_indices, labels, largest_discrepancy_indices,
                         positive_delta_indices, *, seed=42, tail_cap=500):
    """Construct the frozen ARM/x86 stress set deterministically."""
    indices = np.asarray(official_indices, dtype=np.int64)
    labels = np.asarray(labels)
    assert_official_indices(indices, name="official_indices")
    if labels.shape != indices.shape:
        raise AuditInputError("labels must align with official_indices")
    positive = np.asarray(positive_delta_indices, dtype=np.int64)
    if positive.size > tail_cap:
        raise AuditInputError("positive final-Delta input must already be capped/ranked")
    rng = np.random.default_rng(seed)
    random_part = np.concatenate([
        rng.choice(indices[labels == cls], size=20, replace=False)
        for cls in range(10)
    ])
    selected = set(map(int, np.concatenate([
        np.sort(np.asarray(largest_discrepancy_indices, dtype=np.int64))[:100],
        positive, random_part])))
    for cls in range(10):
        members = np.sort(indices[labels == cls])
        present = sum(int(labels[i] == cls) for i in selected)
        selected.update(map(int, members[:max(0, 20 - present)]))
    return np.asarray(sorted(selected), dtype=np.int64)


def capped_positive_delta_indices(train_indices, deltas, *, tail_cap=500):
    """Official indices with final-Delta strictly > 0, capped/ranked for build_stress_indices.

    Rank by delta descending; ties broken by official index ascending.
    Record true_count before cap. Returns (indices_int64_len_le_tail_cap, meta_dict)
    where meta_dict = {"true_count": int, "cap": int, "cap_applied": bool}.
    """
    indices = np.asarray(train_indices, dtype=np.int64)
    values = np.asarray(deltas, dtype=np.float64)
    if indices.ndim != 1 or values.shape != indices.shape:
        raise AuditInputError(
            f"train_indices and deltas must be 1-D and aligned, got "
            f"{indices.shape} vs {values.shape}")
    if not np.all(np.isfinite(values)):
        raise AuditInputError("deltas must be finite")
    if tail_cap < 0:
        raise AuditInputError("tail_cap must be non-negative")
    mask = values > 0.0
    positive_indices = indices[mask]
    positive_deltas = values[mask]
    true_count = int(positive_indices.size)
    # lexsort: last key is primary. Primary = -delta, secondary = index asc.
    order = np.lexsort((positive_indices, -positive_deltas))
    ranked = positive_indices[order]
    cap_applied = true_count > int(tail_cap)
    selected = ranked[:int(tail_cap)] if cap_applied else ranked
    meta = {
        "true_count": true_count,
        "cap": int(tail_cap),
        "cap_applied": bool(cap_applied),
    }
    return selected.astype(np.int64, copy=False), meta


def rank_discrepancy_indices(official_indices, thetas_left, thetas_right):
    """Order official indices by per-image max abs coordinate difference.

    thetas_* shape (n, d), rows aligned to official_indices.
    Sort key: (-max_abs_diff, official_index). Returns int64 array length n.
    """
    indices = np.asarray(official_indices, dtype=np.int64)
    left = np.asarray(thetas_left, dtype=np.float64)
    right = np.asarray(thetas_right, dtype=np.float64)
    if indices.ndim != 1:
        raise AuditInputError("official_indices must be 1-D")
    if left.shape != right.shape:
        raise AuditInputError(
            f"thetas_left/right shape mismatch: {left.shape} vs {right.shape}")
    if left.ndim != 2 or left.shape[0] != indices.shape[0]:
        raise AuditInputError(
            f"thetas must be (n, d) aligned to official_indices; got "
            f"{left.shape} vs n={indices.shape[0]}")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise AuditInputError("thetas must be finite")
    max_abs = np.max(np.abs(left - right), axis=1)
    order = np.lexsort((indices, -max_abs))
    return indices[order].astype(np.int64, copy=False)


def max_abs_difference(left, right):
    """Finite max abs elementwise difference; raises AuditInputError on shape/non-finite."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise AuditInputError(
            f"max_abs_difference shape mismatch: {a.shape} vs {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise AuditInputError("max_abs_difference requires finite inputs")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def evaluate_propagation_halt(delta_g_max_abs_by_graph, *, threshold=CONTRAST_THRESHOLD):
    """Protocol 1 stage-5 halt.

    delta_g_max_abs_by_graph: dict graph -> float (max over stress images of
    |Delta_g_arm - Delta_g_x86|).
    Returns {
      "halt_triggered": bool,  # True if ANY graph value > threshold (strict >)
      "threshold": float,
      "per_graph": {g: {"max_abs_delta_g_diff": float, "exceeds": bool}},
      "exceeding_graphs": [str, ...],  # stable EVOLVED_GRAPHS order
    }
    """
    if not isinstance(delta_g_max_abs_by_graph, dict):
        raise AuditInputError("delta_g_max_abs_by_graph must be a dict")
    threshold = float(threshold)
    per_graph = {}
    exceeding = []
    for graph in EVOLVED_GRAPHS:
        if graph not in delta_g_max_abs_by_graph:
            raise AuditInputError(
                f"delta_g_max_abs_by_graph missing graph {graph!r}")
        value = float(delta_g_max_abs_by_graph[graph])
        if not np.isfinite(value):
            raise AuditInputError(
                f"delta_g max-abs for {graph} is not finite: {value!r}")
        exceeds = bool(value > threshold)  # strict; equality does not halt
        per_graph[graph] = {
            "max_abs_delta_g_diff": value,
            "exceeds": exceeds,
        }
        if exceeds:
            exceeding.append(graph)
    return {
        "halt_triggered": bool(exceeding),
        "threshold": threshold,
        "per_graph": per_graph,
        "exceeding_graphs": exceeding,
    }


def propagation_stage_maxima(
        *,
        theta_arm, theta_x86,
        features_arm, features_x86,
        pred_arm, pred_x86,
        mse_arm, mse_x86,
        delta_g_arm, delta_g_x86,
):
    """Return the frozen five-stage max-abs cross-arch report.

    Stages (always all five):
      1. encoding: max_abs_difference(theta_arm, theta_x86)
      2. evolved_features: per EVOLVED_GRAPHS + pre_evolution
      3. prediction: per ALL_CONDITIONS
      4. per_image_mse: per condition
      5. delta_g: per EVOLVED_GRAPHS
    """
    encoding = max_abs_difference(theta_arm, theta_x86)

    feature_keys = (PRE_EVOLUTION, *EVOLVED_GRAPHS)
    evolved_features = {}
    for key in feature_keys:
        if key not in features_arm or key not in features_x86:
            raise AuditInputError(
                f"features missing condition {key!r} on one or both arches")
        evolved_features[key] = max_abs_difference(
            features_arm[key], features_x86[key])

    prediction = {}
    per_image_mse = {}
    for condition in ALL_CONDITIONS:
        if condition not in pred_arm or condition not in pred_x86:
            raise AuditInputError(
                f"predictions missing condition {condition!r}")
        if condition not in mse_arm or condition not in mse_x86:
            raise AuditInputError(
                f"mse missing condition {condition!r}")
        prediction[condition] = max_abs_difference(
            pred_arm[condition], pred_x86[condition])
        per_image_mse[condition] = max_abs_difference(
            mse_arm[condition], mse_x86[condition])

    delta_g = {}
    for graph in EVOLVED_GRAPHS:
        if graph not in delta_g_arm or graph not in delta_g_x86:
            raise AuditInputError(
                f"delta_g missing graph {graph!r} on one or both arches")
        delta_g[graph] = max_abs_difference(
            delta_g_arm[graph], delta_g_x86[graph])

    return {
        "encoding": encoding,
        "evolved_features": evolved_features,
        "prediction": prediction,
        "per_image_mse": per_image_mse,
        "delta_g": delta_g,
    }



def sensitivity_table(final_delta_by_steps, evaluate_gate):
    """Recompute the gate without re-encoding, varying only abs_conv_eps."""
    table = {}
    for steps in SENSITIVITY_STEPS:
        if steps not in final_delta_by_steps:
            continue
        clean, noisy = final_delta_by_steps[steps]
        table[str(steps)] = {
            str(eps): evaluate_gate(clean, noisy, abs_conv_eps=eps)
            for eps in ALPHA_EPS
        }
    return table


def require_audit_prerequisites(encoded_150, production_ridge_rerun):
    """Refuse execution until the load-bearing sequencing gate is met."""
    missing = [path for path in (encoded_150, production_ridge_rerun)
               if not path or not os.path.exists(path)]
    if missing:
        raise AuditInputError(
            "audit sequencing gate is closed; missing prerequisite(s): "
            + ", ".join(map(str, missing)))
