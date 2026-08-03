"""
Stage 2A: the locked confirmatory evaluation (DESIGN.md, "Confirmatory
endpoint and test"). The official 10,000-image KMNIST test set is
touched here for the first and only time in this project.

No new hyperparameter search: each condition's C was already selected
via full-training-set CV in feasibility stage 3
(stage3_classifier_conditions.pkl). This script only does the single
locked final refit (fresh scaler + classifier on the complete 60,000-
image training set, at the already-selected C) and applies it, unchanged,
to the test set -- per stage2a_classifier.fit_final_at_selected_C.

Primary: T-evolved vs. encoded-pre-evolution (the sole primary Level 3
comparison, DESIGN.md). Secondary: lattice/rewired/curr_random each vs.
encoded-pre-evolution -- graph-specific, cannot rescue a null primary
result, no cross-comparison correction. Raw pixels and both MLP
baselines are reported for context only, never part of the locked
primary/secondary comparisons.
"""
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
from stage2a_classifier import fit_final_at_selected_C, NonConvergenceError

TRAIN_SCRATCH = "/private/tmp/claude-501/-Users-dan-Code-pycharm-bonsai-2026/54a406a1-f8d0-41df-bc2a-d46e08e68715/scratchpad/stage2a_gpu_stage3"
TEST_SCRATCH = "/private/tmp/claude-501/-Users-dan-Code-pycharm-bonsai-2026/54a406a1-f8d0-41df-bc2a-d46e08e68715/scratchpad/stage2a_gpu_stage4_test"
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]
N_RESAMPLES = 20000
BOOTSTRAP_SEED = 42


def compute_feat_post_for_split(encode_local, gpu_results, ref_idx):
    """Reuses stage2a_core's own order_parameter/reference_node_features
    (unchanged) to turn GPU-evolved theta_T into R_post/feat_post per
    topology, for either the train or test split -- identical approach to
    analyze_stage3_results.py's build_results_structure, factored out
    here since both splits need it."""
    n_images = encode_local["n_images"]
    feat_post = {}
    R_post = {}
    solver_failed = {}
    for name in TOPOLOGY_NAMES:
        theta_T = gpu_results["results"][name]["theta_T"]
        success = gpu_results["results"][name]["success"]
        failed = ~success
        n_failed = int(failed.sum())
        assert n_failed == 0, (
            f"[{name}] {n_failed} solver failures found -- this script assumes zero, "
            f"per every prior stage's measured result; failure handling was not "
            f"implemented since it was never needed. Investigate before proceeding.")
        fp = np.stack([s2a.reference_node_features(theta_T[i], ref_idx) for i in range(n_images)])
        rp = np.array([s2a.order_parameter(theta_T[i]) for i in range(n_images)])
        feat_post[name] = fp
        R_post[name] = rp
        solver_failed[name] = failed
    return feat_post, R_post, solver_failed


def per_image_log_loss(y_true, proba, classes):
    """ell_i for each image -- sklearn's log_loss gives only the mean;
    the locked test needs the per-image value d_i is built from."""
    class_to_col = {c: j for j, c in enumerate(classes)}
    cols = np.array([class_to_col[y] for y in y_true])
    p_true = proba[np.arange(len(y_true)), cols]
    eps = 1e-15
    p_true = np.clip(p_true, eps, 1 - eps)
    return -np.log(p_true)


def paired_class_stratified_bootstrap(d, y, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """DESIGN.md's locked primary test: 20,000 paired, class-stratified
    bootstrap resamples (each resample preserves each class's original
    count, drawn with replacement within class), mean per-image d_i on
    each resample, two-sided 95% percentile interval. Vectorized per
    class rather than materializing a full resampled index array per
    draw."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    total_n = len(d)
    sums = np.zeros(n_resamples)
    for c in classes:
        idx_c = np.where(y == c)[0]
        n_c = len(idx_c)
        d_c = d[idx_c]
        draws = rng.integers(0, n_c, size=(n_resamples, n_c))
        sums += d_c[draws].sum(axis=1)
    means = sums / total_n
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "resampled_means": means, "ci_low": float(lo), "ci_high": float(hi),
        "observed_mean": float(np.mean(d)),
    }


def mcnemar_exact(y_true, pred_a, pred_b, label_a, label_b):
    """Exact McNemar's test on the discordant pairs (A wrong/B right vs.
    A right/B wrong), via a two-sided exact binomial test on the
    discordant counts -- the standard 'exact McNemar' construction,
    implemented directly via scipy.stats.binomtest rather than adding a
    new dependency for it."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n_b_only = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n_a_only = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    n_discordant = n_a_only + n_b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        k = min(n_a_only, n_b_only)
        p_value = binomtest(k, n_discordant, 0.5, alternative="two-sided").pvalue
    return {
        f"n_{label_a}_only_correct": n_a_only, f"n_{label_b}_only_correct": n_b_only,
        "n_discordant": n_discordant, "p_value": float(p_value),
    }


def summarize_condition(y_true, y_pred, proba, classes, label):
    acc = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=classes)
    mean_logloss = float(log_loss(y_true, proba, labels=classes))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"  [{label}] test accuracy={acc:.4f}, macro-F1={macro_f1:.4f}, "
          f"mean log-loss={mean_logloss:.4f}")
    print(f"    per-class recall: {np.round(per_class_recall, 4).tolist()}")
    return {
        "accuracy": acc, "macro_f1": macro_f1, "mean_log_loss": mean_logloss,
        "per_class_recall": per_class_recall.tolist(), "confusion_matrix": cm.tolist(),
    }


def main():
    print("=" * 70)
    print("STAGE 2A CONFIRMATORY EVALUATION -- official test set touched "
          "for the first and only time")
    print("=" * 70)

    print("\nLoading training-side artifacts (feasibility stage 3)...")
    with open(os.path.join(TRAIN_SCRATCH, "stage3_encode_local.pkl"), "rb") as f:
        train_encode = pickle.load(f)
    with open(os.path.join(TRAIN_SCRATCH, "stage3_gpu_results.pkl"), "rb") as f:
        train_gpu = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, "stage3_classifier_conditions.pkl"), "rb") as f:
        stage3_conditions = pickle.load(f)
    selected_C = {label: c["selected_C"] for label, c in stage3_conditions["conditions"].items()
                  if c.get("converged", False)}
    print(f"Selected C per condition (from stage 3 CV, reused unchanged): {selected_C}")

    ref_idx = train_encode["ref_idx"]
    y_train = train_encode["labels"]
    print(f"\nComputing training-set feat_post from GPU-evolved theta_T "
          f"({train_encode['n_images']} images x {len(TOPOLOGY_NAMES)} topologies)...")
    t0 = time.time()
    train_feat_post, train_R_post, _train_failed = compute_feat_post_for_split(
        train_encode, train_gpu, ref_idx)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\nLoading test-side artifacts (this project's first and only touch "
          "of the official test set)...")
    with open(os.path.join(TEST_SCRATCH, "stage4_encode_local.pkl"), "rb") as f:
        test_encode = pickle.load(f)
    with open(os.path.join(TEST_SCRATCH, "stage4_gpu_results.pkl"), "rb") as f:
        test_gpu = pickle.load(f)
    assert test_encode["ref_idx"] == ref_idx
    y_test = test_encode["labels"]
    print(f"Official test set: {test_encode['n_images']} images")

    print(f"\nComputing test-set feat_post from GPU-evolved theta_T...")
    test_feat_post, test_R_post, _test_failed = compute_feat_post_for_split(
        test_encode, test_gpu, ref_idx)

    classes = np.unique(y_train)
    assert np.array_equal(classes, np.unique(y_test))

    X_train_by_cond = {
        "raw_pixels": train_encode["raw_feat"],
        "encoded_pre_evolution": train_encode["feat_pre"],
        "evolved_T": train_feat_post["T"],
        "evolved_lattice": train_feat_post["lattice"],
        "evolved_rewired": train_feat_post["rewired"],
        "evolved_curr_random": train_feat_post["curr_random"],
    }
    X_test_by_cond = {
        "raw_pixels": test_encode["raw_feat"],
        "encoded_pre_evolution": test_encode["feat_pre"],
        "evolved_T": test_feat_post["T"],
        "evolved_lattice": test_feat_post["lattice"],
        "evolved_rewired": test_feat_post["rewired"],
        "evolved_curr_random": test_feat_post["curr_random"],
    }

    print("\n" + "=" * 70)
    print("FINAL REFIT (full 60,000-image training set, stage-3-selected C, "
          "no new CV search) AND TEST-SET EVALUATION")
    print("=" * 70)
    condition_results = {}
    for label, best_C in selected_C.items():
        print(f"\nCondition: {label} (C={best_C}, dim={X_train_by_cond[label].shape[1]})")
        t0 = time.time()
        fit = fit_final_at_selected_C(
            X_train_by_cond[label], y_train, X_test_by_cond[label], best_C, label)
        fit_elapsed = time.time() - t0
        proba = fit["classifier"].predict_proba(fit["X_test_standardized"])
        y_pred = fit["classifier"].classes_[np.argmax(proba, axis=1)]
        ell_i = per_image_log_loss(y_test, proba, fit["classifier"].classes_)
        summary = summarize_condition(y_test, y_pred, proba, classes, label)
        condition_results[label] = {
            **summary, "selected_C": best_C, "fit_elapsed_seconds": fit_elapsed,
            "final_n_iter": fit["final_n_iter"], "y_pred": y_pred, "ell_i": ell_i,
        }
        print(f"  final refit + test eval: {fit_elapsed:.1f}s, n_iter={fit['final_n_iter']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("PRIMARY COMPARISON: evolved_T vs. encoded_pre_evolution")
    print("=" * 70)
    d_primary = condition_results["evolved_T"]["ell_i"] - condition_results["encoded_pre_evolution"]["ell_i"]
    boot_primary = paired_class_stratified_bootstrap(d_primary, y_test)
    print(f"Observed mean d_i (evolved_T - pre_evolution): {boot_primary['observed_mean']:.6f}")
    print(f"20,000-resample 95% percentile interval: "
          f"[{boot_primary['ci_low']:.6f}, {boot_primary['ci_high']:.6f}]")
    if boot_primary["ci_high"] < 0:
        primary_verdict = "IMPROVEMENT (entire interval below zero)"
    elif boot_primary["ci_low"] > 0:
        primary_verdict = "PRE-EVOLUTION WINS (entire interval above zero)"
    else:
        primary_verdict = "NULL (interval straddles zero)"
    print(f"PRIMARY VERDICT: {primary_verdict}")

    mcnemar_primary = mcnemar_exact(
        y_test, condition_results["encoded_pre_evolution"]["y_pred"],
        condition_results["evolved_T"]["y_pred"], "pre_evolution", "evolved_T")
    print(f"McNemar's exact test (pre_evolution vs. evolved_T): {mcnemar_primary}")

    print("\n" + "=" * 70)
    print("SECONDARY COMPARISONS: evolved_{lattice,rewired,curr_random} vs. "
          "encoded_pre_evolution (graph-specific, no cross-comparison correction)")
    print("=" * 70)
    secondary_results = {}
    for name in ["evolved_lattice", "evolved_rewired", "evolved_curr_random"]:
        d_sec = condition_results[name]["ell_i"] - condition_results["encoded_pre_evolution"]["ell_i"]
        boot_sec = paired_class_stratified_bootstrap(d_sec, y_test)
        if boot_sec["ci_high"] < 0:
            verdict = "IMPROVEMENT"
        elif boot_sec["ci_low"] > 0:
            verdict = "PRE-EVOLUTION WINS"
        else:
            verdict = "NULL (straddles zero)"
        mcnemar_sec = mcnemar_exact(
            y_test, condition_results["encoded_pre_evolution"]["y_pred"],
            condition_results[name]["y_pred"], "pre_evolution", name)
        print(f"\n[{name} vs. pre_evolution] observed mean d_i={boot_sec['observed_mean']:.6f}, "
              f"95% CI=[{boot_sec['ci_low']:.6f}, {boot_sec['ci_high']:.6f}] -> {verdict}")
        print(f"  McNemar: {mcnemar_sec}")
        secondary_results[name] = {"bootstrap": boot_sec, "verdict": verdict, "mcnemar": mcnemar_sec}

    print("\n" + "=" * 70)
    print("MLP BASELINES (context only -- never part of locked primary/secondary "
          "comparisons)")
    print("=" * 70)
    X_train_raw = train_encode["raw_feat"]
    X_test_raw = test_encode["raw_feat"]
    scaler_mlp = StandardScaler().fit(X_train_raw)
    X_train_raw_s = scaler_mlp.transform(X_train_raw)
    X_test_raw_s = scaler_mlp.transform(X_test_raw)

    mlp_results = {}
    for H, mlp_label in [(13, "MLP_H13_param_matched"), (128, "MLP_H128_competent_context")]:
        print(f"\n{mlp_label} (hidden_layer_sizes=({H},))...")
        n_params = 784 * H + H + H * 10 + 10
        t0 = time.time()
        clf = MLPClassifier(hidden_layer_sizes=(H,), activation="relu", solver="adam",
                             alpha=1e-4, batch_size=256, max_iter=200, random_state=42,
                             early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
        clf.fit(X_train_raw_s, y_train)
        train_elapsed = time.time() - t0
        proba = clf.predict_proba(X_test_raw_s)
        y_pred = clf.classes_[np.argmax(proba, axis=1)]
        summary = summarize_condition(y_test, y_pred, proba, classes, mlp_label)
        mlp_results[mlp_label] = {
            **summary, "n_params": n_params, "hidden_units": H,
            "n_iter": int(clf.n_iter_), "train_elapsed_seconds": train_elapsed,
        }
        print(f"  n_params={n_params}, n_iter={clf.n_iter_}, train_elapsed={train_elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("RAW PIXELS (context only)")
    print("=" * 70)
    print(f"  accuracy={condition_results['raw_pixels']['accuracy']:.4f}, "
          f"macro-F1={condition_results['raw_pixels']['macro_f1']:.4f}, "
          f"log-loss={condition_results['raw_pixels']['mean_log_loss']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "stage4_confirmatory_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "condition_results": {k: {kk: vv for kk, vv in v.items() if kk not in ("y_pred", "ell_i")}
                                   for k, v in condition_results.items()},
            "condition_y_pred": {k: v["y_pred"] for k, v in condition_results.items()},
            "condition_ell_i": {k: v["ell_i"] for k, v in condition_results.items()},
            "primary": {"bootstrap": boot_primary, "verdict": primary_verdict, "mcnemar": mcnemar_primary},
            "secondary": secondary_results,
            "mlp_results": mlp_results,
            "y_test": y_test,
        }, f)
    print(f"\nSaved full confirmatory results to {out_path}")
    print("\n" + "=" * 70)
    print("CONFIRMATORY EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
