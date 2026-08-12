"""The gauge comparison, run on Stage 2A's own train split.

Pre-registered in GAUGE_COMPARISON_2A_PREREGISTRATION.md (committed
917088f, amended 1bbfc89) BEFORE any classifier was fitted under either
gauge. This script implements that registration and nothing else.

What it does: for each of the five GAUGED conditions, fit the locked
Stage 2A CV procedure twice -- once under the locked reference-node gauge,
once under the circular-mean gauge -- holding everything else fixed, and
report whether the condition ordering survives the swap.

TRAIN SPLIT ONLY. Nothing here reads the 10,000-image official test set;
that evaluation was one-shot and is already spent on the locked result.
`raw_pixels` is not fitted at all: it is gauge-invariant, so it carries no
information about a gauge swap and only costs time.

One (condition, gauge) arm per invocation, each writing its own JSON, so
arms parallelize across cores and a killed run loses one arm rather than
all ten. `--combine` reads the arms back and applies the registered rule.

Usage
-----
    python run_gauge_comparison_2a.py --gate
    python run_gauge_comparison_2a.py --verify-cv
    python run_gauge_comparison_2a.py --arm evolved_T:reference
    python run_gauge_comparison_2a.py --combine
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
import stage2a_classifier as s2a_clf
from stage2a_paths import train_scratch_dir

SCRATCH_DIR = train_scratch_dir()
OUT_DIR = os.path.join(_THIS_DIR, "results", "gauge_comparison_2a")

# The five gauged conditions. raw_pixels is deliberately absent -- it is
# gauge-invariant, so it cannot inform a gauge swap.
GAUGED_CONDITIONS = ("encoded_pre_evolution", "evolved_T", "evolved_lattice",
                     "evolved_rewired", "evolved_curr_random")
TOPOLOGY_NAMES = ("T", "lattice", "rewired", "curr_random")
GAUGES = ("reference", "circular_mean")

# Registered in the pre-registration; not derived from any observed value.
THETA = 0.002
TIE_DECIMALS = 5


def locked_record_path():
    """Where stage3_classifier_conditions.pkl actually is.

    The locked record is a gitignored artifact, so a git worktree checked
    out from this repo has the tracked code but not the pickle -- it lives
    only in whichever checkout produced it. Resolve the main checkout via
    git's common dir rather than assuming the record sits next to this file,
    and name both candidates if neither exists.
    """
    candidates = [os.path.join(_THIS_DIR, "results",
                               "stage3_classifier_conditions.pkl")]
    try:
        import subprocess
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=_THIS_DIR, capture_output=True, text=True, check=True).stdout.strip()
        main_root = os.path.dirname(common)          # strip the trailing /.git
        candidates.append(os.path.join(
            main_root, "experiments", "stage2a_dynamics_classification",
            "results", "stage3_classifier_conditions.pkl"))
    except Exception:
        pass

    for path in candidates:
        if os.path.exists(path):
            return path
    raise SystemExit("the locked Stage 2A record was not found. It is gitignored, "
                     "so a fresh worktree will not have it. Looked in:\n  "
                     + "\n  ".join(candidates))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def load_encode_local():
    path = os.path.join(SCRATCH_DIR, "stage3_encode_local.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_theta0():
    """Reassemble theta0 from the upload chunks, in chunk-index order."""
    chunks = sorted(f for f in os.listdir(SCRATCH_DIR)
                    if f.startswith("theta0_chunk_") and f.endswith(".npy"))
    if not chunks:
        raise FileNotFoundError(f"no theta0 chunks in {SCRATCH_DIR}")
    return np.concatenate([np.load(os.path.join(SCRATCH_DIR, c)) for c in chunks])


def load_theta_T():
    path = os.path.join(SCRATCH_DIR, "stage3_gpu_results.pkl")
    with open(path, "rb") as fh:
        gpu = pickle.load(fh)
    theta = {n: np.asarray(gpu["results"][n]["theta_T"]) for n in TOPOLOGY_NAMES}
    success = {n: np.asarray(gpu["results"][n]["success"], dtype=bool)
               for n in TOPOLOGY_NAMES}
    return theta, success


def gauge_features(theta_batch, gauge, ref_idx):
    """Apply a gauge to (n_images, n_nodes) phases.

    Both gauges are called from stage2a_core -- never reimplemented, and in
    particular the inlined JAX gauge in analyze_stage3_results_jax.py is not
    ported here (principle 16, and the pre-registration pins this).
    """
    if gauge == "reference":
        return np.stack([s2a.reference_node_features(t, ref_idx) for t in theta_batch])
    if gauge == "circular_mean":
        return np.stack([s2a.circular_mean_features(t) for t in theta_batch])
    raise ValueError(f"unknown gauge {gauge!r}")


# --------------------------------------------------------------------------
# The pre-fit gate
# --------------------------------------------------------------------------

def run_gate():
    """Reference-gauge features recomputed from the theta0 chunks must equal
    the cached feat_pre EXACTLY. Same function, same inputs -- any difference
    is a chunk-ordering or indexing error, caught before a fit runs."""
    local = load_encode_local()
    theta0 = load_theta0()
    ref_idx = int(local["ref_idx"])
    feat_pre = local["feat_pre"]

    print(f"theta0 {theta0.shape} sha={_sha(theta0)}")
    print(f"feat_pre {feat_pre.shape} sha={_sha(feat_pre)}")
    if theta0.shape[0] != feat_pre.shape[0]:
        raise SystemExit(f"GATE FAILED: {theta0.shape[0]} states vs "
                         f"{feat_pre.shape[0]} cached feature rows")

    recomputed = gauge_features(theta0, "reference", ref_idx)
    exact = np.array_equal(recomputed, feat_pre)
    max_abs = float(np.max(np.abs(recomputed - feat_pre)))
    n_diff = int(np.count_nonzero(recomputed != feat_pre))

    print(f"recomputed sha={_sha(recomputed)}")
    print(f"byte-exact: {exact}   max|diff|={max_abs:.3e}   differing elements={n_diff}")
    if not exact:
        raise SystemExit("GATE FAILED: recomputed reference-gauge features differ "
                         "from the cached feat_pre. Investigate before fitting.")
    print("GATE PASSED: cached feat_pre reproduces exactly from the theta0 chunks.")


# --------------------------------------------------------------------------
# CV that records accuracy alongside the locked log-loss selection
# --------------------------------------------------------------------------

def cv_with_accuracy(X, y, label, c_grid=None, n_folds=None, seed=None):
    """The locked select_C_via_cv procedure, additionally recording per-fold
    validation ACCURACY for every (fold, C).

    This mirrors stage2a_classifier.select_C_via_cv rather than calling it,
    because that function computes accuracy nowhere and discards the fitted
    classifiers. The mirror is not trusted on inspection: --verify-cv asserts
    it reproduces the real function's (best_C, mean_val_loss) exactly on a
    subsample. Every fit goes through the locked s2a_clf._fit_one, and the
    fold splits, scaler placement, grid and tie-break are the locked ones.
    """
    c_grid = list(s2a_clf.C_GRID if c_grid is None else c_grid)
    n_folds = s2a_clf.N_FOLDS if n_folds is None else n_folds
    seed = s2a_clf.SEED if seed is None else seed

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_C_loss = {C: [] for C in c_grid}
    per_C_acc = {C: [] for C in c_grid}
    # Iteration counts and wall times per (fold, C). Recorded rather than only
    # printed: they are the evidence for conditioning and cost claims, and a
    # number that lives only in a console log is not reproducible from
    # committed code (principle 24).
    fits = []
    labels_sorted = sorted(set(y))

    for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]
        scaler = StandardScaler().fit(X_tr)          # fold-safe, as locked
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        for C in c_grid:
            t0 = time.time()
            clf, converged, n_iter = s2a_clf._fit_one(X_tr_s, y_tr, C)
            if not converged:
                # The locked stop-gate. Registered as a halt-and-report, not
                # something to patch around by raising max_iter.
                raise s2a_clf.NonConvergenceError(
                    f"[{label}] fold={fold_idx} C={C}: did not converge in "
                    f"{n_iter} iterations "
                    f"(max_iter={s2a_clf.CLASSIFIER_KWARGS['max_iter']}).")
            proba = clf.predict_proba(X_va_s)
            per_C_loss[C].append(log_loss(y_va, proba, labels=labels_sorted))
            per_C_acc[C].append(float(np.mean(clf.predict(X_va_s) == y_va)))
            fits.append({"fold": fold_idx, "C": float(C),
                         "loss": per_C_loss[C][-1], "acc": per_C_acc[C][-1],
                         "n_iter": int(n_iter), "seconds": time.time() - t0})
            print(f"    fold={fold_idx} C={C:<8g} loss={per_C_loss[C][-1]:.6f} "
                  f"acc={per_C_acc[C][-1]:.6f} iter={n_iter} "
                  f"({fits[-1]['seconds']:.1f}s)", flush=True)

    mean_loss = {C: float(np.mean(v)) for C, v in per_C_loss.items()}
    mean_acc = {C: float(np.mean(v)) for C, v in per_C_acc.items()}
    lo = min(mean_loss.values())
    tied = [C for C, v in mean_loss.items() if abs(v - lo) < 1e-12]
    best_C = min(tied)                                # locked tie-break
    return best_C, mean_loss, mean_acc, per_C_acc, fits


def run_verify_cv(n_sub=2000):
    """Principle 16: the mirror above is glue code, so show it agrees with the
    real locked function rather than assuming it does."""
    local = load_encode_local()
    y = np.asarray(local["labels"])
    X = np.asarray(local["feat_pre"])
    rng = np.random.default_rng(0)
    sub = rng.choice(len(y), size=n_sub, replace=False)
    X, y = X[sub], y[sub]

    print(f"verifying the CV mirror against stage2a_classifier.select_C_via_cv "
          f"on n={n_sub}")
    t0 = time.time()
    mine_C, mine_loss, mine_acc, _, _ = cv_with_accuracy(X, y, "verify")
    t_mine = time.time() - t0
    t0 = time.time()
    real_C, real_loss, _ = s2a_clf.select_C_via_cv(X, y, "verify")
    t_real = time.time() - t0

    ok_C = (mine_C == real_C)
    max_d = max(abs(mine_loss[C] - real_loss[C]) for C in real_loss)
    print(f"\nbest_C   mirror={mine_C}  locked={real_C}  match={ok_C}")
    print(f"max |mean_val_loss difference| = {max_d:.3e}")
    print(f"timing   mirror={t_mine:.1f}s  locked={t_real:.1f}s")
    print(f"accuracy at selected C = {mine_acc[mine_C]:.6f} "
          f"(the locked function computes no accuracy at all)")
    if not ok_C or max_d > 1e-12:
        raise SystemExit("VERIFY FAILED: the CV mirror does not reproduce the "
                         "locked procedure. Do not run any arm.")
    print("VERIFY PASSED: mirror reproduces the locked selection exactly.")


# --------------------------------------------------------------------------
# One arm
# --------------------------------------------------------------------------

def build_X(condition, gauge, local, theta0, theta_T, success):
    ref_idx = int(local["ref_idx"])
    y = np.asarray(local["labels"])
    if condition == "encoded_pre_evolution":
        if gauge == "reference":
            return np.asarray(local["feat_pre"]), y   # cached, gate-verified
        return gauge_features(theta0, gauge, ref_idx), y
    topo = condition[len("evolved_"):]
    mask = success[topo]
    return gauge_features(theta_T[topo][mask], gauge, ref_idx), y[mask]


def run_arm(condition, gauge):
    if condition not in GAUGED_CONDITIONS:
        raise SystemExit(f"{condition!r} is not a gauged condition")
    if gauge not in GAUGES:
        raise SystemExit(f"{gauge!r} is not a registered gauge")
    os.makedirs(OUT_DIR, exist_ok=True)

    local = load_encode_local()
    theta0 = load_theta0() if condition == "encoded_pre_evolution" else None
    if condition.startswith("evolved_"):
        theta_T, success = load_theta_T()
    else:
        theta_T, success = {}, {}

    t0 = time.time()
    X, y = build_X(condition, gauge, local, theta0, theta_T, success)
    t_feat = time.time() - t0
    print(f"[{condition}:{gauge}] X={X.shape} sha={_sha(X)} "
          f"n_valid={len(y)} features built in {t_feat:.1f}s", flush=True)

    t0 = time.time()
    try:
        best_C, mean_loss, mean_acc, per_C_acc, fits = cv_with_accuracy(
            X, y, f"{condition}:{gauge}")
    except s2a_clf.NonConvergenceError as exc:
        # Registered outcome, recorded as a result rather than worked around.
        payload = {"condition": condition, "gauge": gauge,
                   "non_convergence": str(exc), "n_valid": int(len(y)),
                   "n_features": int(X.shape[1]),
                   "elapsed_seconds": time.time() - t0}
        with open(os.path.join(OUT_DIR, f"{condition}__{gauge}.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        raise SystemExit(f"NON-CONVERGENCE (registered halt): {exc}")

    payload = {
        "condition": condition,
        "gauge": gauge,
        "n_valid": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_sha": _sha(X),
        "selected_C": best_C,
        "mean_val_loss_per_C": {str(C): v for C, v in mean_loss.items()},
        "mean_val_acc_per_C": {str(C): v for C, v in mean_acc.items()},
        "per_fold_acc_per_C": {str(C): v for C, v in per_C_acc.items()},
        "accuracy_at_selected_C": mean_acc[best_C],
        "fits": fits,
        "elapsed_seconds": time.time() - t0,
        "feature_seconds": t_feat,
        "non_convergence": None,
    }
    out = os.path.join(OUT_DIR, f"{condition}__{gauge}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[{condition}:{gauge}] selected_C={best_C} "
          f"acc={mean_acc[best_C]:.6f} loss={mean_loss[best_C]:.6f} "
          f"in {payload['elapsed_seconds']:.0f}s -> {out}")


# --------------------------------------------------------------------------
# Combine, and the registered rule
# --------------------------------------------------------------------------

def classify(acc, conditions=GAUGED_CONDITIONS, theta=THETA):
    """The registered rule, as a pure function so it can be tested directly.

    `acc` maps gauge -> condition -> accuracy. Returns
    (class, why, direction, ranking, M_g, max_abs_M). Precedence is
    C (ranking changed) > B (preserved, margins material) > A.
    """
    M = {c: acc["circular_mean"][c] - acc["reference"][c] for c in conditions}
    rank = {g: sorted(conditions, key=lambda c: -round(acc[g][c], TIE_DECIMALS))
            for g in GAUGES}
    max_M = max(abs(v) for v in M.values())

    if rank["reference"] != rank["circular_mean"]:
        cls, why = "C", "ranking changed -- at least one pairwise inversion"
    elif max_M >= theta:
        cls, why = "B", f"ranking preserved, max|M_g|={max_M:.6f} >= theta={theta}"
    else:
        cls, why = "A", f"ranking preserved, max|M_g|={max_M:.6f} < theta={theta}"

    signs = {np.sign(round(v, TIE_DECIMALS)) for v in M.values()} - {0.0}
    if len(signs) == 1:
        direction = ("uniform direction favouring "
                     + ("circular_mean" if signs == {1.0} else "reference"))
    else:
        direction = "mixed direction"
    return cls, why, direction, rank, M, max_M


def _load_arms():
    arms = {}
    for cond in GAUGED_CONDITIONS:
        for g in GAUGES:
            p = os.path.join(OUT_DIR, f"{cond}__{g}.json")
            if os.path.exists(p):
                with open(p) as fh:
                    arms[(cond, g)] = json.load(fh)
    return arms


def run_combine():
    arms = _load_arms()
    missing = [f"{c}:{g}" for c in GAUGED_CONDITIONS for g in GAUGES
               if (c, g) not in arms]
    if missing:
        raise SystemExit(f"cannot combine, {len(missing)} arm(s) missing: {missing}")

    record_path = locked_record_path()
    print(f"locked record: {record_path}\n")
    with open(record_path, "rb") as fh:
        locked = pickle.load(fh)["conditions"]

    # -- Halt condition 3, as restated in amendment 1 -----------------------
    print("=" * 72)
    print("REPRODUCTION GATE: reference arm vs the locked record")
    print("=" * 72)
    gate_ok = True
    for cond in GAUGED_CONDITIONS:
        got = arms[(cond, "reference")]["selected_C"]
        want = locked[cond]["selected_C"]
        same = (float(got) == float(want))
        gate_ok &= same
        dl = max(abs(arms[(cond, "reference")]["mean_val_loss_per_C"][str(C)]
                     - locked[cond]["mean_val_loss_per_C"][C])
                 for C in locked[cond]["mean_val_loss_per_C"])
        print(f"  {cond:<24} selected_C {got:<8g} vs locked {want:<8g} "
              f"{'MATCH' if same else 'MISMATCH'}   max|dloss|={dl:.3e}")
    print(f"\n  gate: {'PASSED' if gate_ok else 'FAILED'}")
    if not gate_ok:
        print("  A mismatch means the REGENERATION, not the gauge, is what this "
              "run measured. Halting per the pre-registration.")
        return

    # -- The registered comparison ------------------------------------------
    print("\n" + "=" * 72)
    print("GAUGE COMPARISON (accuracy at each arm's own selected C)")
    print("=" * 72)
    acc = {g: {c: arms[(c, g)]["accuracy_at_selected_C"] for c in GAUGED_CONDITIONS}
           for g in GAUGES}
    cls, why, direction, rank, M, max_M = classify(acc)

    print(f"  {'condition':<24} {'ref':>10} {'circ':>10} {'M_g':>11}  C_ref/C_cm")
    for c in GAUGED_CONDITIONS:
        print(f"  {c:<24} {acc['reference'][c]:>10.6f} "
              f"{acc['circular_mean'][c]:>10.6f} {M[c]:>+11.6f}  "
              f"{arms[(c,'reference')]['selected_C']:g}/"
              f"{arms[(c,'circular_mean')]['selected_C']:g}")

    print(f"\n  ranking under reference    : {' > '.join(rank['reference'])}")
    print(f"  ranking under circular_mean: {' > '.join(rank['circular_mean'])}")

    print(f"\n  CLASS {cls}: {why}")
    print(f"  direction flag: {direction}")

    # -- the companion frozen-C arm, registered as reported-alongside --------
    # Accuracy at the LOCKED run's selected C for both gauges, so the two arms
    # differ only in the gauge and not also in which C each one chose.
    print("\n" + "=" * 72)
    print("COMPANION ARM (accuracy at the LOCKED run's selected C)")
    print("=" * 72)
    facc = {g: {} for g in GAUGES}
    for c in GAUGED_CONDITIONS:
        C_locked = locked[c]["selected_C"]
        for g in GAUGES:
            facc[g][c] = arms[(c, g)]["mean_val_acc_per_C"][str(C_locked)]
    fcls, fwhy, fdirection, frank, fM, fmax = classify(facc)

    print(f"  {'condition':<24} {'C':>7} {'ref':>10} {'circ':>10} {'M_g':>11}")
    for c in GAUGED_CONDITIONS:
        print(f"  {c:<24} {locked[c]['selected_C']:>7g} {facc['reference'][c]:>10.6f} "
              f"{facc['circular_mean'][c]:>10.6f} {fM[c]:>+11.6f}")
    print(f"\n  CLASS {fcls}: {fwhy}")
    print(f"  direction flag: {fdirection}")
    if fdirection != direction:
        print(f"\n  NOTE: the two arms disagree on direction ('{direction}' vs "
              f"'{fdirection}'). C is selected by log-loss while the comparison "
              f"is accuracy, so re-selection can move accuracy either way.")

    summary = {"companion_frozen_C": {
                   "class": fcls, "why": fwhy, "direction": fdirection,
                   "accuracy": facc, "M_g": fM, "ranking": frank,
                   "max_abs_M": fmax},
               "class": cls, "why": why, "direction": direction,
               "theta": THETA, "max_abs_M": max_M,
               "accuracy": acc, "M_g": M,
               "ranking": rank, "reproduction_gate_passed": bool(gate_ok)}
    out = os.path.join(OUT_DIR, "SUMMARY.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="store_true",
                    help="pre-fit byte-exact feat_pre reproduction gate")
    ap.add_argument("--verify-cv", action="store_true",
                    help="check the CV mirror against the locked function")
    ap.add_argument("--arm", metavar="CONDITION:GAUGE",
                    help="run one (condition, gauge) arm")
    ap.add_argument("--combine", action="store_true",
                    help="apply the registered rule to the completed arms")
    args = ap.parse_args()

    if args.gate:
        run_gate()
    elif args.verify_cv:
        run_verify_cv()
    elif args.arm:
        cond, _, gauge = args.arm.partition(":")
        run_arm(cond, gauge)
    elif args.combine:
        run_combine()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
