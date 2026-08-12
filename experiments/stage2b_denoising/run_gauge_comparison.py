"""The gauge-sensitivity comparison Stage 2A pre-registered and never ran.

Pre-registration, written and committed BEFORE this was run:
`GAUGE_COMPARISON_PREREGISTRATION.md`. Read it first -- the four possible
outcomes and their readings are fixed there, along with the companion
prediction and the caveats, so that whichever way this lands the
interpretation was chosen when it could still have gone the other way.

## What this varies, and what it does not

Exactly one thing changes: the gauge function turning a phase vector into
features. `reference_node_features(theta, 363)` is the locked primary;
`circular_mean_features(theta)` is the pre-registered robustness check.
Both are IMPORTED from `stage2a_core` and neither is reimplemented here
(CLAUDE.md principle 16 -- a verified helper says nothing about glue code
that quietly rewrites it).

Everything else is held: the same cached `theta_T` states, the same target,
the same stratifier, the same folds and seed, the same 13-point alpha grid.

## Why it can run without a GPU evolution

The expensive step is already done and persisted. `stage3/*/theta_T.npz`
holds the evolved phases for all four graphs; this consumes them READ-ONLY
over public HTTPS and recomputes only features and ridge. Nothing here
writes to a lineage path, and nothing re-evolves.

## The gate that runs before any circular-mean number is trusted

`verify_reference_reproduction` recomputes the KNOWN gauge from the same
cached states through this file's own consume path and compares against the
cached `features.npz`. If the gauge this pipeline has always used does not
reproduce here, no number from the other gauge means anything. It reports
the observed maximum difference either way rather than only asserting.

## Scope

TRAIN split only. Descriptive and nominal throughout: no new confirmatory
statistic, no member of any corrected family, nothing that alters the locked
Stage 4 verdict, and no path that reads the test split.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
)
ENV_COMMIT = "BONSAI_COMMIT"

BUCKET_URL = "https://storage.googleapis.com/bonsai-2026-stage2b-cache"
ROOT = "stage2b/train"

OK_SENTINEL = "GAUGE_OK"

# 363 in 505-space, T's median-degree node. Asserted against the topology
# metadata rather than trusted, exactly as run_audit.py does.
EXPECTED_REF_IDX = 363
EXPECTED_N_ACTIVE = 505

# The locked encoder step count; names the published encoded object.
ENCODER_STEPS = 1200

# The five conditions that HAVE a phase, and therefore a gauge. raw_505 and
# raw_784 are pixel baselines and are deliberately absent.
GAUGED_CONDITIONS = ("pre_evolution", "T", "lattice", "rewired", "curr_random")
PATH_SEGMENT = {"pre_evolution": "pre_evolution",
                **{g: f"evolved_{g}" for g in
                   ("T", "lattice", "rewired", "curr_random")}}

# Reproduction of the KNOWN gauge. Expected byte-exact on x86 (same numpy
# code, same input bytes); ~1 ULP on ARM per this project's own measured
# propagation table. Reported either way; only a breach halts.
REPRODUCTION_TOL = 1e-12


def say(line):
    print(line, flush=True)


def fetch(object_name, cache_dir):
    local = os.path.join(cache_dir, object_name.replace("/", "__"))
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        say(f"fetching {object_name}")
        urllib.request.urlretrieve(f"{BUCKET_URL}/{object_name}", local)
    return local


def fetch_npz(object_name, cache_dir):
    with np.load(fetch(object_name, cache_dir), allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def fetch_json(object_name, cache_dir):
    with open(fetch(object_name, cache_dir), encoding="utf-8") as handle:
        return json.load(handle)


def ensure_importable():
    """Local runs import from the checkout; a Colab run arrives as
    transmitted TEXT with no repository on disk, so the pinned commit is
    cloned first. Conditional on the import actually failing, so a machine
    that already has the code does not get a clone imposed on it.

    The sibling Stage 2A directory is added first because that is where the
    two gauge functions live. `__file__` is a synthetic sentinel under
    `mighty-colab exec`, not a real path, so this resolves to nothing there
    and the clone path below takes over -- which is the intended split."""
    here = os.path.dirname(os.path.abspath(__file__))
    for sibling in ("stage2a_dynamics_classification", "stage2b_denoising"):
        candidate = os.path.join(os.path.dirname(here), sibling)
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        import stage2a_core                                  # noqa: F401
        return None
    except ImportError:
        pass
    commit = os.environ.get(ENV_COMMIT)
    if not commit:
        raise SystemExit(
            f"stage2a_core is not importable and {ENV_COMMIT} is unset, so there "
            f"is nothing to bootstrap from.")
    import subprocess
    if not os.path.isdir(os.path.join(CLONE_DIR, ".git")):
        os.makedirs(CLONE_DIR, exist_ok=True)
        for argv in (["git", "init", "-q", CLONE_DIR],
                     ["git", "remote", "add", "origin", REPO_URL],
                     ["git", "fetch", "--depth", "1", "origin", commit],
                     ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(argv, cwd=None if argv[1] == "init" else CLONE_DIR,
                           check=True, capture_output=True, text=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLONE_DIR, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"clone is at {head}, expected {commit}")
    for directory in (*EXPERIMENT_DIRS, "src"):
        entry = os.path.join(CLONE_DIR, directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    say(f"bootstrapped {CLONE_DIR} at {head}")
    return head


def build_features(core, theta, gauge, ref_idx):
    """One gauge applied to every row. `gauge` names which pre-registered
    function to call; both come from `stage2a_core`."""
    if gauge == "reference_node":
        return np.stack([core.reference_node_features(theta[i], ref_idx)
                         for i in range(theta.shape[0])])
    if gauge == "circular_mean":
        return np.stack([core.circular_mean_features(theta[i])
                         for i in range(theta.shape[0])])
    raise ValueError(f"unknown gauge {gauge!r}")


def verify_reference_reproduction(core, theta, cached_X, ref_idx, condition):
    """The gate. Recompute the LOCKED gauge here and compare to the cached
    stage-3 features.

    Principle 16 turned on this driver: `reference_node_features` being
    correct says nothing about whether THIS file consumes, orders and
    restricts the states the way the pipeline did. If the known gauge does
    not reproduce, nothing from the unknown gauge is meaningful."""
    ours = build_features(core, theta, "reference_node", ref_idx)
    if ours.shape != cached_X.shape:
        raise SystemExit(
            f"{condition}: reproduction gate FAILED on shape -- recomputed "
            f"{ours.shape}, cached {cached_X.shape}")
    max_abs = float(np.max(np.abs(ours - cached_X)))
    exact = bool(np.array_equal(ours, cached_X))
    say(f"  reproduction gate {condition}: max|diff| {max_abs:.3e} "
        f"({'byte-exact' if exact else 'not byte-exact'})")
    if max_abs > REPRODUCTION_TOL:
        raise SystemExit(
            f"{condition}: reproduction gate FAILED -- max|diff| {max_abs:.3e} "
            f"exceeds {REPRODUCTION_TOL:.1e}. The locked gauge does not "
            f"reproduce through this driver's consume path, so no "
            f"circular-mean number from it can be trusted.")
    return {"max_abs_diff": max_abs, "byte_exact": exact}


def gauge_offset_stats(theta, ref_idx):
    """Companion: per-image `theta_ref - mu`, wrapped to (-pi, pi] before any
    spread is taken. The raw difference of two angles is discontinuous at the
    branch cut and its std is meaningless -- the same handling
    `measure_gauge_offset.py` uses, reused rather than rewritten."""
    mu = np.angle(np.mean(np.exp(1j * theta), axis=1))
    order = np.abs(np.mean(np.exp(1j * theta), axis=1))
    offset = np.angle(np.exp(1j * (theta[:, ref_idx] - mu)))
    return {"R_mean": float(order.mean()),
            "offset_mean": float(offset.mean()),
            "offset_std": float(offset.std()),
            "offset_min": float(offset.min()),
            "offset_max": float(offset.max())}


def ridge_for(ridge, X, Y, y_strat, frozen_alpha):
    """Both alpha arms, on identical features.

    Re-selected is the honest pipeline-versus-pipeline comparison. Frozen
    isolates feature geometry from selection effects. They are computed from
    the SAME cross-validation output so the two arms cannot drift apart."""
    cv = ridge.cross_validate_alpha(X, Y, y_strat)
    alphas = list(ridge.ALPHA_GRID)
    mean_val = np.asarray(cv["mean_clipped_val_mse"]
                          if "mean_clipped_val_mse" in cv
                          else np.mean(np.asarray(cv["fold_clipped_val_mse"]), axis=0))
    selected = float(cv["alpha"])
    frozen_idx = alphas.index(frozen_alpha)
    return {
        "alpha_reselected": selected,
        "mse_reselected": float(mean_val[alphas.index(selected)]),
        "alpha_frozen": float(frozen_alpha),
        "mse_frozen": float(mean_val[frozen_idx]),
        "mean_clipped_val_mse_by_alpha": [float(v) for v in mean_val],
        "at_grid_endpoint": bool(selected in (alphas[0], alphas[-1])),
    }


def main_run(args):
    commit = ensure_importable()
    import stage2a_core as core
    import stage2b_ridge as ridge

    stage = args.stage
    n_limit = args.n_limit or None
    say(f"gauge comparison: stage {stage}, "
        f"n_limit={n_limit or 'all'}, commit={commit or 'local'}")

    topo = fetch_npz(f"{ROOT}/stage1/common/topologies.npz", args.cache_dir)
    # Read from the topology's own metadata rather than trusting the
    # constant, then assert -- the same check run_audit.py makes.
    ref_idx = int(json.loads(str(topo["summary_json"]))["nodes_T"]["median"])
    if ref_idx != EXPECTED_REF_IDX:
        raise SystemExit(f"reference node is {ref_idx}, expected {EXPECTED_REF_IDX}")
    say(f"reference node {ref_idx} (505-space, from topology metadata)")

    corpus = fetch_npz(f"{ROOT}/stage{stage}/common/corpus.npz", args.cache_dir)
    active = np.asarray(topo["active_indices"])
    labels = np.asarray(corpus["labels"])
    Y = np.asarray(corpus["images_01"]).reshape(labels.shape[0], 784)[:, active]

    frozen = fetch_json(f"{ROOT}/stage3/common/ridge_cv.json", args.cache_dir)
    frozen_alphas = {c: float(v["cv"]["alpha"])
                     for c, v in frozen["conditions"].items()}

    if n_limit:
        labels, Y = labels[:n_limit], Y[:n_limit]
    say(f"target Y {Y.shape}, stratifier {labels.shape}")

    results, companions, gates, skipped = {}, {}, {}, []
    for condition in GAUGED_CONDITIONS:
        segment = PATH_SEGMENT[condition]
        say(f"--- {condition} ---")
        if condition == "pre_evolution":
            # The encoded phases are published for stage 3 only; stage 1
            # publishes its pre_evolution FEATURES but not the states they
            # came from. Rather than reconstruct them (which would re-run the
            # encoder and stop being a read-only consume), the condition is
            # skipped where its input does not exist, and the skip is
            # reported rather than silently dropping a row.
            enc_name = f"{ROOT}/stage{stage}/common/encoded_train_s{ENCODER_STEPS}.npz"
            try:
                theta = np.asarray(fetch_npz(enc_name, args.cache_dir)["thetas_505"])
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                say(f"  SKIPPED: {enc_name} is not published at this stage, so "
                    f"pre_evolution has no states to gauge")
                skipped.append(condition)
                continue
        else:
            theta = np.asarray(
                fetch_npz(f"{ROOT}/stage{stage}/{segment}/theta_T.npz",
                          args.cache_dir)["theta_T"])
        if n_limit:
            theta = theta[:n_limit]
        if theta.shape[1] != EXPECTED_N_ACTIVE:
            raise SystemExit(f"{condition}: theta is {theta.shape}, expected "
                             f"(*, {EXPECTED_N_ACTIVE})")

        cached = fetch_npz(f"{ROOT}/stage{stage}/{segment}/features.npz",
                           args.cache_dir)
        cached_X = np.asarray(cached["X"])[:theta.shape[0]]
        gates[condition] = verify_reference_reproduction(
            core, theta, cached_X, ref_idx, condition)

        companions[condition] = gauge_offset_stats(theta, ref_idx)
        say(f"  R={companions[condition]['R_mean']:.4f}  "
            f"offset std={companions[condition]['offset_std']:.4f} rad")

        per_gauge = {}
        for gauge in ("reference_node", "circular_mean"):
            X = (cached_X if gauge == "reference_node"
                 else build_features(core, theta, gauge, ref_idx))
            if not np.all(np.isfinite(X)):
                raise SystemExit(f"{condition}/{gauge}: non-finite features")
            out = ridge_for(ridge, X, Y, labels, frozen_alphas[condition])
            out["feature_dim"] = int(X.shape[1])
            out["min_col_std"] = float(np.min(np.std(X, axis=0)))
            per_gauge[gauge] = out
            say(f"  {gauge:15s} dim={X.shape[1]} "
                f"alpha_resel={out['alpha_reselected']:g} "
                f"mse_resel={out['mse_reselected']:.6e} "
                f"mse_frozen={out['mse_frozen']:.6e} "
                f"min_col_std={out['min_col_std']:.3e}")
        results[condition] = per_gauge

    summary = {
        "stage": stage, "n_limit": n_limit, "commit": commit,
        "reference_node": ref_idx,
        "reproduction_gate": gates,
        "gauge_offset": companions,
        "ridge": results,
        "scope": ("TRAIN split only; descriptive and nominal; no confirmatory "
                  "statistic; consumes cached states read-only"),
        "preregistration": "GAUGE_COMPARISON_PREREGISTRATION.md",
    }
    measured = [c for c in GAUGED_CONDITIONS if c in results]
    summary["skipped_conditions"] = skipped
    for arm in ("reselected", "frozen"):
        ranking = sorted(measured,
                         key=lambda c: results[c]["reference_node"][f"mse_{arm}"])
        ranking_cm = sorted(measured,
                            key=lambda c: results[c]["circular_mean"][f"mse_{arm}"])
        summary[f"ranking_{arm}"] = {"reference_node": ranking,
                                     "circular_mean": ranking_cm,
                                     "preserved": ranking == ranking_cm}
        say(f"ranking [{arm}] reference_node : {' < '.join(ranking)}")
        say(f"ranking [{arm}] circular_mean  : {' < '.join(ranking_cm)}")
        say(f"ranking [{arm}] PRESERVED: {ranking == ranking_cm}")

    observed = sorted([c for c in measured if c != "pre_evolution"],
                      key=lambda c: -companions[c]["offset_std"])
    summary["companion_offset_ordering"] = {
        "predicted": ["T", "lattice", "curr_random", "rewired"],
        "observed": observed,
    }
    say(f"companion offset-std ordering predicted: T > lattice > curr_random > rewired")
    say(f"companion offset-std ordering observed : {' > '.join(observed)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        say(f"wrote {args.out}")
    say(OK_SENTINEL)
    return 0


def main():
    """Flags locally, environment variables remotely.

    `mighty-colab exec` transmits this file's TEXT into a live kernel: it has
    an `--env` option and no way to pass argv, and the argv that does exist
    belongs to ipykernel. So every setting defaults from an environment
    variable and unrecognised arguments are tolerated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int,
                        default=int(os.environ.get("GAUGE_STAGE", "3")))
    parser.add_argument("--n-limit", type=int,
                        default=int(os.environ.get("GAUGE_N_LIMIT", "0")))
    parser.add_argument("--cache-dir",
                        default=os.environ.get("GAUGE_CACHE_DIR",
                                               "results/_gauge_cache"))
    parser.add_argument("--out",
                        default=os.environ.get("GAUGE_OUT",
                                               "results/gauge_comparison.json"))
    args, _unrecognised = parser.parse_known_args()
    return main_run(args)


if __name__ == "__main__" or os.environ.get(ENV_COMMIT):
    # Exit only on FAILURE. `SystemExit(0)` raised inside an IPython kernel is
    # reported as an exception; under `mighty-colab --json` that is now
    # separable from a CLI failure, but a zero exit still costs nothing and
    # every other driver here ends this way.
    _status = main()
    if _status:
        sys.exit(_status)
