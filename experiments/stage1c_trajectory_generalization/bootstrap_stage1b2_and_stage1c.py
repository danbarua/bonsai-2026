"""
Cold-clone bootstrap for Stage 1B.2 + Stage 1C.

A fresh `git clone` of the public GitHub repository has none of the
result artifacts these two stages need to run their analyses --
`.pkl`/`.npz` files and `datasets/*/` are all gitignored (regenerable
local-only data, per this project's convention -- see .gitignore and
CLAUDE.md's "datasets/" section). Until now, reproducing Stage 1B.2 and
Stage 1C from a cold clone required manually copying or renaming an
ignored pickle from another stage -- this script replaces that with a
single, idempotent, resumable command.

Runs, in order, skipping any step whose output already exists (so
re-running this script after an interruption, or in a repo that already
has some of this cached, only does the missing work -- and, critically,
NEVER regenerates Stage 1B.2's results once they exist, since Stage 1B.2
is this project's frozen reference trajectory; see
docs/PROJECT_MEMORY.md's repeated "Stage 1B2 is frozen" convention):

1. Check datasets/kmnist/ has the 4 raw KMNIST IDX files. If not, this
   script stops with instructions -- it does not fetch them itself (this
   project acquires datasets manually; see datasets/notmnist/ for the
   equivalent precedent with notMNIST).
2. Build experiments/stage1b2_structured_transformation/results/class0_constructions.pkl
   via src/bonsai/dynamics/construction_bundle.py's
   build_class_construction_bundle() directly from the raw KMNIST class-0
   images (n_per_class=200, the historically recovered hyperparameter;
   rewired_seed=1, random_seed=1, matching
   stage0_simulator_calibration/build_all_class_topologies.py's
   convention) -- if it doesn't already exist.
3. Run Stage 1B.2 (run_stage1b2.py) to produce
   results/stage1b2_results.pkl -- if it doesn't already have all 432
   trials.
4. Run Stage 1C's 9 new trajectories (run_stage1c.py, no arguments --
   its own default already excludes seed=3000; see run_stage1c.py's
   NEW_BASELINE_SEEDS / STAGE1B2_REFERENCE_SEED guard) -- for whichever
   of the 9 don't already have all 432 trials.
5. Run analyze_stage1c.py to produce results/stage1c_final_analysis.pkl
   covering all 10 trajectories -- if it doesn't already exist.

Each of steps 3-5 is invoked as a subprocess with an explicit `cwd`
(rather than imported and called directly), matching how these scripts
are actually documented to be run (`python3 run_stage1b2.py` from within
its own directory) and avoiding relative-path bugs those scripts'
existing hardcoded relative paths would otherwise hit if imported from
elsewhere.

Usage: python3 bootstrap_stage1b2_and_stage1c.py
       (safe to re-run at any point; every step is a no-op if its output
       already exists)
"""
import os
import pickle
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_STAGE1B2_DIR = os.path.join(_REPO_ROOT, "experiments", "stage1b2_structured_transformation")
_STAGE1C_DIR = _THIS_DIR

KMNIST_DIR = os.path.join(_REPO_ROOT, "datasets", "kmnist")
KMNIST_REQUIRED_FILES = [
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
]

CLASS0_CONSTRUCTIONS_PATH = os.path.join(_STAGE1B2_DIR, "results", "class0_constructions.pkl")
STAGE1B2_RESULTS_PATH = os.path.join(_STAGE1B2_DIR, "results", "stage1b2_results.pkl")
STAGE1C_FINAL_ANALYSIS_PATH = os.path.join(_STAGE1C_DIR, "results", "stage1c_final_analysis.pkl")

N_PER_CLASS = 200
REWIRED_SEED = 1
RANDOM_SEED = 1
EXPECTED_TRIALS = 432


def log(msg):
    print(f"[bootstrap] {msg}")


def step1_check_kmnist():
    log("Step 1: checking for raw KMNIST data...")
    missing = [f for f in KMNIST_REQUIRED_FILES if not os.path.exists(os.path.join(KMNIST_DIR, f))]
    if missing:
        log(f"MISSING: {missing} under {KMNIST_DIR}")
        log("This script does not fetch datasets itself. Obtain the KMNIST IDX files "
            "(the official KMNIST dataset repository, rois-codh/kmnist on GitHub, publishes "
            "them in this exact IDX format) and place them at the paths above -- matching "
            "this project's existing convention of manual dataset acquisition (see "
            "datasets/notmnist/ for the equivalent precedent with notMNIST).")
        sys.exit(1)
    log("OK: all 4 raw KMNIST files present.")


def step2_ensure_class0_constructions():
    log("Step 2: checking for class0_constructions.pkl...")
    if os.path.exists(CLASS0_CONSTRUCTIONS_PATH):
        log(f"OK: already present at {CLASS0_CONSTRUCTIONS_PATH} -- skipping.")
        return
    log("Not present -- building from raw KMNIST class-0 images "
        f"(n_per_class={N_PER_CLASS}, rewired_seed={REWIRED_SEED}, random_seed={RANDOM_SEED})...")
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
    import numpy as np
    from bonsai.data.mnist_loader import load_mnist
    from bonsai.dynamics.construction_bundle import build_class_construction_bundle

    X_train, y_train, _, _ = load_mnist(KMNIST_DIR, gz=False)
    idx = np.where(y_train == 0)[0][:N_PER_CLASS]
    images = X_train[idx].astype(np.float64) / 255.0
    bundle = build_class_construction_bundle(images, rewired_seed=REWIRED_SEED, random_seed=RANDOM_SEED)

    os.makedirs(os.path.dirname(CLASS0_CONSTRUCTIONS_PATH), exist_ok=True)
    with open(CLASS0_CONSTRUCTIONS_PATH, "wb") as f:
        pickle.dump({0: bundle}, f)
    log(f"Built and saved {CLASS0_CONSTRUCTIONS_PATH} (n_active={bundle['n_active']}).")


def _trial_count(path):
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return len(pickle.load(f))


def step3_ensure_stage1b2_results():
    log("Step 3: checking Stage 1B.2 results...")
    count = _trial_count(STAGE1B2_RESULTS_PATH)
    if count == EXPECTED_TRIALS:
        log(f"OK: {STAGE1B2_RESULTS_PATH} already has all {EXPECTED_TRIALS} trials -- skipping. "
            f"NOT regenerating (Stage 1B.2 is this project's frozen reference).")
        return
    log(f"{count}/{EXPECTED_TRIALS} trials present -- running run_stage1b2.py "
        f"(cwd={_STAGE1B2_DIR})...")
    subprocess.run([sys.executable, "run_stage1b2.py"], cwd=_STAGE1B2_DIR, check=True)
    final_count = _trial_count(STAGE1B2_RESULTS_PATH)
    if final_count != EXPECTED_TRIALS:
        raise RuntimeError(f"run_stage1b2.py finished but only {final_count}/{EXPECTED_TRIALS} "
                            f"trials are present -- investigate before proceeding.")
    log(f"Stage 1B.2 complete: {final_count}/{EXPECTED_TRIALS} trials.")


def step4_ensure_stage1c_new_trajectories():
    log("Step 4: checking Stage 1C's 9 new trajectories...")
    sys.path.insert(0, _STAGE1C_DIR)
    from run_stage1c import NEW_BASELINE_SEEDS, checkpoint_path

    incomplete = [s for s in NEW_BASELINE_SEEDS if _trial_count(checkpoint_path(s)) != EXPECTED_TRIALS]
    if not incomplete:
        log(f"OK: all {len(NEW_BASELINE_SEEDS)} new trajectories already complete -- skipping.")
        return
    log(f"{len(incomplete)}/{len(NEW_BASELINE_SEEDS)} trajectories incomplete: {incomplete} "
        f"-- running run_stage1c.py (cwd={_STAGE1C_DIR})...")
    subprocess.run([sys.executable, "run_stage1c.py"] + [str(s) for s in incomplete],
                    cwd=_STAGE1C_DIR, check=True)
    still_incomplete = [s for s in incomplete if _trial_count(checkpoint_path(s)) != EXPECTED_TRIALS]
    if still_incomplete:
        raise RuntimeError(f"run_stage1c.py finished but trajectories {still_incomplete} are "
                            f"still incomplete -- investigate before proceeding.")
    log("All 9 new trajectories complete.")


def step5_ensure_stage1c_analysis():
    log("Step 5: checking Stage 1C's final aggregate analysis...")
    if os.path.exists(STAGE1C_FINAL_ANALYSIS_PATH):
        log(f"OK: already present at {STAGE1C_FINAL_ANALYSIS_PATH} -- skipping.")
        return
    log(f"Not present -- running analyze_stage1c.py (cwd={_STAGE1C_DIR})...")
    subprocess.run([sys.executable, "analyze_stage1c.py"], cwd=_STAGE1C_DIR, check=True)
    if not os.path.exists(STAGE1C_FINAL_ANALYSIS_PATH):
        raise RuntimeError("analyze_stage1c.py finished but stage1c_final_analysis.pkl "
                            "was not produced -- investigate before proceeding.")
    log("Stage 1C analysis complete.")


def main():
    step1_check_kmnist()
    step2_ensure_class0_constructions()
    step3_ensure_stage1b2_results()
    step4_ensure_stage1c_new_trajectories()
    step5_ensure_stage1c_analysis()
    log("Bootstrap complete: Stage 1B.2 and Stage 1C are both fully reproduced.")


if __name__ == "__main__":
    main()
