"""
Tests for experiments/stage1c_trajectory_generalization/bootstrap_stage1b2_and_stage1c.py.

Tier 2 only (skip-if-missing) -- there's no meaningful Tier-1 synthetic
version of "does the full cold-clone bootstrap pipeline work," and
exercising the actual build/run steps against real data other than via
skip-detection would mean either regenerating Stage 1B.2 (never allowed
-- it's this project's frozen reference) or spending real compute on a
full run in every test suite invocation (not appropriate for a fast,
always-run test). What IS safe and meaningful to check unconditionally
against a fully-populated local checkout: that running the bootstrap
script when everything already exists is a true no-op -- in particular,
that it never touches (mtime-wise) Stage 1B.2's frozen results file.
"""
import os
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE1C_DIR = _REPO_ROOT / "experiments" / "stage1c_trajectory_generalization"
sys.path.insert(0, str(_STAGE1C_DIR))

import bootstrap_stage1b2_and_stage1c as bootstrap  # noqa: E402

_ALL_ARTIFACTS_PRESENT = (
    all(os.path.exists(os.path.join(bootstrap.KMNIST_DIR, f)) for f in bootstrap.KMNIST_REQUIRED_FILES)
    and os.path.exists(bootstrap.CLASS0_CONSTRUCTIONS_PATH)
    and bootstrap._trial_count(bootstrap.STAGE1B2_RESULTS_PATH) == bootstrap.EXPECTED_TRIALS
    and os.path.exists(bootstrap.STAGE1C_FINAL_ANALYSIS_PATH)
)


@pytest.mark.skipif(not _ALL_ARTIFACTS_PRESENT,
                     reason="not all Stage 1B2/1C artifacts present locally -- bootstrap no-op check needs a fully-populated checkout")
def test_bootstrap_is_a_true_noop_and_never_touches_stage1b2s_frozen_file():
    stage1b2_mtime_before = os.path.getmtime(bootstrap.STAGE1B2_RESULTS_PATH)
    class0_mtime_before = os.path.getmtime(bootstrap.CLASS0_CONSTRUCTIONS_PATH)

    bootstrap.main()  # must complete without raising and without regenerating anything

    assert os.path.getmtime(bootstrap.STAGE1B2_RESULTS_PATH) == stage1b2_mtime_before, (
        "bootstrap.main() modified Stage 1B2's frozen results file -- it must only ever "
        "be read, never regenerated, once it exists")
    assert os.path.getmtime(bootstrap.CLASS0_CONSTRUCTIONS_PATH) == class0_mtime_before


def test_expected_trial_count_matches_stage1c_and_stage1b2_design():
    assert bootstrap.EXPECTED_TRIALS == 432


def test_construction_hyperparameters_match_the_historically_recovered_convention():
    # n_per_class=200 / seed=1 for both rewired and random is the same convention
    # documented and tested in build_all_class_topologies.py -- keeping these two
    # in sync (rather than each picking its own value) matters because the two
    # scripts are expected to produce the same class-0 T.
    assert bootstrap.N_PER_CLASS == 200
    assert bootstrap.REWIRED_SEED == 1
    assert bootstrap.RANDOM_SEED == 1
