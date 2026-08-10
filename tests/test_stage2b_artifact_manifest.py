"""Tier 1/2 checks on `generate_stage2b_artifact_manifest.py` -- the
committed index of the artifacts behind Stage 2B's two locked results
(stage 4's official confirmatory result, stage 5's amendment-impact
audit), mirroring
`experiments/stage2a_dynamics_classification/generate_artifact_manifest.py`'s
purpose. Named distinctly from that file (not just directory-scoped)
because a Makefile-recipe scanner elsewhere in this suite
(`test_stage2b_gcs_makefile.py`) matches targets to scripts by basename,
and the two experiments' otherwise-identical script names collided under
it -- caught by running the suite, not by design review. Tier 1 (pure
functions, the `ARTIFACTS` list's own shape) always runs. Tier 2
(`test_main_produces_a_manifest_matching_the_real_bucket`) touches the
real, public-read bucket anonymously and is `@pytest.mark.slow`.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(STAGE2B_DIR))

import generate_stage2b_artifact_manifest as gam  # noqa: E402


def test_artifacts_list_has_no_duplicate_labels():
    labels = [entry[0] for entry in gam.ARTIFACTS]
    assert len(labels) == len(set(labels)), "duplicate artifact label in ARTIFACTS"


def test_artifacts_list_covers_both_locked_results():
    labels = {entry[0] for entry in gam.ARTIFACTS}
    assert "stage4_official_result_json" in labels
    assert "stage5_trigger_verdict" in labels


def test_only_stage4_entries_opt_into_the_test_split():
    """Stage 5 is train-side only -- the audit never touches the official
    test corpus (run_audit.py's own module docstring). A stage-5 entry
    with allow_test_split=True would be a silent scope violation."""
    for label, stage, _split, _condition, _kind, _ext, allow_test_split in gam.ARTIFACTS:
        if stage == 5:
            assert not allow_test_split, f"{label}: stage 5 must never opt into the test split"
        if stage == 4:
            assert allow_test_split, f"{label}: stage 4's official artifacts are test-split"


def test_strip_long_lists_leaves_short_lists_untouched():
    obj = {"a": [1, 2, 3], "b": "x"}
    assert gam.strip_long_lists(obj, threshold=20) == obj


def test_strip_long_lists_replaces_long_lists_with_a_count():
    obj = {"resampled_means": list(range(20000)), "observed_mean": -0.0044}
    stripped = gam.strip_long_lists(obj, threshold=20)
    assert stripped["observed_mean"] == -0.0044
    assert isinstance(stripped["resampled_means"], str)
    assert "20000" in stripped["resampled_means"]


def test_strip_long_lists_recurses_into_nested_dicts_and_lists():
    obj = {"outer": {"inner": {"big": list(range(50)), "small": [1, 2]}}}
    stripped = gam.strip_long_lists(obj, threshold=20)
    assert isinstance(stripped["outer"]["inner"]["big"], str)
    assert stripped["outer"]["inner"]["small"] == [1, 2]


def test_strip_long_lists_break_confirmation_a_bootstrap_array_would_bloat_the_file():
    """Break-confirmation: without stripping, the real stage-4 official
    result's 20,000-element bootstrap resample arrays alone would make
    the committed manifest ~20x larger (measured: 745 KB unstripped vs
    ~44 KB stripped, on the real artifact). This test pins the mechanism
    on synthetic data shaped like the real bloat, not the live download."""
    fake_bootstrap_heavy = {
        "primary": {"resampled_means": list(range(20000)), "observed_mean": -0.0044},
        "one_graph_wins": {"unique_winner": "T"},
    }
    unstripped_size = len(json.dumps(fake_bootstrap_heavy))
    stripped_size = len(json.dumps(gam.strip_long_lists(fake_bootstrap_heavy)))
    assert stripped_size < unstripped_size / 10, (
        "stripping did not meaningfully shrink a bootstrap-array-heavy payload")


@pytest.mark.slow
def test_main_produces_a_manifest_matching_the_real_bucket():
    """Runs the real generator against the real, public-read bucket
    (anonymous, no credentials, no billing) and checks the committed
    manifest's own headline fields against a fresh run."""
    bucket = gam.gcs.get_bucket(anonymous=True)
    manifest = {"artifacts": {}, "frozen_results": {},
               "environment": gam.get_environment_metadata()}
    for label, stage, split, condition, kind, ext, allow_test_split in gam.ARTIFACTS:
        manifest["artifacts"][label] = gam.entry_for(
            label, stage, split, condition, kind, ext, allow_test_split, bucket=bucket)

    present = {label: entry["present"] for label, entry in manifest["artifacts"].items()}
    # The two run-scoped reports carry no manifest by design (see
    # run_audit.py::step9_report / run_ladder_stage4.py's report step,
    # neither passes fingerprint= for its report artifacts) -- everything
    # else must have one.
    for label, is_present in present.items():
        if "official_result" in label or "audit_report" in label:
            continue
        assert is_present, f"{label} unexpectedly has no manifest"

    trigger_name = manifest["artifacts"]["stage5_trigger_verdict"]["object"]
    trigger = gam.read_json_object(trigger_name, bucket=bucket, allow_test_split=False)
    assert trigger["combined_triggered"] is False

    committed = json.loads((STAGE2B_DIR / "ARTIFACT_MANIFEST.json").read_text())
    assert (committed["frozen_results"]["stage5_trigger_verdict_summary"]["combined_triggered"]
            == trigger["combined_triggered"])
