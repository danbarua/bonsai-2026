"""Tier 1 checks on the amendment-impact audit driver (`run_audit.py`), run
locally with no network, no GPU, no Colab session. This audit's real
compute (evolving the 150-step budget and the 60,000-image OOF ridge) needs
an A100, a bucket, and Dan's explicit release -- mirroring
`test_stage2b_ladder_stage4.py`'s own framing of what CAN be pinned locally:
that the module stays importable without cloud dependencies, that its pure
decision functions do what they claim (including on synthetic data
specifically constructed to break them -- principle 21's corollary, a guard
never seen to fail is not yet a guard), and that the single load-bearing
invariant this driver adds -- it NEVER opts into the test split -- actually
holds across the whole file rather than being asserted only in prose.

One Tier 2 exception, marked `@pytest.mark.slow` and excluded from the
default suite exactly like the round-trip test: `step5_stage12_cross_check`
is the one piece of `run_audit.py` logic that needs no GPU at all, only
network access to the (public-read) bucket -- so it is exercised here for
real, against real stage-1/stage-2 artifacts, rather than left as something
only ever exercised by an actual audit run on an A100. Per principle 20:
hand-verified functionality (this WAS hand-verified, live, before this test
existed) becomes an executable test once confirmed, so it stops living only
in a session transcript.
"""
import ast
import hashlib
import importlib
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
DRIVER_PATH = STAGE2B_DIR / "run_audit.py"

sys.path.insert(0, str(STAGE2B_DIR))

import stage2b_audit as audit  # noqa: E402
import stage2b_gcs as gcs  # noqa: E402
import stage2b_ridge as ridge  # noqa: E402


@pytest.fixture(scope="module")
def driver():
    """Import with BONSAI_COMMIT guaranteed absent: the driver's entry
    guard is `__name__ == "__main__" or os.environ.get(ENV_COMMIT)`, so an
    inherited BONSAI_COMMIT would run the whole audit on import."""
    previous = os.environ.pop("BONSAI_COMMIT", None)
    try:
        yield importlib.import_module("run_audit")
    finally:
        if previous is not None:
            os.environ["BONSAI_COMMIT"] = previous


@pytest.fixture(scope="module")
def tree():
    return ast.parse(DRIVER_PATH.read_text())


# ---- module scope is safe to import under the transmitted-text model ----

def test_module_scope_imports_only_stdlib_and_numpy(tree):
    """`mighty-colab exec -f` transmits this file's TEXT into an existing
    kernel, so nothing from this repo exists on disk until bootstrap_repo()
    has cloned it -- which happens INSIDE main(). Mirrors
    `test_stage2b_ladder_stage4.py`'s identically-named test."""
    allowed = {"hashlib", "json", "os", "subprocess", "sys", "threading", "time",
               "traceback", "types", "contextlib", "itertools", "numpy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed, node.module


def test_no_dunder_file_at_module_scope(tree):
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "__file__":
                pytest.fail("__file__ referenced at module scope")


def test_the_driver_never_forces_an_overwrite(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "force":
            assert not (isinstance(node.value, ast.Constant) and node.value.value is True)


def test_importing_the_driver_does_not_run_main(driver):
    assert driver.OK_SENTINEL == "AUDIT_OK"


# ---- this driver never touches the test split -----------------------------

def test_no_call_site_passes_allow_test_split(tree):
    """Unlike stage 4, this driver reads stage 1/2/3's TRAIN-side artifacts
    and writes its own train-side artifacts under `LADDER_STAGE=5`. No
    object this file names or reads should ever be a test-split object."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "allow_test_split":
            pytest.fail("run_audit.py must never pass allow_test_split at all")


def test_split_constant_is_train(driver):
    assert driver.SPLIT == "train"


# ---- sentinels and constants ----------------------------------------------

def test_sentinels_are_audit_specific(driver):
    assert driver.OK_SENTINEL == "AUDIT_OK"
    assert driver.FAIL_SENTINEL == "AUDIT_FAIL"
    assert driver.OK_SENTINEL not in ("STAGE3_OK", "STAGE4_OK")


def test_ladder_stage_and_upstream_stages(driver):
    assert driver.LADDER_STAGE == 5
    assert driver.TRAIN_STAGE == 3
    assert driver.STAGE1_STAGE == 1
    assert driver.STAGE2_STAGE == 2


def test_ladder_stage_5_is_actually_a_valid_object_path_stage(driver):
    """A constant matching what the driver EXPECTS is not the same claim as
    `stage2b_gcs.object_path` actually ACCEPTING that value -- the exact gap
    that broke the first real GPU run: `LADDER_STAGE = 5` was correct as a
    driver constant while `stage2b_gcs.LADDER_STAGES` still hard-coded
    `(1, 2, 3, 4)`, so every artifact-writing call in `step3_evolve_150`
    failed on its first attempt, after real evolution compute had already
    run. `_check_stage` is reached through `object_path` itself, not
    re-implemented here."""
    name = gcs.object_path(stage=driver.LADDER_STAGE, condition="evolved_T",
                           kind="theta_T_s150", ext="npz", split=driver.SPLIT)
    assert name == "stage2b/train/stage5/evolved_T/theta_T_s150.npz"


def test_corpus_constants(driver):
    assert driver.EXPECTED_N == 60_000
    assert driver.EXPECTED_N_ACTIVE == 505
    assert driver.EXPECTED_FEATURE_DIM == 1008
    assert driver.EXPECTED_REF_IDX == 363


def test_audit_budgets_match_the_frozen_protocol(driver):
    assert driver.AUDIT_NEW_STEPS == 150
    assert driver.PRODUCTION_STEPS == 1200
    assert (driver.AUDIT_NEW_STEPS, driver.PRODUCTION_STEPS) == audit.AUDIT_STEPS


def test_reference_node_matches_stage2b_audit_gauge_node(driver):
    """`step1b_topologies` asserts this equality at run time; pinned here
    too so a drift in either constant fails fast, locally, for free."""
    assert driver.EXPECTED_REF_IDX == audit.GAUGE_NODE


def test_evolve_chunk_divides_the_corpus_exactly(driver):
    assert driver.EXPECTED_N % driver.EVOLVE_CHUNK == 0


def test_driver_filename_matches_the_real_file(driver):
    assert driver.DRIVER_FILENAME == DRIVER_PATH.name


def test_no_one_shot_lock_object_kind_is_named(tree):
    """Design decision: this audit is diagnostic, not a locked confirmatory
    result -- no `official_result`-style write-once object, unlike stage 4.
    Checks CODE, not the module docstring (which discusses the decision in
    prose): no function named after stage 4's one-shot guard, and no
    string literal `"official_result"` used as an artifact kind anywhere."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.name != "refuse_if_official_result_exists"
        if isinstance(node, ast.Constant) and node.value == "official_result":
            pytest.fail("official_result used as a literal artifact kind")


# ---- PINNED_SHA256: the pre-contract stage-1/2 digest table ---------------

EXPECTED_PIN_KEYS = {
    "stage1/topologies", "stage1/corpus", "stage1/ridge_cv",
    "stage1/features/pre_evolution", "stage1/features/T",
    "stage1/features/lattice", "stage1/features/rewired",
    "stage1/features/curr_random",
    "stage2/corpus", "stage2/ridge_cv",
    "stage2/features/pre_evolution", "stage2/features/T",
    "stage2/features/lattice", "stage2/features/rewired",
    "stage2/features/curr_random",
}


def test_pinned_sha256_covers_exactly_what_the_cross_check_needs(driver):
    """Principle 21: the pin table is a hand-maintained list standing in
    for a derivable set (everything `step5_stage12_cross_check` and
    `step1b_topologies` read pre-contract) -- assert it equals that set in
    both directions rather than trusting it."""
    assert set(driver.PINNED_SHA256) == EXPECTED_PIN_KEYS


def test_every_pin_is_a_well_formed_sha256_hex_digest(driver):
    for key, digest in driver.PINNED_SHA256.items():
        assert isinstance(digest, str), key
        assert len(digest) == 64, key
        assert digest == digest.lower(), key
        int(digest, 16)  # raises ValueError if not valid hex


def test_stage1_topologies_pin_matches_stage3s_own_pin():
    """Same immutable object, reused verbatim -- not re-pinned independently,
    which would silently tolerate the two drifting apart."""
    stage3 = importlib.import_module("run_ladder_stage3")
    import_driver = importlib.import_module("run_audit")
    assert (import_driver.PINNED_SHA256["stage1/topologies"]
            == stage3.PINNED_SHA256["stage1/topologies"])


# ---- pure decision functions: evaluate_probe -------------------------------

def test_evaluate_probe_passes_within_every_budget(driver):
    measured = {"jax_svd_s": 1.0, "device_peak_bytes": 1_000_000}
    _proj, reasons = driver.evaluate_probe(measured, elapsed_so_far_s=0.0)
    assert reasons == []


def test_evaluate_probe_halts_on_projected_ridge_over_budget(driver):
    """Break-confirmation (principle 21 corollary): construct a measurement
    whose projection is deliberately over budget and confirm the specific
    expected halt reason fires -- a guard never seen to fail is not yet a
    guard."""
    huge = driver.PROBE_RIDGE_BUDGET_S / driver.PROBE_JAX_SVD_COUNT + 1.0
    measured = {"jax_svd_s": huge, "device_peak_bytes": 0}
    proj, reasons = driver.evaluate_probe(measured, elapsed_so_far_s=0.0)
    assert reasons
    assert any("ridge" in r for r in reasons)
    assert proj["ridge_projected_s"] > driver.PROBE_RIDGE_BUDGET_S


def test_evaluate_probe_halts_on_device_peak_over_budget(driver):
    measured = {"jax_svd_s": 0.01,
               "device_peak_bytes": driver.PROBE_DEVICE_PEAK_BUDGET_BYTES + 1}
    _proj, reasons = driver.evaluate_probe(measured, elapsed_so_far_s=0.0)
    assert any("device peak" in r for r in reasons)


def test_evaluate_probe_projection_uses_the_derived_svd_count(driver):
    """100 = 2 budgets x 5 conditions x (5 CV-fold + 5 OOF-fold) SVDs -- no
    sklearn leg, unlike stage 3's ridge step. Pinned here so a silent change
    to the call structure this count is derived from is caught."""
    assert driver.PROBE_JAX_SVD_COUNT == 100
    proj = driver.probe_projections({"jax_svd_s": 2.0, "device_peak_bytes": None})
    assert proj["ridge_projected_s"] == pytest.approx(200.0)


# ---- pure decision function: combine_trigger_regimes ----------------------

def _mse(pre, t, lattice, rewired, curr):
    return {"pre_evolution": np.asarray(pre), "T": np.asarray(t),
            "lattice": np.asarray(lattice), "rewired": np.asarray(rewired),
            "curr_random": np.asarray(curr)}


def _verdict(sign_before, sign_after):
    """A `trigger_verdict` built from real `stage2b_audit` machinery
    (never reimplemented here) whose `triggered` flag is controlled by the
    caller via the sign of the primary contrast at each budget."""
    base = np.zeros(8)
    before = _mse(base, np.full(8, sign_before), np.full(8, sign_before - .01),
                  np.full(8, sign_before - .02), np.full(8, sign_before - .03))
    after = _mse(base, np.full(8, sign_after), np.full(8, sign_after - .01),
                 np.full(8, sign_after - .02), np.full(8, sign_after - .03))
    return audit.trigger_verdict({150: before, 1200: after})


def test_combine_regimes_requires_all_six_pairwise_comparisons(driver):
    """Break-confirmation for `binding_gate.0570557b8fa8`: a verdict with a
    pairwise entry missing must fail the completeness check rather than
    silently passing with fewer than six comparisons covered."""
    fixed = _verdict(-0.2, -0.2)
    reselected = _verdict(-0.2, -0.2)
    del fixed["pairwise"][next(iter(fixed["pairwise"]))]
    with pytest.raises(driver.AuditHalt, match="expected 6 pairwise"):
        driver.combine_trigger_regimes(fixed, reselected)


def test_combine_regimes_ors_across_regimes_not_ands(driver):
    """Break-confirmation for `binding_gate.b6e8be29a571`: a reversal
    present in ONLY the fixed regime (reselected never reverses) must still
    set the COMBINED verdict `triggered=True` -- 'check once on whichever
    regime runs first' would silently halve this clause's reach."""
    fixed_triggering = _verdict(-0.2, 0.2)          # sign reversal: triggers
    reselected_calm = _verdict(-0.2, -0.2)          # no reversal: does not
    assert fixed_triggering["triggered"] is True
    assert reselected_calm["triggered"] is False

    combined = driver.combine_trigger_regimes(fixed_triggering, reselected_calm)
    assert combined["combined_triggered"] is True
    assert combined["combination_rule"] == "fixed.triggered OR reselected.triggered"


def test_combine_regimes_calm_in_both_is_not_triggered(driver):
    fixed_calm = _verdict(-0.2, -0.2)
    reselected_calm = _verdict(-0.2, -0.2)
    combined = driver.combine_trigger_regimes(fixed_calm, reselected_calm)
    assert combined["combined_triggered"] is False


def test_combine_regimes_covers_all_six_pairs_when_intact(driver):
    fixed = _verdict(-0.2, -0.2)
    reselected = _verdict(-0.2, -0.2)
    combined = driver.combine_trigger_regimes(fixed, reselected)
    assert len(combined["fixed"]["pairwise"]) == 6
    assert len(combined["reselected"]["pairwise"]) == 6
    assert len(combined["fixed"]["pairwise"]) == len(
        list(combinations(("T", "lattice", "rewired", "curr_random"), 2)))


# ---- grid_tag: consistency with the other drivers' identical helper -------

def test_grid_tag_matches_stage3s_own_for_the_frozen_grid(driver):
    stage3 = importlib.import_module("run_ladder_stage3")
    grid = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
    assert driver.grid_tag(grid) == stage3.grid_tag(grid) == "g13_88edf9ac"


# ---- Tier 2: the stage-1/2 historical cross-check, against the real bucket

def _evidence(line):
    """Plain `print`: under `-s` it streams live; without `-s` pytest
    captures it and replays it on failure. This test is `@pytest.mark.slow`
    and deselected from the default suite, so verbosity costs nothing here
    -- mirrors `test_stage2b_gcs_roundtrip.py::_evidence` exactly."""
    print(line, flush=True)


def _require_storage_library():
    try:
        gcs._storage_module()
    except ImportError as exc:
        pytest.skip(f"google-cloud-storage is needed to reach the bucket and is not "
                    f"installed locally. It ships in this project's `gpu` dependency "
                    f"group -- run under `uv run --group gpu` to enable this test. "
                    f"({exc.__class__.__name__})")


@pytest.mark.slow
def test_step5_cross_check_reproduces_stage1_and_stage2_fold_aggregates(driver, tmp_path):
    """The actual computation `step5_stage12_cross_check` performs, against
    the real bucket -- no GPU needed for this half of the driver, only
    network. Reads every object anonymously (`gcs.get_bucket(anonymous=True)`),
    since the bucket is public-read and this is exactly the un-credentialed
    access pattern the driver itself would use for these pre-contract
    reads. Verifies digests against `driver.PINNED_SHA256` before trusting
    anything downloaded, then runs `ridge.oof_per_image_mse` on each
    stage's OWN historical (pre-amendment, nine-decade) grid and checks it
    against that stage's own stored `fold_clipped_val_mse` via
    `audit.assert_oof_matches_fold_aggregates` -- the exact call the driver
    makes, not a reimplementation of it.

    Run this with `-s` for the per-condition evidence."""
    _require_storage_library()
    bucket = gcs.get_bucket(anonymous=True)
    full_grid = driver.FULL_GRID

    topo_name = gcs.object_path(stage=driver.STAGE1_STAGE, condition=None,
                                kind="topologies", ext="npz", split=driver.SPLIT)
    topo_local = tmp_path / "topologies.npz"
    gcs.download_file(topo_name, str(topo_local), bucket=bucket)
    with open(topo_local, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == driver.PINNED_SHA256["stage1/topologies"]
    active_indices = np.load(topo_local, allow_pickle=False)["active_indices"]
    _evidence(f"topologies      : sha256 {digest[:16]}... matches pin, "
              f"{active_indices.size} active indices")

    for stage, stage_key, index_key in ((driver.STAGE1_STAGE, "stage1", "stage1_indices"),
                                        (driver.STAGE2_STAGE, "stage2", "stage2_indices")):
        corpus_name = gcs.object_path(stage=stage, condition=None, kind="corpus",
                                      ext="npz", split=driver.SPLIT)
        corpus_local = tmp_path / f"{stage_key}_corpus.npz"
        gcs.download_file(corpus_name, str(corpus_local), bucket=bucket)
        with open(corpus_local, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        assert digest == driver.PINNED_SHA256[f"{stage_key}/corpus"]

        cv_name = gcs.object_path(stage=stage, condition=None, kind="ridge_cv",
                                  ext="json", split=driver.SPLIT)
        cv_local = tmp_path / f"{stage_key}_ridge_cv.json"
        gcs.download_file(cv_name, str(cv_local), bucket=bucket)
        with open(cv_local, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        assert digest == driver.PINNED_SHA256[f"{stage_key}/ridge_cv"]

        corpus = np.load(corpus_local, allow_pickle=False)
        images, labels = corpus["images_01"], corpus["labels"]
        n = images.shape[0]
        Y = images.reshape(n, full_grid)[:, active_indices]
        with open(cv_local, "r", encoding="utf-8") as handle:
            ridge_cv = json.load(handle)
        _evidence(f"{stage_key}       : n={n}, sha256s match pins")

        for condition in ("pre_evolution", "T", "lattice", "rewired", "curr_random"):
            seg = "pre_evolution" if condition == "pre_evolution" else f"evolved_{condition}"
            feat_name = gcs.object_path(stage=stage, condition=seg, kind="features",
                                        ext="npz", split=driver.SPLIT)
            feat_local = tmp_path / f"{stage_key}_{condition}_features.npz"
            gcs.download_file(feat_name, str(feat_local), bucket=bucket)
            with open(feat_local, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            assert digest == driver.PINNED_SHA256[f"{stage_key}/features/{condition}"]

            X = np.load(feat_local, allow_pickle=False)["X"]
            stored_cv = ridge_cv["conditions"][condition]["cv"]
            historical_alphas = np.asarray(stored_cv["alphas"], dtype=np.float64)
            oof = ridge.oof_per_image_mse(X, Y, labels, alphas=historical_alphas)
            check = audit.assert_oof_matches_fold_aggregates(oof, stored_cv)
            assert check["passed"]
            _evidence(f"  {condition:<13} max_abs_diff={check['max_abs_diff']:.3e} "
                      f"(atol {check['atol']:.0e}), grid={len(historical_alphas)} decades")
