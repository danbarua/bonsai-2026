"""Tier 1 checks on the ladder stage-4 driver (`run_ladder_stage4.py`),
run locally with no network, no GPU, no Colab session, and NO official
test-split object read, written or named -- this file verifies a driver
that touches the test split, without itself touching it.

The driver cannot be exercised here -- it needs an A100, a bucket, the
full corpus, and Dan's explicit release (`AUDIT_PROTOCOL.md`: "Stage 4
stays blocked behind the package review and explicit release"). What CAN
be pinned locally is everything discoverable by reading rather than
spending money: that its module scope stays importable without cloud
dependencies, that the single opt-in site invariant actually holds across
the whole directory (not just asserted in a docstring), that its pure
helper functions do what they claim, and that its call sites bind against
real signatures.
"""
import ast
import importlib
import inspect
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
DRIVER_PATH = STAGE2B_DIR / "run_ladder_stage4.py"

sys.path.insert(0, str(STAGE2B_DIR))


@pytest.fixture(scope="module")
def driver():
    """Import with BONSAI_COMMIT guaranteed absent: the driver's entry
    guard is `__name__ == "__main__" or os.environ.get(ENV_COMMIT)`, so an
    inherited BONSAI_COMMIT would run the whole ladder on import."""
    previous = os.environ.pop("BONSAI_COMMIT", None)
    try:
        yield importlib.import_module("run_ladder_stage4")
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
    has cloned it -- which happens INSIDE main(). A module-scope import of
    anything else fails before main() ever starts. Mirrors
    `test_stage2b_ladder_stage3.py`'s identically-named test."""
    allowed = {"hashlib", "json", "multiprocessing", "os", "subprocess", "sys",
               "threading", "time", "traceback", "types", "contextlib", "numpy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed, node.module


def test_no_dunder_file_at_module_scope(tree):
    """`__file__` is undefined under this execution model."""
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "__file__":
                pytest.fail("__file__ referenced at module scope")


def test_the_driver_never_forces_an_overwrite(tree):
    """Every scientific artifact here is create-once. A `force=True` would
    raise WriteOnceViolation at run time; catching it here costs nothing."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "force":
            assert not (isinstance(node.value, ast.Constant)
                        and node.value.value is True)


def test_importing_the_driver_does_not_run_main(driver):
    """Import alone must not have called `main()` -- if it had, `_RUN_T0`
    would be the only observable effect locally, but the real risk is a
    driver whose entry guard silently fires under pytest collection."""
    assert driver.OK_SENTINEL == "STAGE4_OK"


# ---- sentinels and constants ----

def test_sentinels_are_stage_four_specific(driver):
    assert driver.OK_SENTINEL == "STAGE4_OK"
    assert driver.FAIL_SENTINEL == "STAGE4_FAIL"
    assert driver.OK_SENTINEL != "STAGE3_OK"
    assert driver.FAIL_SENTINEL != "STAGE3_FAIL"


def test_ladder_stage_and_split(driver):
    assert driver.LADDER_STAGE == 4
    assert driver.SPLIT == "test"
    assert driver.TRAIN_STAGE == 3
    assert driver.TRAIN_SPLIT == "train"


def test_corpus_constants(driver):
    assert driver.EXPECTED_N_TEST == 10_000
    assert driver.EXPECTED_N_TRAIN == 60_000
    assert driver.EXPECTED_N_ACTIVE == 505
    assert driver.EXPECTED_FEATURE_DIM == 1008
    assert driver.ENCODER_STEPS == 1200


def test_evolve_chunk_divides_the_test_corpus_exactly(driver):
    assert driver.EXPECTED_N_TEST % driver.EVOLVE_CHUNK == 0


def test_encode_chunk_divides_the_test_corpus_exactly(driver):
    assert driver.EXPECTED_N_TEST % driver.TEST_ENCODE_CHUNK == 0


def test_driver_filename_matches_the_real_file(driver):
    assert driver.DRIVER_FILENAME == DRIVER_PATH.name


# ---- the single opt-in site, derived rather than asserted in prose ----

def _experiment_py_files():
    return sorted(p for p in STAGE2B_DIR.glob("*.py") if p.is_file())


def _files_containing_literal_allow_test_split_true():
    """Every `.py` file directly under `experiments/stage2b_denoising/`
    whose AST contains a keyword argument `allow_test_split=True` (the
    literal opt-in, not merely the parameter name -- a function that
    DECLARES `allow_test_split=False` as a default, or forwards a
    caller-supplied value, does not itself opt in).

    Derived from the AST rather than a substring search or a hand-
    maintained list, per CLAUDE.md principle 21: the claim under test is
    exactly "this file, and only this file, ever passes the literal
    True", and a list of "the files I remember touching this" is the
    failure mode that principle names."""
    found = set()
    for path in _experiment_py_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.keyword) and node.arg == "allow_test_split"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True):
                found.add(path.name)
                break
    return found


# Files other than the driver itself that legitimately pass the literal
# `allow_test_split=True`, each named and justified per CLAUDE.md
# principle 21 ("any exemption gets a named constant and a reason, plus
# its own test that the exemption still refers to something real"),
# rather than silently widening the driver-only claim below.
_NON_DRIVER_EXEMPTIONS = {
    "smoke_stage2b_gcs.py":
        "a pre-existing, hand-run infrastructure probe for stage2b_gcs's "
        "transport layer itself (round-trips a throwaway object through "
        "every stage/split combination, including a deliberate test-side "
        "round trip it deletes immediately afterward). Exercises the "
        "OPT-IN MECHANISM, not the science -- produces no scientific "
        "artifact and is not a driver.",
    "generate_stage2b_artifact_manifest.py":
        "reads (never writes) stage 4's already-published official_result "
        "-- the ONE locked result this manifest indexes -- through the "
        "same validated consume path everything else uses, with the same "
        "named require_manifest=False opt-out run-scoped reports need. "
        "Read-only provenance indexing, not a driver and not new science.",
    "plot_cnn_denoising.py":
        "renders ten held-out characters as PIXELS. Corrupts test images "
        "to draw them; computes no metric, writes no artifact, and no "
        "number it prints enters any record. DESIGN.md's lock is on "
        "EVALUATING against the test corpus, and held-out images are the "
        "honest choice for a demonstration -- training-split characters "
        "would show in-sample behaviour. Dan's ruling, 2026-08-11: "
        "\"erm, I don't see why not? It's just for visualisation\".",
    "animate_graph_dynamics.py":
        "the same ten corrupted inputs under four topologies, animated as "
        "a phase field. Same standing as plot_cnn_denoising.py above and "
        "the same ruling: pixels, no metric, no artifact. The order "
        "parameters it prints are descriptive readings of ten images at "
        "one encoder seed, not a result.",
}


def test_every_exemption_still_refers_to_something_real():
    """Each exemption is only honest if the file it names still exists and
    still contains the literal it is excused for -- an exemption for code
    that moved or lost the opt-in would silently stop meaning anything."""
    found = _files_containing_literal_allow_test_split_true()
    for name, reason in _NON_DRIVER_EXEMPTIONS.items():
        assert reason, f"{name} is exempt with no reason"
        assert (STAGE2B_DIR / name).is_file(), f"exemption {name} names a missing file"
        assert name in found, f"{name} is exempted but no longer contains the literal"


def test_allow_test_split_true_appears_in_exactly_this_driver_plus_the_exemptions():
    """The module docstring's claim, checked rather than trusted: across
    every `.py` file in the directory, the literal `allow_test_split=True`
    appears in `run_ladder_stage4.py`, in the named non-driver exemptions
    above, and nowhere else.

    Break-confirmed both directions by construction of the test itself:
    adding the literal to any other, unexempted file in the directory
    would grow the derived set past the expected set and fail the first
    assertion; removing every occurrence from this driver (e.g. reverting
    to `allow_test_split=False` throughout, silently disabling the whole
    test-corpus path) would drop it from the set and fail the second."""
    found = _files_containing_literal_allow_test_split_true()
    expected = {"run_ladder_stage4.py", *_NON_DRIVER_EXEMPTIONS}
    assert found == expected, (
        f"expected exactly {sorted(expected)} to pass the literal "
        f"allow_test_split=True opt-in, found {sorted(found)}")


def test_the_driver_names_the_test_split(tree):
    """The mirror image of stage 3's `test_the_driver_never_names_the_test_split`:
    this driver MUST reach the test split, or the whole point of writing it
    is unmet."""
    source = DRIVER_PATH.read_text()
    assert "allow_test_split" in source
    assert 'SPLIT = "test"' in source


def test_train_side_reads_never_pass_the_literal_opt_in(tree):
    """Every call whose first argument is a `_obj_train`-style name, or
    which downloads a stage-3 artifact, must not carry
    `allow_test_split=True` -- that opt-in belongs to test-side objects
    only. Checked by asserting every `allow_test_split=True` keyword in
    the file sits inside a call whose function is one of the small set
    that legitimately writes or reads a TEST-side object."""
    test_side_functions = {
        "object_path", "ensure_artifact", "ensure_npz", "ensure_json",
        "ensure_text", "corrupt_corpus", "consume_validated", "object_exists",
        "parent_map",
    }
    for node in ast.walk(tree):
        if not (isinstance(node, ast.keyword) and node.arg == "allow_test_split"
                and isinstance(node.value, ast.Constant) and node.value.value is True):
            continue
        # The keyword's parent Call is found by walking again and matching
        # identity -- ast.walk gives no parent pointers.
        parents = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and node in getattr(n, "keywords", [])]
        assert parents, "orphaned allow_test_split=True keyword"
        for call in parents:
            name = call.func.attr if isinstance(call.func, ast.Attribute) else (
                call.func.id if isinstance(call.func, ast.Name) else None)
            assert name in test_side_functions, (
                f"allow_test_split=True passed to unexpected callee {name!r}")


# ---- CNN reproduction gate ----

def test_cnn_reproduction_matches_on_identical_summaries(driver):
    original = {"best_seed": 1, "best_epoch": 42, "best_clipped_val_mse": 0.01}
    reproduced = dict(original)
    assert driver.cnn_reproduction_mismatch_reason(reproduced, original) is None


def test_cnn_reproduction_halts_on_seed_mismatch(driver):
    original = {"best_seed": 1, "best_epoch": 42, "best_clipped_val_mse": 0.01}
    reproduced = {"best_seed": 2, "best_epoch": 42, "best_clipped_val_mse": 0.01}
    reason = driver.cnn_reproduction_mismatch_reason(reproduced, original)
    assert reason is not None
    assert "seed=2" in reason and "seed=1" in reason


def test_cnn_reproduction_halts_on_epoch_mismatch_even_with_matching_seed(driver):
    original = {"best_seed": 0, "best_epoch": 42, "best_clipped_val_mse": 0.01}
    reproduced = {"best_seed": 0, "best_epoch": 41, "best_clipped_val_mse": 0.01}
    reason = driver.cnn_reproduction_mismatch_reason(reproduced, original)
    assert reason is not None
    assert "best_epoch=41" in reason and "best_epoch=42" in reason


def test_cnn_reproduction_does_not_gate_on_val_mse_alone(driver):
    """A large `best_clipped_val_mse` difference with matching seed and
    epoch must NOT halt -- this function has no calibrated numeric
    tolerance and must not invent one (AUDIT_PROTOCOL.md's Freeze 1)."""
    original = {"best_seed": 0, "best_epoch": 42, "best_clipped_val_mse": 0.01}
    reproduced = {"best_seed": 0, "best_epoch": 42, "best_clipped_val_mse": 0.5}
    assert driver.cnn_reproduction_mismatch_reason(reproduced, original) is None


def test_cnn_reproduction_checks_seed_before_epoch(driver):
    """Ordering matters for a readable halt message: a run with both a
    different seed AND a different epoch should report the seed
    mismatch, not the epoch one, since seed determines everything
    downstream of it."""
    original = {"best_seed": 0, "best_epoch": 42, "best_clipped_val_mse": 0.01}
    reproduced = {"best_seed": 1, "best_epoch": 99, "best_clipped_val_mse": 0.01}
    reason = driver.cnn_reproduction_mismatch_reason(reproduced, original)
    assert "seed=1" in reason and "seed=0" in reason
    assert "best_epoch=99" not in reason  # the epoch branch never ran


# ---- grid_tag agrees with stage 3's own, unimported copy ----

def test_grid_tag_is_deterministic(driver):
    tag1 = driver.grid_tag((1e-6, 1e-3, 1.0))
    tag2 = driver.grid_tag((1e-6, 1e-3, 1.0))
    assert tag1 == tag2


def test_grid_tag_changes_with_the_grid(driver):
    assert driver.grid_tag((1e-6, 1e-3, 1.0)) != driver.grid_tag((1e-6, 1e-2, 1.0))


def test_grid_tag_matches_stage_three_bit_for_bit():
    """Both drivers compute this independently (see the module docstring
    on why code cannot be shared across the bootstrap boundary). If the
    two implementations ever diverge, stage 4 would silently look for a
    `ridge_final_test_*` object under a tag stage 3 never wrote a
    `ridge_final_*` counterpart for, and every downstream alpha lookup
    would 404 instead of reading the frozen production values."""
    stage3 = importlib.import_module("run_ladder_stage3")
    stage4 = importlib.import_module("run_ladder_stage4")
    grid = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
    assert stage3.grid_tag(grid) == stage4.grid_tag(grid)


# ---- parent_map forwards allow_test_split -- regression for the first
# real run's actual failure: `read_manifest` refused a test-side parent
# because parent_map never threaded the opt-in through to it. ----

class _StubGCSManifest:
    def __init__(self):
        self.calls = []

    def read_manifest(self, name, *, bucket, allow_test_split=False):
        self.calls.append((name, allow_test_split))
        return {"payload_sha256": "deadbeef"}


def test_parent_map_forwards_allow_test_split(driver):
    mods = types_namespace(driver, gcs=_StubGCSManifest())
    driver.parent_map(mods, bucket=object(), names=("x",), allow_test_split=True)
    assert mods.gcs.calls == [("x", True)]


def test_parent_map_defaults_to_no_opt_in(driver):
    """The default stays False -- a future train-side caller of parent_map
    must ask for the opt-in explicitly rather than inherit it."""
    mods = types_namespace(driver, gcs=_StubGCSManifest())
    driver.parent_map(mods, bucket=object(), names=("x",))
    assert mods.gcs.calls == [("x", False)]


# ---- the official result is written only on a genuine OK verdict --
# regression for the first real run's second actual failure: a halted,
# pre-inference attempt wrote `official_result` anyway, which then
# permanently blocked every subsequent attempt via
# `refuse_if_official_result_exists`. ----

class _StubGCSReport:
    def __init__(self, exists=False):
        self._exists = exists
        self.written_kinds = []

    def object_path(self, *, kind, **_kwargs):
        return kind

    def object_exists(self, name, *, bucket, allow_test_split=False):
        return self._exists

    def ensure_artifact(self, name, local_path, *, produce, bucket, fingerprint=None,
                        parents=None, allow_test_split=False, force=False):
        produce(local_path)
        self.written_kinds.append(name)
        return _StubResult(local_path)


class _StubResult:
    def __init__(self, local_path):
        self.local_path = local_path

    def summary(self):
        return "stub"


def _report_record(verdict):
    return {"run": {"run_id": "20260101T000000Z", "head_sha": "abc123"},
            "timings": {}, "verdict": verdict, "halt_reason": None}


def test_official_result_is_not_written_on_a_failed_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "local_path_for",
                        lambda name: str(tmp_path / name.replace("/", "__")))
    gcs = _StubGCSReport(exists=False)
    mods = types_namespace(driver, gcs=gcs)
    driver.step11_report(mods, bucket=object(), record=_report_record(driver.FAIL_SENTINEL))
    assert "official_result" not in gcs.written_kinds
    assert any(k.startswith("stage4_report_") for k in gcs.written_kinds)


def test_official_result_is_written_on_a_successful_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "local_path_for",
                        lambda name: str(tmp_path / name.replace("/", "__")))
    gcs = _StubGCSReport(exists=False)
    mods = types_namespace(driver, gcs=gcs)
    driver.step11_report(mods, bucket=object(), record=_report_record(driver.OK_SENTINEL))
    assert "official_result" in gcs.written_kinds


# ---- refusal to re-run the official result ----

class _StubGCS:
    def __init__(self, exists):
        self._exists = exists
        self.calls = []

    def object_exists(self, name, *, bucket, allow_test_split=False):
        self.calls.append((name, bucket, allow_test_split))
        return self._exists


class _StubMods:
    def __init__(self, exists):
        self.gcs = _StubGCS(exists)


def test_refuses_when_the_official_result_already_exists(driver):
    mods = _StubMods(exists=True)
    with pytest.raises(driver.Stage4Halt, match="already exists"):
        driver.refuse_if_official_result_exists(mods, bucket=object(), name="x")
    assert mods.gcs.calls == [("x", mods.gcs.calls[0][1], True)]


def test_does_not_raise_when_the_official_result_is_absent(driver):
    mods = _StubMods(exists=False)
    driver.refuse_if_official_result_exists(mods, bucket=object(), name="x")  # no raise


# ---- object-path construction ----

class _StubGCSPaths:
    def __init__(self):
        self.calls = []

    def object_path(self, **kwargs):
        self.calls.append(kwargs)
        return "/".join([str(kwargs.get(k)) for k in
                         ("stage", "split", "condition", "kind", "ext")])


def test_obj_test_always_opts_in_and_targets_stage_four_split_test(driver):
    mods = types_namespace(driver, gcs=_StubGCSPaths())
    driver._obj_test(mods, "kind", "npz")
    call = mods.gcs.calls[-1]
    assert call["stage"] == 4
    assert call["split"] == "test"
    assert call["allow_test_split"] is True


def test_obj_train_never_opts_in_and_targets_train_split(driver):
    mods = types_namespace(driver, gcs=_StubGCSPaths())
    driver._obj_train(mods, "kind", "npz")
    call = mods.gcs.calls[-1]
    assert call["stage"] == 3
    assert call["split"] == "train"
    assert "allow_test_split" not in call


def types_namespace(driver, **kwargs):
    import types
    return types.SimpleNamespace(**kwargs)


# ---- call sites bind against real signatures ----

# ---- CNN test-corpus evaluation actually runs -- regression for the
# second real run's actual failure: `clipped_validation_per_image_mse`
# was called directly on raw (n, 28, 28) arrays, never passed through
# `as_image_batch` first (its own docstring names this exact mistake:
# "Any caller reaching ... clipped_validation_mse directly should pass
# its arrays through here first"). `equinox`'s Conv rejected the
# unbatched-channel shape 663.65s into a real GPU run, after the CNN had
# already been retrained from all three seeds. This is the one test in
# this file that actually executes JAX/equinox, on CPU, rather than only
# checking signatures or ASTs -- a static check would not have caught a
# runtime shape error one function call deep in a library. ----

def test_cnn_test_evaluation_runs_on_raw_corpus_arrays():
    """Exercises the exact call `step7_test_cnn` makes: a real model, a
    real mask, and raw (n, 28, 28) arrays straight off an `.npz` load --
    not pre-shaped by any caller, matching what `corrupt_corpus` and
    `images_01` actually hand back."""
    import numpy as np
    import stage2b_ridge  # noqa: F401 -- enables jax_enable_x64 at import,
                          # required by clipped_validation_per_image_mse's
                          # float64 accumulation; must precede stage2b_cnn,
                          # exactly as run_ladder_stage4.py's load_modules
                          # orders its own imports.
    import stage2b_cnn as cnn

    n, active_indices = 4, np.array([0, 1, 27, 28, 55, 700], dtype=np.int64)
    model = cnn.make_model(cnn.seed_keys(0)[0])
    mask = cnn.build_active_support_mask(active_indices)
    rng = np.random.default_rng(0)
    x_raw = rng.uniform(0, 1, size=(n, 28, 28)).astype(np.float64)
    y_raw = rng.uniform(0, 1, size=(n, 28, 28)).astype(np.float64)

    with pytest.raises(ValueError, match="rank 3|shape"):
        cnn.clipped_validation_per_image_mse(model, x_raw, y_raw, mask)

    result = cnn.clipped_validation_per_image_mse(
        model, cnn.as_image_batch(x_raw, "x"), cnn.as_image_batch(y_raw, "y"), mask)
    result = np.asarray(result)
    assert result.shape == (n,)
    assert np.all(np.isfinite(result))


# ---- train_raw_pixel_conditions builds X from CORRUPTED train images --
# regression for the PR #29 finding: `step6_test_ridge` built raw_505/
# raw_784's TRAIN-side X from clean `images_01` directly (the same array
# `Y_train` comes from), fitting Y-on-Y. The primary test, denoising
# gate, both Holm families, and one_graph_wins never touch raw_505/
# raw_784 (mse_by_condition's six keys are pre_evolution, the four
# evolved graphs, and identity), so this bug never reached the locked
# result -- but it silently collapsed the raw-pixel descriptive
# baseline toward the identity baseline, which is exactly the "no sign
# the phase representation is lossy" reading the official write-up drew
# from it. This test runs the real `stage2b_corruption.corrupt_corpus`,
# not a stub -- the failure mode is a data-flow mistake a signature
# check or an AST check cannot see. ----

def test_train_raw_pixel_conditions_uses_corrupted_not_clean_images(driver):
    import numpy as np
    import types
    import stage2b_corruption as corruption

    rng = np.random.default_rng(0)
    n = 32
    images_train = rng.uniform(0.2, 0.8, size=(n, 28, 28))
    train_indices = np.arange(n, dtype=np.int64)
    active_indices = np.array([0, 1, 27, 28, 55, 400, 700], dtype=np.int64)
    mods = types.SimpleNamespace(corruption=corruption)

    raw_784, raw_505, y_train = driver.train_raw_pixel_conditions(
        mods, images_train, train_indices, active_indices)

    clean_784 = images_train.reshape(n, 784)
    # The bug this regresses: raw_784 identical to the clean array. A
    # genuine corruption (nonzero alpha_bar noise) makes this a real,
    # not merely formal, difference.
    assert not np.allclose(raw_784, clean_784)
    # Y is unaffected by the fix -- targets are always the clean pixels.
    np.testing.assert_array_equal(y_train, clean_784[:, active_indices])
    np.testing.assert_array_equal(raw_505, raw_784[:, active_indices])
    assert raw_784.shape == (n, 784)


@pytest.mark.parametrize("module_name,func,expected", [
    ("stage2b_ridge", "fit_final", ("X_train", "Y_train", "alpha")),
    ("stage2b_ridge", "ridge_predict", ("fit", "X_scaled", "alpha_index")),
    ("stage2b_ridge", "clipped_per_image_mse", ("pred", "target")),
    ("stage2b_cnn", "train_cnn_for_seed", ("x_fit", "y_fit", "x_val", "y_val", "mask")),
    ("stage2b_cnn", "select_best_seed", ("seeds", "clipped_val_mses")),
    ("stage2b_cnn", "clipped_validation_per_image_mse", ("model", "x", "y", "mask")),
    ("stage2b_cnn", "build_active_support_mask", ("active_indices",)),
    ("stage2b_corruption", "corrupt_corpus", ("images", "split", "indices")),
    ("stage2b_corruption", "rescaled_identity", ("x_t_clip",)),
    ("stage2b_corruption", "corruption_diagnostics",
     ("x0", "x_t", "x_t_clip", "active_indices")),
    ("stage2b_partition", "index_join", ("source_indices", "target_indices")),
    ("stage2b_stats", "run_stage2b_inference", ("mse_by_condition", "y")),
])
def test_call_sites_bind_against_real_signatures(module_name, func, expected):
    """Every one of these is called by the driver and cannot be exercised
    without a GPU. A renamed positional parameter would surface only
    after provisioning."""
    module = importlib.import_module(module_name)
    params = list(inspect.signature(getattr(module, func)).parameters)
    assert params[:len(expected)] == list(expected)
