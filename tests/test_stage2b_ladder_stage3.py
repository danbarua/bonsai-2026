"""Tier 1 checks on the ladder stage-3 (Phase B) driver, run locally with
no network, no GPU and no Colab session.

The driver cannot be exercised here -- it needs an A100, a bucket and the
full corpus. What CAN be pinned locally is everything that would otherwise
only be discovered by spending money: that its constants agree with the
Makefile, that its module scope stays importable without the cloud
dependencies, that the sizing probe's arithmetic and halt behaviour are
what the plan says, and that every call site binds against real
signatures.

Each test names the specific failure it exists to prevent. None of them
restates the driver's own source.
"""
import ast
import importlib
import inspect
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
DRIVER_PATH = STAGE2B_DIR / "run_ladder_stage3.py"

sys.path.insert(0, str(STAGE2B_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _makefile import make_var as _make_var, recipes as _recipes  # noqa: E402

TARGET = "stage2b-ladder-stage3"


def _expanded(body, passes=3):
    for _ in range(passes):
        def substitute(match):
            value = _make_var(match.group(1))
            return value if value is not None else match.group(0)
        expanded = re.sub(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", substitute, body)
        if expanded == body:
            break
        body = expanded
    return body


@pytest.fixture(scope="module")
def driver():
    """Import with BONSAI_COMMIT guaranteed absent: the driver's entry
    guard is `__name__ == "__main__" or os.environ.get(ENV_COMMIT)`, so an
    inherited BONSAI_COMMIT would run the whole ladder on import."""
    previous = os.environ.pop("BONSAI_COMMIT", None)
    try:
        yield importlib.import_module("run_ladder_stage3")
    finally:
        if previous is not None:
            os.environ["BONSAI_COMMIT"] = previous


@pytest.fixture(scope="module")
def tree():
    return ast.parse(DRIVER_PATH.read_text())


# ---- constants the driver and the design must agree on ----

def test_corpus_constants(driver):
    assert driver.LADDER_STAGE == 3
    assert driver.EXPECTED_N == 60_000
    assert driver.EXPECTED_N_ACTIVE == 505
    assert driver.EXPECTED_REF_IDX == 363
    assert driver.EXPECTED_FEATURE_DIM == 1008 == 2 * 505 - 2
    assert driver.ENCODER_STEPS == 1200
    assert driver.SPLIT == "train"


def test_evolve_chunk_divides_the_corpus_exactly(driver):
    """The driver halts on a non-divisible corpus, so a chunk size that did
    not divide 60,000 would fail only after provisioning."""
    assert driver.EXPECTED_N % driver.EVOLVE_CHUNK == 0
    assert driver.EXPECTED_N // driver.EVOLVE_CHUNK == 240


def test_roles_match_the_locked_partition(driver):
    import stage2b_partition as partition
    assert partition.N_FIT + partition.N_VALIDATION == driver.EXPECTED_N


def test_sentinels_are_stage_three_specific(driver):
    """A stage-2 sentinel would make the make target accept a stage-2 run's
    output as this stage's success."""
    assert driver.OK_SENTINEL == "STAGE3_OK"
    assert driver.FAIL_SENTINEL == "STAGE3_FAIL"
    assert driver.OK_SENTINEL not in ("STAGE1_OK", "STAGE2_OK")


# ---- the sizing probe ----

def test_probe_projects_each_leg_on_its_own_multiplier(driver):
    """A single blended rate is what principle 18 forbids, and this
    pipeline is its own example: the two legs differ by 7.5x in count and
    run on different processors."""
    proj = driver.probe_projections({"jax_svd_s": 2.0, "sklearn_fit_s": 10.0})
    assert proj["jax_projected_s"] == 2.0 * driver.PROBE_JAX_SVD_COUNT
    assert proj["sklearn_projected_s"] == 10.0 * driver.PROBE_SKLEARN_FIT_COUNT
    assert proj["ridge_projected_s"] == proj["jax_projected_s"] + proj["sklearn_projected_s"]
    assert driver.PROBE_JAX_SVD_COUNT != driver.PROBE_SKLEARN_FIT_COUNT


def test_probe_multipliers_match_the_design_accounting(driver):
    """42 = 35 fold-level + 7 refits, over SEVEN conditions -- the count
    the plan originally got wrong as six."""
    import stage2b_conditions as conditions
    import stage2b_ridge as ridge
    n_conditions = len(conditions.ALL_CONDITIONS) + len(driver.RAW_CONDITIONS)
    assert n_conditions == 7
    assert driver.PROBE_JAX_SVD_COUNT == n_conditions * ridge.N_SPLITS + n_conditions
    assert driver.PROBE_SKLEARN_FIT_COUNT == (
        n_conditions * ridge.N_SPLITS * len(ridge.ALPHA_GRID))


def test_probe_passes_within_budget(driver):
    _proj, reasons = driver.evaluate_probe(
        {"jax_svd_s": 1.0, "sklearn_fit_s": 5.0, "device_peak_bytes": 2 * 1024**3}, 300.0)
    assert reasons == []


def test_probe_halts_on_an_over_budget_ridge_projection(driver):
    """The guard seen to fire. A probe that cannot halt is a measurement,
    not a gate -- and the run it was meant to stop would proceed."""
    over = driver.PROBE_RIDGE_BUDGET_S / driver.PROBE_SKLEARN_FIT_COUNT * 2
    _proj, reasons = driver.evaluate_probe(
        {"jax_svd_s": 1.0, "sklearn_fit_s": over, "device_peak_bytes": 1024**3}, 300.0)
    assert any("projected ridge" in r for r in reasons)


def test_probe_halts_on_an_over_budget_device_peak(driver):
    _proj, reasons = driver.evaluate_probe(
        {"jax_svd_s": 1.0, "sklearn_fit_s": 1.0,
         "device_peak_bytes": driver.PROBE_DEVICE_PEAK_BUDGET_BYTES + 1}, 300.0)
    assert any("device peak" in r for r in reasons)


def test_probe_halts_on_an_over_budget_run_total(driver):
    """Elapsed time already spent counts against the run budget: a probe
    that ignored it would pass a run already too far behind to finish."""
    _proj, reasons = driver.evaluate_probe(
        {"jax_svd_s": 1.0, "sklearn_fit_s": 1.0, "device_peak_bytes": 1024**3},
        driver.PROBE_RUN_BUDGET_S + 1)
    assert any("elapsed-plus-ridge" in r for r in reasons)


def test_probe_treats_an_unreported_device_peak_as_absent_not_zero(driver):
    """A CPU backend reports no memory stats. Recording that as 0 would
    read as 'measured, and tiny' -- a passing gate on no measurement."""
    proj, reasons = driver.evaluate_probe(
        {"jax_svd_s": 1.0, "sklearn_fit_s": 1.0, "device_peak_bytes": None}, 300.0)
    assert proj["device_peak_bytes"] is None
    assert not any("device peak" in r for r in reasons)


def test_probe_budgets_are_the_values_the_plan_committed(driver):
    """Fixed before the probe ran, so the verdict cannot be chosen after a
    number exists. If these change, the plan changes with them."""
    assert driver.PROBE_RIDGE_BUDGET_S == 7_200.0
    assert driver.PROBE_RUN_BUDGET_S == 9_000.0
    assert driver.PROBE_DEVICE_PEAK_BUDGET_BYTES == 12 * 1024**3

    plan = (STAGE2B_DIR / "PHASE_B_PLAN.md").read_text()
    assert "7,200 s" in plan and "9,000 s" in plan and "12 GB" in plan


def test_probe_runs_before_the_expensive_steps(tree):
    """A probe that could only run after evolution and feature extraction
    could not stop the run from paying for them."""
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    # Ordered by source line: `ast.walk` is breadth-first and does NOT
    # preserve source order, which would make this test assert nothing in
    # particular about sequence.
    calls = [(n.lineno, n.args[0].value)
             for n in ast.walk(main)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "timed_step"
             and n.args and isinstance(n.args[0], ast.Constant)]
    steps = [name for _lineno, name in sorted(calls)]
    assert "2b_sizing_probe" in steps, steps
    assert steps.index("2b_sizing_probe") < steps.index("5_evolution")
    assert steps.index("2b_sizing_probe") < steps.index("7_ridge")
    assert steps.index("2b_sizing_probe") < steps.index("6_features")


# ---- the pre-contract consumes ----

def test_pinned_digests_are_full_sha256(driver):
    for key, digest in driver.PINNED_SHA256.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), key


def test_every_pre_contract_consume_is_pinned(tree):
    """`require_manifest=False` turns off every provenance check there is.
    Each use must therefore either be a KMNIST staging read (whose bytes
    the loader itself validates) or go through `consume_pinned`.

    Derived from the AST rather than a hand-listed set of call sites --
    principle 21: a list standing in for a derivable set under-covers, and
    the next consume added is exactly what it would miss."""
    unpinned = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(isinstance(k, ast.keyword) and k.arg == "require_manifest"
                   and isinstance(k.value, ast.Constant) and k.value.value is False
                   for k in node.keywords):
            continue
        parent_fn = None
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and node in ast.walk(fn):
                parent_fn = fn.name
        unpinned.append(parent_fn)
    assert set(unpinned) <= {"consume_pinned", "stage_kmnist", "step2_corruption"}, (
        f"an unpinned pre-contract consume appeared in {sorted(set(unpinned))}; route "
        f"it through consume_pinned or justify it explicitly")


def test_consume_pinned_refuses_a_digest_mismatch(driver, tmp_path, monkeypatch):
    """The pin is the only thing standing between this run and silently
    different input, since these objects carry no manifest at all."""
    payload = tmp_path / "topologies.npz"
    payload.write_bytes(b"not the pinned bytes")

    class FakeGcs:
        @staticmethod
        def consume_validated(name, local, **kw):
            return None, False

    mods = type("M", (), {"gcs": FakeGcs})()
    with pytest.raises(driver.Stage3Halt, match="pinned digest"):
        driver.consume_pinned(mods, None, "obj", "stage1/topologies",
                              local_path=str(payload))


def test_consume_pinned_accepts_the_pinned_bytes(driver, tmp_path):
    import hashlib
    payload = tmp_path / "p.npz"
    payload.write_bytes(b"whatever")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    class FakeGcs:
        @staticmethod
        def consume_validated(name, local, **kw):
            return None, False

    mods = type("M", (), {"gcs": FakeGcs})()
    driver.PINNED_SHA256["__test__"] = digest
    try:
        assert driver.consume_pinned(mods, None, "obj", "__test__",
                                     local_path=str(payload)) == str(payload)
    finally:
        del driver.PINNED_SHA256["__test__"]


# ---- parents pin bytes ----

def test_parent_map_refuses_an_unresolvable_parent(driver, monkeypatch):
    """A parent recorded as None is a lineage link that constrains nothing
    -- the manifest would claim provenance it cannot check."""
    class FakeGcs:
        @staticmethod
        def read_manifest(name, **kw):
            return None

    mods = type("M", (), {"gcs": FakeGcs})()
    monkeypatch.setattr(driver, "local_path_for", lambda n: "/nonexistent/path")
    with pytest.raises(driver.Stage3Halt, match="pin nothing"):
        driver.parent_map(mods, None, ("stage2b/train/stage3/common/x.npz",))


def test_parent_map_takes_the_digest_from_the_manifest(driver):
    class FakeGcs:
        @staticmethod
        def read_manifest(name, **kw):
            return {"payload_sha256": "a" * 64}

    mods = type("M", (), {"gcs": FakeGcs})()
    assert driver.parent_map(mods, None, ("x",)) == {"x": "a" * 64}


# ---- the official-index join is shared, not rewritten ----

def test_the_driver_joins_through_the_shared_helper(tree):
    """Principle 16's exact failure shape: two artifacts built from
    differently-ordered index lists align row-for-row, agree on shape, and
    compare entirely wrong numbers with nothing raised anywhere. The join
    lives in one tested place; the driver must not grow its own."""
    source = DRIVER_PATH.read_text()
    assert "partition.index_join(" in source
    assert "index_join" in source
    # a hand-rolled position map is the shape of the reimplementation
    assert "for i, v in enumerate(" not in source


def test_index_join_agrees_with_a_brute_force_join():
    """The helper itself, against an independent implementation on
    deliberately scattered indices -- never a prefix."""
    import stage2b_partition as partition
    rng = np.random.default_rng(0)
    target = np.arange(1000)
    source = rng.choice(target, size=50, replace=False)
    rows, report = partition.index_join(source, target)
    expected = np.array([int(np.flatnonzero(target == v)[0]) for v in source])
    assert np.array_equal(rows, expected)
    assert report["n_overlap"] == 50


def test_index_join_refuses_a_missing_image():
    import stage2b_partition as partition
    with pytest.raises(ValueError, match="absent"):
        partition.index_join(np.array([1, 999]), np.arange(10))


def test_index_join_refuses_duplicate_target_indices():
    import stage2b_partition as partition
    with pytest.raises(ValueError, match="duplicates"):
        partition.index_join(np.array([1]), np.array([1, 1, 2]))


# ---- module scope stays importable without cloud dependencies ----

def test_module_scope_imports_only_stdlib_and_numpy(tree):
    """`mighty-colab exec -f` transmits this file's TEXT into an existing
    kernel, so nothing from this repo exists on disk until bootstrap_repo()
    has cloned it -- which happens INSIDE main(). A module-scope import of
    anything else fails before main() ever starts."""
    allowed = {"hashlib", "json", "os", "subprocess", "sys", "threading", "time",
               "traceback", "types", "contextlib", "numpy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed, node.module


def test_no_dunder_file_at_module_scope(tree):
    """`__file__` is undefined under this execution model -- confirmed on a
    real run at stage 2, before main() ever started."""
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "__file__":
                pytest.fail("__file__ referenced at module scope")


# ---- call sites bind against real signatures ----

@pytest.mark.parametrize("module_name,func,expected", [
    ("stage2b_ridge", "svd_ridge_fit", ("X_train_scaled", "Y_train")),
    ("stage2b_ridge", "sklearn_ridge_predict",
     ("X_train_scaled", "Y_train", "X_eval_scaled")),
    ("stage2b_ridge", "cross_validate_alpha", ("X", "Y", "y_strat")),
    ("stage2b_ridge", "ridge_equivalence_check", ("X", "Y", "y_strat")),
    ("stage2b_cnn", "train_cnn_for_seed", ("x_fit", "y_fit", "x_val", "y_val", "mask")),
    ("stage2b_corruption", "corruption_diagnostics",
     ("x0", "x_t", "x_t_clip", "active_indices")),
])
def test_call_sites_bind_against_real_signatures(module_name, func, expected):
    """Every one of these is called by the driver and cannot be exercised
    without a GPU. A renamed positional parameter would surface only after
    provisioning."""
    module = importlib.import_module(module_name)
    params = list(inspect.signature(getattr(module, func)).parameters)
    assert params[:len(expected)] == list(expected)


# ---- the test split is not reachable from this driver ----

def test_the_driver_never_names_the_test_split():
    source = DRIVER_PATH.read_text()
    assert "testsplit" not in source
    assert "allow_test_split" not in source
    assert 'split="test"' not in source


def test_the_driver_never_forces_an_overwrite(tree):
    """Every scientific artifact here is LINEAGE and create-once. A
    `force=True` would raise WriteOnceViolation at run time; catching it
    here costs nothing."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "force":
            assert not (isinstance(node.value, ast.Constant)
                        and node.value.value is True)


# ---- the Makefile target ----

def test_target_exists_and_runs_the_closure_check():
    body = _expanded(_recipes()[TARGET])
    assert "stage2b_fingerprint.py --check-closure" in body
    assert "run_ladder_stage3.py" in body


def test_target_passes_an_exec_timeout():
    """The omission that made all three Stage 2A GPU targets unable to ever
    complete. Derived guards exist for this; this one is the specific
    check that the override is actually wired."""
    body = _expanded(_recipes()[TARGET])
    assert "--timeout" in body
    assert _make_var("STAGE3_EXEC_TIMEOUT") == "10800"
    assert "10800" in body


def test_target_requires_the_stage_three_sentinel():
    body = _recipes()[TARGET]
    assert "STAGE3_OK" in body
    assert "STAGE2_OK" not in body


def test_target_refuses_a_commit_not_on_a_remote():
    """The runtime clones from origin, so an unpushed HEAD would fetch
    something that does not exist -- after provisioning."""
    body = _recipes()[TARGET]
    assert "branch -r --contains" in body
    assert "REFUSING" in body


def test_target_tears_down_unconditionally_and_checks_the_status():
    body = _recipes()[TARGET]
    assert "stop -s" in body
    assert "check_teardown" in body


def test_the_test_file_is_in_the_stage2b_list():
    """`STAGE2B_TEST_FILES` is a narrowing, and verifying a narrowing with
    the broader form (`pytest tests/`) proves nothing about it."""
    assert "tests/test_stage2b_ladder_stage3.py" in _make_var("STAGE2B_TEST_FILES")


# ---- the probe's measurement must survive its own halt ----

def test_probe_publishes_its_measurement_before_raising_the_halt(driver, tree):
    """The ordering is the point, not a style choice.

    The probe's numbers used to live only in the run record, written at
    teardown -- so a session dying after measuring lost the measurement and
    the next run re-measured on a metered GPU. A HALTING probe is exactly
    the case where the evidence matters most and would otherwise be the
    case most likely to lose it.

    Asserted structurally, by source order within the function, because the
    property is "the write happens first" and no return value exposes it."""
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "step2b_sizing_probe")

    publish_lines = [n.lineno for n in ast.walk(fn)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == "ensure_json"]
    halt_lines = [n.lineno for n in ast.walk(fn)
                  if isinstance(n, ast.Raise)]

    assert publish_lines, "the probe publishes no artifact at all"
    assert halt_lines, "the probe has no halt path"
    assert min(publish_lines) < min(halt_lines), (
        "the probe raises its halt before publishing its measurement, so a "
        "halting run loses the very evidence the halt should be argued from")


def test_probe_artifact_is_run_scoped_and_carries_the_run_id(driver):
    """Nothing consumes it and it is never a parent, so RUN_SCOPED is
    right. The run id is what stops a resumed attempt displacing its
    predecessor's measurement."""
    import stage2b_gcs as gcs
    kind = "probe_sizing_20260808T000000Z"
    name = gcs.object_path(stage=3, condition=None, kind=kind, ext="json",
                           split="train")
    assert gcs.artifact_class(name) == gcs.RUN_SCOPED
    assert "probe_sizing_" in name


def test_probe_publication_failure_never_masks_the_halt(driver, tree):
    """A bucket problem must not turn a halting probe into a passing one.
    The publish is wrapped; the halt is not inside that wrapper."""
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "step2b_sizing_probe")
    for handler in [n for n in ast.walk(fn) if isinstance(n, ast.Try)]:
        raises_inside = [n for n in ast.walk(handler) if isinstance(n, ast.Raise)]
        assert not raises_inside, (
            "the halt is raised inside the try/except that guards publication, "
            "so a publish failure would swallow it")


# ---- the amended grid, and the silent-reuse trap it would otherwise spring ----

def test_alpha_grid_is_the_amended_thirteen_decades(driver):
    import stage2b_ridge as ridge
    assert len(ridge.ALPHA_GRID) == 13
    assert min(ridge.ALPHA_GRID) == 1e-6 and max(ridge.ALPHA_GRID) == 1e6


def test_sklearn_multiplier_follows_the_grid(driver):
    """455 = 7 x 5 x 13. A multiplier left at 315 would under-project the
    dominant cost leg by 44% on the very run the probe exists to size."""
    import stage2b_conditions as conditions
    import stage2b_ridge as ridge
    n_conditions = len(conditions.ALL_CONDITIONS) + len(driver.RAW_CONDITIONS)
    assert driver.PROBE_SKLEARN_FIT_COUNT == (
        n_conditions * ridge.N_SPLITS * len(ridge.ALPHA_GRID)) == 455


def test_grid_tag_changes_when_the_grid_changes(driver):
    """The guard against a silent no-op re-run.

    `ridge_cv.json` from the nine-decade run is already in the bucket WITH
    a valid manifest, and `ensure_json` passes no `expected_fingerprint` --
    so an untagged name would hit `ensure_artifact`'s skip branch, hand
    back the SUPERSEDED results, recompute nothing and report STAGE3_OK.

    Derived, not hand-set: any change to the grid must move the name."""
    nine = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
    thirteen = (1e-6, 1e-5, 1e-4, 1e-3, *nine)
    assert driver.grid_tag(nine) != driver.grid_tag(thirteen)
    # stable for the same grid, and order-sensitive
    assert driver.grid_tag(thirteen) == driver.grid_tag(list(thirteen))
    assert driver.grid_tag(thirteen) != driver.grid_tag(tuple(reversed(thirteen)))
    # a single changed value moves it too
    almost = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3,
              1e4, 1e5, 1e7)
    assert driver.grid_tag(almost) != driver.grid_tag(thirteen)
    assert driver.grid_tag(thirteen).startswith("g13_")


def test_ridge_artifact_names_carry_the_grid_tag(tree):
    """Both ridge artifacts, not just one: a tagged cv table beside an
    untagged final would leave the final silently reused."""
    source = DRIVER_PATH.read_text()
    assert 'f"ridge_cv_{tag}"' in source
    assert 'f"ridge_final_{tag}"' in source
    # Only OBJECT-NAME construction may not use the untagged kind. The bare
    # strings still appear as record keys (`record["ridge_final"]`), which
    # are internal and carry no reuse risk -- an earlier version of this
    # assertion banned the substring outright and failed on exactly that.
    assert '_obj(mods, "ridge_cv"' not in source
    assert '_obj(mods, "ridge_final"' not in source


def test_the_nine_decade_artifacts_are_not_reachable_by_the_new_names(driver):
    """Concretely: the name this run will write differs from the name Phase
    B wrote, so the old tables survive as history untouched."""
    import stage2b_gcs as gcs
    import stage2b_ridge as ridge
    tag = driver.grid_tag(ridge.ALPHA_GRID)
    new = gcs.object_path(stage=3, condition=None, kind=f"ridge_cv_{tag}",
                          ext="json", split="train")
    old = "stage2b/train/stage3/common/ridge_cv.json"
    assert new != old
    assert gcs.artifact_class(new) == gcs.LINEAGE


# ---- the smoke-or-skip decision is explicit and recorded ----

def test_stats_smoke_decision_is_explicit_and_carries_its_reason(driver):
    """PHASE_B_PLAN.md's forward rule. Phase B's own deviation was
    invisible precisely because nothing recorded that a choice was made."""
    assert driver.STATS_SMOKE in ("run", "skip")
    assert driver.STATS_SMOKE == "skip"
    reason = driver.STATS_SMOKE_REASON
    assert "776.6" in reason and "non-inferential" in reason
    assert len(reason) > 100, "a decision without a reason is an omission"
