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
    assert any("run total" in r for r in reasons)


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
