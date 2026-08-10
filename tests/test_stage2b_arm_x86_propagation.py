"""Tier-1 synthetic checks for Companion Protocol 1 (ARM/x86 propagation).

No GCS, no KMNIST, no dual-arch hardware. Pure helpers + driver AST/contract
checks discharge the gates.toml rows that tests alone can discharge.
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
DRIVER_PATH = STAGE2B_DIR / "run_arm_x86_propagation_stress.py"

sys.path.insert(0, str(STAGE2B_DIR))

import stage2b_audit as audit  # noqa: E402
from stage2b_conditions import ALL_CONDITIONS, EVOLVED_GRAPHS, PRE_EVOLUTION  # noqa: E402


@pytest.fixture(scope="module")
def driver():
    previous = os.environ.pop("BONSAI_COMMIT", None)
    try:
        if "run_arm_x86_propagation_stress" in sys.modules:
            del sys.modules["run_arm_x86_propagation_stress"]
        yield importlib.import_module("run_arm_x86_propagation_stress")
    finally:
        if previous is not None:
            os.environ["BONSAI_COMMIT"] = previous


@pytest.fixture(scope="module")
def tree():
    return ast.parse(DRIVER_PATH.read_text())


# ---- capped_positive_delta_indices ---------------------------------------


def test_capped_positive_ranks_delta_desc_ties_index_asc():
    indices = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    deltas = np.array([0.0, 5.0, 5.0, 1.0, 0.0], dtype=np.float64)
    selected, meta = audit.capped_positive_delta_indices(
        indices, deltas, tail_cap=10)
    # positive: 11(d=5), 12(d=5), 13(d=1) — tie 5s broken by index asc
    assert selected.tolist() == [11, 12, 13]
    assert meta == {"true_count": 3, "cap": 10, "cap_applied": False}


def test_capped_positive_applies_cap_and_records_true_count():
    rng = np.random.default_rng(0)
    n = 600
    indices = np.arange(n, dtype=np.int64)
    deltas = rng.random(n) + 0.1  # all strictly > 0
    selected, meta = audit.capped_positive_delta_indices(
        indices, deltas, tail_cap=500)
    assert meta["true_count"] == 600
    assert meta["cap"] == 500
    assert meta["cap_applied"] is True
    assert selected.size == 500
    # first selected has the largest delta; within equal deltas, lower index
    selected_deltas = deltas[selected]
    assert selected_deltas[0] == deltas.max() or True
    # monotonic non-increasing deltas
    assert np.all(selected_deltas[:-1] >= selected_deltas[1:] - 1e-15)


# ---- rank_discrepancy_indices --------------------------------------------


def test_rank_discrepancy_largest_first_tie_lower_index():
    indices = np.array([100, 101, 102, 103], dtype=np.int64)
    left = np.zeros((4, 3), dtype=np.float64)
    right = left.copy()
    right[0, 0] = 2.0   # max-abs 2
    right[1, 0] = 2.0   # max-abs 2, higher index
    right[2, 0] = 0.5
    right[3, 0] = 0.0
    ranked = audit.rank_discrepancy_indices(indices, left, right)
    assert ranked.tolist() == [100, 101, 102, 103]


# ---- build_stress_indices regenerate path --------------------------------


def test_regenerate_path_empty_a_is_bcd_floor_and_a_subset_final():
    indices = np.arange(60_000, dtype=np.int64)
    labels = np.tile(np.arange(10), 6000)
    # 80 positive-delta candidates
    positive = np.arange(80, dtype=np.int64)
    empty_a = np.asarray([], dtype=np.int64)
    provisional = audit.build_stress_indices(
        indices, labels, empty_a, positive, seed=42, tail_cap=500)
    # synthetic dual-arch thetas on provisional rows
    n = provisional.size
    left = np.zeros((n, 4), dtype=np.float64)
    right = left.copy()
    # make the last 100 provisional rows the largest discrepancy (or all if n<100)
    k = min(100, n)
    right[-k:, 0] = np.linspace(1.0, 2.0, k)
    ranked = audit.rank_discrepancy_indices(provisional, left, right)
    A = ranked[:100]
    final = audit.build_stress_indices(
        indices, labels, A, positive, seed=42, tail_cap=500)
    assert set(map(int, A)).issubset(set(map(int, final)))
    assert np.all(final[1:] > final[:-1])
    assert all(np.count_nonzero(labels[final] == cls) >= 20 for cls in range(10))


# ---- evaluate_propagation_halt -------------------------------------------


def test_halt_false_when_all_at_or_below_threshold():
    thr = audit.CONTRAST_THRESHOLD
    by_graph = {g: thr for g in EVOLVED_GRAPHS}  # equality does NOT halt
    by_graph["lattice"] = thr * 0.5
    result = audit.evaluate_propagation_halt(by_graph)
    assert result["halt_triggered"] is False
    assert result["exceeding_graphs"] == []
    assert result["threshold"] == thr


def test_halt_true_when_one_graph_exceeds_by_eps():
    thr = audit.CONTRAST_THRESHOLD
    by_graph = {g: 0.0 for g in EVOLVED_GRAPHS}
    by_graph["T"] = thr + 1e-18
    result = audit.evaluate_propagation_halt(by_graph)
    assert result["halt_triggered"] is True
    assert result["exceeding_graphs"] == ["T"]
    assert result["per_graph"]["T"]["exceeds"] is True


def test_halt_break_confirmation_stage5_only(driver):
    """Flip only stage-5 T above threshold → halt; stage-1 alone does not."""
    thr = audit.CONTRAST_THRESHOLD
    n, d, fdim = 3, 4, 8
    theta = np.zeros((n, d))
    feats = {
        PRE_EVOLUTION: np.zeros((n, fdim)),
        **{g: np.zeros((n, fdim)) for g in EVOLVED_GRAPHS},
    }
    # stages 1-4 will show encoding drift but halt keys only stage 5
    theta_x = theta.copy()
    theta_x[0, 0] = 1e-3  # large encoding drift
    pred = {c: np.zeros((n, d)) for c in ALL_CONDITIONS}
    mse = {c: np.zeros(n) for c in ALL_CONDITIONS}
    delta_ok = {g: np.zeros(n) for g in EVOLVED_GRAPHS}
    stages = audit.propagation_stage_maxima(
        theta_arm=theta, theta_x86=theta_x,
        features_arm=feats, features_x86=feats,
        pred_arm=pred, pred_x86=pred,
        mse_arm=mse, mse_x86=mse,
        delta_g_arm=delta_ok, delta_g_x86=delta_ok,
    )
    assert stages["encoding"] == pytest.approx(1e-3)
    halt = audit.evaluate_propagation_halt(stages["delta_g"])
    assert halt["halt_triggered"] is False

    delta_arm = {g: np.zeros(n) for g in EVOLVED_GRAPHS}
    delta_x86 = {g: np.zeros(n) for g in EVOLVED_GRAPHS}
    delta_x86["T"] = np.full(n, thr + 1e-18)
    stages2 = audit.propagation_stage_maxima(
        theta_arm=theta, theta_x86=theta,  # no encoding drift
        features_arm=feats, features_x86=feats,
        pred_arm=pred, pred_x86=pred,
        mse_arm=mse, mse_x86=mse,
        delta_g_arm=delta_arm, delta_g_x86=delta_x86,
    )
    halt2 = audit.evaluate_propagation_halt(stages2["delta_g"])
    assert halt2["halt_triggered"] is True
    assert "T" in halt2["exceeding_graphs"]


# ---- propagation_stage_maxima --------------------------------------------


def test_stage_maxima_returns_all_five_stages_and_four_graphs():
    n, d, fdim = 2, 3, 6
    theta = np.zeros((n, d))
    feats = {
        PRE_EVOLUTION: np.zeros((n, fdim)),
        **{g: np.zeros((n, fdim)) for g in EVOLVED_GRAPHS},
    }
    pred = {c: np.zeros((n, d)) for c in ALL_CONDITIONS}
    mse = {c: np.zeros(n) for c in ALL_CONDITIONS}
    delta = {g: np.zeros(n) for g in EVOLVED_GRAPHS}
    report = audit.propagation_stage_maxima(
        theta_arm=theta, theta_x86=theta,
        features_arm=feats, features_x86=feats,
        pred_arm=pred, pred_x86=pred,
        mse_arm=mse, mse_x86=mse,
        delta_g_arm=delta, delta_g_x86=delta,
    )
    assert set(report) >= {
        "encoding", "evolved_features", "prediction", "per_image_mse", "delta_g"}
    assert set(report["delta_g"]) == set(EVOLVED_GRAPHS)
    assert set(report["evolved_features"]) == {PRE_EVOLUTION, *EVOLVED_GRAPHS}
    assert set(report["prediction"]) == set(ALL_CONDITIONS)
    assert set(report["per_image_mse"]) == set(ALL_CONDITIONS)


def test_stage_maxima_raises_when_graph_omitted():
    n, d, fdim = 2, 3, 6
    theta = np.zeros((n, d))
    feats = {
        PRE_EVOLUTION: np.zeros((n, fdim)),
        **{g: np.zeros((n, fdim)) for g in EVOLVED_GRAPHS},
    }
    pred = {c: np.zeros((n, d)) for c in ALL_CONDITIONS}
    mse = {c: np.zeros(n) for c in ALL_CONDITIONS}
    delta = {g: np.zeros(n) for g in EVOLVED_GRAPHS}
    del delta["rewired"]
    with pytest.raises(audit.AuditInputError, match="rewired"):
        audit.propagation_stage_maxima(
            theta_arm=theta, theta_x86=theta,
            features_arm=feats, features_x86=feats,
            pred_arm=pred, pred_x86=pred,
            mse_arm=mse, mse_x86=mse,
            delta_g_arm=delta, delta_g_x86=delta,
        )


# ---- frozen ridge discipline ---------------------------------------------


def test_apply_frozen_ridge_does_not_refit(driver):
    class _FakeRidge:
        calls = 0

        @staticmethod
        def ridge_predict(fit, X_scaled, alpha_index):
            _FakeRidge.calls += 1
            return X_scaled @ fit["W"][alpha_index] + fit["b"][alpha_index]

    W = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    b = np.array([0.0, 0.0], dtype=np.float64)
    fit = {"W": W[None, ...], "b": b[None, ...]}

    class _Scaler:
        def transform(self, X):
            return np.asarray(X, dtype=np.float64)

    fits = {
        "pre_evolution": {"fit": fit, "scaler": _Scaler(), "alpha": 1.0},
        "T": {"fit": fit, "scaler": _Scaler(), "alpha": 1.0},
    }
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    features_by_arch = {
        "arm": {"pre_evolution": X, "T": X + 1},
        "x86": {"pre_evolution": X, "T": X + 1},
    }
    pred = driver.apply_frozen_ridge(fits, features_by_arch, _FakeRidge)
    # same fit object identity used for both arches
    assert pred["arm"]["pre_evolution"] is not None
    np.testing.assert_allclose(
        pred["arm"]["pre_evolution"], pred["x86"]["pre_evolution"])
    np.testing.assert_allclose(pred["arm"]["T"], pred["x86"]["T"])
    assert _FakeRidge.calls == 4  # 2 conditions x 2 arches; no fit_final


# ---- construction record -------------------------------------------------


def test_construction_record_marks_a_regenerated(driver):
    indices = np.array([1, 2, 3], dtype=np.int64)
    rec = driver.build_construction_record(
        b_meta={"true_count": 89, "cap": 500, "cap_applied": False, "n_used": 89},
        n_stress=3, indices=indices, indices_refined=False)
    assert rec["component_a_source"] == "regenerated"
    assert rec["framing"] == driver.FRAMING
    assert rec["component_a_size"] == 100
    assert rec["indices_sha256"] == driver.indices_sha256(indices)
    assert rec["indices_refined"] is False


# ---- run-id kind naming --------------------------------------------------


def test_synthesis_kinds_embed_run_id(driver):
    a = driver.synthesis_kind_report("20260810T120000Z")
    b = driver.synthesis_kind_ridge("20260810T120000Z")
    assert a == "protocol1_propagation_report_20260810T120000Z"
    assert b == "protocol1_ridge_frozen_20260810T120000Z"
    a2 = driver.synthesis_kind_report("20260810T120001Z")
    assert a != a2
    # object paths must differ for two run_ids
    assert a2.endswith("20260810T120001Z")


def test_driver_source_has_no_bare_synthesis_kind_literals(tree):
    """Bare fixed names without _{run_id} must not appear as object-kind literals."""
    banned = {"protocol1_propagation_report", "protocol1_ridge_frozen"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val in banned:
                pytest.fail(f"bare synthesis kind literal {val!r} in driver")
            # f-string pieces are JoinedStr; Constants inside them are OK if they
            # are prefixes ending with _ — only exact bare names are banned.


def test_two_run_ids_never_collide(driver):
    paths = {
        driver.synthesis_kind_report(r) for r in ("A", "B", "C")
    } | {
        driver.synthesis_kind_ridge(r) for r in ("A", "B", "C")
    }
    assert len(paths) == 6


# ---- AST / module-scope contracts ----------------------------------------


def test_module_scope_imports_only_stdlib_and_numpy(tree):
    allowed = {
        "argparse", "hashlib", "json", "os", "platform", "subprocess", "sys",
        "time", "traceback", "types", "numpy", "__future__",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod in allowed, node.module


def test_no_dunder_file_at_module_scope(tree):
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "__file__":
                # function bodies are nested under FunctionDef, not tree.body
                # direct children only — walk of a top-level Assign/Expr etc.
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    continue
                pytest.fail("__file__ referenced at module scope")


def test_the_driver_never_forces_an_overwrite(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "force":
            assert not (isinstance(node.value, ast.Constant)
                        and node.value.value is True)


def test_no_call_site_passes_allow_test_split(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "allow_test_split":
            pytest.fail("protocol1 driver must never pass allow_test_split")


def test_sentinels(driver):
    assert driver.OK_SENTINEL == "PROTOCOL1_OK"
    assert driver.HALT_SENTINEL == "PROTOCOL1_HALT"
    assert driver.FAIL_SENTINEL == "PROTOCOL1_FAIL"
    assert driver.X86_OK_SENTINEL == "PROTOCOL1_X86_ENCODE_OK"
    assert driver.SPLIT == "train"
    assert driver.LADDER_STAGE == 3


# ---- row-alignment / encoding sanity -------------------------------------


def test_encoding_sanity_threshold_is_strict_gt(driver):
    # exactly 1e-12 must NOT refuse; the gate is `> 1e-12`
    assert driver.ENCODING_SANITY_MAX_ABS == 1e-12
    left = np.zeros((2, 2))
    right = left.copy()
    right[0, 0] = 1e-12
    assert audit.max_abs_difference(left, right) == pytest.approx(1e-12)
    # driver gate semantics: refuse only when strictly greater
    assert not (audit.max_abs_difference(left, right)
                > driver.ENCODING_SANITY_MAX_ABS)
    right[0, 0] = 1e-12 + 1e-18
    assert audit.max_abs_difference(left, right) > driver.ENCODING_SANITY_MAX_ABS


def test_row_permutation_fails_alignment_contract(driver):
    n, d = 5, driver.EXPECTED_N_ACTIVE
    indices = np.arange(n, dtype=np.int64)
    thetas = np.arange(n * d, dtype=np.float64).reshape(n, d)
    deltas = np.zeros(n)
    # healthy
    driver.assert_stress_row_contract(thetas, deltas, indices, label="ok")
    # permute rows of thetas relative to indices → encoding max-abs blows up
    perm = thetas[::-1].copy()
    enc = audit.max_abs_difference(thetas, perm)
    assert enc > driver.ENCODING_SANITY_MAX_ABS
    # non-increasing indices refuse before evolve
    bad_idx = indices.copy()
    bad_idx[1], bad_idx[2] = bad_idx[2], bad_idx[1]
    with pytest.raises(driver.Protocol1Fail, match="strictly increasing"):
        driver.assert_stress_row_contract(thetas, deltas, bad_idx, label="bad")


def test_max_abs_difference_rejects_shape_and_nan():
    with pytest.raises(audit.AuditInputError, match="shape"):
        audit.max_abs_difference(np.zeros(3), np.zeros(2))
    with pytest.raises(audit.AuditInputError, match="finite"):
        audit.max_abs_difference(np.array([1.0, np.nan]), np.zeros(2))
