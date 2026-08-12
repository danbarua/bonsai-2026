"""Tests for the Stage 2A gauge comparison driver.

Two-tier, per this project's convention. Tier 1 is self-contained and
synthetic and always runs; Tier 2 verifies against the local-only
regenerated states and skips cleanly when they are absent.

The properties pinned here were all established by hand while building the
driver -- the CV mirror reproducing the locked selection, the gauges'
dimensions, the circular-mean rank deficiency, and the registered
classification rule. Principle 20: a property confirmed in a session
transcript is real but unfalsifiable later, so each one is an assertion
here rather than a paragraph in a log.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE2A = os.path.join(_REPO, "experiments", "stage2a_dynamics_classification")

pytest.importorskip("sklearn")
sys.path.insert(0, _STAGE2A)


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "run_gauge_comparison_2a", os.path.join(_STAGE2A, "run_gauge_comparison_2a.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


drv = _load_driver()
import stage2a_core as s2a          # noqa: E402
import stage2a_classifier as s2a_clf  # noqa: E402


# ---------------------------------------------------------------- Tier 1 ---

def _synthetic(n=240, n_nodes=24, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n // n_classes)
    centres = rng.uniform(0, 2 * np.pi, size=(n_classes, n_nodes))
    theta = np.stack([centres[c] + 0.4 * rng.standard_normal(n_nodes) for c in y])
    return theta, y


def test_cv_mirror_reproduces_the_locked_selection_exactly():
    """cv_with_accuracy is glue code around the locked procedure. Principle
    16: the risk is not that _fit_one is wrong, it is that the loop around it
    quietly diverges -- so assert agreement rather than inspect it."""
    theta, y = _synthetic()
    X = np.stack([s2a.reference_node_features(t, 3) for t in theta])
    grid = (1e-2, 1e0, 1e2)

    best_C, mean_loss, mean_acc, per_fold = drv.cv_with_accuracy(
        X, y, "test", c_grid=grid)
    real_C, real_loss, _ = s2a_clf.select_C_via_cv(X, y, "test", )

    # Restrict the locked call to the same grid by comparing shared keys.
    assert set(mean_loss) == set(grid)
    for C in grid:
        assert mean_loss[C] == pytest.approx(real_loss[C], abs=1e-12), (
            f"C={C}: mirror {mean_loss[C]} vs locked {real_loss[C]}")

    # Selection agrees whenever the restricted grid contains the locked
    # argmin; otherwise the mirror must still pick the argmin of its own grid.
    assert best_C == min(mean_loss, key=lambda C: (mean_loss[C], C))
    if real_C in grid:
        assert best_C == real_C

    assert set(mean_acc) == set(grid)
    assert all(0.0 <= a <= 1.0 for a in mean_acc.values())
    assert all(len(v) == s2a_clf.N_FOLDS for v in per_fold.values())


def test_the_mirror_would_notice_a_changed_fold_split():
    """The guard above only means something if a divergence breaks it."""
    theta, y = _synthetic()
    X = np.stack([s2a.reference_node_features(t, 3) for t in theta])
    grid = (1e0,)

    _, honest, _, _ = drv.cv_with_accuracy(X, y, "test", c_grid=grid)
    _, drifted, _, _ = drv.cv_with_accuracy(X, y, "test", c_grid=grid, seed=1)
    assert honest[1e0] != drifted[1e0], (
        "changing the fold seed left mean_val_loss identical -- this test "
        "cannot detect a split divergence")


def test_the_two_gauges_have_their_registered_dimensions():
    theta, _ = _synthetic()
    n_nodes = theta.shape[1]
    ref = drv.gauge_features(theta, "reference", 3)
    cm = drv.gauge_features(theta, "circular_mean", 3)
    assert ref.shape == (len(theta), 2 * n_nodes - 2)
    assert cm.shape == (len(theta), 2 * n_nodes)


def test_circular_mean_is_rank_deficient_by_exactly_one():
    """Amendment 1's correction: the gauge's apparent +2 column edge is
    really +1 independent direction, because sum_i sin(theta_i - mu) = 0
    identically. If this ever stops holding, the caveat in the
    pre-registration is wrong and the write-up needs revisiting."""
    theta, _ = _synthetic(n=400, n_nodes=30)
    n_nodes = theta.shape[1]
    cm = drv.gauge_features(theta, "circular_mean", 3)
    ref = drv.gauge_features(theta, "reference", 3)

    sin_sums = cm[:, n_nodes:].sum(axis=1)
    assert np.max(np.abs(sin_sums)) < 1e-10, (
        f"sin columns no longer sum to zero (max {np.max(np.abs(sin_sums)):.3e})")
    assert np.linalg.matrix_rank(cm) == cm.shape[1] - 1
    assert np.linalg.matrix_rank(ref) == ref.shape[1]


def test_the_gauges_are_the_real_ones_not_a_local_copy():
    """principle 16 again, at the other call site: the driver must not
    reimplement the gauge the way analyze_stage3_results_jax.py inlines it."""
    theta, _ = _synthetic(n=8)
    ref = drv.gauge_features(theta, "reference", 3)
    cm = drv.gauge_features(theta, "circular_mean", 3)
    for i, t in enumerate(theta):
        assert np.array_equal(ref[i], s2a.reference_node_features(t, 3))
        assert np.array_equal(cm[i], s2a.circular_mean_features(t))


def test_an_unknown_gauge_is_refused():
    theta, _ = _synthetic(n=4)
    with pytest.raises(ValueError):
        drv.gauge_features(theta, "whatever", 3)


# -- the registered rule ----------------------------------------------------

def _acc(ref, cm):
    conds = tuple(ref)
    return {"reference": dict(ref), "circular_mean": dict(cm)}, conds


def test_rule_class_A_when_ranking_holds_and_margins_are_small():
    acc, conds = _acc({"a": 0.90, "b": 0.80, "c": 0.70},
                      {"a": 0.9005, "b": 0.8005, "c": 0.7005})
    cls, _, direction, _, _, max_M = drv.classify(acc, conditions=conds)
    assert cls == "A"
    assert max_M < drv.THETA
    assert direction == "uniform direction favouring circular_mean"


def test_rule_class_B_when_ranking_holds_but_a_margin_is_material():
    acc, conds = _acc({"a": 0.90, "b": 0.80, "c": 0.70},
                      {"a": 0.9050, "b": 0.8005, "c": 0.7005})
    cls, _, _, _, _, max_M = drv.classify(acc, conditions=conds)
    assert cls == "B"
    assert max_M >= drv.THETA


def test_rule_class_C_takes_precedence_over_a_material_margin():
    """C outranks B: an inversion is reported as an inversion even when some
    margin is also large."""
    acc, conds = _acc({"a": 0.90, "b": 0.80, "c": 0.70},
                      {"a": 0.75, "b": 0.85, "c": 0.70})
    cls, _, _, rank, _, max_M = drv.classify(acc, conditions=conds)
    assert cls == "C"
    assert max_M >= drv.THETA, "this case must also satisfy B, to test precedence"
    assert rank["reference"] != rank["circular_mean"]


def test_rule_reports_mixed_direction_when_signs_disagree():
    acc, conds = _acc({"a": 0.90, "b": 0.80, "c": 0.70},
                      {"a": 0.9050, "b": 0.7950, "c": 0.7005})
    _, _, direction, _, _, _ = drv.classify(acc, conditions=conds)
    assert direction == "mixed direction"


def test_float_noise_below_the_tie_resolution_cannot_manufacture_an_inversion():
    """The registered tie rule: accuracies equal to 5 decimal places count as
    preserved. Two conditions separated by 1e-9 must not read as a swap."""
    acc, conds = _acc({"a": 0.800000001, "b": 0.800000000},
                      {"a": 0.800000000, "b": 0.800000001})
    cls, _, _, _, _, _ = drv.classify(acc, conditions=conds)
    assert cls == "A"


# ---------------------------------------------------------------- Tier 2 ---

_SCRATCH = None
try:
    from stage2a_paths import train_scratch_dir
    _SCRATCH = train_scratch_dir()
except Exception:  # pragma: no cover - import guard only
    pass

_ENCODE = os.path.join(_SCRATCH, "stage3_encode_local.pkl") if _SCRATCH else ""
_HAVE_STATES = bool(_SCRATCH) and os.path.exists(_ENCODE) and any(
    f.startswith("theta0_chunk_") for f in os.listdir(_SCRATCH))


@pytest.mark.skipif(not _HAVE_STATES,
                    reason="regenerated Stage 2A train states not present locally")
def test_cached_feat_pre_reproduces_from_the_theta0_chunks():
    """The driver's pre-fit gate, as a test. Catches a chunk-ordering or
    indexing error before it can reach a classifier -- the cached feat_pre and
    a fresh recomputation come from the same function on the same inputs, so
    anything other than exact equality is a defect in the reassembly."""
    local = drv.load_encode_local()
    theta0 = drv.load_theta0()
    assert theta0.shape[0] == local["feat_pre"].shape[0]

    recomputed = drv.gauge_features(theta0, "reference", int(local["ref_idx"]))
    assert np.array_equal(recomputed, local["feat_pre"]), (
        f"max|diff|={np.max(np.abs(recomputed - local['feat_pre'])):.3e}")
