import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "stage2b_denoising"))

import stage2b_audit as audit  # noqa: E402


def _mse(pre, t, lattice, rewired, curr):
    return {"pre_evolution": np.asarray(pre), "T": np.asarray(t),
            "lattice": np.asarray(lattice), "rewired": np.asarray(rewired),
            "curr_random": np.asarray(curr)}


def test_index_alignment_is_by_official_index():
    values = np.array([[10], [20], [30]])
    np.testing.assert_array_equal(
        audit.align_by_official_index(values, [5, 2, 9], [9, 5, 2]),
        [[30], [10], [20]])


def test_index_alignment_rejects_missing_index():
    with pytest.raises(audit.AuditInputError, match="absent"):
        audit.align_by_official_index([1, 2], [5, 6], [7])


def test_artifacts_may_only_differ_in_encoder_steps():
    audit.assert_same_audit_inputs({"encoder_steps": 150, "seed": 42},
                                   {"encoder_steps": 1200, "seed": 42})
    with pytest.raises(audit.AuditInputError, match="outside"):
        audit.assert_same_audit_inputs({"encoder_steps": 150, "seed": 1},
                                       {"encoder_steps": 1200, "seed": 2})


def test_trigger_checks_all_graphs_and_pairwise_orders():
    base = np.zeros(8)
    low = np.full(8, -0.2)
    high = np.full(8, 0.2)
    before = _mse(base, low, low - .01, low - .02, low - .03)
    after = _mse(base, high, high - .01, high - .02, high - .03)
    verdict = audit.trigger_verdict({150: before, 1200: after})
    assert verdict["primary_sign_reversal"]
    assert verdict["graph_sign_reversals"]["T"]
    assert verdict["triggered"]
    assert len(verdict["pairwise"]) == 6


def test_feature_distance_reports_per_image_distributions():
    left = np.zeros((2, 4))
    right = np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]])
    result = audit.feature_distances(
        left, right, reference_column=0,
        features_left=np.zeros((2, 6)), features_right=np.ones((2, 6)))
    assert set(result) == {"phase_max", "phase_rms", "cos_sin_euclidean"}
    assert result["phase_max"]["max"] > 0
    assert result["cos_sin_euclidean"]["median"] == pytest.approx(np.sqrt(6))


def test_wrapped_phase_difference_rejects_incompatible_shapes():
    with pytest.raises(audit.AuditInputError, match="broadcast-compatible"):
        audit.wrapped_phase_difference(np.zeros((2, 4)), np.zeros((3, 1)))


def test_stress_set_is_sorted_deduplicated_and_class_covered():
    indices = np.arange(60000)
    labels = np.arange(60000) % 10
    result = audit.build_stress_indices(indices, labels, [1, 2, 3], [4, 5])
    assert np.array_equal(result, np.sort(result))
    assert len(result) == len(set(result))
    assert all(np.count_nonzero(labels[result] == cls) >= 20 for cls in range(10))


def test_sensitivity_table_only_uses_supplied_steps_and_epsilons():
    calls = []

    def gate(clean, noisy, *, abs_conv_eps):
        calls.append(abs_conv_eps)
        return {"passed": abs_conv_eps == 1e-12}

    result = audit.sensitivity_table({75: ([1], [2]), 1200: ([3], [4])}, gate)
    assert set(result) == {"75", "1200"}
    assert len(calls) == 8


def test_prerequisite_guard_names_missing_inputs(tmp_path):
    with pytest.raises(audit.AuditInputError, match="sequencing gate"):
        audit.require_audit_prerequisites(tmp_path / "encoded150.npz",
                                          tmp_path / "ridge.json")


def test_oof_fold_cross_check_rejects_wrong_stored_aggregate():
    oof = {"oof_clipped_mse": np.array([[1., 2.], [3., 4.]]),
           "fold_index": np.array([0, 1])}
    stored = {"fold_clipped_val_mse": np.array([[1., 2.], [3., 4.]])}
    assert audit.assert_oof_matches_fold_aggregates(oof, stored)["passed"]
    stored["fold_clipped_val_mse"][0, 0] += 1e-3
    with pytest.raises(audit.AuditInputError, match="reproduce"):
        audit.assert_oof_matches_fold_aggregates(oof, stored)
