"""Tier 1: the regeneration acceptance test's index join.

`compare_stage3_regeneration.py` decides whether the 60,000-image
regeneration reproduced the 54,000-image baseline. It can only mean that
if it compares the RIGHT 54,000 rows, and `AUDIT_PROTOCOL.md` is explicit
that the match is by official KMNIST image index and never by positional
prefix.

The failure this pins is the quiet one: `new[:54000]` returns an array of
exactly the right shape and dtype, and comparing it produces a perfectly
confident verdict about the wrong rows. Nothing raises. So the join is
tested on synthetic artifacts small enough to reason about, including the
degenerate case where a prefix would have worked -- which must be refused
rather than accepted, because a join that silently accepts the prefix
case is a join nobody can distinguish from `[:54000]`.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE2B = os.path.join(REPO_ROOT, "experiments", "stage2b_denoising")
sys.path.insert(0, STAGE2B)

cmp_mod = pytest.importorskip("compare_stage3_regeneration")


def _artifacts(n_total=20, n_old=14, order=None, rng_seed=0):
    """A synthetic (new, old) pair with a known correct alignment.

    `new` covers `range(n_total)` in ascending order, the real artifact's
    layout. `old` covers a scattered subset, as the real fit side does.
    Old rows are FILLED FROM the new rows they correspond to, so a correct
    join finds them bit-exact and an incorrect one does not."""
    rng = np.random.default_rng(rng_seed)
    new_idx = np.arange(n_total)
    thetas = rng.normal(size=(n_total, 5))
    deltas = rng.random(n_total)
    if order is None:
        order = np.sort(rng.choice(n_total, size=n_old, replace=False))
    old_idx = np.asarray(order)
    new = {"train_indices": new_idx, "thetas_505": thetas, "deltas": deltas}
    old = {"fit_indices": old_idx,
           "thetas_505": thetas[old_idx].copy(),
           "deltas": deltas[old_idx].copy()}
    return new, old


@pytest.fixture(autouse=True)
def small_baseline(monkeypatch):
    """The real script asserts a 54,000-row overlap. Point that constant
    at the fixture's size rather than building 54,000 synthetic rows."""
    monkeypatch.setattr(cmp_mod, "BASELINE_TAIL_N", 14)


def test_the_join_recovers_the_baseline_rows_bit_exactly():
    new, old = _artifacts()
    rows, report = cmp_mod.align(old, new)
    findings = cmp_mod.compare(old, new, rows)
    assert all(entry["bit_exact"] for entry in findings), findings
    assert report["n_overlap"] == 14
    assert report["alignment_is_a_prefix"] is False
    assert report["n_rows_moved"] > 0


def test_the_positional_prefix_gives_a_different_and_wrong_answer():
    """The whole reason the join exists: taking the first 14 rows is not
    the same 14 images, and nothing about shape or dtype reveals it."""
    new, old = _artifacts()
    rows, _ = cmp_mod.align(old, new)
    prefix = np.arange(old["fit_indices"].size)
    assert not np.array_equal(rows, prefix)
    correct = cmp_mod.compare(old, new, rows)
    wrong = cmp_mod.compare(old, new, prefix)
    assert all(entry["bit_exact"] for entry in correct)
    assert not any(entry["bit_exact"] for entry in wrong), (
        "the prefix comparison passed, so this fixture cannot distinguish a "
        "correct join from a positional one and proves nothing")
    # And the wrong comparison still reports plausible shapes and dtypes.
    assert wrong[0]["shape_old"] == wrong[0]["shape_new"]


def test_an_alignment_that_IS_a_prefix_is_refused_rather_than_accepted():
    """The degenerate case. If the baseline's indices happen to be
    `range(14)`, the join and the prefix coincide -- and a passing result
    would then be indistinguishable from never having joined at all. The
    script refuses, because the real fit side is scattered and a prefix
    alignment means the artifact's row order changed underneath it."""
    new, old = _artifacts(order=np.arange(14))
    with pytest.raises(ValueError, match="positional prefix"):
        cmp_mod.align(old, new)


def test_a_baseline_image_missing_from_the_regeneration_is_named():
    new, old = _artifacts()
    old["fit_indices"] = np.append(old["fit_indices"], 999)
    with pytest.raises(ValueError, match="absent from the new artifact"):
        cmp_mod.align(old, new)


def test_duplicate_indices_in_the_regeneration_are_refused():
    """A duplicate would make the index -> row map lossy, and the loss
    would be silent: the map keeps the last occurrence and the comparison
    proceeds against rows nobody chose."""
    new, old = _artifacts()
    new["train_indices"] = new["train_indices"].copy()
    new["train_indices"][5] = new["train_indices"][4]
    with pytest.raises(ValueError, match="duplicates"):
        cmp_mod.align(old, new)


def test_an_overlap_of_the_wrong_size_is_refused():
    """The vacuity guard: a join over 3 images would compare cleanly and
    say nothing about the 54,000 the acceptance test is named for."""
    new, old = _artifacts(n_old=3, order=np.array([1, 7, 13]))
    with pytest.raises(ValueError, match="expected an overlap"):
        cmp_mod.align(old, new)


def test_a_single_changed_coordinate_is_caught_and_quantified():
    new, old = _artifacts()
    old["thetas_505"] = old["thetas_505"].copy()
    old["thetas_505"][3, 2] = np.nextafter(old["thetas_505"][3, 2], np.inf)
    rows, _ = cmp_mod.align(old, new)
    findings = {entry["array"]: entry for entry in cmp_mod.compare(old, new, rows)}
    assert findings["thetas_505"]["bit_exact"] is False
    assert findings["thetas_505"]["n_differing"] == 1
    assert 0.0 < findings["thetas_505"]["max_abs_diff"] < 1e-15
    assert findings["deltas"]["bit_exact"] is True, (
        "one array differing must not contaminate the verdict on another")


def test_the_comparison_is_on_bytes_not_on_equality():
    """`==` treats two NaNs as different and -0.0 as equal to 0.0. A
    regeneration check has to see both, so the digest is over bytes."""
    new, old = _artifacts()
    old["deltas"] = old["deltas"].copy()
    old["deltas"][0] = -0.0
    new["deltas"] = new["deltas"].copy()
    new["deltas"][old["fit_indices"][0]] = 0.0
    rows, _ = cmp_mod.align(old, new)
    findings = {entry["array"]: entry for entry in cmp_mod.compare(old, new, rows)}
    assert findings["deltas"]["bit_exact"] is False, (
        "-0.0 and 0.0 compared equal, so the check is doing `==` and not bytes")


def test_the_baseline_object_is_never_the_regeneration_target():
    """A regeneration that wrote to the baseline's name would destroy the
    thing it is compared against, irreversibly, via exactly the
    `force=True` trust-point bypass this project has documented."""
    enc = pytest.importorskip("encode_stage3_local")
    assert cmp_mod.BASELINE_KIND == "encoded_fit_s1200"
    assert cmp_mod.BASELINE_KIND not in enc.object_name_for(1200)
