"""Tests for experiments/stage2b_denoising/stage2b_partition.py -- the
locked fit/validation partition of the official KMNIST training split and
the feasibility-ladder development subsets drawn from it.

Tier 1 (self-contained, always run) only: Stage 2B has no historical
cached artifact to verify against. Every test here is synthetic; the
official-size cases use synthetic labels of length 60,000, which is the
only property of the real corpus this module depends on.

The ordering tests are the point of the file. DESIGN.md requires the
`StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)` to
be created BEFORE any feasibility subset is drawn; these check that the
API makes the reverse sequence raise rather than merely discouraging it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedShuffleSplit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_partition as partition  # noqa: E402


def _official_labels(seed=0):
    """Synthetic labels of the official training split's length, with the
    class imbalance that makes stratification observable."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(10, size=partition.N_OFFICIAL_TRAIN,
                        p=np.array([12, 8, 11, 9, 10, 10, 10, 10, 10, 10]) / 100.0)
    return labels


def _small_labels(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5, n)


# ---- locked constants ----

def test_locked_partition_constants():
    assert partition.N_PARTITION_SPLITS == 1
    assert partition.VALIDATION_TEST_SIZE == 0.10
    assert partition.PARTITION_SEED == 42
    assert partition.N_OFFICIAL_TRAIN == 60_000
    assert partition.N_FIT == 54_000
    assert partition.N_VALIDATION == 6_000
    assert partition.STAGE1_SUBSET_SIZE == 1_000
    assert partition.STAGE2_SUBSET_SIZE == 5_000


# ---- the partition itself ----

def test_official_partition_is_54000_fit_and_6000_validation():
    p = partition.Stage2BTrainingPartition(_official_labels())
    assert p.fit_indices.size == partition.N_FIT
    assert p.validation_indices.size == partition.N_VALIDATION


def test_partition_is_disjoint_and_covers_every_training_index():
    p = partition.Stage2BTrainingPartition(_official_labels())
    assert np.intersect1d(p.fit_indices, p.validation_indices).size == 0
    np.testing.assert_array_equal(
        np.union1d(p.fit_indices, p.validation_indices),
        np.arange(partition.N_OFFICIAL_TRAIN))


def test_partition_matches_the_literal_locked_splitter():
    """The spec is a literal `StratifiedShuffleSplit` configuration, so the
    partition must equal what that configuration produces -- constructed
    independently here, not read back from the module's own constants."""
    labels = _official_labels()
    p = partition.Stage2BTrainingPartition(labels)
    expected_fit, expected_val = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
        .split(np.zeros((labels.size, 1)), labels))
    np.testing.assert_array_equal(p.fit_indices, np.sort(expected_fit))
    np.testing.assert_array_equal(p.validation_indices, np.sort(expected_val))


def test_partition_is_reproducible_across_constructions():
    labels = _official_labels()
    a = partition.Stage2BTrainingPartition(labels)
    b = partition.Stage2BTrainingPartition(labels)
    np.testing.assert_array_equal(a.fit_indices, b.fit_indices)
    np.testing.assert_array_equal(a.validation_indices, b.validation_indices)


def test_partition_is_stratified_by_class():
    labels = _official_labels()
    p = partition.Stage2BTrainingPartition(labels)
    overall = np.bincount(labels, minlength=10) / labels.size
    val = np.bincount(p.validation_labels(), minlength=10) / p.validation_indices.size
    np.testing.assert_allclose(val, overall, atol=1e-3)


def test_returned_index_arrays_are_read_only():
    """A caller that could edit these in place could change the locked
    validation partition after the fact, invisibly."""
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    for arr in (p.fit_indices, p.validation_indices):
        with pytest.raises(ValueError):
            arr[0] = -1


def test_indices_are_positions_in_the_official_training_split():
    """The index space is the one `stage2b_corruption.corrupt_corpus`
    requires: 0..n-1 over the whole official split, not 0..subset_size-1."""
    p = partition.Stage2BTrainingPartition(_official_labels())
    assert p.fit_indices.min() >= 0
    assert max(p.fit_indices.max(), p.validation_indices.max()) == partition.N_OFFICIAL_TRAIN - 1
    assert p.fit_indices.dtype == np.int64


# ---- the ordering requirement, structurally ----

def test_constructing_on_an_already_drawn_subset_raises():
    """The failure this class exists to prevent: drawing a 5,000-image
    feasibility subset first and partitioning THAT. The subset is a
    perfectly valid label array, so nothing except the size guard can
    notice."""
    labels = _official_labels()
    subset_labels = labels[:5_000]
    with pytest.raises(ValueError, match="BEFORE any feasibility subset"):
        partition.Stage2BTrainingPartition(subset_labels)


def test_non_official_size_is_allowed_only_with_the_explicit_flag():
    labels = _small_labels()
    with pytest.raises(ValueError, match="refusing to partition"):
        partition.Stage2BTrainingPartition(labels)
    p = partition.Stage2BTrainingPartition(labels, allow_non_official_size=True)
    assert p.n_images == labels.size
    assert p.fit_indices.size + p.validation_indices.size == labels.size


def test_the_split_happens_at_construction_with_no_separate_step():
    """There is no `split()` to call in the wrong order, or twice with
    different arguments: the partition exists as soon as the object does."""
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    assert p.fit_indices.size > 0 and p.validation_indices.size > 0
    assert not hasattr(p, "split")
    assert not hasattr(partition, "split_training_set")


def test_drawing_a_subset_does_not_disturb_the_locked_validation_partition():
    """Feasibility work shrinks the fit side only. The validation corpus at
    ladder stage 2 must be the same one used at stage 3."""
    p = partition.Stage2BTrainingPartition(_official_labels())
    before = np.array(p.validation_indices, copy=True)
    p.development_subset(partition.STAGE1_SUBSET_SIZE, seed=0, stratified=True)
    p.development_subset(partition.STAGE2_SUBSET_SIZE, seed=0, stratified=False)
    np.testing.assert_array_equal(p.validation_indices, before)


# ---- development subsets ----

@pytest.mark.parametrize("stratified", [True, False])
def test_development_subset_never_touches_the_validation_partition(stratified):
    p = partition.Stage2BTrainingPartition(_official_labels())
    subset = p.development_subset(partition.STAGE2_SUBSET_SIZE, seed=42,
                                  stratified=stratified)
    assert subset.size == partition.STAGE2_SUBSET_SIZE
    assert np.intersect1d(subset, p.validation_indices).size == 0
    assert np.setdiff1d(subset, p.fit_indices).size == 0


@pytest.mark.parametrize("size", [partition.STAGE1_SUBSET_SIZE,
                                  partition.STAGE2_SUBSET_SIZE])
def test_both_ladder_subset_sizes_are_drawable(size):
    p = partition.Stage2BTrainingPartition(_official_labels())
    assert p.development_subset(size, seed=7, stratified=True).size == size


def test_development_subset_requires_the_draw_rule_to_be_stated():
    """DESIGN.md does not lock how the subset is drawn, so neither argument
    is defaulted -- calling without them is a TypeError, not a silent
    choice made by this module."""
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    with pytest.raises(TypeError):
        p.development_subset(10)
    with pytest.raises(TypeError):
        p.development_subset(10, seed=0)
    with pytest.raises(TypeError):
        p.development_subset(10, stratified=True)


def test_development_subset_rejects_a_non_bool_stratified():
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    with pytest.raises(TypeError, match="explicit bool"):
        p.development_subset(10, seed=0, stratified="yes")


@pytest.mark.parametrize("stratified", [True, False])
def test_development_subset_is_reproducible_and_seed_dependent(stratified):
    p = partition.Stage2BTrainingPartition(_official_labels())
    a = p.development_subset(1_000, seed=0, stratified=stratified)
    b = p.development_subset(1_000, seed=0, stratified=stratified)
    c = p.development_subset(1_000, seed=1, stratified=stratified)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stratified_subset_preserves_class_proportions_and_unstratified_need_not():
    """The two draw rules are genuinely different objects, not one with a
    cosmetic flag -- which is why the choice has to be stated rather than
    defaulted."""
    labels = _official_labels()
    p = partition.Stage2BTrainingPartition(labels)
    fit_props = np.bincount(p.fit_labels(), minlength=10) / p.fit_indices.size
    strat = labels[p.development_subset(5_000, seed=3, stratified=True)]
    np.testing.assert_allclose(np.bincount(strat, minlength=10) / strat.size,
                               fit_props, atol=2e-3)
    plain = labels[p.development_subset(5_000, seed=3, stratified=False)]
    assert not np.array_equal(np.bincount(plain, minlength=10),
                              np.bincount(strat, minlength=10))


@pytest.mark.parametrize("stratified", [True, False])
def test_development_subset_is_sorted_and_duplicate_free(stratified):
    p = partition.Stage2BTrainingPartition(_official_labels())
    subset = p.development_subset(2_000, seed=5, stratified=stratified)
    np.testing.assert_array_equal(subset, np.sort(subset))
    assert np.unique(subset).size == subset.size


def test_development_subset_rejects_impossible_sizes():
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    with pytest.raises(ValueError, match="fit side"):
        p.development_subset(0, seed=0, stratified=True)
    with pytest.raises(ValueError, match="fit side"):
        p.development_subset(p.fit_indices.size + 1, seed=0, stratified=True)


def test_development_subset_of_the_whole_fit_side_is_the_fit_side():
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    subset = p.development_subset(p.fit_indices.size, seed=0, stratified=True)
    np.testing.assert_array_equal(subset, p.fit_indices)


# ---- provenance and input validation ----

def test_summary_records_the_locked_splitter_configuration():
    p = partition.Stage2BTrainingPartition(_official_labels())
    s = p.summary()
    assert s == {"n_images": 60_000, "n_fit": 54_000, "n_validation": 6_000,
                 "splitter": "StratifiedShuffleSplit", "n_splits": 1,
                 "test_size": 0.10, "random_state": 42,
                 "index_space": "official KMNIST training split (0-based)",
                 "is_official_size": True}


def test_rejects_non_1d_labels():
    with pytest.raises(ValueError, match="1-D"):
        partition.Stage2BTrainingPartition(np.zeros((100, 2), dtype=int),
                                           allow_non_official_size=True)
