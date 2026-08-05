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
import subprocess
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


def _balanced_official_labels(seed=2):
    """The real corpus's shape: KMNIST's training split is exactly 6,000
    images per class. Shuffled, so official index order carries no class
    information -- `_class_blocked_labels` is the opposite case."""
    labels = np.repeat(np.arange(10), partition.N_OFFICIAL_TRAIN // 10)
    np.random.default_rng(seed).shuffle(labels)
    return labels


def _class_blocked_labels():
    """Official index order that tracks class perfectly. A plausible
    corpus layout, and the one that makes a naive prefix of a stratified
    draw degenerate."""
    return np.repeat(np.arange(10), partition.N_OFFICIAL_TRAIN // 10)


def _max_prefix_deviation(indices, labels):
    """The largest gap, over EVERY prefix length, between a class's count
    in the prefix and its exact quota `p * n_c / N`."""
    classes = np.unique(labels)
    present = labels[indices][:, None] == classes[None, :]
    counts = np.cumsum(present, axis=0)
    p = np.arange(1, indices.size + 1)[:, None]
    quota = p * (counts[-1] / indices.size)[None, :]
    return float(np.abs(counts - quota).max())


def _ladder(p, size=partition.STAGE2_SUBSET_SIZE,
            prefix_size=partition.STAGE1_SUBSET_SIZE, seed=None, stratified=None):
    return p.nested_development_subsets(
        size=size, prefix_size=prefix_size,
        seed=partition.LADDER_SUBSET_SEED if seed is None else seed,
        stratified=partition.LADDER_SUBSET_STRATIFIED if stratified is None else stratified)


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


# ---- the nested ladder draw ----

def test_ladder_draw_rule_constants():
    """The decided values, so a call site has one obvious right answer to
    pass rather than a defaulted one it never sees."""
    assert partition.LADDER_SUBSET_SEED == 42
    assert partition.LADDER_SUBSET_STRATIFIED is True


def test_stage1_is_exactly_the_prefix_of_stage2():
    s = _ladder(partition.Stage2BTrainingPartition(_balanced_official_labels()))
    assert s.stage2_indices.size == partition.STAGE2_SUBSET_SIZE
    assert s.stage1_indices.size == partition.STAGE1_SUBSET_SIZE
    np.testing.assert_array_equal(
        s.stage1_indices, s.stage2_indices[:partition.STAGE1_SUBSET_SIZE])


def test_nesting_is_structural_not_two_arrays_that_happen_to_agree():
    """`stage1_indices` is a VIEW of `stage2_indices`, sharing its buffer.
    Nesting is therefore a property of the objects: there is no second
    array that could drift out of agreement with the first."""
    s = _ladder(partition.Stage2BTrainingPartition(_balanced_official_labels()))
    assert s.stage1_indices.base is s.stage2_indices


def test_the_ladder_is_one_draw_not_two():
    """Stage 2's images are exactly the single `development_subset` draw at
    the same seed -- reordered, not redrawn."""
    p = partition.Stage2BTrainingPartition(_balanced_official_labels())
    s = _ladder(p)
    np.testing.assert_array_equal(
        np.sort(s.stage2_indices),
        p.development_subset(partition.STAGE2_SUBSET_SIZE,
                             seed=partition.LADDER_SUBSET_SEED, stratified=True))


def test_both_levels_are_stratified_on_the_real_corpus_shape():
    """The guarantee is exact when the drawn class counts are equal, which
    is the real case: 6,000 images per class in the official training
    split gives 500 per class at stage 2 and 100 per class at stage 1."""
    labels = _balanced_official_labels()
    s = _ladder(partition.Stage2BTrainingPartition(labels))
    assert s.stage2_class_counts == (500,) * 10
    assert s.stage1_class_counts == (100,) * 10
    np.testing.assert_array_equal(np.bincount(labels[s.stage2_indices], minlength=10),
                                  np.full(10, 500))
    np.testing.assert_array_equal(np.bincount(labels[s.stage1_indices], minlength=10),
                                  np.full(10, 100))


def test_both_levels_are_stratified_on_an_imbalanced_corpus():
    """Stage 2 tracks the fit side's proportions; the stage 1 prefix tracks
    stage 2's -- to within one image per class, on counts, not on a
    tolerance that would hide a skew."""
    labels = _official_labels()
    p = partition.Stage2BTrainingPartition(labels)
    s = _ladder(p)

    fit_props = np.bincount(p.fit_labels(), minlength=10) / p.fit_indices.size
    stage2 = np.array(s.stage2_class_counts)
    np.testing.assert_allclose(stage2 / stage2.sum(), fit_props, atol=2e-3)

    stage1 = np.array(s.stage1_class_counts)
    quota = partition.STAGE1_SUBSET_SIZE * stage2 / stage2.sum()
    assert np.abs(stage1 - quota).max() < 1.0
    assert stage1.sum() == partition.STAGE1_SUBSET_SIZE


def test_a_naive_prefix_of_the_same_draw_would_be_skewed():
    """Why the ordering exists. On a corpus whose index order tracks class,
    the first 1,000 of the SORTED draw covers two classes out of ten --
    while still being a perfectly valid nested prefix of a stratified
    5,000. Nesting alone does not make a prefix representative."""
    labels = _class_blocked_labels()
    p = partition.Stage2BTrainingPartition(labels)
    s = _ladder(p)

    naive = np.sort(s.stage2_indices)[:partition.STAGE1_SUBSET_SIZE]
    naive_counts = np.bincount(labels[naive], minlength=10)
    assert np.count_nonzero(naive_counts) == 2

    assert s.stage1_class_counts == (100,) * 10


@pytest.mark.parametrize("labels_fn", [_balanced_official_labels, _official_labels])
def test_every_prefix_length_tracks_the_draw_proportions(labels_fn):
    """The guarantee is over all prefix lengths, not just 1,000 -- swept
    rather than assumed, so a later change to the ordering that happens to
    keep 1,000 correct still fails here."""
    labels = labels_fn()
    s = _ladder(partition.Stage2BTrainingPartition(labels))
    assert _max_prefix_deviation(s.stage2_indices, labels) < 1.0


def test_the_prefix_is_not_the_corpus_earliest_images():
    """Within-class order is randomized. Ordering each class's draw by
    ascending official index would nest correctly and stratify correctly
    and still make stage 1 the earliest images in the corpus."""
    labels = _balanced_official_labels()
    s = _ladder(partition.Stage2BTrainingPartition(labels))
    earliest = np.sort(s.stage2_indices)[:partition.STAGE1_SUBSET_SIZE]
    assert np.intersect1d(s.stage1_indices, earliest).size < 400
    assert np.median(s.stage1_indices) > 0.3 * partition.N_OFFICIAL_TRAIN


def test_nested_subsets_are_read_only():
    s = _ladder(partition.Stage2BTrainingPartition(_balanced_official_labels()))
    for arr in (s.stage1_indices, s.stage2_indices):
        with pytest.raises(ValueError):
            arr[0] = -1


def test_nested_subsets_stay_in_the_official_index_space():
    p = partition.Stage2BTrainingPartition(_balanced_official_labels())
    s = _ladder(p)
    for arr in (s.stage1_indices, s.stage2_indices):
        assert arr.dtype == np.int64
        assert arr.min() >= 0 and arr.max() < partition.N_OFFICIAL_TRAIN
        assert np.unique(arr).size == arr.size


@pytest.mark.parametrize("stratified", [True, False])
def test_nested_subsets_never_touch_the_validation_partition(stratified):
    """Both levels stay strictly inside the fit side. Neither the draw nor
    the ordering can reach an image the locked validation partition holds."""
    p = partition.Stage2BTrainingPartition(_balanced_official_labels())
    before = np.array(p.validation_indices, copy=True)
    s = _ladder(p, stratified=stratified)
    for arr in (s.stage1_indices, s.stage2_indices):
        assert np.intersect1d(arr, p.validation_indices).size == 0
        assert np.setdiff1d(arr, p.fit_indices).size == 0
    np.testing.assert_array_equal(p.validation_indices, before)


def test_nested_subsets_cannot_precede_the_split():
    """Same structural argument as `development_subset`: it is a method on
    a partitioned object, with no module-level twin to call first."""
    assert not hasattr(partition, "nested_development_subsets")
    assert hasattr(partition.Stage2BTrainingPartition, "nested_development_subsets")


def test_nested_subsets_are_reproducible_and_seed_dependent():
    p = partition.Stage2BTrainingPartition(_balanced_official_labels())
    a, b, c = _ladder(p), _ladder(p), _ladder(p, seed=7)
    np.testing.assert_array_equal(a.stage2_indices, b.stage2_indices)
    np.testing.assert_array_equal(a.stage1_indices, b.stage1_indices)
    assert not np.array_equal(a.stage2_indices, c.stage2_indices)


def test_nested_subsets_are_reproducible_across_processes():
    """A fresh interpreter, with hash randomization at its default, must
    reproduce the same ladder -- the subsets are quoted in FINDINGS-level
    results and cannot depend on which process drew them."""
    p = partition.Stage2BTrainingPartition(_balanced_official_labels())
    code = (
        f"import sys; sys.path.insert(0, {str(_STAGE2B_DIR)!r});"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r});"
        "import numpy as np, stage2b_partition as P;"
        "from test_stage2b_partition import _balanced_official_labels, _ladder;"
        "p = P.Stage2BTrainingPartition(_balanced_official_labels());"
        "print(repr(_ladder(p).stage1_indices[:8].tolist()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert eval(out) == _ladder(p).stage1_indices[:8].tolist()


def test_nested_subsets_require_the_draw_rule_to_be_stated():
    """Every argument is keyword-only with no default, exactly as on
    `development_subset` -- including the two sizes."""
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    for kwargs in ({}, {"size": 20}, {"size": 20, "prefix_size": 5},
                   {"size": 20, "prefix_size": 5, "seed": 42},
                   {"size": 20, "prefix_size": 5, "stratified": True}):
        with pytest.raises(TypeError):
            p.nested_development_subsets(**kwargs)
    with pytest.raises(TypeError):
        p.nested_development_subsets(20, 5, seed=42, stratified=True)


def test_nested_subsets_reject_a_prefix_that_would_not_nest():
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    with pytest.raises(ValueError, match="nests inside"):
        p.nested_development_subsets(size=20, prefix_size=21, seed=42, stratified=True)
    with pytest.raises(ValueError, match="nests inside"):
        p.nested_development_subsets(size=20, prefix_size=0, seed=42, stratified=True)
    with pytest.raises(ValueError, match="fit side"):
        p.nested_development_subsets(size=p.fit_indices.size + 1, prefix_size=5,
                                     seed=42, stratified=True)


def test_nested_subsets_of_equal_sizes_are_the_same_array():
    p = partition.Stage2BTrainingPartition(_small_labels(), allow_non_official_size=True)
    s = p.nested_development_subsets(size=40, prefix_size=40, seed=42, stratified=True)
    np.testing.assert_array_equal(s.stage1_indices, s.stage2_indices)


def test_unstratified_nested_draw_still_interleaves_its_own_proportions():
    """`stratified=False` changes what is drawn, not the promise about the
    prefix: the prefix tracks whatever proportions the draw came out with."""
    labels = _official_labels()
    p = partition.Stage2BTrainingPartition(labels)
    s = _ladder(p, stratified=False)
    assert _max_prefix_deviation(s.stage2_indices, labels) < 1.0
    stage2 = np.array(s.stage2_class_counts)
    quota = partition.STAGE1_SUBSET_SIZE * stage2 / stage2.sum()
    assert np.abs(np.array(s.stage1_class_counts) - quota).max() < 1.0


def test_nested_summary_records_the_draw_rule_and_what_it_achieved():
    s = _ladder(partition.Stage2BTrainingPartition(_balanced_official_labels()))
    assert s.summary() == {
        "n_stage1": 1_000, "n_stage2": 5_000, "seed": 42, "stratified": True,
        "nesting": "stage1_indices is a read-only view of stage2_indices[:n_stage1]",
        "order": "class-proportional interleave, draw order (not sorted)",
        "index_space": "official KMNIST training split (0-based)",
        "classes": tuple(range(10)),
        "stage1_class_counts": (100,) * 10,
        "stage2_class_counts": (500,) * 10,
    }


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
