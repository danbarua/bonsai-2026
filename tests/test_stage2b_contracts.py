"""Cross-module contracts between Stage 2B's modules -- the properties
that hold BETWEEN `stage2b_partition`, `stage2b_corruption`,
`stage2b_stats` and `stage2b_gcs` rather than inside any one of them.

Tier 1 (self-contained, always run) only: synthetic labels and synthetic
images throughout, no dataset, no network.

Each module's own test file exercises it in isolation, with hand-built
inputs. That is what leaves these contracts untested: the partition's
tests never corrupt anything, the corruption tests never see a partition
object, and the statistics and the object-path scheme name the same
conditions in two different spellings without either file ever comparing
them. The failures below are all of the same kind -- every module
individually correct, everything downstream still computing, and the
number quietly wrong.

**No image is corrupted with `split="test"` here, and no object path is
built with `allow_test_split=True`.**
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_conditions as conditions  # noqa: E402
import stage2b_corruption as corr  # noqa: E402
import stage2b_gcs as gcs  # noqa: E402
import stage2b_partition as partition  # noqa: E402
import stage2b_stats as stats  # noqa: E402

_N_SYNTHETIC = 400
_SUBSET_SIZE = 40
_PREFIX_SIZE = 12


def _partition_and_images(seed=3):
    """A synthetic corpus, its labels, and the partition over them.

    `allow_non_official_size=True` is the deliberate synthetic-data
    escape hatch the partition documents; the contract under test is
    about index PAIRING, which does not depend on the corpus's size."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, _N_SYNTHETIC)
    images = rng.random((_N_SYNTHETIC, 784))
    return partition.Stage2BTrainingPartition(labels, allow_non_official_size=True), images


def _nested_subsets(part):
    return part.nested_development_subsets(
        size=_SUBSET_SIZE, prefix_size=_PREFIX_SIZE,
        seed=partition.LADDER_SUBSET_SEED,
        stratified=partition.LADDER_SUBSET_STRATIFIED)


# ---- partition -> corruption: the index pairing ----

def test_nested_subset_indices_pair_elementwise_with_corrupt_corpus():
    """`nested_development_subsets` returns official indices in DRAW
    order -- the nesting prefix is load-bearing, so they are deliberately
    not sorted -- and `corrupt_corpus` pairs `images[i]` with
    `indices[i]` elementwise. Gathering the images with the same array
    that is passed as `indices` is therefore the whole contract, and this
    is the check that it holds end to end rather than in each module's
    own hand-built fixtures.
    """
    part, images = _partition_and_images()
    subset = _nested_subsets(part)
    idx = subset.stage1_indices

    # Precondition, asserted rather than assumed: if the draw ever came
    # back ascending, sorting would be a no-op and the test below it
    # would pass while demonstrating nothing.
    assert not np.array_equal(idx, np.sort(idx))

    x_t, x_t_clip = corr.corrupt_corpus(images[idx], "train", idx)
    assert x_t.shape == (_PREFIX_SIZE, 784)
    for row, official in enumerate(idx):
        expected_t, expected_c = corr.corrupt_image(images[official], "train", int(official))
        np.testing.assert_array_equal(x_t[row], expected_t)
        np.testing.assert_array_equal(x_t_clip[row], expected_c)


def test_sorting_one_side_only_silently_mispairs_images_and_noise():
    """The trap the contract above guards against, made explicit.

    A driver that sorts the images it gathers but passes the indices in
    draw order (or the reverse) attaches every image to another image's
    noise realization. Nothing raises: the corpus is the right shape, it
    is exactly reproducible, each realization is still the locked one for
    the index it was drawn from, and -- because the noise itself is drawn
    correctly -- the censoring diagnostics stay in their expected range.
    Only the MSE against `x_0` is wrong.
    """
    part, images = _partition_and_images()
    idx = _nested_subsets(part).stage1_indices
    sorted_idx = np.sort(idx)
    assert not np.array_equal(idx, sorted_idx)

    correct, correct_clip = corr.corrupt_corpus(images[idx], "train", idx)
    # the bug: images gathered in sorted order, indices still in draw order
    mispaired, mispaired_clip = corr.corrupt_corpus(images[sorted_idx], "train", idx)

    assert not np.allclose(mispaired, correct)
    assert not np.allclose(mispaired_clip, correct_clip)

    # exactly the rows whose gathered image changed differ, and no others
    expected_rows = np.flatnonzero(sorted_idx != idx)
    observed_rows = np.flatnonzero(
        [not np.array_equal(mispaired[i], correct[i]) for i in range(idx.size)])
    assert expected_rows.size > 0
    np.testing.assert_array_equal(observed_rows, expected_rows)

    # and the reason it survives review: the diagnostics do not move
    rates_correct = corr.empirical_clip_rates(correct)
    rates_mispaired = corr.empirical_clip_rates(mispaired)
    assert abs(rates_correct["total"] - rates_mispaired["total"]) < 0.03


def test_corrupt_corpus_accepts_the_nested_prefix_and_its_superset_alike():
    """Stage 1's subset is a read-only view of stage 2's. Both are valid
    `indices` arguments, and an image common to both gets the same
    realization at both ladder stages -- "one realization per image,
    reused identically" across the ladder, not only within one call."""
    part, images = _partition_and_images()
    subset = _nested_subsets(part)
    stage1, stage2 = subset.stage1_indices, subset.stage2_indices

    x1, _ = corr.corrupt_corpus(images[stage1], "train", stage1)
    x2, _ = corr.corrupt_corpus(images[stage2], "train", stage2)
    np.testing.assert_array_equal(x1, x2[:_PREFIX_SIZE])


# ---- statistics <-> object paths: one condition, two spellings ----

def test_every_stats_condition_has_exactly_one_path_segment_and_back():
    """The bijection. `stage2b_stats` keys results by `"T"`;
    `stage2b_gcs`'s paths carry `"evolved_T"` (DESIGN.md's own spelling
    of the primary contrast). `stage2b_conditions` is the only place that
    says those are the same condition."""
    stats_keys = (stats.PRE_EVOLUTION, *stats.EVOLVED_GRAPHS)
    assert set(conditions.CONDITION_PATH_SEGMENT) == set(stats_keys)
    assert conditions.ALL_CONDITIONS == stats_keys

    segments = list(conditions.CONDITION_PATH_SEGMENT.values())
    assert len(set(segments)) == len(segments) == len(stats_keys)
    assert set(conditions.PATH_SEGMENT_CONDITION) == set(segments)

    for key in stats_keys:
        assert conditions.condition_for_path_segment(conditions.path_segment(key)) == key
    for segment in segments:
        assert conditions.path_segment(conditions.condition_for_path_segment(segment)) == segment


def test_stats_takes_its_condition_vocabulary_from_the_shared_module():
    """One source of truth, not two lists that agree today."""
    assert stats.EVOLVED_GRAPHS is conditions.EVOLVED_GRAPHS
    assert stats.CONTROL_GRAPHS is conditions.CONTROL_GRAPHS
    assert stats.PRE_EVOLUTION is conditions.PRE_EVOLUTION
    assert stats.PRIMARY_GRAPH is conditions.PRIMARY_GRAPH


def test_the_two_path_segments_design_md_names_literally():
    """DESIGN.md writes `d_i = MSE_i(evolved_T) - MSE_i(pre_evolution)`.
    Those two segments are transcription; the other three follow the same
    `evolved_{graph}` rule as a disclosed extension (see
    `stage2b_conditions`'s module docstring)."""
    assert conditions.path_segment("T") == "evolved_T"
    assert conditions.path_segment("pre_evolution") == "pre_evolution"
    assert conditions.PRIMARY_PATH_SEGMENT == "evolved_T"
    for graph in conditions.CONTROL_GRAPHS:
        assert conditions.path_segment(graph) == f"evolved_{graph}"


def test_every_path_segment_survives_the_transport_layer_shape_check():
    """`stage2b_gcs` validates a condition token by shape, not against a
    vocabulary -- correct, and not weakened here. What has to be checked
    is the other direction: that this vocabulary passes that shape check
    and renders the expected path. Confirmed by running the real
    `object_path` -- which is where the token check is enforced -- not by
    restating its pattern here."""
    for segment in conditions.CONDITION_PATH_SEGMENT.values():
        name = gcs.object_path(split="train", stage=2, condition=segment,
                               kind="features", ext="npz")
        assert name == f"{gcs.TRAIN_ROOT}/stage2/{segment}/features.npz"
    primary = gcs.object_path(split="train", stage=2,
                              condition=conditions.PRIMARY_PATH_SEGMENT,
                              kind="features", ext="npz")
    assert primary == "stage2b/train/stage2/evolved_T/features.npz"


def test_the_reserved_common_segment_is_not_a_condition():
    """`common` is the object-path scheme's segment for artifacts that
    belong to no condition. It must not collide with one."""
    assert gcs.COMMON_CONDITION not in conditions.PATH_SEGMENT_CONDITION
    assert gcs.COMMON_CONDITION not in conditions.CONDITION_PATH_SEGMENT


def test_unknown_conditions_and_segments_raise_rather_than_pass_through():
    """An unrecognized key silently becoming its own path segment is how
    a run writes half its artifacts under `.../T/` and half under
    `.../evolved_T/`, with every upload succeeding."""
    with pytest.raises(ValueError, match="unknown Stage 2B condition"):
        conditions.path_segment("evolved_T")      # the segment, not the key
    with pytest.raises(ValueError, match="unknown Stage 2B path segment"):
        conditions.condition_for_path_segment("T")   # the key, not the segment
    with pytest.raises(ValueError):
        conditions.path_segment("curr-random")
    with pytest.raises(ValueError):
        conditions.condition_for_path_segment(None)


def test_the_condition_maps_are_read_only():
    """A mutable module-level mapping is a shared source of truth only
    until something mutates it."""
    with pytest.raises(TypeError):
        conditions.CONDITION_PATH_SEGMENT["T"] = "T"
    with pytest.raises(TypeError):
        conditions.PATH_SEGMENT_CONDITION["evolved_T"] = "lattice"
