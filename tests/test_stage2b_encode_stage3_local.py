"""Tier 1: `encode_stage3_local.py`'s chunking, population roles and tail
reporting, on small synthetic data.

The full run is nine minutes on 60,000 real images and needs the KMNIST
dataset, so the properties that matter are pulled out and pinned here
instead. Three of them:

- **Chunking must not move the numbers.** CLAUDE.md principle 19: a
  chunked draw is not automatically the same stream as the unchunked one,
  and both produce entirely plausible output either way. The regeneration
  changes the chunk count (54,000/5,400 = 10; 60,000/6,000 = 10, but a
  `--chunk` flag exists and someone will use it), so this is exactly the
  situation the principle names.
- **The population assertion must be able to fail.** Freeze 2's roles are
  the thing that scoped Phase A wrongly the first time.
- **The tail interval must be the exact binomial one**, because the
  counts are small and near zero, where a normal approximation is wrong
  in a direction that flatters agreement.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE2B = os.path.join(REPO_ROOT, "experiments", "stage2b_denoising")
sys.path.insert(0, STAGE2B)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments",
                                "stage2a_dynamics_classification"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments",
                                "stage1d_topology_specificity"))

enc = pytest.importorskip("encode_stage3_local")


# ------------------------------------------------------- chunk invariance

def _tiny_corpus(n=7, rng_seed=0):
    """Synthetic 28x28 uint8 images. Real shape, no dataset required."""
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 7, 10])
def test_chunking_does_not_change_a_single_bit(chunk):
    """The pin. `encode_with_final_delta_batch` passes a CONSTANT seed
    into every per-image job and the encoder builds a fresh
    `default_rng(seed)` per call, so an image's perturbation depends on
    the image and the seed and nothing else -- not on which chunk it
    landed in, not on how many workers ran.

    That is a property of the current implementation, not a guarantee of
    the API, which is why it is asserted rather than commented. The sweep
    includes a chunk larger than the corpus, a chunk of one, and two
    non-divisors of 7."""
    images = _tiny_corpus()
    indices = np.arange(images.shape[0])
    active = np.arange(20)
    reference, ref_deltas, _ = enc.encode_indices(
        images, indices, active, steps=3, n_workers=1, chunk=len(indices),
        progress=False)
    thetas, deltas, _ = enc.encode_indices(
        images, indices, active, steps=3, n_workers=1, chunk=chunk,
        progress=False)
    assert thetas.tobytes() == reference.tobytes(), (
        f"chunk={chunk} changed the encoded output. A chunked and an unchunked "
        f"run must be the same stream (CLAUDE.md principle 19).")
    assert deltas.tobytes() == ref_deltas.tobytes(), (
        f"chunk={chunk} changed the final-Deltas")


def test_the_chunk_invariance_check_can_actually_fail():
    """The guard on the guard: if the encoding were position-dependent,
    the sweep above would have to notice. Encoding a permuted corpus and
    un-permuting it must reproduce the original -- and it does only
    because the seed is per-image. A positional seed would break this,
    which is the mutation the sweep is watching for."""
    images = _tiny_corpus()
    indices = np.arange(images.shape[0])
    active = np.arange(20)
    straight, _, _ = enc.encode_indices(images, indices, active, steps=3,
                                        n_workers=1, chunk=99, progress=False)
    order = np.array([4, 0, 6, 2, 5, 1, 3])
    shuffled, _, _ = enc.encode_indices(images, indices[order], active, steps=3,
                                        n_workers=1, chunk=99, progress=False)
    restored = np.empty_like(shuffled)
    restored[order] = shuffled
    assert restored.tobytes() == straight.tobytes(), (
        "encoding depends on an image's POSITION, not just its identity -- "
        "which would make the chunk sweep above meaningless")
    # ...and a genuinely different image really does encode differently,
    # so the equality above is not trivially satisfied by a constant.
    assert straight[0].tobytes() != straight[1].tobytes()


def test_corruption_is_keyed_on_official_indices_not_positions():
    """`corrupt_corpus` takes the ORIGINAL dataset indices. Passing a
    positional counter instead would give every subset a different
    corruption from the full run, silently."""
    # Rows 10-13 are byte-identical copies of rows 0-3, so the ONLY thing
    # separating the two encodings below is the official index the
    # corruption is keyed on.
    images = _tiny_corpus(n=14)
    images[10:14] = images[0:4]
    active = np.arange(20)
    a, _, _ = enc.encode_indices(images, np.array([0, 1, 2, 3]), active, steps=3,
                                 n_workers=1, chunk=99, progress=False)
    b, _, _ = enc.encode_indices(images, np.array([10, 11, 12, 13]), active,
                                 steps=3, n_workers=1, chunk=99, progress=False)
    assert np.array_equal(images[0:4], images[10:14]), "the fixture is wrong"
    assert a.tobytes() != b.tobytes(), (
        "identical images under different official indices encoded identically, "
        "so corruption is not keyed on the index it is documented to use")


# ------------------------------------------------------------ tail report

def test_clopper_pearson_brackets_the_point_estimate():
    lo, hi = enc.clopper_pearson(79, 54_000)
    rate = 79 / 54_000
    assert lo < rate < hi
    # Sanity against the published interval for this count.
    assert 0.0011 < lo < 0.0013 and 0.0017 < hi < 0.0019, (lo, hi)


def test_clopper_pearson_handles_the_zero_count_without_a_negative_bound():
    """The case a normal approximation gets wrong: 0/6,000 would give a
    symmetric interval straddling zero."""
    lo, hi = enc.clopper_pearson(0, 6_000)
    assert lo == 0.0
    assert 0.0 < hi < 0.001


def test_a_rare_rate_is_much_less_precise_at_6000_than_at_54000():
    """Why the protocol demands uncertainty rather than bare percentages:
    the same rate measured on 6,000 images has a visibly wider interval,
    so two proportions that look different may not be."""
    n_small = 6_000
    k_small = round(79 / 54_000 * n_small)
    lo_s, hi_s = enc.clopper_pearson(k_small, n_small)
    lo_l, hi_l = enc.clopper_pearson(79, 54_000)
    assert (hi_s - lo_s) > 2 * (hi_l - lo_l), (
        f"width at n=6,000 is {hi_s - lo_s:.2e}, at n=54,000 "
        f"{hi_l - lo_l:.2e} -- the point of reporting an interval")


def test_the_tail_report_splits_by_role_and_never_averages_them():
    deltas = np.zeros(100)
    deltas[:3] = [1e-14, 1e-11, 1e-9]          # three nonzero, all in fit
    in_fit = np.zeros(100, dtype=bool)
    in_fit[:90] = True
    report = enc.tail_report(deltas, in_fit)

    assert report["fit"]["n"] == 90 and report["fit"]["n_nonzero"] == 3
    assert report["validation"]["n"] == 10 and report["validation"]["n_nonzero"] == 0
    assert report["all"]["n"] == 100 and report["all"]["n_nonzero"] == 3
    # The thresholded counts follow the design table's ladder.
    assert report["fit"]["n_gt_1e_13"] == 2
    assert report["fit"]["n_gt_1e_12"] == 2
    assert report["fit"]["n_gt_1e_10"] == 1
    assert report["validation"]["max"] == 0.0
    # A zero-count role still gets a real interval rather than nan.
    assert report["validation"]["ci95_upper"] > 0.0


def test_the_tail_report_counts_only_strictly_positive_deltas():
    """`> 0.0`, not `>= 0.0`: exact float64 zero is the converged case and
    the whole point of the tail is what is NOT at zero."""
    report = enc.tail_report(np.array([0.0, 0.0, 5e-16]),
                             np.array([True, True, True]))
    assert report["all"]["n_nonzero"] == 1


# ------------------------------------------------------- population roles

def test_the_object_name_no_longer_claims_the_fit_side():
    """Two things at once: the name must not say "fit" for a 60,000-image
    array, and it must differ from the baseline object so a regeneration
    cannot overwrite the artifact it is compared against."""
    name = enc.object_name_for(1200)
    assert "encoded_train_s1200" in name
    assert "encoded_fit" not in name
    baseline = name.replace("encoded_train_s1200", "encoded_fit_s1200")
    assert name != baseline


def test_the_declared_population_is_the_official_training_split():
    assert enc.N_OFFICIAL_TRAIN == 60_000
    # 54,000 + 6,000, the roles AUDIT_PROTOCOL.md Freeze 2 fixes.
    assert enc.N_OFFICIAL_TRAIN - 54_000 == 6_000
