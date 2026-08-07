"""Stage 2B's locked fit/validation partition of the official KMNIST
TRAINING split, and the feasibility-ladder development subsets drawn from
it -- DESIGN.md's "Training, literal" ordering requirement:

    StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42),
    created BEFORE feasibility subsets are drawn (the 5,000-image
    development subset comes only from the remaining 90%); at stage 3,
    "full training" = 54,000 fit + 6,000 locked validation.

## Why this is a module and not a run-script convention

The ordering is a correctness requirement, not discipline. Drawing a
1,000- or 5,000-image feasibility subset first and partitioning THAT
would produce a validation set contaminated by the ladder's own subset
structure, and would give feasibility stages a different validation
corpus from stage 3 -- silently, since every downstream number would
still compute.

So the ordering is encoded in the API rather than documented next to it:

- `Stage2BTrainingPartition.__init__` performs the split immediately, on
  construction, over the WHOLE official training corpus. There is no
  separate `split()` call a caller could forget, reorder, or run twice
  with different arguments.
- It refuses a corpus that is not the official 60,000 images unless the
  caller passes `allow_non_official_size=True` deliberately -- so
  constructing the partition on an already-drawn subset raises rather
  than quietly partitioning the wrong thing. Synthetic tests opt in
  loudly; the production path cannot drift into it.
- `development_subset` and `nested_development_subsets` are METHODS on
  that object and draw only from `fit_indices`. A subset that has not
  been preceded by the split is unrepresentable, because there is nothing
  to call the method on.

The validation partition is identical at every ladder stage: shrinking
the fit side to 1,000 or 5,000 images for feasibility work leaves the
6,000 validation images untouched. That is what "locked validation"
means here.

## The ladder's two subsets are one draw

The same pattern applies one rung further down. Stage 1's 1,000 images
are NESTED inside stage 2's 5,000, and the nesting is produced by a
single draw rather than by two independent calls that agree:
`nested_development_subsets` draws the 5,000 once and returns stage 1's
1,000 as a read-only VIEW of its first 1,000 entries. Two separate
stratified draws would have to be cross-checked against each other for
consistency; one draw makes the nesting true by construction, and there
is no arrangement of arguments that yields a stage 1 subset outside the
stage 2 one.

Because the prefix is load-bearing, that method returns its indices in
DRAW order, not sorted -- see `nested_development_subsets` for the
ordering rule and the stratification it guarantees at every prefix
length. `development_subset` is unchanged and still returns sorted
indices.

## Index semantics, load-bearing

Every index this module returns is a position within the OFFICIAL
training split (0..59,999) -- the same index space
`stage2b_corruption.corrupt_corpus` requires, so a subset's indices can
be passed straight to it and each image keeps the one corruption
realization it is entitled to. They are NOT positions within a subset.

## Test-use scope

This module partitions the official KMNIST TRAINING split and nothing
else. It has no test-split code path, takes no split label, and needs no
test-side data: the 10% validation partition here is carved out of
training images and is not the official test corpus. DESIGN.md locks
that no Stage 2B test-side result is accessed during feasibility stages
1-3; nothing here can reach one.

dtype: indices are int64 throughout; no image data passes through this
module.
"""
from typing import NamedTuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ---- Locked constants (DESIGN.md, "Training, literal" / "Feasibility ladder") ----
N_OFFICIAL_TRAIN = 60_000      # the official KMNIST training split
VALIDATION_TEST_SIZE = 0.10    # StratifiedShuffleSplit(test_size=...)
PARTITION_SEED = 42            # StratifiedShuffleSplit(random_state=...)
N_PARTITION_SPLITS = 1         # StratifiedShuffleSplit(n_splits=...)
N_FIT = 54_000                 # "full training" fit side at ladder stage 3
N_VALIDATION = 6_000           # the locked validation partition
STAGE1_SUBSET_SIZE = 1_000     # feasibility ladder stage 1
STAGE2_SUBSET_SIZE = 5_000     # feasibility ladder stage 2 (development subset)

# ---- The ladder draw rule (a project decision, not DESIGN.md) ----
# DESIGN.md gives the ladder's sizes and where the subset comes from, and
# says nothing about how it is drawn. The decision -- stratified, one
# consistent seed, stage 1 nested inside stage 2 -- is recorded on
# `nested_development_subsets`. These constants exist so that the call
# site still states the rule explicitly (nothing here is defaulted) while
# having one obvious right answer to state.
LADDER_SUBSET_SEED = 42          # the design's one seed, shared with the corruption RNG
LADDER_SUBSET_STRATIFIED = True  # both levels stratified, not just the outer draw

# The interleave's within-class shuffle draws from a stream derived from
# the same seed but distinct from `default_rng(seed)`, so ordering never
# shares a stream with the unstratified draw at the same seed.
_INTERLEAVE_STREAM = 1


class NestedDevelopmentSubsets(NamedTuple):
    """The feasibility ladder's stage 1 and stage 2 subsets, from one draw.

    `stage1_indices` is a read-only VIEW of `stage2_indices[:n_stage1]` --
    `stage1_indices.base is stage2_indices`. The nesting is therefore a
    property of the objects rather than something a caller has to check:
    there is no second array that could disagree with the first.

    Both arrays are official training indices (the index space
    `stage2b_corruption.corrupt_corpus` requires) in DRAW order, not
    sorted. `corrupt_corpus` pairs images with indices elementwise and
    does not care about order; the prefix relation does.

    `classes` / `stage1_class_counts` / `stage2_class_counts` record what
    the draw actually achieved per class, so a run log carries the
    realized stratification rather than the intent (CLAUDE.md principle 7).
    """
    stage1_indices: np.ndarray
    stage2_indices: np.ndarray
    seed: int
    stratified: bool
    classes: tuple
    stage1_class_counts: tuple
    stage2_class_counts: tuple

    def summary(self):
        """Provenance for the run log."""
        return {
            "n_stage1": int(self.stage1_indices.size),
            "n_stage2": int(self.stage2_indices.size),
            "seed": int(self.seed),
            "stratified": bool(self.stratified),
            "nesting": "stage1_indices is a read-only view of stage2_indices[:n_stage1]",
            "order": "class-proportional interleave, draw order (not sorted)",
            "index_space": "official KMNIST training split (0-based)",
            "classes": self.classes,
            "stage1_class_counts": self.stage1_class_counts,
            "stage2_class_counts": self.stage2_class_counts,
        }


class Stage2BTrainingPartition:
    """The fit/validation partition of the official training split, made
    at construction time.

    Parameters
    ----------
    labels : (n,) integer class labels for the official training split,
        in official index order -- `labels[i]` is the label of training
        image `i`.
    allow_non_official_size : bool
        Guard, default False. The partition is specified over the whole
        60,000-image official training split; a corpus of any other size
        is either a subset (the ordering error this class exists to make
        impossible) or synthetic test data. Passing True says the caller
        knows which.

    Attributes
    ----------
    fit_indices : (n_fit,) sorted official training indices -- the 90%
        the model may be fitted on, and the only pool
        `development_subset` draws from.
    validation_indices : (n_val,) sorted official training indices -- the
        locked 10%, identical at every ladder stage. Both arrays are
        returned read-only; a caller that needs to modify one must copy
        it, so the partition cannot be edited in place after the fact.
    """

    def __init__(self, labels, allow_non_official_size=False):
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
        n = int(labels.shape[0])
        if n != N_OFFICIAL_TRAIN and not allow_non_official_size:
            raise ValueError(
                f"refusing to partition {n} images: DESIGN.md's validation partition is "
                f"defined over the whole {N_OFFICIAL_TRAIN}-image official training split "
                f"and must be created BEFORE any feasibility subset is drawn. A corpus of "
                f"another size is most likely an already-drawn subset, which would give "
                f"feasibility stages a different validation corpus from stage 3. Pass "
                f"allow_non_official_size=True only for synthetic data, deliberately.")

        splitter = StratifiedShuffleSplit(
            n_splits=N_PARTITION_SPLITS, test_size=VALIDATION_TEST_SIZE,
            random_state=PARTITION_SEED)
        # StratifiedShuffleSplit needs an X of matching length only for its shape;
        # the stratification is entirely on `labels`.
        fit_idx, val_idx = next(splitter.split(np.zeros((n, 1)), labels))

        self._labels = labels
        self._n = n
        self._fit_indices = self._frozen(np.sort(fit_idx))
        self._validation_indices = self._frozen(np.sort(val_idx))
        self._verify_partition()

    # ---- construction-time invariants ----

    @staticmethod
    def _frozen(idx):
        out = np.asarray(idx, dtype=np.int64)
        out.flags.writeable = False
        return out

    def _verify_partition(self):
        """Disjointness and coverage, checked rather than assumed -- a
        partition that silently dropped or duplicated an image would leave
        every downstream number computable and wrong."""
        fit, val = self._fit_indices, self._validation_indices
        if np.intersect1d(fit, val).size:
            raise AssertionError("fit and validation partitions overlap")
        union = np.union1d(fit, val)
        if union.size != self._n or union[0] != 0 or union[-1] != self._n - 1:
            raise AssertionError(
                f"partition does not cover 0..{self._n - 1} exactly "
                f"({union.size} distinct indices)")
        if self._n == N_OFFICIAL_TRAIN and (fit.size, val.size) != (N_FIT, N_VALIDATION):
            raise AssertionError(
                f"official-size partition must be {N_FIT} fit / {N_VALIDATION} "
                f"validation, got {fit.size} / {val.size}")

    # ---- the partition ----

    @property
    def fit_indices(self):
        return self._fit_indices

    @property
    def validation_indices(self):
        return self._validation_indices

    @property
    def n_images(self):
        return self._n

    def fit_labels(self):
        return self._labels[self._fit_indices]

    def validation_labels(self):
        return self._labels[self._validation_indices]

    # ---- feasibility-ladder subsets, drawable only from the fit side ----

    def development_subset(self, size, *, seed, stratified):
        """A feasibility-ladder subset of `size` official training indices,
        drawn only from `fit_indices`.

        DESIGN.md locks where the subset comes from (the 90% fit side,
        after the split) and the sizes the ladder uses (1,000 at stage 1,
        5,000 at stage 2). It does NOT lock how the draw is made.
        `seed` and `stratified` are therefore required keyword arguments
        with no defaults: the choice gets recorded at the call site rather
        than silently locked here, and remains a disclosed decision for
        the ladder's run scripts to make and log.

        `stratified=True` preserves the fit side's class proportions via
        `StratifiedShuffleSplit(n_splits=1, train_size=size,
        random_state=seed)`; `stratified=False` draws uniformly without
        replacement from `fit_indices` using
        `numpy.random.default_rng(seed)`. Both are reproducible functions
        of `seed`; neither can reach `validation_indices`.

        Returns sorted official training indices."""
        size = int(size)
        if not 0 < size <= self._fit_indices.size:
            raise ValueError(
                f"development subset size must be in 1..{self._fit_indices.size} "
                f"(the fit side); got {size}")
        if not isinstance(stratified, (bool, np.bool_)):
            raise TypeError("stratified must be an explicit bool -- DESIGN.md does not "
                            "lock the draw rule, so it is stated, not defaulted")

        fit = self._fit_indices
        if stratified:
            if size == fit.size:
                chosen = np.arange(fit.size)
            else:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, train_size=size, random_state=int(seed))
                chosen, _rest = next(
                    splitter.split(np.zeros((fit.size, 1)), self.fit_labels()))
        else:
            rng = np.random.default_rng(int(seed))
            chosen = rng.permutation(fit.size)[:size]
        subset = np.sort(fit[np.asarray(chosen, dtype=np.int64)])
        assert not np.intersect1d(subset, self._validation_indices).size
        return subset

    def nested_development_subsets(self, *, size, prefix_size, seed, stratified):
        """The ladder's stage 2 subset and, nested inside it, stage 1's --
        one draw, returned as a `NestedDevelopmentSubsets`.

        ## The decision this implements

        DESIGN.md fixes the ladder's sizes (1,000 at stage 1, 5,000 at
        stage 2) and the pool they come from (the fit side, after the
        split), and is silent on how they are drawn or how they relate.
        Decided: **stratified, seed 42, with the 1,000 nested inside the
        5,000, achieved by a single draw rather than two independent
        calls.** Seed 42 is the design's existing seed -- the corruption
        RNG and the fold assignment already run on it -- so the ladder
        adds no second magic number. `LADDER_SUBSET_SEED` and
        `LADDER_SUBSET_STRATIFIED` name those values; they are still
        passed explicitly, like `development_subset`'s arguments, so the
        choice is recorded at the call site.

        The reason for one draw is the same one that puts the validation
        split at construction time: two stratified draws would agree only
        as long as someone kept checking that they did, whereas a prefix
        of one draw cannot come apart. It is the CNN validation
        partition's rule -- never silently re-cross an earlier partition
        boundary -- one rung further down the ladder. It also keeps a
        diagnostic that is worth having on purpose: when something breaks
        between stage 1 and stage 2, the 1,000 already-vetted images and
        the 4,000 new ones are distinguishable by position.

        ## What the prefix guarantees

        A naive prefix of a stratified draw is NOT itself stratified: if
        the drawn indices arrive grouped by class, or in any order that
        tracks class, "the first 1,000" can be severely skewed while
        every nesting check still passes. So the draw is reordered by a
        class-proportional interleave: the drawn indices are shuffled
        within class (from `seed`), each gets a within-class rank `j` out
        of its class's `n_c` drawn images, and all are ordered by the key
        `(j + 0.5) / n_c`, ties broken by class label.

        At EVERY prefix length `p`, this puts class `c`'s count within one
        image of its exact quota `p * n_c / N`, where `N` is the drawn
        size -- so the prefix reproduces the proportions of the draw it
        came from, whatever those are, rather than forcing equal counts on
        an imbalanced corpus.

        On the real corpus the guarantee is exact, not approximate: the
        KMNIST training split holds 6,000 images per class, so the fit
        side is 5,400 per class, a stratified 5,000 is 500 per class, all
        `n_c` are equal, the keys tie, and the interleave degenerates to
        plain round-robin -- the 1,000-image prefix is exactly 100 images
        per class. The within-one-image statement is what remains when the
        drawn class counts are unequal, and it is measured over every
        prefix length in the tests rather than assumed. It is not a proven
        bound for arbitrary distributions: a deliberately pathological
        synthetic corpus (classes from 40% down to 0.2%) measures a worst
        case of 1.41 images. Nothing in this design approaches that.

        The within-class shuffle matters as much as the interleave. Taking
        each class's drawn images in ascending official-index order would
        make stage 1 systematically the corpus's earliest images -- an
        ordering the corpus is free to have structure in -- so the order
        is randomized from a stream derived from `seed`.

        Parameters
        ----------
        size, prefix_size : stage 2's and stage 1's sizes;
            `0 < prefix_size <= size <= fit_indices.size`.
        seed, stratified : the draw rule, as on `development_subset`.
            Required keyword arguments with no defaults.

        Neither returned array can reach `validation_indices`: the draw is
        `development_subset`, unchanged, and this method only reorders it.
        """
        size = int(size)
        prefix_size = int(prefix_size)
        if not 0 < prefix_size <= size:
            raise ValueError(
                f"prefix (stage 1) size must be in 1..{size} so that it nests inside "
                f"the stage 2 subset; got {prefix_size}")

        # The draw itself is `development_subset`, called once -- not
        # reimplemented here (CLAUDE.md principle 16). It validates `size`
        # against the fit side, applies the stated draw rule, and checks
        # the result against the validation partition. This method adds
        # ordering and nothing else.
        drawn = self.development_subset(size, seed=seed, stratified=stratified)
        ordered = self._interleave_by_class(drawn, seed)

        if not np.array_equal(np.sort(ordered), drawn):
            raise AssertionError(
                "interleaving changed the drawn subset's membership -- the order "
                "must be a permutation of the draw, nothing else")

        stage2 = self._frozen(ordered)
        stage1 = stage2[:prefix_size]   # a read-only VIEW: stage1.base is stage2
        assert not np.intersect1d(stage2, self._validation_indices).size

        classes = np.unique(self._labels)
        return NestedDevelopmentSubsets(
            stage1_indices=stage1,
            stage2_indices=stage2,
            seed=int(seed),
            stratified=bool(stratified),
            classes=tuple(classes.tolist()),
            stage1_class_counts=self._counts_by_class(stage1, classes),
            stage2_class_counts=self._counts_by_class(stage2, classes),
        )

    def _interleave_by_class(self, subset, seed):
        """Orders `subset` so that every prefix tracks the subset's own
        class proportions -- see `nested_development_subsets`."""
        subset = np.asarray(subset, dtype=np.int64)
        n = subset.size
        rng = np.random.default_rng([int(seed), _INTERLEAVE_STREAM])
        shuffled = subset[rng.permutation(n)]

        classes, class_pos = np.unique(self._labels[shuffled], return_inverse=True)
        counts = np.bincount(class_pos, minlength=classes.size)
        rank = np.empty(n, dtype=np.int64)
        for c in range(classes.size):
            in_class = class_pos == c
            rank[in_class] = np.arange(int(counts[c]))

        # Ordering by (j + 0.5) / n_c interleaves the classes in
        # proportion to their counts; `np.unique` returns classes sorted,
        # so equal keys (the equal-counts case) break by class label.
        key = (rank + 0.5) / counts[class_pos]
        return shuffled[np.lexsort((class_pos, key))]

    def _counts_by_class(self, indices, classes):
        labels = self._labels[indices]
        return tuple(int(np.count_nonzero(labels == c)) for c in classes)

    # ---- provenance ----

    def summary(self):
        """The partition's provenance as a dict, for the run log (CLAUDE.md
        principle 7: a partition with no explicit record of how it was made
        is the same class of hazard as an array with no record of its
        normalization state)."""
        return {
            "n_images": self._n,
            "n_fit": int(self._fit_indices.size),
            "n_validation": int(self._validation_indices.size),
            "splitter": "StratifiedShuffleSplit",
            "n_splits": N_PARTITION_SPLITS,
            "test_size": VALIDATION_TEST_SIZE,
            "random_state": PARTITION_SEED,
            "index_space": "official KMNIST training split (0-based)",
            "is_official_size": bool(self._n == N_OFFICIAL_TRAIN),
        }


# ---- joining two artifacts by official image index ----

def index_join(source_indices, target_indices, *, source_name="source",
               target_name="target"):
    """Rows of `target` holding `source`'s images, in `source`'s order.

    Returns `(rows, report)` such that `target_array[rows]` lines up with
    `source_array`, image for image.

    AUDIT_PROTOCOL.md requires that all cross-artifact comparison happen
    **by official KMNIST image index, never by positional prefix**. This is
    that join, in one tested place, because writing it fresh at each call
    site is CLAUDE.md principle 16's exact failure shape: two artifacts
    built from differently-ordered index lists align row-for-row, agree on
    shape, and compare entirely wrong numbers with no error raised
    anywhere.

    The report records `alignment_is_a_prefix` and `n_rows_moved` rather
    than deciding what they mean. A prefix alignment is legitimate for some
    pairs and evidence of a degenerated join for others, so the caller
    holds that judgement -- `compare_stage3_regeneration.align` refuses one,
    and a caller joining a subset drawn in ascending order should expect
    one."""
    source = np.asarray(source_indices)
    target = np.asarray(target_indices)
    if target.size != np.unique(target).size:
        raise ValueError(f"{target_name} indices contain duplicates, so a join "
                         f"against them is ambiguous")
    if source.size != np.unique(source).size:
        raise ValueError(f"{source_name} indices contain duplicates")

    position = {int(v): i for i, v in enumerate(target)}
    missing = [int(v) for v in source if int(v) not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} of {source_name}'s images are absent from "
            f"{target_name} (first few: {missing[:5]}). The two artifacts do not "
            f"cover the populations this comparison assumes.")
    rows = np.array([position[int(v)] for v in source], dtype=np.int64)
    if np.unique(rows).size != rows.size:
        raise ValueError(f"the alignment maps two {source_name} images to one "
                         f"{target_name} row")
    return rows, {
        "n_overlap": int(rows.size),
        "n_target_total": int(target.size),
        "alignment_is_a_prefix": bool(np.array_equal(rows, np.arange(rows.size))),
        "n_rows_moved": int(np.count_nonzero(rows != np.arange(rows.size))),
        "first_five_source_indices": [int(v) for v in source[:5]],
        "their_rows_in_the_target": [int(v) for v in rows[:5]],
    }
