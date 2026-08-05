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
- `development_subset` is a METHOD on that object and draws only from
  `fit_indices`. A subset that has not been preceded by the split is
  unrepresentable, because there is nothing to call the method on.

The validation partition is identical at every ladder stage: shrinking
the fit side to 1,000 or 5,000 images for feasibility work leaves the
6,000 validation images untouched. That is what "locked validation"
means here.

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
