# Preliminary Investigation of Archived Multi-threshold notMNIST Artefacts (Unverified -- Not Yet a Finding)

**Epistemic status, stated plainly before anything else**: everything in
this document is derived from files whose provenance could not be
established. They are internally consistent (each block scores
sensibly alone; column-wise statistics match a plausible pipeline output)
but internal consistency is not the same as verified validity. Per
review feedback, this should not be read as an experimental finding until
a from-scratch rerun, built entirely from verified, current-pipeline
artifacts, either reproduces or fails to reproduce the result reported
below. That rerun is reported separately, following this document, and
is the only part of this investigation that should currently be trusted
as evidence.

## Why this check mattered

Capacity Experiment I concluded, from Fashion-MNIST alone, that naive
concatenation of multi-threshold topology scores does not detectably
improve accuracy beyond generic dimensionality effects -- but is
essentially harmless (all conditions within about 1 percentage point of
each other). Before treating that as a general property of the
combination approach, the same question was tested on a second dataset.
**The archived artefacts suggest the conclusion may change completely --
but this remains unconfirmed pending independent verification.** This is
exactly the kind of check this project's whole transfer-testing history
argues for.

## What was found (not yet completed independently)

Partially-built artifacts for a 60-dimensional notMNIST representation
(three thresholds: 0.85, 0.90, 0.95) were discovered already present on
disk -- hybrid-head training features for only 6 of 10 classes, no test-set
scoring yet done. Origin unclear (plausibly earlier work in this same long
session, not fully visible in the immediate conversation history) -- also
plausibly a partially-completed abandoned branch, or output from an
earlier, possibly buggy, implementation. **These four possibilities were
not distinguished before the artifacts were completed and evaluated in
the original pass**, which is the gap this document now flags rather than
papers over. Column-wise statistics were checked against a baseline file
of the same unverified provenance -- a self-consistency check, not
independent validation. The remaining 4 classes of hybrid-head training
data plus all test-set scoring were completed following what appeared to
be the same protocol as the existing 6 classes.

## Result: catastrophic collapse, not a null result

| Condition | Accuracy |
|---|---|
| 20D baseline (re-confirmed) | 80.40% |
| **60D combined (0.90+0.85+0.95)** | **36.00%** |
| 60D duplicated-20D control (3x copy) | 80.80% |
| 60 random-feature control | 85.00% |

This is not the small, controls-explain-it pattern found on Fashion-MNIST.
The real combined representation collapses to less than half the 20D
baseline's accuracy, while both controls perform normally. The confusion
matrix reveals the mechanism is not generic noise: **the fitted model
never predicts 6 of the 10 classes at all** -- every test image is
assigned to one of only 4 classes (G, H, I, J). This is a degenerate
multinomial decision boundary, not diffuse confusion.

## Ruling out the obvious explanations, in order

- **Convergence failure**: no warnings at max_iter=1000; identical result
  at max_iter=5000. Not a stopped-too-early optimization.
- **Regularization**: stronger regularization (C=0.1, C=0.01) gives
  essentially the same collapsed result (35.8-36.2%). Not fixable by
  standard L2 tuning.
- **Train/test misalignment or a baseline-file bug**: raw column
  statistics were checked directly against the baseline file and against
  each other (train vs. test) at multiple column positions -- all
  consistent, no evidence of a wrong-order or mismatched-topology bug.
- **Each block is individually broken**: tested directly -- each of the
  three threshold blocks, scored alone, gives reasonable standalone
  accuracy (69.6%, 73.4%, 77.8%). The data itself is fine in isolation.
- **Number of blocks (3 vs. 2)**: tested directly by extracting just two
  of the three blocks in every combination (0.90+0.95, matching Fashion-
  MNIST's exact experimental setup; 0.90+0.85; 0.85+0.95). **All three
  2-block combinations also collapse** (36.2-39.8%), ruling this out --
  the problem is not specific to combining three thresholds rather than
  two.
- **Simple multicollinearity-severity story**: checked by comparing
  cross-threshold correlation between the two datasets. Fashion-MNIST's
  0.90-vs-0.95 correlation is actually **higher** (0.963) than any of
  notMNIST's threshold-pair correlations (0.579-0.811) -- the opposite of
  what a naive "more correlation causes more collapse" story would
  predict. Whatever is happening is not simply explained by how
  correlated the combined blocks are with each other.

**None of the standard explanations account for this. The honest state of
knowledge is: this is a real, reproducible, dataset-dependent failure mode
whose mechanism is not yet identified.**

## What this would mean, if independently reproduced

If confirmed by a from-scratch rerun, the Fashion-MNIST-only conclusion
from Capacity Experiment I -- "concatenating multiple topology-derived
functionals is safe, if not obviously beneficial" -- would not generalize.
On notMNIST, the identical procedure would be actively destructive,
collapsing a working 80% classifier to 36% by making it forget half its
classes exist. Until independently reproduced, this remains a hypothesis
generated by unverified artifacts, not an established property of the
pipeline -- see the verification report following this document.

## Honest limitations

- The root mechanism is unidentified. Candidate directions for future
  investigation: the multinomial (10-class) softmax objective's behavior
  under near-degenerate feature geometry specific to notMNIST's class
  structure; possible near-linear-dependence in a subset of the 60
  (or 40) dimensions not captured by simple pairwise correlation; some
  interaction with notMNIST's own known class-confusability structure
  (E/F, I/J) that a correlated multi-threshold representation amplifies
  rather than resolves. None of these have been tested.
- This was diagnosed on one specific collapse; it has not yet been
  checked whether Kuzushiji-MNIST or MNIST show the same pattern, a
  milder version, or none at all.
- The origin of the partially-completed artifacts this check built upon
  is not fully accounted for -- they were verified for correctness before
  use (topology edge counts, test-set integrity, column-wise statistics)
  but their provenance in this session's history is not fully traced.

## Reproducing these results

`notmnist_60d_hybrid_p{1..5}.pkl`, `notmnist_60d_test_features.pkl`,
`notmnist_baselines_60d.pkl`, `notmnist_topologies_{085,095}.pkl`,
`notmnist_class_topologies_200.pkl` (0.90 threshold), plus the existing
frozen `notmnist_test_sub.npy` / `notmnist_test_labels.npy`.
