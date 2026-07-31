# Capacity Experiment II, Replicated: Graph Smoothness Across Four Datasets

## The question this answers

Does graph smoothness provide a robust, genuinely distinct readout across
datasets, and does it add information beyond the original 20D topology-
matching representation? Tested identically on Fashion-MNIST, notMNIST,
Kuzushiji-MNIST, and MNIST -- same five conditions, same diagnostics,
same frozen 500-image test convention throughout.

## A second data-integrity bug caught mid-experiment, using the same discipline just established

Before any smoothness result could be trusted, the MNIST 20D baseline
came back at exactly 10.0% -- chance level. This was not reported; it was
diagnosed immediately. Cause: a cached file (`hybrid_20feat_batch1.pkl`)
was already z-score normalized (raw column means near 0, stds near 1),
but was being treated as raw and normalized a second time, destroying the
signal. The calibration baseline file itself was independently verified
correct (a fresh rebuild matched it exactly). The fix was the same one
used for the notMNIST archived artifacts: rebuild the suspect component
from scratch from verified raw ingredients (`class_topologies_200.pkl` and
a fresh calibration pass), rather than patch around an unexplained
mismatch. The corrected MNIST 20D baseline (89.6%) is consistent with
this project's established MNIST results. This is flagged here explicitly
because it is the same lesson recurring, not a one-time fix: an
unverified assumption about a cached file's format silently produced a
catastrophic, obviously-wrong number, and it was caught only because a
result that should have been ~85-90% and came back at exactly chance
level was treated as suspicious rather than reported.

## Results across all four datasets

| Dataset | 20D | 10D smoothness | 30D combined | 30D duplicated | 30 random |
|---|---|---|---|---|---|
| Fashion-MNIST | 71.4% | 70.2% | 71.8% | 71.6% | 72.0% |
| notMNIST | 80.4% | 76.4% | 80.2% | 80.4% | 80.6% |
| Kuzushiji-MNIST | 52.4% | 46.2% | 52.4% | 52.6% | 44.2% |
| MNIST | 89.6% | 87.6% | 89.4% | 89.0% | 74.0% |

**Smoothness alone is consistently, substantially above chance on every
dataset tested, remaining within 1.2-6.2 percentage points of the 20D
baseline** (Fashion-MNIST: 1.2pp; notMNIST: 4.0pp; Kuzushiji-MNIST:
6.2pp; MNIST: 2.0pp) -- never once collapsing toward chance, on domains
as different as clothing, font glyphs, hiragana, and digits. This answers
the first of the three questions cleanly: **smoothness is independently
effective, and this is not a Fashion-MNIST-specific artifact.**

**Combined 30D essentially never beats the 20D baseline.** The four
datasets show four different relationships to the random-feature control
(comparable on Fashion-MNIST and notMNIST, clearly better on MNIST and
Kuzushiji-MNIST, where random projections are weak), but the comparison
that matters most for the actual research question -- 20D alone versus
20D+smoothness combined -- shows no meaningful movement in any of the four
cases.

## Paired McNemar tests (exact binomial on discordant pairs, same 500 test images per dataset)

| Dataset | 20D vs. smoothness | 20D vs. combined-30D | Combined-30D vs. random |
|---|---|---|---|
| Fashion-MNIST | p=0.307 (n.s.) | p=0.688 (n.s.) | p=1.000 (n.s.) |
| notMNIST | **p=0.0029** | p=1.000 (n.s.) | p=0.914 (n.s.) |
| Kuzushiji-MNIST | **p=0.00024** | p=1.000 (n.s.) | **p=0.0023** |
| MNIST | p=0.087 (borderline n.s.) | p=1.000 (n.s.) | **p<0.00001** |

**The single most consistent result across all four datasets: 20D vs.
combined-30D is never significant** -- but the correct reading of this is
about discordant counts, not just the p-value. Reporting the actual
off-diagonal counts, not just "no disagreement": Fashion-MNIST (both
correct=355, 20D-only=2, combined-only=4, both wrong=139, 6 discordant
total), notMNIST (both correct=399, 20D-only=3, combined-only=2, both
wrong=96, 5 discordant total), Kuzushiji-MNIST (both correct=260,
20D-only=2, combined-only=2, both wrong=236, 4 discordant total), MNIST
(both correct=446, 20D-only=2, combined-only=1, both wrong=51, 3
discordant total). In this specific case the discordant counts genuinely
are small (3-6 out of 500 per dataset) -- but this should be read off the
counts directly, not inferred from a p-value of 1.0, which only indicates
a balanced split between the two directions of disagreement and would be
equally consistent with a much larger, evenly-split discordant set.
Whether 20D beats smoothness alone varies by dataset (significant on notMNIST and Kuzushiji-MNIST, not on Fashion-MNIST
or MNIST), and whether the combined representation beats a random-feature
control depends entirely on how strong that control is on a given dataset
(weak random controls on MNIST and Kuzushiji-MNIST get clearly beaten;
strong ones on Fashion-MNIST and notMNIST do not) -- but concatenating
smoothness onto the original 20D representation itself never produces a
statistically detectable change from 20D alone, anywhere.

## Prediction agreement and effective rank

| Dataset | 20D/smoothness agreement | Effective rank of combined 30D (of 30 nominal dims) | Components for 95% variance |
|---|---|---|---|
| Fashion-MNIST | 92.4% | 2.88 | 4 |
| notMNIST | 89.0% | 4.52 | 7 |
| Kuzushiji-MNIST | 70.6% | 3.60 | 8 |
| MNIST | 93.2% | 4.63 | 8 |

**Every dataset shows severe redundancy in the joint 30D space** -- an
entropy-based effective rank of 2.9-4.6 out of a nominal 30 dimensions,
needing only 4-8 principal components to explain 95% of the variance.
Mean absolute correlation between individual original and smoothness
features ranged 0.51-0.70 across datasets (max single-pair correlations
0.97-0.99 in every case), confirming the two functionals are picking up
heavily overlapping, not orthogonal, structure -- even though they are
computed by genuinely different operations on the same topology.

Kuzushiji-MNIST stands out with the lowest 20D/smoothness prediction
agreement (70.6%, versus 89-93% elsewhere) -- the two functionals
disagree with each other far more often on this dataset than any other,
despite showing similarly low effective rank. This is worth flagging as a
place where the redundancy story may be less complete than the other
three datasets suggest, though it does not change the central finding
(20D vs. combined-30D still shows no significant difference here either).

## Two things the diagnostics support, and two things they don't

Low effective rank and high feature correlation are evidence of
**substantial shared structure**, not proof that smoothness adds zero
unique information -- a small, low-variance component of the joint space
could still carry class-relevant signal a linear probe fails to isolate.
Similarly, prediction agreement is not a direct redundancy measure:
Fashion-MNIST and MNIST's high 20D/smoothness agreement (92.4%, 93.2%)
partly reflects both classifiers getting the same easy examples right,
not necessarily representational overlap specifically. This makes
Kuzushiji-MNIST's low agreement (70.6%) the most informative single
number so far -- the two readouts genuinely disagree often there, which
is exactly the condition under which naive concatenation's continued
failure to help is most surprising and most worth investigating further
via oracle complementarity, below.

## Correction: "modest," not "substantial," oracle headroom

The review is right to push back on this framing. The headroom is real
but limited: 1.8pp (Fashion-MNIST), 2.2pp (notMNIST), 3.8pp
(Kuzushiji-MNIST), 1.8pp (MNIST). The smoothness-only-correct counts that
define the entire opportunity available to any fusion method are small in
absolute terms too: 9, 11, 19, and 9 images out of 500 respectively. Any
fusion method's job is to identify a small minority of cases where
smoothness is specifically preferable, not to find a large pool of
recoverable value.

## Oracle complementarity, for reference

| Dataset | 20D acc | Smoothness acc | Oracle acc (either correct) | Both correct | 20D-only | Smoothness-only | Both wrong | Headroom |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 71.4% | 70.2% | 73.2% | 342 | 15 | 9 | 134 | 1.8pp |
| notMNIST | 80.4% | 76.4% | 82.6% | 371 | 31 | 11 | 87 | 2.2pp |
| Kuzushiji-MNIST | 52.4% | 46.2% | 56.2% | 212 | 50 | 19 | 219 | 3.8pp |
| MNIST | 89.6% | 87.6% | 91.4% | 429 | 19 | 9 | 43 | 1.8pp |

## Confidence-gating diagnostic: the cheap check, done first

For every disagreement between 20D and smoothness, compare choosing
whichever model has higher softmax confidence against always trusting
20D:

| Dataset | N disagreements | Confidence-gate accuracy on disagreements | Always-20D accuracy on disagreements | Gate vs. 20D overall |
|---|---|---|---|---|
| Fashion-MNIST | 38 | 34.2% | 39.5% | 71.0% vs. 71.4% (worse) |
| notMNIST | 55 | 52.7% | 56.4% | 80.0% vs. 80.4% (worse) |
| Kuzushiji-MNIST | 147 | 29.9% | 34.0% | 51.2% vs. 52.4% (worse) |
| MNIST | 34 | 61.8% | 55.9% | 90.0% vs. 89.6% (better) |

**Raw confidence is not a usable gating signal on three of four
datasets** -- including Kuzushiji-MNIST specifically, the dataset with the
most headroom to recover. This was a discouraging early signal for the
more elaborate fusion methods that followed, and it held up.

## Three bounded fusion methods, tested as specified

All three used a proper held-out validation split (250/class for fitting
base classifiers, 50/class held out for method selection, reusing the
existing cached hybrid-head training data rather than computing new
images -- zero leakage into the frozen test set).

**1. Validation-selected probability weighting** (alpha grid: 0.5-1.0):

| Dataset | Selected alpha | Test accuracy: 20D vs. fusion | Discordant counts | McNemar p |
|---|---|---|---|---|
| Fashion-MNIST | **1.0** (no smoothness weight) | 70.0% vs. 70.0% | 0 vs. 0 | 1.000 |
| notMNIST | **1.0** (no smoothness weight) | 79.8% vs. 79.8% | 0 vs. 0 | 1.000 |
| Kuzushiji-MNIST | 0.7 | 51.6% vs. 52.2% | 6 only-20D, 9 only-fusion | 0.607 |
| MNIST | 0.5 | 88.6% vs. 89.2% | 4 only-20D, 7 only-fusion | 0.549 |

Validation correctly selected alpha=1.0 -- explicitly excluding smoothness
entirely -- on two of four datasets, exactly the null result the grid was
designed to be able to detect. On the other two, a real non-trivial
weight was selected with a small, directionally positive but not
statistically significant gain. (Note: the 20D accuracy figures here use
only 250/class, the reduced training set from carving out the validation
split, so they differ slightly from the 300/class main results elsewhere
in this document -- an expected, disclosed side effect of this specific
test, not an inconsistency.)

**2. Calibrated logit fusion** (temperature-scaled per model, tested only
on the two datasets where method 1 found a non-trivial alpha):

| Dataset | T(20D) | T(smoothness) | Equal-calibrated accuracy | Weighted-calibrated accuracy | Best McNemar p |
|---|---|---|---|---|---|
| Kuzushiji-MNIST | 1.017 | 0.987 | 51.2% (worse) | 51.8% (+0.2pp) | 1.000 |
| MNIST | 0.913 | 0.647 | 88.8% (+0.2pp) | 89.0% (+0.4pp) | 0.774 |

The smoothness classifier's logits needed real cooling on MNIST (T=0.647)
-- confirming a genuine calibration mismatch existed -- but correcting for
it did not change the conclusion: gains remain small and not
statistically significant.

**3. Out-of-fold stacked fusion** (5-fold out-of-fold base predictions,
regularized multinomial meta-classifier on the 20 resulting probabilities,
full 300/class training data, no leakage):

| Dataset | 20D | Stacked | Discordant counts | McNemar p |
|---|---|---|---|---|
| Kuzushiji-MNIST | 52.4% | 52.4% (identical) | 6 vs. 6 | 1.000 |
| MNIST | 89.6% | 89.4% (slightly worse) | 5 vs. 4 | 1.000 |

The most rigorous of the three methods shows the flattest result of all --
exactly even on Kuzushiji-MNIST, slightly worse on MNIST.

## Kuzushiji-MNIST deep dive: a partially interpretable, not purely scattered, pattern

Inspecting all 19 smoothness-only-correct cases directly: true classes
rescued are spread across 8 of 10 categories (no single class dominates,
maximum 4 cases), consistent with "scattered." But **20D's wrong
prediction on these rescued cases is class 5 (は, "ha") in 8 of 19 cases
(42%)** -- a real concentration, not noise -- and within those, class 2
(す, "su") is the true class 4 separate times. This is a genuine, narrow,
recognizable regime: 20D has a specific bias toward confusing certain
characters (especially す) with は that smoothness does not share. It is
real and interpretable, but narrow enough (a handful of cases around one
specific character pair) that it does not translate into a broadly
exploitable gating signal, consistent with the confidence-gating
diagnostic's failure on this exact dataset.

## Applying the stopping rule

Per the pre-specified conditions: **the stopping rule is met.** Validation
weighting selected alpha at or near 1.0 on half the datasets; where it did
not, calibrated fusion's gains were small and not significant; stacking,
the most rigorous method, produced no significant improvement on either
remaining candidate dataset (Kuzushiji-MNIST: exactly even; MNIST:
slightly worse). The correct conclusion, in the review's own framing:
**smoothness provides an independently effective but mostly overlapping
readout. Its complementary errors are real (confirmed by consistent,
if modest, oracle headroom across all four datasets, and by a genuine,
interpretable confusion-pattern difference on Kuzushiji-MNIST) but too
sparse or insufficiently identifiable for the tested fusion mechanisms to
exploit reliably.**

## What this justifies next

Per the decision rule, this specific fusion branch is closed -- three
bounded methods were tested, and the stopping conditions were met. This
does not mean smoothness or the broader graph-representation direction is
exhausted, only that raw-feature concatenation and the three late-fusion
mechanisms tested here are not the way to combine it with the original
20D readout. Any future work combining the two would need either a
fundamentally different mechanism than tested here, or a different
justification for why one of the specific narrow regimes found (like
Kuzushiji-MNIST's su/ha confusion) would generalize into something worth
building a specialized gate for.

## Honest limitations

- Single run per condition per dataset -- no repeated seeds anywhere in
  this replication, consistent with every other result in this project.
- The MNIST frozen 500-image test set was newly created for this
  experiment (stratified, seed=2024, matching the other three datasets'
  convention) -- it is not the same sample used in MNIST's original
  full-10,000-image evaluation, so the 89.6% figure here should be read as
  internally consistent with this replication's own conditions, not
  directly compared to the earlier 90.89% full-test-set result.
- Kuzushiji-MNIST's unusually low 20D/smoothness prediction agreement
  (70.6%) is reported but not further investigated -- it does not change
  the central conclusion but is flagged as worth understanding before any
  future work builds specifically on Kuzushiji-MNIST's smoothness
  behavior.
- The effective-rank calculation uses an entropy-based measure on the
  training-set covariance; a hard-threshold rank measure (e.g., number of
  eigenvalues above some cutoff) was not computed and could give a
  somewhat different number, though the qualitative conclusion (severe
  redundancy, well below the nominal 30 dimensions) is unlikely to change.

## Reproducing these results

For each dataset: existing cached 20D artifacts (topologies, baselines,
hybrid-training and test raw scores) plus newly-built smoothness
calibration, training, and test features via `graph_smoothness.py`,
following the identical protocol across all four datasets. MNIST's 20D
artifacts were rebuilt fresh in this experiment
(`mnist_baselines_20d_VERIFIED.pkl`, `mnist_20d_VERIFIED_train_b{1,2}.pkl`)
after the normalization bug was caught -- the pre-existing
`hybrid_20feat_batch{1,2}.pkl` and `baselines_both_modes.pkl` files should
not be reused together without accounting for the normalization mismatch
identified here.
