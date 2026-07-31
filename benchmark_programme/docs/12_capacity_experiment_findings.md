# Capacity Experiment I: Multi-threshold Readout -- Does a Second Pruning Threshold Recover Information the 20D Readout Misses?

## The question, precisely

Following the notMNIST result (topology overlap correlated significantly
with an independently-trained MLP's confusion, but not with Bonsai's own),
the natural next question was reframed, correctly, away from "is overlap
meaningful" and toward: **does a richer, pre-specified summary of the
existing learned topologies extract additional useful information already
present in the graph, without changing the topology-construction process
itself?** This document reports one clean, controlled test of that
question on Fashion-MNIST, plus a separate sensitivity analysis on the
notMNIST "I" outlier flagged in the prior transfer report.

## Methodology, following the review's ordering exactly

**1. Audit first.** Before building anything, the cache was checked
directly rather than assumed. Confirmed present: raw (pre-threshold, full
784x784 continuous correlation) population statistics for Fashion-MNIST at
200 images/class; ink/background masks; calibration baselines for the
primary threshold; *raw* (pre-normalization) 20D score matrices for both
the hybrid-head training images and the frozen test set. This mattered in
practice -- it meant a second pruning threshold could be built by
re-thresholding already-computed statistics, at essentially no
recomputation cost, rather than rebuilding population statistics from
scratch.

**2. Evaluation protocol frozen.** The second threshold (0.95, notably
tighter than the primary 0.9) was chosen before looking at any results and
was not tuned against the test set. The same frozen 500-image test sample
used throughout the Fashion-MNIST work was reused unchanged.

**3. One clean comparison, not a combined feature dump.** Exactly as
specified: the new 40D representation is the original 20D (simple+cosine
against the 0.90-threshold topology) concatenated with 20D of the same two
scores against a 0.95-threshold topology built from the identical
population statistics -- one variable changed, not several at once.

**4. Proper controls included.**
- 20D baseline (re-confirmed, reproduces 71.40% exactly).
- 40D duplicated-20D control (the original 20D features concatenated with
  an exact copy of themselves) -- tests whether dimensionality alone,
  with zero new information, changes anything.
- 40 random-linear-projection control -- tests the same question via a
  completely different, information-free 40D space.

## Result 1: accuracy -- a clean negative result for the specific hypothesis

| Condition | Accuracy | Change from 20D |
|---|---|---|
| 20D baseline | 71.40% | -- |
| **40D combined (real second threshold)** | **72.20%** | **+0.80pp** |
| 40D duplicated-20D (pure redundancy) | 71.80% | +0.40pp |
| **40 random-feature control** | **72.40%** | **+1.00pp** |

**The real second-threshold information does not exceed either control.**
The random-feature control achieved a similarly small improvement (+1.0
percentage point versus +0.8 for the second threshold) -- from a single
run each, this should be read as reinforcing that gains of this magnitude
can arise without introducing any additional topology-derived information
at all, not as a meaningful ranking between the two controls. Paired comparison: the 20D and 40D-combined classifiers
agree on 93.2% of the same 500 test images, meaning most of the apparent
gain is a handful of flipped predictions, consistent with generic
capacity/regularization effects of a shallow linear model gaining more
input dimensions, not with newly-exploited structural information. **The
honest answer to the stated primary research question, for this specific
richer functional (a second fixed pruning threshold): no, it does not
detectably recover additional useful information beyond what generic
dimensionality inflation already provides.**

## Result 2: diagnostic correlations show the same pattern, not a different story

Per the review's framing, these are diagnostics, not optimization
targets -- reported for what they show, not because a stronger overlap
correlation was the goal.

| Condition | Confusion vs. topology overlap (rho, p) | Confusion vs. MLP confusion (rho, p) |
|---|---|---|
| 20D baseline | 0.424, p=0.024 | 0.794, p<0.0001 |
| 40D combined (real) | 0.459, p=0.013 | 0.874, p<0.0001 |
| 40D duplicated-20D control | 0.433, p=0.018 | 0.812, p<0.0001 |

The real 40D representation's diagnostics are directionally slightly
higher than either the 20D baseline or the duplicated control -- but the
duplicated (zero-new-information) control shows nearly the same uptick.
**These differences between rho values (0.424 vs. 0.459 vs. 0.433; 0.794
vs. 0.874 vs. 0.812) are descriptive only** -- no statistical test was run
to compare one Mantel correlation against another, and none should be
inferred. Without repeated runs to establish whether a gap this small (0.459 vs.
0.433; 0.874 vs. 0.812) exceeds ordinary run-to-run variation, **this
should not be read as evidence that the second threshold specifically
contributes real structural information beyond what any additional
dimensions provide to this shallow classifier.**

## What this does and does not establish

**Does not establish**: that the learned topology lacks additional
recoverable information beyond the current 20D readout -- this experiment
tested one specific candidate functional (a second fixed pruning
threshold) and found it indistinguishable from generic dimensionality
effects. Per the review's own staged plan, richer graph functionals
(degree-profile features, spectral/Laplacian smoothness, community
structure) remain untested and were deliberately not folded into this
first, deliberately simple experiment.

**Does establish**: a second pruning threshold, specifically, is not the
functional that recovers whatever information the notMNIST result
suggested might be present but unexploited. That is useful negative
information -- it rules out the simplest, cheapest next hypothesis before
investing in more complex graph summaries, exactly the kind of result the
review characterized as a diagnostic rather than a failure.

## Separate deliverable: notMNIST "I" outlier sensitivity analysis

Recomputed all Mantel correlations for notMNIST excluding class 8 ("I"),
per the review's request, before drawing any further conclusions from the
borderline results in the prior report:

| Comparison | Full 10 classes | Excluding "I" (9 classes) |
|---|---|---|
| Bonsai confusion vs. raw overlap | rho=0.270, p=0.116 | rho=0.108, p=0.513 |
| Bonsai confusion vs. Jaccard overlap | rho=0.140, p=0.402 | rho=0.256, p=0.112 |
| MLP confusion vs. raw overlap | rho=0.397, p=0.0024 | rho=0.352, p=0.020 |
| MLP confusion vs. Jaccard overlap | rho=0.308, p=0.033 | rho=0.368, p=0.020 |
| Bonsai confusion vs. MLP confusion (direct) | rho=0.291, p=0.051 | rho=0.246, p=0.156 |

**One important correction this sensitivity check produces**: the
previously-reported "borderline" direct Bonsai-MLP agreement (p=0.051,
reported with appropriate hedging as "borderline" in the prior document)
does **not** survive excluding the outlier class -- it drops to p=0.156,
clearly non-significant. That borderline finding depended substantially on
the "I" outlier and should not have been leaned on as suggestive evidence.
**The more central finding is robust to the exclusion**: the MLP's
confusion correlates significantly with topology overlap both with and
without "I" (p=0.002-0.033 throughout), while Bonsai's own confusion does
not (p=0.11-0.51 throughout) -- the core notMNIST finding stands
independent of this specific outlier.

## Honest limitations

- Single run throughout -- no repeated seeds for any of the four
  conditions in the capacity experiment, consistent with (and sharing the
  same caveat as) every other unreplicated result in this project.
- Only one alternative functional (a second fixed threshold) was tested,
  exactly as scoped by the review's staged plan -- richer graph summaries
  remain for a future experiment, not evidence of absence here.
- The "why does class I reach near-total ink coverage" question was not
  investigated further, per the review's explicit instruction not to
  build interpretation around it without first confirming it via
  sensitivity analysis (done) or investigating the underlying cause
  (not done in this round).

## Reproducing these results

All Fashion-MNIST artifacts reused from the existing cache
(`fmnist_raw_stats_200.pkl`, `fmnist_ink_masks.pkl`,
`fmnist_hybrid_train_batch{1,2}.pkl`, `fmnist_test_features.pkl`,
`fmnist_baselines.pkl`) plus newly-built threshold-0.95 equivalents built
by the identical pipeline pointed at the same cached raw statistics.
