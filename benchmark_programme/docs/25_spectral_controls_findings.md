# Capacity Experiment III, Controlled: Spectral Information Survives the Sharpest Test, With One Honest Wrinkle

## Where this picks up

Capacity Experiment III found the first positive capacity result in this
series: 20D+spectral beat the 20D baseline significantly on three of four
datasets. Per the review's explicit instruction, this section runs the
missing controls before treating that as established -- duplicate,
random-Gaussian, and shuffled-spectral, followed by the more direct
residualization test.

## Controls: a real, but not fully clean, pattern

| Dataset | 20D+spectral vs. duplicate | vs. random Gaussian | vs. shuffled-spectral (3 seeds) |
|---|---|---|---|
| Fashion-MNIST | **p=0.0094** | p=0.188 (n.s.) | **p=0.0014, 0.0005, 0.0001** |
| notMNIST | **p=0.0002** | p=0.080 (n.s.) | **p<0.0001 (all three)** |
| Kuzushiji-MNIST | **p=0.0003** | p=0.059 (n.s.) | **p=0.0008, 0.0002, 0.0003** |
| MNIST | p=0.115 (n.s.) | p=0.824 (n.s.) | p=0.013, 0.405, 0.210 (mixed) |

**On all three primary datasets, 20D+spectral significantly beats both the
duplicate control and the shuffled-spectral control (which preserves
every distributional property of the spectral block while destroying
per-image correspondence) -- consistently, across all three shuffle
seeds.** This is real evidence against "any 10 extra numbers help" and
against "only the spectral block's marginal statistics matter, not which
image gets which value."

**It does not, however, clearly beat the random-Gaussian control** (p=0.06-0.19,
all above conventional significance on every dataset). This is the honest
wrinkle, not smoothed over: a generic random linear projection of the raw
pixels performs comparably to genuine spectral features when combined
with 20D. Per the review's own decision framework, this partially
resembles (without matching exactly) the "beats random and duplicate but
not shuffled" branch -- here it is the reverse pattern (beats duplicate and
shuffled, not clearly random), which the pre-specified branches did not
anticipate. The most defensible reading: **some of spectral's value may be
explainable by "any sufficiently rich, sample-specific alternative view of
the raw image helps," not uniquely by spectral computation specifically**
-- which is exactly why the residualization test, a more targeted
instrument, matters more here than the control comparisons alone.

## Residualization: the more direct, and more decisive, test

Each spectral feature was fit from the 20D features via ridge regression
(training data only); R² per feature ranged 0.28-0.80 (Fashion-MNIST),
0.28-0.48 (notMNIST), 0.29-0.45 (Kuzushiji-MNIST) -- confirming a real,
substantial, but incomplete linear relationship exists, consistent with
the moderate cross-correlations found earlier. The residual (spectral
minus its 20D-predictable component) was then tested alone and combined
with 20D:

| Dataset | Residual-spectral alone | 20D | 20D + residual-spectral | McNemar p |
|---|---|---|---|---|
| Fashion-MNIST | 33.2% | 71.4% | **74.4%** (+3.0pp) | **0.0357** |
| notMNIST | 41.4% | 80.4% | **85.4%** (+5.0pp) | **0.00017** |
| Kuzushiji-MNIST | 29.0% | 52.4% | **58.0%** (+5.6pp) | **0.00203** |

**On every one of the three primary datasets, the residual -- everything
in spectral that is NOT linearly predictable from 20D -- still produces a
statistically significant improvement when added to 20D.** notMNIST
recovers essentially the entire original gain (85.4%, identical to the
full-spectral combination); Kuzushiji-MNIST recovers most of it (58.0% vs.
the original 59.2%); Fashion-MNIST recovers about three-quarters (74.4%
vs. 75.4%). This is the review's strongest validating branch, and it is
confirmed directly: **genuinely additional representational content, not
explainable by whatever spectral shares with the original 20D readout, is
present and useful on all three primary datasets.**

## Mechanistic confound check: active-node count

Class-level correlation between each class's active-node count (the size
of its ink-involving subgraph) and its mean spectral score across the
training population:

| Dataset | Correlation (active-node count, mean spectral score) |
|---|---|
| Fashion-MNIST | 0.187 (weak) |
| notMNIST | 0.270 (weak) |
| Kuzushiji-MNIST | **0.635 (moderate)** |

Fashion-MNIST and notMNIST show only a weak relationship -- spectral
energy is not simply tracking how large each class's active subgraph is.
Kuzushiji-MNIST shows a real, moderate correlation worth flagging plainly,
though this is a class-level statistic computed from only 10 data points
and should be treated as suggestive, not conclusive. Per the review's
explicit instruction, this is reported as a mechanistic diagnostic, not
used to replace or adjust the primary result -- the normalized-energy
control (dividing projection energy by total active-subgraph input
energy) remains for future work, not run in this pass.

## Where this leaves the capacity claim

Not the strongest possible outcome the review outlined ("survives all
controls including beating random Gaussian cleanly") -- the random-Gaussian
comparison is a genuine, unresolved soft spot. But also clearly stronger
than "beats baseline but not controls" (duplicate and shuffled-spectral
are both cleanly beaten) or "distributional properties only" (shuffled-
spectral losing decisively rules this out directly). **The residualization
result is the most direct test of the actual scientific question asked --
does spectral structure contain class-relevant information beyond what
20D already captures -- and it answers yes, on all three primary datasets,
with statistical significance.** The random-Gaussian result is a real,
disclosed complication for the *specifically-spectral* framing, not a
reason to discard the finding: it suggests part of spectral's value may
generalize to "any sufficiently informative alternative linear view of the
image," while the residualization test confirms real information exists
in spectral beyond what 20D captures, regardless of how that compares to
an arbitrary alternative transform.

## MNIST, deliberately not investigated further

Per explicit instruction: MNIST already shows high baseline accuracy,
minimal oracle headroom (2.8pp, only 14 recoverable errors), and a
non-significant result that plausibly reflects ceiling effects and
limited statistical power at 500 test images, not evidence of a genuinely
different underlying effect. No separate investigation was undertaken;
this remains an open question for a larger frozen evaluation, not a
finding requiring explanation now.

## Honest limitations

- Single run per condition (though three shuffle seeds were used for the
  shuffled-spectral control specifically, per the review's instruction to
  avoid a single random draw).
- The random-Gaussian control used one seed only -- per the review's own
  observation that "a single random control has already behaved
  erratically across datasets" (true again here: p=0.06-0.19, an unstable
  range), multiple random seeds for this control specifically would
  strengthen confidence in whether it's a genuine near-tie or noise.
- The active-node-count confound check is a coarse, class-level (n=10)
  correlation, not the full per-image normalized-energy control the
  review specified -- flagged as a gap, not resolved here.
- Residualization used ridge regression with a single, untuned
  regularization strength (alpha=1.0) -- not validated against
  alternatives.

## Immediate next steps

1. Multiple-seed random-Gaussian control, to determine whether the
   current near-miss (p=0.06-0.19) is a stable near-tie or an artifact of
   one unlucky/lucky draw -- directly relevant given the review's own
   prior observation about random controls being unstable.
2. The full normalized-energy mechanistic control, per-image, not just
   the class-level confound check done here.
3. Given the residualization result is now the strongest confirmed
   evidence, the mechanistic follow-up questions the review outlined
   (number of eigenvectors K, low- vs. mid-frequency bands, eigenvalue
   weighting) are reasonable next steps -- but the random-Gaussian
   ambiguity is the more pressing gap to close first.

## Reproducing these results

All artifacts reused from Capacity Experiment III's cache
(`{dataset}_spectral_train.pkl` / `_test.pkl` / `_baselines_spectral.pkl`,
all with provenance metadata via `feature_provenance.py`), plus the
existing 20D artifacts for each dataset.
