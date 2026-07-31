# The T-E Residualization Test: Resolving the Dependence/Importance Tension, and It Generalizes

## The precise question this answers

Kuzushiji-MNIST showed an apparent tension: T and E are unusually strongly
dependent (dCor=0.860) while E's correct pairing is simultaneously the
single most important factor found anywhere in this line of work
(shuffling E there costs 6.06pp on average, with McNemar significance
levels as low as 1.0x10^-7). The review's proposed resolution: most of
E's variation overlaps T, but the small non-overlapping fraction carries
disproportionate discriminative value. Tested directly by fitting E from
T via ridge regression, forming the residual E_perp = E - Ê(T), and
comparing T alone, T+predicted-E, T+residual-E, and T+full-E.

## Result: confirmed, and not specific to Kuzushiji-MNIST

| Dataset | R² of E from T (range across dims) | T alone | T+predicted-E | T+residual-E | T+full-E | % of full-E gain recovered by residual alone |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.702-0.809 | 70.00% | 70.60% | 73.60% | 74.40% | 81.8% |
| notMNIST | 0.368-0.393 | 79.40% | 79.20% | 83.00% | 83.60% | 85.7% |
| Kuzushiji-MNIST | 0.741-0.801 | 49.40% | 49.40% | 58.40% | 59.60% | **88.2%** |

**T+predicted-E never meaningfully beats T alone** (identical on
Kuzushiji-MNIST, small deviations of ±0.6-0.8pp elsewhere) -- this is
mainly a useful implementation check, not new evidence on its own: Ê(T)
is a linear transformation of T, so it cannot enlarge the linear span
available to an unregularized linear classifier, and the small observed
deviations are attributable to regularization geometry, numerical
conditioning, or finite optimization behavior rather than genuine
information. The evidential weight belongs entirely to the residual
condition, below. **On every dataset, the residual alone recovers 82-88% of E's
full benefit**, despite R² of E from T varying enormously across datasets
(37-39% on notMNIST, 70-81% on Fashion-MNIST and Kuzushiji-MNIST). This is
not a Kuzushiji-MNIST-specific quirk -- it is a general property of how E
relates to T: E's classification value is concentrated in whatever
fraction is NOT linearly recoverable from T, regardless of how large or
small that fraction is in absolute terms.

## Statistical confirmation (T alone vs. T+residual-E)

| Dataset | McNemar p-value |
|---|---|
| Fashion-MNIST | 6.43x10^-3 |
| notMNIST | 3.93x10^-3 |
| Kuzushiji-MNIST | 2.15x10^-8 |

All three reach significance; Kuzushiji-MNIST's is dramatically stronger,
consistent with its outsized E-dependence found throughout this line of
work.

## The precise resolution of the apparent tension

Kuzushiji-MNIST's high T-E dCor (0.860) reflects that a large share of
E's *total variance* overlaps T -- true and confirmed directly (R²=0.74-0.80
here). But dependence measured over the whole block does not describe how
that dependence is distributed with respect to classification value.
**A precision correction on the variance claim**: "roughly 20-26% of E's
variance" is only accurate dimension by dimension, from the reported R²
range -- the aggregate residual variance for a 10-dimensional block
depends on each dimension's variance and the correlations between E's
dimensions, not just the per-dimension R² values. The safer, correct
statement: *each E coordinate retained approximately 20-26% unexplained
variance under the fitted linear model on Kuzushiji-MNIST.* A
variance-weighted multivariate R² or reconstruction-error score would be
needed to support a block-level percentage claim, which this document
does not make. What is established without that caveat: this small
per-dimension unexplained fraction carries the overwhelming majority of
E's discriminative content (88.2% of its total classification gain).

**A second precision point: this residual is about linear recoverability
specifically.** E_perp = E - ridge-predicted-E(T) removes only the
component of E linearly predictable from T -- it does not remove all
nonlinear dependence, and the earlier distance-correlation measurements
indicate substantial whole-block dependence may remain beyond what this
linear model captures. The exact, defensible claim is: *E contains
label-relevant information not linearly recoverable from T.* A nonlinear
residualization would test a stronger claim but was not necessary to
establish this one.

High aggregate dependence and high residual importance are not in tension
once properly decomposed -- they describe different things (how much of
E's variance is shared, versus where E's linearly-non-recoverable,
label-relevant information is concentrated), and this result shows
precisely how both can be true at once.

## Honest limitations

- Ridge regression with a single untuned regularization strength
  (alpha=1.0), consistent with every other residualization in this
  project but not validated against alternatives.
- The T+predicted-E condition's near-zero gain is expected mechanically
  for a linear classifier reading a linear projection of T -- this
  confirms the decomposition is behaving as designed, not an independent
  finding.
- This resolves the dependence/importance tension descriptively; it does
  not explain *why* Kuzushiji-MNIST's overlap fraction is so much larger
  than notMNIST's, which remains an open, unexplained structural
  difference between the datasets' active-support geometry.

## Immediate next steps, per the review's recommended order

1. Global ink and class-independent energy controls (still owed).
2. Frequency-band comparison with E held fixed, now sharpened further:
   given this result, comparing T+E+R_band across bands should also track
   whether R's informative residual (relative to T) behaves the same way
   E's does.
3. Optionally, repeat the dCor measurements with T reduced to 10
   dimensions via PCA, to confirm the E,R-lowest-dependence ordering is
   not an artifact of T's larger dimensionality.

## Reproducing these results

Ridge regression (`sklearn.linear_model.Ridge`, alpha=1.0) fit from
existing cached T (20D topology-matching) to existing cached E
(active-support energy) on training data for each dataset; residual
formed and evaluated identically to prior residualization tests in this
project. No new per-image computation required.
