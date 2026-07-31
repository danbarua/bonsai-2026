# Quantifying the Channels: Dependence Measures and the Decomposition Narrative

## Why this measurement matters, distinct from everything before it

Every prior result in this line of work demonstrated usefulness through
classifier performance -- accuracy, McNemar tests, oracle headroom. None
of it directly measured how statistically distinct the three channels
(T: topology matching, E: support energy, R: normalized spectral
allocation) actually are. This closes that gap using two complementary
measures: CCA (linear shared structure) and distance correlation (linear
and nonlinear dependence in one scalar).

## A necessary calibration step before interpreting anything

A synthetic sanity check surfaced a real methodological trap before real
data was touched: with a few hundred samples and ten dimensions, even
**genuinely independent** random blocks give a top canonical correlation
around 0.27 and a comparable distance correlation -- a well-known
finite-sample selection bias (the "top" canonical correlation is a
maximum over many directions, and it does not converge to zero just
because the true relationship is null). Every real measurement below is
therefore reported alongside a shuffled-null baseline (5 seeds, breaking
the pairing while preserving each block's own marginal structure) rather
than compared against a naive assumption that zero means independence.

## Results: all three pairs are genuinely, massively dependent -- but not equally

| Dataset | T,E: CCA (null) / dCor (null) | T,R: CCA (null) / dCor (null) | E,R: CCA (null) / dCor (null) |
|---|---|---|---|
| Fashion-MNIST | 0.966 (0.129) / 0.522 (0.080) | 0.900 (0.125) / 0.579 (0.110) | 0.832 (0.102) / 0.467 (0.089) |
| notMNIST | 0.771 (0.124) / 0.423 (0.087) | 0.834 (0.128) / 0.507 (0.123) | 0.815 (0.098) / 0.386 (0.081) |
| Kuzushiji-MNIST | 0.920 (0.119) / **0.860** (0.088) | 0.778 (0.125) / 0.513 (0.122) | 0.768 (0.102) / 0.310 (0.107) |

Every observed dependence value greatly exceeded all shuffled-null values
across all five seeds -- T, E, and R all share substantial real structure
with each other on every dataset. (The stronger "5 to 10 standard
deviations" framing is avoided deliberately: five null draws give a noisy
variance estimate, and a safe claim about exceeding every observed null
value is better supported than a precise multiple-of-sigma claim would
be.) None of these channels are anywhere close to independent in an
absolute sense. All features (T, E, and R alike) were standardized using
training-set statistics before every dependence measurement in this
document, matching the normalization already applied throughout this
project's classifier evaluations.

**But E,R consistently shows the lowest distance correlation of the three
pairs, on every single dataset** -- 0.467 vs. 0.522/0.579 (Fashion-MNIST),
0.386 vs. 0.423/0.507 (notMNIST), 0.310 vs. 0.860/0.513 (Kuzushiji-MNIST).
**The top-CCA ordering should not be combined with this** -- it tracks a
different quantity (the single strongest linearly-aligned direction,
versus dCor's summary of whole-block linear-and-nonlinear dependence) and
is dataset-dependent: E,R is lowest on Fashion-MNIST (0.832 vs. 0.900/0.966),
but T,E is lowest on notMNIST (0.771 vs. 0.815/0.834), and on
Kuzushiji-MNIST E,R (0.768) is only marginally below T,R (0.778). The
precise statement is that whole-block dependence (dCor) is consistently
lowest between E and R, while the single strongest linear relationship
(top CCA) varies by dataset -- two different, both informative,
observations, not one ranking.

This is a clean, cross-dataset confirmation, independent of any classifier,
that E and R carry more statistically distinct information from each
other than either does from the original topology-matching
representation -- consistent with the classifier-based finding, while not
itself establishing that the less-shared information is label-relevant.
Distance correlation answers how strongly two blocks are statistically
associated, not whether the less-shared portion carries information
useful for classification -- that second question is what the
residualization, shuffle, and classification controls (before and after
this document) actually answer. The dependence analysis supplies
descriptive structure around those causal controls; it does not replace
them.

## A striking, dataset-specific finding worth flagging rather than explaining away

Kuzushiji-MNIST's T,E distance correlation (0.860) is far higher than the
same pair on either other dataset (0.522, 0.423) -- support energy and
topology-matching scores are unusually tightly linked there. This sits in
apparent tension with the earlier finding that E's *correct pairing*
matters more on Kuzushiji-MNIST than anywhere else (the largest, most
significant shuffle-sensitivity of any component on any dataset). These
are not actually contradictory: high dependence is not the same as
complete redundancy -- a dCor of 0.86 leaves real room for E to carry
additional information beyond what T captures, and the classifier
evidence indicates the model is using exactly that residual information.
This is offered as a resolution of an apparent tension, not a fully
explained mechanism, and is worth keeping in mind for any future
Kuzushiji-MNIST-specific mechanistic work.

## The reframed narrative, adopted directly

The three-stage structure proposed for this whole line of work is more
useful than the loose "experiment, then controls" framing this document's
history has drifted through, and is worth stating plainly:

- **Stage A -- does graph-spectral information exist beyond topology
  matching?** Yes (Capacity Experiment III's original result, confirmed
  across three datasets, one non-significant).
- **Stage B -- is the effect genuine?** Yes. The effect survives
  duplication, correspondence shuffling, linear residualization, and
  matched-dimensional random-projection controls -- not one of these
  produced a result inconsistent with a real effect on notMNIST and
  Kuzushiji-MNIST, and Fashion-MNIST's more limited status was itself
  confirmed consistently across every control, not contradicted by any of
  them. Dependence measurements (CCA, distance correlation) then show
  that the useful channels are strongly overlapping rather than
  statistically independent -- a descriptive characterization of the
  relationship between channels, kept conceptually separate from the
  performance-validation controls that actually established the effect
  is genuine.
- **Stage C -- what is the representation?** Not S = E×R. The
  representation is (T, E, R). These channels are neither independent nor
  redundant. They are overlapping transformations of the same image and
  learned topology, with distinct label-relevant residual information.
  The multiplicative product was an information-losing bottleneck, not
  the natural unit of representation.

**The central claim has shifted, and the shift is real, not just
rhetorical**: this is no longer primarily a claim about spectral features
being useful. It is a claim that a compressed statistic (S) was hiding two
separable channels whose separate identities carry more classification
value than their product -- a representational finding about information
compression, not a feature-engineering result specific to graph spectra.
Any classifier built on a similarly-compressed graph statistic may be
subject to the same loss.

## Honest limitations

- CCA and distance correlation are two specific dependence measures among
  many reasonable choices (the review also suggested HSIC and kernel
  alignment) -- consistent results between these two increases confidence
  but does not exhaust the space of ways to quantify dependence.
- Distance correlation was computed on an 800-sample subsample per
  dataset for computational tractability (O(n^2) cost) -- not the full
  ~3000-sample training set, though the shuffled-null comparison uses the
  identical subsampling procedure, keeping the comparison fair.
- The Kuzushiji-MNIST T-E dependence finding is reported and given a
  plausible resolution, not a tested mechanism -- it was not anticipated
  before running this measurement and has not been independently verified
  by a second method.
- The class-independent global ink-energy control and the redesigned
  frequency-band ablation (now cleanly formulated as (T,E,R_low) vs.
  (T,E,R_mid) vs. (T,E,R_high), with E held fixed per the review's
  updated design) remain the next steps and were not run in this pass.

## Reproducing these results

`dependence_measures.py` (new): `cca_canonical_correlations` and
`distance_correlation`, both verified against synthetic
independent/linearly-dependent/nonlinearly-dependent test cases before
use on real data. Applied to the existing cached T (20D topology-matching),
E (active-support energy), and R (normalized spectral allocation)
features for all three primary datasets, with a 5-seed shuffled-null
baseline computed identically for every comparison.
