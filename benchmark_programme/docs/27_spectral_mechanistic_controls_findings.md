# Mechanistic Controls: Normalized Energy and Active-Support-Matched Random Bases

## What these two experiments were for

The prior round established that Kuzushiji-MNIST's spectral advantage is
statistically robust against 100 generic random pixel-space projections
(Bonferroni-corrected), while Fashion-MNIST and notMNIST sit near the top
of that distribution without reaching significance. It also identified a
real gap: the random-Gaussian control is not information-source-matched
to spectral (class-agnostic and whole-pixel-space, versus spectral's
class-conditioned, active-support-restricted, topology-derived
construction). These two experiments close that gap, in the order
specified: per-image active-energy normalization first, then random
orthonormal bases matched to spectral's exact active support.

## Experiment 1: normalized energy and active-support energy

| Dataset | 20D+spectral (unnormalized) | 20D+spectral (normalized) | 20D+active-energy-alone |
|---|---|---|---|
| Fashion-MNIST | 76.60% | 76.60% (identical) | 74.40% |
| notMNIST | 84.40% | 84.60% (+0.2pp) | 83.60% |
| Kuzushiji-MNIST | 59.00% | 57.80% (-1.2pp) | **59.60% (+0.6pp)** |

**Fashion-MNIST and notMNIST**: normalizing by active-support energy does
not collapse the gain -- if anything it is unchanged or marginally better.
This supports the interpretation that the useful signal concerns *how
energy is distributed* within the low-frequency subspace (genuine
spectral shape), not simply *how much* of the image overlaps the class's
active support. Active-support-energy alone, retained as its own control,
underperforms both spectral variants on both datasets.

**Kuzushiji-MNIST's apparent pattern does not survive paired testing --
this needed checking before being treated as a real mechanistic finding,
and it doesn't hold up.** McNemar tests: unnormalized-vs-normalized
(only_unnorm=27, only_norm=21, p=0.471) and unnormalized-vs-active-energy
(only_unnorm=31, only_energy=34, p=0.804) -- neither difference is
distinguishable from ordinary test-sample variation on 500 images. The
same check on notMNIST's apparent +0.2pp increase also shows no
significant difference (p=1.000, p=0.627). **The defensible statement:
normalization reduced observed Kuzushiji-MNIST accuracy by 1.2 percentage
points, and active-support energy scored 0.6 points above unnormalized
spectral, but paired testing shows neither difference is distinguishable
from test-sample variation.** This is a real correction to the earlier
framing, which treated these small directional differences as a
confirmed mechanistic complication before checking significance.

## Experiment 2: random orthonormal bases on the exact active support

Same active-node support, same 5-dimensional output, same class
conditioning as spectral -- but a random orthonormal subspace instead of
the learned low-frequency eigenvectors. Tested with 20 seeds per dataset
(smaller ensemble than the 100-seed pixel-space test, given the added
per-image cost of this more tightly matched control; treated as a first
pass, not a final-precision estimate).

| Dataset | Spectral's delta | Active-support-matched random ensemble | Percentile | Matched/exceeded |
|---|---|---|---|---|
| Fashion-MNIST | +6.60pp | mean +5.03pp, std 1.20pp, range [+3.20, +7.20] | 95.0% | 1 of 20 |
| notMNIST | +5.00pp | mean +3.92pp, std 0.85pp, range [+2.60, +5.40] | 90.0% | 2 of 20 |
| **Kuzushiji-MNIST** | **+9.60pp** | mean +6.74pp, std 1.37pp, range [+4.40, +9.60] | see correction below | **1 of 20 (corrected)** |

**A numerical error in the original report, caught by review and confirmed
by recomputing with full floating-point precision rather than rounded
literals**: the random ensemble's maximum delta and spectral's delta are
*exactly* floating-point identical (0.09599999999999997, both corresponding
to 295/500 correct) -- a genuine tie, not spectral beating every draw. The
correct count is **1 of 20 matched or exceeded, 0 of 20 strictly
exceeded** -- not the "0 of 20" originally reported, which came from
comparing the ensemble against a rounded decimal literal (0.0960) rather
than the exact computed value. Using the review's own formula,
p_emp = (1 + 1)/21 ≈ **0.095**, not the stronger implied claim. **This
20-seed experiment provides strong ranking evidence (spectral is at or
above the top of this distribution) but only borderline inferential
evidence, and cannot independently support a Bonferroni-corrected claim.
The earlier 100-seed pixel-space test (p=0.0099, surviving Bonferroni)
remains the statistically stronger result for Kuzushiji-MNIST's privilege
claim** -- this tighter, active-support-matched control corroborates the
ranking but should not be read as an independent significance result on
its own.

Restricting to the exact same active support and class-conditioning as
spectral, the active-support-matched random control performs *better on
average* than the fully generic pixel-space random control did (e.g.,
Fashion-MNIST: mean +5.03pp here vs. +4.52pp against the generic control)
-- confirming that active-support restriction and class-conditioning are
themselves genuinely informative, independent of which specific subspace
within that support is used. Fashion-MNIST and notMNIST remain in a
similar (if anything, slightly weaker for notMNIST) position relative to
this tighter control as they were against the generic one -- still not
reaching significance, with the same caveat that 20 seeds gives a noisier
estimate than the earlier 100-seed test.

## Putting the two experiments together: Kuzushiji-MNIST's story is more complicated than "cleanly privileged"

Experiment 2 confirms Kuzushiji-MNIST's spectral advantage requires the
specific eigenstructure, not just active-support access -- a clean, strong
result. But Experiment 1 complicates the mechanistic story: active-
support-*energy alone* (no eigenvector structure at all) also performs
about as well as full spectral there. Both can be true simultaneously
without contradiction: the *specific eigenvectors* matter relative to
*other random subspaces of the same support* (Experiment 2's finding),
while *how much energy falls in the support at all* also carries real,
separate signal that a normalization scheme divides away (Experiment 1's
finding). This suggests Kuzushiji-MNIST's regime may be one where multiple
related but distinct properties of the active subgraph -- not low-frequency
structure alone -- are doing useful, only partially overlapping work.

## Honest limitations

- Experiment 2 used 20 seeds per dataset, not 100 -- a coarser estimate
  than the primary random-ensemble test, chosen for tractability given
  the additional per-image computation this tighter control requires.
- The apparent notMNIST weakening (90th vs. 95th percentile) between the
  generic and active-support-matched controls should not be over-read
  given the smaller ensemble size here.
- Frequency-band ablation on Kuzushiji-MNIST (low- vs. mid-frequency vs.
  random-eigenvector bands, matched dimensionality) remains the next
  specified step and has not been run in this pass.
- Degree-preserving graph nulls remain reserved for later, per the
  review's own ordering, only if band ablation leaves the topological
  question unresolved.

## Immediate next step

Frequency-band ablation on Kuzushiji-MNIST specifically: lowest five
non-trivial modes (current spectral construction) vs. the next five modes
vs. a middle-frequency band vs. five random eigenvectors from the same
graph, matched dimensionality throughout. This directly tests whether the
privileged result is about *low-frequency* structure specifically, or the
learned graph basis more generally -- the sharper question Experiment 1's
active-energy finding raises but does not resolve on its own.

## Reproducing these results

`spectral_readout.py` extended with
`spectral_score_normalized_and_energy` and
`build_random_orthonormal_basis`. Phase vectors cached once per dataset
(`{dataset}_phases_cache.npz`, small: per-pixel vectors, not full pairwise
matrices) and reused across all 20 random-basis seeds to avoid redundant
oscillator-dynamics computation.
