# Corrections and the E/R Decomposition: A Cleaner Capacity Expansion Than the Original Spectral Score

## Two corrections, resolved with exact computation rather than argument

**The KMNIST random-basis "0 of 20" was wrong.** Recomputing with full
floating-point precision instead of a rounded literal: the random
ensemble's maximum delta and spectral's delta are exactly identical
(0.09599999999999997, both 295/500 correct) -- a genuine tie. The correct
count is 1 of 20 matched or exceeded, 0 of 20 strictly exceeded. Using the
review's own formula, p_emp = 2/21 ≈ 0.095 -- borderline ranking evidence,
not independent significance. **The 100-seed pixel-space test (p=0.0099,
surviving Bonferroni) remains the statistically load-bearing result for
Kuzushiji-MNIST's privilege claim**; this tighter 20-seed control
corroborates the ranking without adding independent statistical weight.

**Experiment 1's apparent Kuzushiji-MNIST pattern does not survive paired
testing.** McNemar tests on the unnormalized-vs-normalized comparison
(p=0.471) and unnormalized-vs-active-energy comparison (p=0.804) both show
no significant difference -- the previously-reported "-1.2pp decline" and
"active-energy beats spectral" findings are indistinguishable from
ordinary test-sample variation on 500 images. Same for notMNIST's
apparent +0.2pp increase (p=1.000, p=0.627). Both corrected sections are
updated in the mechanistic-controls document; this document does not
repeat that detail, only notes it was fixed before proceeding.

## The E/R decomposition: the review's flagged "central new condition," and it delivers

The unnormalized spectral score decomposes as S_c(x) = E_c(x) * R_c(x),
where E_c is active-support energy and R_c is the energy fraction
captured by the low-frequency subspace. Both were already computed
(Experiment 1's "active-energy" and "normalized spectral" features,
respectively) -- what remained untested was whether giving the classifier
**both as separate features** (20D+E+R, 20 dimensions total beyond the
base 20D) exposes information the multiplicative product S compresses
away.

| Dataset | 20D+E | 20D+R | 20D+S (=E×R, original) | **20D+E+R** |
|---|---|---|---|---|
| Fashion-MNIST | 74.40% | 76.60% | 76.60% | **77.40%** |
| notMNIST | 83.60% | 84.60% | 84.40% | **86.80%** |
| Kuzushiji-MNIST | 59.60% | 57.80% | 59.00% | **63.60%** |

### Paired McNemar tests, E+R against each individual condition

| Dataset | E+R vs. E | E+R vs. R | E+R vs. S |
|---|---|---|---|
| Fashion-MNIST | **p=0.044** | p=0.424 (n.s.) | p=0.636 (n.s.) |
| **notMNIST** | **p=0.0070** | **p=0.0127** | **p=0.0290** |
| **Kuzushiji-MNIST** | **p=0.0205** | **p=0.00004** | **p=0.0044** |

**On notMNIST and Kuzushiji-MNIST, E+R significantly beats all three of
E alone, R alone, and the original product S -- the strongest outcome the
review's decision framework specified.** The multiplicative product
formulation was compressing two genuinely complementary signals (how much
image energy falls on the class's active support, and how that energy is
distributed across the class's low-frequency modes) into a single scalar,
losing real information in the process. Simply giving the classifier both
components separately recovers that information without any new
computation -- E and R were already being computed as intermediate
quantities inside the original spectral score.

**Fashion-MNIST is more limited**: E+R beats E significantly, but not R or
S -- consistent with Experiment 1's finding that Fashion-MNIST's gain was
already substantially carried by R alone there (normalized spectral
matched unnormalized exactly, 76.60% both), leaving less room for E to add
distinguishable value on top.

## Where this leaves the mechanistic picture

Per the review's own decision rule, "E + R_low beats E" is now confirmed,
significantly, on all three datasets -- the strongest listed outcome:
*support overlap and low-frequency structure contribute complementary
information*. On two of three datasets this extends further: E+R beats
not just E but R and the original S as well, meaning this is not merely
"E adds a little to R" but a genuine reformulation that recovers real,
previously-compressed information. This reframes the practical
recommendation directly: **the E+R decomposition (20 dimensions: 10
active-energy + 10 normalized-spectral) is a better-performing, no-new-
computation replacement for the original 10-dimensional unnormalized
spectral score**, not merely a diagnostic that explained an anomaly.

## Honest limitations

- Single run per condition, as throughout -- though the pattern (E+R
  beating all three individual conditions) is consistent and significant
  on two of three datasets, not a borderline or single-comparison result.
- Fashion-MNIST's more limited outcome (E+R beats only E) means the
  decomposition's practical value is dataset-dependent, not universal --
  consistent with the broader pattern in this project that capacity
  findings need per-dataset confirmation, not a single global claim.
- The frequency-band ablation on Kuzushiji-MNIST (low vs. mid vs. high vs.
  random-index bands, both normalized and unnormalized, with 20D+E+low-R
  as the central condition per the review's updated design) remains the
  next specified step and has not been run in this pass -- this
  decomposition result changes what that ablation needs to establish: not
  just whether low-frequency beats other bands, but whether *R restricted
  to the low band* adds to E beyond what other bands would.
- A class-independent ink-energy control (to separate class-specific
  support overlap from generic stroke density, per the review's note) has
  not yet been run.

## Immediate next steps, updated given this result

1. Frequency-band ablation on Kuzushiji-MNIST, using the review's expanded
   design (low/low-mid/middle/high/random-index bands, both normalized and
   unnormalized, with 20D+E+R_low as the central condition) -- now
   additionally informative because E+R_low can be compared against
   E+R_band for every other band, isolating whether low-frequency R
   specifically is what complements E, or whether any band's R would do.
2. Class-independent ink-energy control, to confirm E's value is genuinely
   about class-specific support overlap and not simply generic stroke
   density.
3. Extend the E/R decomposition test to check whether it also improves
   the random-orthonormal-basis and random-ensemble comparisons -- if E+R
   beats the random ensembles more decisively than S did, that would be
   additional, independent evidence for the decomposition's value beyond
   the paired comparisons already run.

## Reproducing these results

All features reused directly from the existing cache
(`{dataset}_active_energy_{train,test}.pkl`,
`{dataset}_spectral_normalized_{train,test}.pkl`,
`{dataset}_spectral_{train,test}.pkl`, all with provenance metadata) --
no new per-image computation was required for this decomposition test,
only new combinations of already-computed features.
