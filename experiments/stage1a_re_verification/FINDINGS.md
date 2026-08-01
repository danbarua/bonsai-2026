# Stage 1A Re-Verification: Findings

Full design, notation, and pre-registered decision rule in `DESIGN.md`.
This document reports what the design's own robustness checks concluded
-- including where they disagree with each other, which is the central
finding here.

## What was run

All 770 planned instances (10 classes x (T + lattice, 1 each) + 10
classes x 3 stochastic controls x 25 seeds) completed with zero errors
in ~2.5 minutes (9 parallel workers, ~5.2 instances/s). Before committing
to the full run, a single worst-case instance (class 9, n_active=616, T
construction) was timed serially at 2.0s, giving a conservative upper
bound (~1540s serial) that the actual parallel run came in well under.

Before building any control, each class's T was independently rebuilt
from the raw KMNIST images (`build_class_topology`, n_per_class=200) and
asserted byte-exact against the cached `stage1a_all_classes.pkl` artifact
-- this passed for all 10 classes, confirming the rebuild (needed to
recover `active_indices` and the ink mask, neither of which the cached
pkl stores) reproduces the same T used elsewhere in this project without
silently drifting from it.

theta0 (one shared initial phase vector per class, reused across every
construction and seed for that class) was freshly generated per class
(seed = 4000 + class_idx) -- the original Stage 1A run's initial
conditions were never recorded (see
`stage1a_infinitesimal_response/FINDINGS.md`'s "Honest limitations"), so
this is a new, documented choice, not a recovery of the original one.

## Primary test results (mean-aggregated, Holm-corrected across 4)

| Comparison | Median d | HL estimate | Sign+ /10 | W | p_exact | p_holm | Holm-significant? |
|---|---|---|---|---|---|---|---|
| T vs. historical half-edge random, coupling-budget normalized | -68.56 | -238.45 | 0/10 | 0.0 | 0.00195 | 0.00781 | **Yes** |
| T vs. degree-preserving rewiring | -41.03 | -70.13 | 1/10 | 4.0 | 0.01367 | 0.04102 | **Yes** |
| T vs. current edge-count-matched random | -33.98 | -33.98 | 1/10 | 6.0 | 0.02734 | 0.05469 | No |
| T vs. lattice | -2.82 | -7.51 | 2/10 | 12.0 | 0.13086 | 0.13086 | No |

Two of four comparisons (historical random, rewiring) are nominally
significant after Holm correction, at a difference of sign from the
historical Stage 1A conclusion for those two controls. Full per-class
values are in `results/stage1a_reverification_analysis.pkl`.

## Robustness checks: this significance does not survive median aggregation

| Comparison | Mean-agg. p_exact | Mean-agg. sign+ | Median-agg. p_exact | Median-agg. sign+ | Sign-flip p |
|---|---|---|---|---|---|
| Historical random | 0.00195 | 0/10 | 0.92188 | 5/10 | 0.00195 |
| Rewired | 0.01367 | 1/10 | 0.19336 | 3/10 | 0.00977 |
| Current random | 0.02734 | 1/10 | 0.49219 | 4/10 | 0.02539 |
| Lattice | 0.13086 | 2/10 | (same, deterministic) | 2/10 | 0.15625 |

For all three stochastic comparisons, switching from within-class mean
to within-class median aggregation collapses the result from
(nominally) significant to solidly non-significant, and the sign count
moves from heavily one-sided (0-1 positive out of 10) to roughly even
(3-5 positive out of 10). The exact sign-flip test (which uses the same
mean-based statistic as the primary test) agrees with the primary test
in each case, as expected -- it isn't an independent check of the
mean-vs-median question, only of whether the *mean* statistic's observed
value is extreme relative to its own sign-permutation null.

The 25-seed-count stability diagnostic explains why: within-class mean
AUC has not converged by seed 25 for most (class, stochastic-control)
combinations. The single most dramatic example: historical-random class
2's running mean is 60.7 (k=5), 43.8 (k=10), 32.1 (k=15), 40.0 (k=20),
then jumps to 658.6 at k=25 -- five additional seed draws more than
doubling an estimate that had looked stable for the previous 15. This is
not an isolated case; comparable multi-fold jumps between successive
seed-count checkpoints appear across most classes and both random
controls (see `results/stage1a_reverification_analysis.pkl`'s
`stability_diagnostic` field for the full table).

**Within-class MCSE is not small relative to the class-level differences
it's meant to be tested against** -- one of the decision rule's own
required conditions. MCSE exceeds |d| outright in 2/10 classes for
historical random and 1/10 for current random (0/10 for rewired, though
still 40-80% of |d| in most of its classes). Concretely: historical
random's class 6 has MCSE=93.5 against a class-level difference of only
58.9 -- the seed noise is larger than the effect being measured.

## Decision rule verdict (per DESIGN.md's pre-registered criteria)

DESIGN.md states the historical null conclusion should be considered
robust only if consistent across the primary mean-aggregated Wilcoxon,
median aggregation, the exact sign-flip test, and small within-class MCSE
relative to the class-level differences. Applying that same bar to any
claimed *effect*, not just to the null:

- **Historical random, current random, rewiring: INCONSISTENT.**
  Primary and sign-flip agree with each other (both mean-based), but
  median aggregation disagrees for all three, and MCSE is not small
  relative to |d| for at least one class in two of the three. Per the
  gating logic specified in DESIGN.md ("if the above three are
  consistent, also fit [the mixed model]"), the tertiary mixed model was
  correctly skipped for all three -- fitting a random-intercept model on
  top of a mean-aggregated quantity already shown to be unstable would
  not have added information.
- **Lattice: CONSISTENT** (primary, sign-flip, and bootstrap all
  non-significant/CI-includes-zero; no seed axis exists for this
  deterministic construction, so the median-vs-mean and MCSE questions
  don't apply). This is the one comparison in this design that carries
  no within-class seed-aggregation ambiguity at all, and it reproduces
  the original Stage 1A conclusion (no significant T-vs-lattice
  difference) cleanly.

**Conclusion: this re-verification does not confirm a T-vs-random or
T-vs-rewiring difference, and it does not confirm the original null for
those two comparisons either.** The nominal Holm-significant p-values for
historical random and rewiring are artifacts of aggregating a
heavy-right-tailed AUC distribution by arithmetic mean across only 25
seeds -- exactly the seed-sensitivity flagged as a risk by the earlier
class-0-only pilot (`stage1a_infinitesimal_response/FINDINGS.md`'s
"Post-hoc robustness note"), now confirmed at full scale (10 classes, a
pre-registered decision rule) rather than suggested by one class's
20-seed sweep. The only comparison this design can speak to with
confidence is T vs. lattice, where the original null holds up.

## What this does and does not establish

**Does not establish**: that learned topology T differs from random or
rewired controls in perturbation persistence (the "significant" p-values
are not trustworthy under this design's own robustness gate). Does
**not** establish the opposite either -- that there's definitely no
difference for these two comparisons -- since the instability documented
here means this particular design (arithmetic-mean aggregation of 25 raw
AUC draws) is underpowered to distinguish a real effect from
aggregation noise, not that no effect exists.

**Does establish**: T vs. lattice shows no significant difference,
consistently across every check available for that comparison, matching
the original Stage 1A finding for that specific pair. It also establishes,
concretely and for the first time at full scale, that this
infinitesimal-response AUC metric is heavy-tailed enough that raw
arithmetic-mean aggregation across a plausible sample size (25 seeds) is
not sufficient to get a stable read on the two stochastic-control
comparisons -- a design property of the metric and the aggregation
choice, not of the underlying dynamics being compared.

## What a properly powered version of this comparison would need

The natural next step suggested by this instability -- aggregating AUC
on a log scale (as the original Stage 1A FINDINGS.md's secondary paired-
t-test on log-AUC already did, precisely to handle "AUC values span
roughly three orders of magnitude") rather than the raw arithmetic mean
DESIGN.md specified -- was not run here. DESIGN.md's primary test was
implemented exactly as written (raw-scale mean and median aggregation);
a log-scale variant is a natural robustness check but is outside this
design's pre-specified scope, and is not added here retroactively. This
is a design question for a future iteration, not a build-out of the
current one.

## Honest limitations

- theta0 is a new, documented choice for this re-verification (seed
  4000 + class_idx per class) -- not a recovery of Stage 1A's original,
  unrecorded initial conditions. Direct numeric comparison to the
  original per-class table in `stage1a_infinitesimal_response/FINDINGS.md`
  is not meaningful for this reason alone, independent of the
  seed-aggregation questions above.
- The primary test, as specified in DESIGN.md, aggregates AUC on the raw
  (non-log) scale. Given the now-confirmed heavy right skew, this choice
  materially drives the mean-vs-median disagreement reported above; it
  was followed as written, not modified mid-analysis.
- The mixed model (DESIGN.md's tertiary check) never ran, for any of the
  four comparisons -- three were gated out by the consistency check
  failing, and the fourth (lattice) has no seed axis for a class random
  intercept to be meaningfully distinguished from residual noise. Its
  absence here is a consequence of the other checks' outcomes, not an
  oversight.
- 25 seeds per class is DESIGN.md's pre-specified count; the stability
  diagnostic shows this was, in retrospect, not enough for the mean
  statistic to have converged for several (class, control) pairs. This
  document does not extend the seed count post hoc to see whether more
  seeds would resolve the mean/median disagreement -- doing so after
  seeing this result would not be a pre-registered check.
