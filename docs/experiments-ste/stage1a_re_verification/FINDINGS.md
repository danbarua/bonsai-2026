Simplified Technical English version of `experiments/stage1a_re_verification/FINDINGS.md`.

# Stage 1A Re-Verification: Findings

The full design, the notation, and the pre-registered decision rule
are in `DESIGN.md`. A pre-registered decision rule is a rule the team
wrote down and locked before seeing the results. This document reports
what the design's own robustness checks concluded. The checks disagree
with each other in places. This disagreement is the central finding of
this document.

## What the Team Ran

The team ran all 770 planned instances. This total is 10 classes times
(T plus lattice, 1 each), plus 10 classes times 3 stochastic controls
times 25 seeds. All 770 runs completed with zero errors. The full run
took about 2.5 minutes, using 9 parallel workers, at about 5.2
instances per second. Before committing to the full run, the team
timed one worst-case instance serially: class 9, with 616 active
nodes, T construction. It took 2.0 seconds. This gave a conservative
upper bound of about 1540 seconds for a full serial run. The actual
parallel run finished well under this bound.

Before building any control, the team rebuilt each class's T
independently from the raw KMNIST images. The team used the function
`build_class_topology`, with 200 images per class. The team checked
that each rebuilt T matched the cached `stage1a_all_classes.pkl` file
byte for byte. This check passed for all 10 classes. This confirms
that the rebuild reproduces the same T used elsewhere in this project,
without drifting from it. The rebuild was needed because the cached
file does not store `active_indices` or the ink mask.

theta0 is one shared starting-phase vector for each class. The team
reused the same theta0 across every construction and seed within a
class. The team generated a fresh theta0 for each class, using seed =
4000 plus the class index. The original Stage 1A run's starting
conditions were never recorded (see
`stage1a_infinitesimal_response/FINDINGS.md`'s "Honest limitations"
section). So this theta0 choice is new and documented here. It is not
a recovery of the original starting conditions.

## Primary Test Results (Mean-Aggregated, Holm-Corrected Across 4)

| Comparison | Median d | HL estimate | Sign+ /10 | W | p_exact | p_holm | Holm-significant? |
|---|---|---|---|---|---|---|---|
| T vs. historical half-edge random, coupling-budget normalized | -68.56 | -238.45 | 0/10 | 0.0 | 0.00195 | 0.00781 | **Yes** |
| T vs. degree-preserving rewiring | -41.03 | -70.13 | 1/10 | 4.0 | 0.01367 | 0.04102 | **Yes** |
| T vs. current edge-count-matched random | -33.98 | -33.98 | 1/10 | 6.0 | 0.02734 | 0.05469 | No |
| T vs. lattice | -2.82 | -7.51 | 2/10 | 12.0 | 0.13086 | 0.13086 | No |

"HL estimate" is the Hodges-Lehmann estimate, a robust estimate of the
typical difference. "Sign+ /10" is the number of the 10 class-level
differences that are positive. "W" is the Wilcoxon W statistic.

Two of the four comparisons (historical random and rewiring) are
nominally significant after Holm correction. For these two, the sign
of the effect is different from the original Stage 1A conclusion. The
full per-class values are in
`results/stage1a_reverification_analysis.pkl`.

## Robustness Checks: This Significance Does Not Survive Median Aggregation

| Comparison | Mean-agg. p_exact | Mean-agg. sign+ | Median-agg. p_exact | Median-agg. sign+ | Sign-flip p |
|---|---|---|---|---|---|
| Historical random | 0.00195 | 0/10 | 0.92188 | 5/10 | 0.00195 |
| Rewired | 0.01367 | 1/10 | 0.19336 | 3/10 | 0.00977 |
| Current random | 0.02734 | 1/10 | 0.49219 | 4/10 | 0.02539 |
| Lattice | 0.13086 | 2/10 | (same, deterministic) | 2/10 | 0.15625 |

For all three stochastic comparisons, switching from within-class mean
aggregation to within-class median aggregation changes the result. The
result goes from nominally significant to solidly not significant. The
sign count also changes. It moves from heavily one-sided (0 to 1
positive out of 10) to roughly even (3 to 5 positive out of 10).

The exact sign-flip test uses the same mean-based statistic as the
primary test. It agrees with the primary test in each case, as
expected. This is not an independent check of the mean-vs-median
question. It only checks whether the mean statistic's observed value
is extreme, compared to its own sign-permutation null distribution.

The 25-seed stability check explains why this happens. Within-class
mean AUC has not settled down by seed 25, for most (class,
stochastic-control) combinations. The most dramatic example: for
historical random, class 2's running mean was 60.7 at 5 seeds, 43.8 at
10 seeds, 32.1 at 15 seeds, and 40.0 at 20 seeds. Then it jumped to
658.6 at 25 seeds. Five extra seed draws more than doubled an estimate
that had looked stable for the previous 15 seeds. This is not a single
unusual case. Similar large jumps between seed-count checkpoints
appear across most classes and both random controls. See the
`stability_diagnostic` field in
`results/stage1a_reverification_analysis.pkl` for the full table.

**The within-class MCSE is not small compared to the class-level
differences it is meant to be tested against.** This is one of the
decision rule's own required conditions, and it fails here. MCSE is
larger than the absolute value of d in 2 of 10 classes for historical
random, and in 1 of 10 classes for current random. It is 0 of 10
classes for rewiring, though it still reaches 40 to 80 percent of the
absolute value of d in most of rewiring's classes. One concrete
example: for historical random, class 6 has an MCSE of 93.5, against a
class-level difference of only 58.9. Here, the noise from seed choice
is larger than the effect being measured.

## Decision Rule Verdict (Per DESIGN.md's Pre-Registered Criteria)

`DESIGN.md` states that the historical null conclusion should be
treated as robust only if these four checks agree: the primary
mean-aggregated Wilcoxon test, the median aggregation check, the exact
sign-flip test, and a small within-class MCSE compared to the
class-level differences. The team applies this same bar to any claimed
effect, not only to the null result.

- **Historical random, current random, and rewiring: INCONSISTENT.**
  The primary test and the sign-flip test agree with each other
  (both are mean-based). But median aggregation disagrees for all
  three comparisons. MCSE is not small compared to the absolute value
  of d, for at least one class, in two of the three comparisons. Per
  the design's own rule ("if the above three are consistent, also fit
  [the mixed model]"), the team correctly skipped the third-priority
  mixed model for all three. Fitting a random-intercept model on top
  of a mean-aggregated quantity already shown to be unstable would not
  have added information.
- **Lattice: CONSISTENT.** The primary test, the sign-flip test, and
  the bootstrap are all not significant, and the bootstrap confidence
  interval includes zero. Lattice has no seed axis, since it is a
  deterministic construction. So the mean-vs-median question and the
  MCSE question do not apply to it. Lattice is the one comparison in
  this design with no within-class seed-aggregation ambiguity at all.
  It reproduces the original Stage 1A conclusion cleanly: no
  significant T-vs-lattice difference.

**Conclusion: this re-verification does not confirm a T-vs-random or
T-vs-rewiring difference. It also does not confirm the original null
result for those two comparisons.** The nominally Holm-significant
p-values for historical random and rewiring are artifacts. They come
from aggregating a heavy-right-tailed AUC distribution using the
arithmetic mean, across only 25 seeds. This is exactly the
seed-sensitivity problem flagged earlier by the class-0-only pilot
test (see `stage1a_infinitesimal_response/FINDINGS.md`'s "Post-hoc
robustness note"). That earlier warning is now confirmed at full
scale: 10 classes, using a pre-registered decision rule, rather than
suggested by one class's 20-seed sweep. The only comparison this
design can speak to with confidence is T vs. lattice, where the
original null result holds up.

## What This Finding Does and Does Not Establish

**Does not establish**: that the learned topology T differs from the
random or rewired controls in how long a perturbation persists. The
"significant" p-values found here are not trustworthy, by this
design's own robustness check. This finding also does **not**
establish the opposite: it does not show there is definitely no
difference for these two comparisons. The instability documented here
means this particular design, arithmetic-mean aggregation of 25 raw
AUC draws, is underpowered. It cannot reliably tell apart a real
effect from aggregation noise. This is not the same as showing no
effect exists.

**Does establish**: T vs. lattice shows no significant difference.
This result is consistent across every check available for that
comparison. It matches the original Stage 1A finding for that pair. It
also establishes something new, concretely and for the first time at
full scale: this infinitesimal-response AUC metric has a heavy-tailed
distribution. Raw arithmetic-mean aggregation across a reasonable
sample size, 25 seeds, is not enough to get a stable read on the two
stochastic-control comparisons. This is a property of the metric and
the choice of aggregation method. It is not a property of the
underlying dynamics being compared.

## What a Properly Powered Version of This Comparison Would Need

The instability found here suggests a natural next step: aggregating
AUC on a log scale, rather than using the raw arithmetic mean that
`DESIGN.md` specified. The original Stage 1A `FINDINGS.md` already
used a log-scale check for the same reason. It ran a secondary paired
t-test on log-AUC, specifically to handle the fact that "AUC values
span roughly three orders of magnitude." That log-scale approach was
not run in this document. `DESIGN.md`'s primary test was implemented
exactly as written: raw-scale mean and median aggregation. A log-scale
variant is a natural robustness check, but it falls outside this
design's pre-specified scope. The team does not add it here after the
fact. This is a design question for a future iteration, not part of
this one.

## Honest Limitations

- theta0 is a new, documented choice for this re-verification (seed =
  4000 plus the class index, for each class). It is not a recovery of
  Stage 1A's original, unrecorded starting conditions. For this reason
  alone, a direct numeric comparison to the original per-class table in
  `stage1a_infinitesimal_response/FINDINGS.md` is not meaningful. This
  is separate from the seed-aggregation questions above.
- The primary test, as specified in `DESIGN.md`, aggregates AUC on the
  raw, non-log scale. Given the now-confirmed heavy right skew, this
  choice is a major driver of the mean-vs-median disagreement reported
  above. The team followed the design as written. The team did not
  modify it partway through the analysis.
- The mixed model, `DESIGN.md`'s third-priority check, never ran for
  any of the four comparisons. Three comparisons were excluded because
  the consistency check failed. The fourth (lattice) has no seed axis,
  so there is no meaningful way to separate a class random intercept
  from residual noise for it. The mixed model's absence here is a
  result of the other checks' outcomes. It is not an oversight.
- 25 seeds per class is `DESIGN.md`'s pre-specified count. The
  stability check shows that, in hindsight, this was not enough for
  the mean statistic to settle down for several (class, control)
  pairs. This document does not extend the seed count after the fact
  to check whether more seeds would resolve the mean-versus-median
  disagreement. Doing so after seeing this result would not be a
  pre-registered check.
