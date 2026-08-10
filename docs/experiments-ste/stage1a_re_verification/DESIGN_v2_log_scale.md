Simplified Technical English version of `experiments/stage1a_re_verification/DESIGN_v2_log_scale.md`.

# Stage 1A Re-Verification: Log-Scale Iteration (v2)

The team committed and locked this document before running any
log-scale analysis. This document states what the team will compute.
It states how the team will draw conclusions. The team wrote it before
any of those numbers existed. The team will not revise this document's
decision rule after seeing the log-scale results.

## Why This Iteration Exists

`DESIGN.md` (version 1) specified the primary analysis on the raw AUC
scale. AUC (area under the curve) is a single number that adds up the
strength of a response over time. That analysis found that 2 of the 4
Holm-corrected comparisons were nominally significant. These were T
vs. historical random and T vs. rewiring. But this significance
collapsed completely under median aggregation. It also failed the
design's own MCSE gate. MCSE is the Monte Carlo standard error, a
measure of noise from using a limited number of seeds.

The team has now confirmed the cause of this problem. It is not just a
guess. AUC values have a heavy right tail. This means a single extreme
seed draw can more than double a class-level mean, even after 20 seeds
already looked stable. One concrete example: for historical random,
class 2's running mean was 40.0 at seed count 20. It jumped to 658.6 at
seed count 25. The arithmetic mean is not robust to this kind of jump.
The arithmetic mean of the logarithm of the values is robust to it.
This is the same as the geometric mean of the raw values.

This concern is not new to this document. Stage 1A's original
`FINDINGS.md` already ran a second test, a paired t-test on log-AUC.
It did this specifically because, in that document's words, "AUC
values span roughly three orders of magnitude." This version 2 (v2)
applies that same, already-used transformation to the re-verification
data. It applies the same level of care that version 1 (v1) used.

**This is planned as the second and final round of aggregation-scale
robustness checking.** If the mean-aggregation result and the
median-aggregation result still disagree after the log-scale check,
the honest conclusion is: "genuinely inconclusive under any tested
aggregation scheme." This document commits in advance to stopping
there. It does not commit to trying further transformations, such as
Box-Cox or rank-based methods, to search for one that resolves the
disagreement.

## Data Source: No New Simulation

**This iteration re-analyzes data the team already collected.** All
770 raw AUC values from v1's run are reused directly. These values
come from `results/stage1a_reverification_analysis.pkl`, or an
equivalent file holding the raw per-instance output. The team runs no
new `joint_tangent_matrix_response` calls. The team uses no new seeds.
The team builds no new graph constructions. Only the aggregation
method applied to the already-fixed numbers changes.

This is an important distinction. This is not "collect more data until
something becomes significant." This is "re-combine the same,
already-fixed data using a different, pre-specified transformation
method."

If the raw per-seed AUC values are not directly available, and only
the v1 class-level aggregates were saved, the driver script must
reload or re-derive the raw values from the existing v1 results file.
The team must check what `run_stage1a_reverification.py` actually
saved before deciding whether any recomputation is truly needed. The
team must flag clearly if anything beyond re-aggregation turns out to
be necessary.

## Scope: 3 of the 4 Original Comparisons

**This iteration excludes T vs. lattice.** Lattice has no seed axis.
This means lattice is a deterministic construction, so aggregation
method cannot matter for it. Version 1 found no disagreement between
aggregation methods for lattice, because none is possible for a
deterministic value. Lattice already passed every check in v1: the
primary test, the sign-flip test, and the bootstrap. Re-running it on
the log scale cannot change that result. The v1 result for T vs.
lattice stands as final.

This iteration covers these 3 comparisons:

1. T vs. historical half-edge random, coupling-budget normalized.
2. T vs. current edge-count-matched random.
3. T vs. degree-preserving rewiring.

The team applies Holm correction across these 3 comparisons, not 4.
This is a different, smaller family of tests than v1 used, because
lattice is not part of this family.

## Transformation and Statistic

For each class c and stochastic control g:

  L_cgs = the natural logarithm of A_cgs (each raw AUC value)
  L_cT = the natural logarithm of A_cT (T has no seed axis, so there
  is one log value per class)

**Within-class log-mean aggregation (primary method):**

  Lbar_cg = (1/25) times the sum of L_cgs for s = 0 through 24

Note: Lbar_cg equals the logarithm of the geometric mean of A_cgs.
This is what makes it robust to a single extreme multiplicative
outlier, in a way that the raw arithmetic mean is not.

**Class-level log-difference:**

  d_cg_log = L_cT minus Lbar_cg

This equals the logarithm of (A_cT divided by the geometric mean of
A_cgs across seeds). This is a log-ratio. A log-ratio is the natural,
symmetric way to compare positive quantities that vary across a
multiplicative scale.

## Primary Test

The team runs an exact two-sided paired Wilcoxon signed-rank test
across the 10 class-level log-differences (d_cg_log, for c = 0 through
9). The team runs this test for each comparison. The team applies Holm
correction across the 3 comparisons in scope.

For each comparison, the team reports:

- All 10 class-level log-differences.
- The median log-difference.
- The Hodges-Lehmann estimate on the log scale, AND the same estimate
  converted back with the exponential function (exp) into a
  multiplicative effect size. For example: "T's AUC is estimated to be
  exp(HL) times the geometric mean of [control]'s AUC." The team
  reports both values. The log value alone is not easy to interpret
  for a reader who has not been following the log transform closely.
- The sign count.
- The W statistic, the exact p-value, and the Holm-corrected p-value.

## Robustness Checks

The team applies these checks in the same order as v1, this time to
the log-transformed data.

### 1. Median Seed Aggregation (Sensitivity Check)

The team uses the within-class median of L_cgs, not of A_cgs, as the
aggregate value. The team repeats the primary test with this value.
The team reports whether the conclusions change from the log-mean
result.

### 2. Exact Class-Level Sign-Flip Test

The team applies all 1024 (2 to the power of 10) possible sign flips
to the 10 d_cg_log values. The team uses a mean-based statistic. This
test is exact at N=10.

### 3. Within-Class MCSE, on the Log Scale

  MCSE_cg_log = the standard deviation of L_cgs across seeds, divided
  by the square root of 25

The team reports this value for each class and comparison. The team
also reports whether it is small compared to the absolute value of
d_cg_log. This is the same check v1 used, applied here in log space.

### 4. Hierarchical Bootstrap

The team resamples classes with replacement. Then, for each selected
class, the team resamples seeds with replacement. The team recomputes
the log-mean class differences and the overall effect on each draw.
The team uses 10,000 bootstrap draws (B = 10,000). The team reports
the 95% confidence interval for the mean log-difference. The team also
reports this interval converted back with the exponential function
into a multiplicative form.

### 5. Mixed Model (Third Priority, Same Rule as v1)

The team fits this model only if the primary test, the median check,
the sign-flip test, and the MCSE gate all agree with each other in log
space. If any of these checks disagree, the team states this
explicitly and does not fit the mixed model, matching what v1 did.

## Decision Rule

For each of the 3 comparisons in scope, the team reports one of two
outcomes:

- **Resolved, consistent**: The primary log-mean Wilcoxon test, the
  median-aggregation sensitivity check, and the MCSE gate all agree.
  Either all three point to a real difference, or all three point to
  no difference. This outcome would mean the log transform did what
  the team expected. The team states the resulting conclusion plainly,
  either "real difference" or "no difference," along with the
  converted-back effect size if a difference is found.
- **Still inconsistent**: The log-scale aggregation does not resolve
  the mean-vs-median disagreement either. The team states this plainly
  as the final answer for that comparison, for this re-verification
  effort. Per this document's advance commitment, the team does not
  propose a third transformation. The team reports this as a genuine
  property of the data. It means this specific perturbation-response
  metric, under this specific design, cannot support a confident
  T-vs-control conclusion for that comparison, at 10 classes and 25
  seeds. It is not a failure of analysis technique.

## What This Design Does Not Do

- It does not re-run any simulation. It does not generate any new
  graph construction. It reuses v1's already-collected raw AUC values
  exactly.
- It does not revisit T vs. lattice. That comparison is closed, per
  v1.
- It does not extend the seed count beyond 25. Extending the seed
  count would be a different, separate design change. It is not
  justified yet, and it is not part of this iteration.
- It does not try transformations other than the natural logarithm, if
  the log transform does not resolve the disagreement.

## Files to Create

```
experiments/stage1a_re_verification/
  DESIGN_v2_log_scale.md      <- this document
  analyze_stage1a_log_scale.py  <- re-aggregation and all tests, reusing v1's raw data
  FINDINGS_v2_log_scale.md     <- filled in after the analysis runs, cross-references v1's FINDINGS.md
```
