# Stage 1A Re-Verification: Log-Scale Iteration (v2)

*Committed and locked before running any log-scale analysis. This
document specifies what will be computed and how conclusions will be
drawn, before those numbers exist. Do not revise this document's
decision rule after seeing log-scale results.*

## Why this iteration exists

`DESIGN.md` (v1)'s primary analysis, run on the raw AUC scale as
specified, found 2 of 4 Holm-corrected comparisons nominally significant
(T vs. historical random, T vs. rewiring), but this significance
collapsed entirely under median aggregation and failed the design's own
MCSE gate. The diagnostic cause is now established, not speculative:
AUC is heavy-tailed enough that a single extreme seed draw can more than
double a 20-seed-converged class-level mean (documented concretely:
historical-random class 2's running mean 40.0 at k=20 to 658.6 at k=25).
Arithmetic mean is not robust to this; arithmetic mean of the *logarithm*
(equivalently, the geometric mean of the raw values) is.

This is not a new concern invented for this document. Stage 1A's
original `FINDINGS.md` already ran a secondary paired t-test on log-AUC,
specifically because "AUC values span roughly three orders of
magnitude." v2 applies that same, already-precedented transformation to
this re-verification's data, with the same rigor v1 used.

**This is planned as the second and final iteration of aggregation-scale
robustness checking.** If disagreement between mean and median
aggregation persists under the log scale, the honest conclusion is
"genuinely inconclusive under any tested aggregation scheme" -- this
document pre-commits to stopping there, not to trying further
transformations (Box-Cox, rank-based, or otherwise) in search of one
that resolves the disagreement.

---

## Data source: no new simulation

**This iteration re-analyzes already-collected data.** All 770 raw AUC
values from v1's run (`results/stage1a_reverification_analysis.pkl` or
equivalent raw per-instance output) are reused directly. No new
`joint_tangent_matrix_response` calls, no new seeds, no new graph
constructions. Only the aggregation function applied to already-fixed
numbers changes. This is an important methodological distinction: this
is not "collect more data until something is significant," it is
"re-aggregate identical, already-fixed data under a differently
justified, pre-specified transform."

If the raw per-seed AUC values are not directly available (only the v1
class-level aggregates were saved), the driver must re-load or
re-derive them from the existing v1 results file -- check what
`run_stage1a_reverification.py` actually persisted before deciding
whether any recomputation is genuinely required. Flag explicitly if
anything beyond re-aggregation turns out to be necessary.

---

## Scope: 3 of the 4 original comparisons, not 4

**T vs. lattice is excluded from this iteration.** Lattice has no seed
axis (deterministic construction) -- v1 found no aggregation-method
disagreement for it because none is possible, and it already passed
every check (primary, sign-flip, bootstrap) consistently. Re-running it
in log-scale cannot change that; the v1 result for T-vs-lattice stands
as final.

This iteration covers:
1. T vs. historical half-edge random, coupling-budget normalized
2. T vs. current edge-count-matched random
3. T vs. degree-preserving rewiring

Holm correction applied across these 3, not 4 (a different family size
than v1, since lattice is not part of this family of tests).

---

## Transformation and statistic

For each class c and stochastic control g:

  L_cgs = log(A_cgs)          [natural log of each raw AUC value]
  L_cT  = log(A_cT)           [T has no seed axis; one log value per class]

**Within-class log-mean aggregation (primary):**

  L̄_cg = (1/25) * Σ_{s=0..24} L_cgs

Note L̄_cg = log(geometric mean of A_cgs) -- this is what makes it
robust to a single extreme multiplicative outlier in a way the raw
arithmetic mean is not.

**Class-level log-difference:**

  d_cg^log = L_cT - L̄_cg

This equals log(A_cT / geometric_mean_s(A_cgs)) -- a log-ratio, the
natural symmetric comparison for positive, multiplicative-scale
quantities.

---

## Primary test

Exact two-sided paired Wilcoxon signed-rank test across the 10
class-level log-differences {d_cg^log : c=0..9}, per comparison, Holm
correction across the 3 comparisons in scope.

Report, per comparison:
- All 10 class-level log-differences
- Median log-difference
- Hodges-Lehmann estimate on the log scale, AND back-transformed via
  exp() to a multiplicative effect size (e.g. "T's AUC is estimated to
  be exp(HL) times the geometric mean of [control]'s AUC") -- report
  both; the log value alone is not interpretable to a reader who hasn't
  been tracking the log transform closely.
- Sign count
- W statistic, exact p-value, Holm-corrected p-value

---

## Robustness checks, same order as v1, applied to log-transformed data

### 1. Median seed aggregation (sensitivity analysis)

Within-class median of L_cgs (not of A_cgs) as the aggregate, repeat the
primary test. Report whether conclusions change from log-mean.

### 2. Exact class-level sign-flip test

All 2^10 = 1024 sign flips of the 10 d_cg^log values, mean-based
statistic, exact at N=10.

### 3. Within-class MCSE, on the log scale

  MCSE_cg^log = SD_s(L_cgs) / sqrt(25)

Report per class per comparison, and whether it is small relative to
|d_cg^log| -- same gate as v1, applied in log space.

### 4. Hierarchical bootstrap

Resample classes with replacement, then seeds within each selected class
with replacement, recompute log-mean class differences and overall
effect, B=10,000. Report 95% CI for the mean log-difference AND its
back-transformed (exponentiated) multiplicative form.

### 5. Mixed model (tertiary, same gating logic as v1)

Only fit if primary + median + sign-flip + MCSE-gate are mutually
consistent in log space. If gated out, state that explicitly, as v1 did.

---

## Decision rule

For each of the 3 comparisons in scope, report one of:

- **Resolved, consistent**: primary log-mean Wilcoxon, median-aggregation
  sensitivity check, and MCSE gate all agree (either all point to a real
  difference, or all point to no difference). This would mean the log
  transform did what it was expected to do -- state the resulting
  conclusion (real difference / no difference) plainly, with the
  back-transformed effect size if a difference is found.
- **Still inconsistent**: log-scale aggregation does not resolve the
  mean-vs-median disagreement either. State this plainly as the final
  answer for that comparison under this re-verification effort -- per
  this document's pre-commitment, do not propose a third transformation.
  Report this as a genuine property of the data (this specific
  perturbation-response metric, under this specific design, cannot
  support a confident T-vs-control conclusion for this comparison at
  n=10 classes / 25 seeds), not as a failure of analysis technique.

---

## What this does not do

- Does not re-run any simulation or generate any new graph construction
  -- reuses v1's already-collected raw AUC values exactly.
- Does not revisit T-vs-lattice -- that comparison is closed, per v1.
- Does not extend the seed count beyond 25 -- that would be a different,
  separate, and not-yet-justified design change, not part of this
  iteration.
- Does not try transformations other than natural log if this doesn't
  resolve the disagreement.

## Files to create

```
experiments/stage1a_re_verification/
  DESIGN_v2_log_scale.md      <- this document
  analyze_stage1a_log_scale.py  <- re-aggregation + all tests, reusing v1's raw data
  FINDINGS_v2_log_scale.md     <- populated after analysis, cross-references v1's FINDINGS.md
```