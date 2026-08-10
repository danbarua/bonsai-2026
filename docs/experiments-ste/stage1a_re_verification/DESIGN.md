Simplified Technical English version of `experiments/stage1a_re_verification/DESIGN.md`.

# Stage 1A Re-Verification Design Document

The team wrote this document before they wrote any code for this test.
They will commit it to `experiments/stage1a_re_verification/DESIGN.md`.
They will do this after they confirm that the 10-class construction
pipeline works.

## Why This Document Exists

The original Stage 1A design used one random seed for each class. A
seed is a starting value for a random-number generator. The design used
this one seed for each "stochastic control." A stochastic control is a
comparison graph that the code builds using randomness. Two examples
are matched-sparsity random and degree-preserving rewiring.

A pilot test on class 0 checked this design. The pilot test is a
post-hoc robustness note in
`experiments/stage1a_infinitesimal_response/FINDINGS.md`. The pilot
test compared T against random controls. T is the learned topology,
the graph built from real image data (see Notation, below). The
pilot test used a score called AUC (area under the curve). AUC is a
single number for each class and construction. It adds up the
strength of the response over the whole time window.

The pilot test found a problem. The direction of the T-vs-random
comparison is not stable across seeds for class 0.

- Under the historical random control, 7 of 20 seeds gave the
  opposite sign.
- Under the current random control, 2 of 20 seeds gave the opposite
  sign.

This result does not contradict the original conclusion. The
original conclusion was that T and the controls do not differ. A
control this sensitive to seed choice would struggle to give a
stable signal anyway. But the result showed a source of variance
within each class. The original design did not plan for this
variance. The original design did not document this variance either.

This re-verification fixes that gap. It sweeps many graph seeds for
each class. It combines the seeds within each class into one summary
value before running any test. This is called aggregation. The main
test stays the same as in Stage 1A: a paired Wilcoxon signed-rank
test across 10 class-level differences. A Wilcoxon signed-rank test
is a statistical test. It compares paired differences without
assuming the data follow a normal distribution. Keeping this test the
same lets the team compare the new results directly against the old
ones.

## Notation

- c is a KMNIST class. c can be 0 through 9.
- g is a construction. g can be T, rewired, hist_random, curr_random,
  or lattice. See the project glossary for the plain-English meaning
  of T, lattice, rewired, hist_random, and curr_random.
- s is a graph seed. s can be 0 through 24. Seeds apply only to the
  stochastic controls.
- A_cgs is the Stage 1A AUC value for class c, construction g, and
  seed s.
- A_cT is the AUC value for T. T is deterministic. Deterministic means
  it does not use a seed. Each class has one A_cT value.

## Planned Comparisons

The team plans 4 comparisons. They apply Holm correction across all 4
of them together. Holm correction is a way to adjust p-values when you
run more than one test at the same time. It lowers the chance of a
false positive from running several tests.

1. T vs. historical half-edge random, coupling-budget normalized. The
   code uses `generate_historical_matched_sparsity_random` from
   `src/bonsai/dynamics/historical_matched_sparsity_random.py`.
2. T vs. current edge-count-matched random. The code uses
   `generate_matched_sparsity_topology` from
   `src/bonsai/dynamics/matched_sparsity_ablation.py`.
3. T vs. degree-preserving rewiring. The code uses
   `degree_preserving_rewire` from
   `src/bonsai/dynamics/degree_preserving_rewiring.py`.
4. T vs. lattice. The code uses `build_lattice_topology` from
   `src/bonsai/dynamics/lattice_construction.py`.

The team keeps these 4 comparisons separate. The two random controls
are different null models. A null model is a simple baseline used to
test whether a real effect exists. The team must not combine the two
random controls into one pool of seeds.

## Seeds

The team uses 25 independent graph seeds for each class, for each
stochastic control. The seeds are numbered 0 through 24. The team uses
the same 25 seed numbers for every class and every stochastic
construction. Using matching seed numbers makes the work easier to
check. It does not mean the team treats the results as statistically
paired across construction types.

The deterministic controls are T and lattice. Each has one value per
class. The team does not sweep seeds for these two. The team does not
generate 25 identical copies of them either. All 25 copies would be
equal by construction, so this would add no information.

The rewiring control needs a special note. The function
`degree_preserving_rewire` takes a seed parameter, so it is technically
stochastic. But Stage 1A's original design used a fixed seed, seed=1,
for rewiring. This re-verification includes a full 25-seed sweep for
rewiring too. The team does this for two reasons. First, for
consistency with the other stochastic controls. Second, to test
whether the rewiring comparison is also sensitive to seed choice. The
team will report this test explicitly.

## Combining Seeds Within Each Class (Primary Method)

For each stochastic control g and class c, the team computes an
arithmetic mean:

  Abar_cg = (1/25) times the sum of A_cgs for s = 0 through 24

For the deterministic controls (T and lattice), the "aggregate" value
is just the single value the team already has.

**Seed-count stability check (for description only).** After
computing all 25 seeds, the team also computes Abar_cg using only the
first 5, 10, 15, and 20 seeds. This shows how the class-level mean
estimate changes as more seeds are added. This check is for
description only. The team must not use it to stop early. The team
must not use it to choose a seed count based on the result it shows.

**Within-class Monte Carlo standard error (MCSE).** MCSE measures how
much noise comes from using a limited number of seeds, rather than
every possible seed. The team computes it as:

  MCSE_cg = the standard deviation of A_cgs across seeds, divided by
  the square root of 25

The team reports MCSE for every class and stochastic control
combination.

## Class-Level Differences

  d_cg = A_cT minus Abar_cg

The team computes one value of d_cg for each class, for each of the 4
planned comparisons.

## Primary Test

The team runs an exact two-sided paired Wilcoxon signed-rank test
across the 10 class-level differences (one difference per class, c =
0 through 9). The team runs this test separately for each of the 4
planned comparisons. The team then applies Holm correction across the
4 results.

For each comparison, the team reports:

- All 10 class-level differences.
- The median difference.
- The Hodges-Lehmann estimate. This is a robust estimate of the
  typical difference between the two groups being compared.
- The sign count: how many of the 10 differences are positive.
- The Wilcoxon W statistic.
- The exact p-value. The team uses the exact distribution, not a
  normal approximation. N=10 is small enough for an exact calculation.
- The Holm-corrected p-value.

## Robustness Checks

The team runs these checks in the order listed below, not in a
different order.

### 1. Median Seed Aggregation (Sensitivity Check)

The team repeats the primary test. This time it uses the within-class
median instead of the within-class mean. The team reports whether the
conclusions change.

### 2. Exact Class-Level Sign-Flip Test

For each comparison, the team applies all 1024 (2 to the power of 10)
possible sign flips to the 10 d_cg values. A sign-flip test builds a
comparison distribution by randomly flipping the sign of each
difference. The team uses the mean class difference as the test
statistic. This test is exact at N=10. It makes no assumptions about
the shape of the data. The team reports the two-sided p-value.

### 3. Hierarchical Bootstrap

A bootstrap is a method that draws random samples with replacement to
estimate uncertainty. "With replacement" means the same item can be
drawn more than once in a sample. The team resamples in two steps:

1. Draw 10 classes with replacement from the 10 classes (0 through 9).
2. For each selected class, draw 25 seeds with replacement from its 25
   available seeds.

The team recomputes the class means and the overall mean class
difference on each draw. The team uses 10,000 bootstrap draws (B =
10,000). The team reports the 95% confidence interval for the mean
class-level difference. The team also states whether zero falls
inside or outside this interval.

### 4. Mixed Model (Third Priority Only)

The team fits this model only if the first three checks above agree
with each other:

  d_cgs = A_cT minus A_cgs = mu + u_c + e_cgs

Here mu is the fixed effect, u_c is a random effect for class c (it
follows a normal distribution with variance sigma-squared_c), and
e_cgs is the leftover error for each seed. A mixed model is a
statistical model with both a fixed effect and effects that vary
randomly between groups, here between classes. The team reports the
fixed-effect estimate mu and its 95% confidence interval.

The team notes explicitly in the report: 10 classes is a small basis
for estimating how much the class-level random effect varies. So this
check is a third-priority check, not the main analysis.

## Prerequisites

1. The file
   `experiments/stage0_simulator_calibration/results/stage1a_all_classes.pkl`
   must exist. It must contain all 10 KMNIST classes' T, rewired,
   random, and lattice constructions, in the historical format. The
   script `build_all_class_topologies.py` produces this file. This is
   the 10-class pipeline currently being built.
2. All four construction modules must be importable from
   `src/bonsai/dynamics`:
   `historical_matched_sparsity_random.py`,
   `matched_sparsity_ablation.py`, `degree_preserving_rewiring.py`,
   and `lattice_construction.py`.
3. The function `joint_tangent_matrix_response` must be importable
   from `src/bonsai/dynamics/graph_oscillator_field.py`.

## Computational Scope

**Stochastic control constructions:** The team builds 3 stochastic
controls (hist_random, curr_random, rewired), times 25 seeds, times 10
classes. This gives 750 construction runs.

**Deterministic constructions:** T and lattice, 10 classes each. This
gives 20 runs. These 20 runs already exist in the pipeline output.

**AUC computations:** The team has 750 plus 20, or 770, construction
instances in total. For each instance, the team runs Stage 1A's
`joint_tangent_matrix_response` function per class. The team estimates
the runtime per class from Stage 1A's original method. The team does
this before committing to the full run.

**Checkpoint and parallelize:** These runs are embarrassingly
parallel. This means the runs for each (class, construction, seed)
combination do not depend on each other, so many can run at the same
time. The team uses the same SeedSequence pattern that Stage 1B2 used
for its parallel Monte Carlo work. The team saves a checkpoint file
after each batch, in case the run is interrupted.

## Decision Rule

The team will treat the original Stage 1A conclusion as robust only if
all of these agree:

- The primary mean-aggregated Wilcoxon test.
- The median seed aggregation check.
- The exact class-level sign-flip test.
- Reasonable within-class seed variability. This means the MCSE is
  small compared to the class-level differences.

The original Stage 1A conclusion was: no significant difference
between T and the controls.

If the historical random control and the current random control
disagree with each other, on either direction or significance, the
team will report this as a genuine scientific finding. This would mean
the two null models are not interchangeable. The team will not average
over this disagreement. The team will not ignore this disagreement.

## Files to Create

```
experiments/stage1a_re_verification/
  DESIGN.md           <- this document
  run_stage1a_reverification.py   <- driver
  analyze_stage1a_reverification.py  <- analysis and all 4 tests and robustness checks
  FINDINGS.md         <- filled in after the analysis runs
  results/            <- saved data files, not tracked in git
```

## What This Design Does Not Do

- It does not extend Stage 1C. Stage 1C looks at trajectory
  generalization. That is a separate question about Stage 1B2's
  result, not about Stage 1A's result.
- It does not re-run the original Stage 1A computation. It re-runs the
  comparison against the controls, this time with proper seed
  accounting.
- It does not replace the original Stage 1A FINDINGS.md file. This
  document lives in its own experiment folder. It cross-references the
  original findings document.
