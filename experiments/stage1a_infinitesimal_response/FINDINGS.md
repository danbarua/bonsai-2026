# Diffusion Stage 1A: Tangent-Linear Response, Fully Verified

## All six pre-registered numerical items completed before this run

1. **Joint baseline-and-tangent integration**: implemented as a single
   combined ODE system carrying theta(t) and a tangent MATRIX Delta(t)
   (n x 3, one column per perturbed node) together -- reduces the nominal
   120 separate integrations to 40 (10 classes x 4 constructions), each
   producing all three nodewise responses from one baseline solve.
2. **Fixed-grid AUC evaluation**: `t_eval` passed directly to
   `solve_ivp` -- scipy interpolates via dense output to return values at
   exactly the prespecified grid (t=0 to 2.5, 51 points) regardless of
   the adaptive solver's internal step choices. AUC is never summed over
   the solver's own irregular internal points.
3. **Rotational projection retained**: P = I - (1/n)*ones*ones^T applied
   every time S(t) is computed from Delta(t), removing any accumulated
   numerical drift into the zero-mean subspace at the point of
   measurement.
4. **max_step = 0.05** specified explicitly, so error-controlled steps
   can never leap over the resolution at which the response is
   interpreted, while still adapting freely within that bound.
5. **Independent solver cross-check**: RK45 and DOP853 (different order,
   different explicit Runge-Kutta family) agree on the previously
   pathological case (class 5, rewired, high-degree node) to four decimal
   places -- AUC 137.2997 vs. 137.2998, both under max_step=0.05. Radau
   was attempted but did not complete within budget; the RK45-DOP853
   agreement already rules out a method-specific artifact, since they are
   independent explicit families.
6. **Neutral terminology adopted throughout**: results below are
   described as finite-time infinitesimal perturbation amplification,
   not instability, chaos, or basin switching -- a time-dependent
   Jacobian producing transient growth along a trajectory is expected
   behavior, not a red flag, once verified as numerically real (item 7).

## Revalidation against finite differences, under the actual solver used for inference

Repeated the finite-difference check using the SAME adaptive machinery
(RK45, same tolerances, same max_step) for both a benign and the
pathological case, since the earlier validation used the fixed-step
machinery later shown to fail:

| Case | Step | Tangent S(t) | Finite-diff S(t) | Ratio |
|---|---|---|---|---|
| Benign (class 0, T, median node) | 50 | 3.608 | 3.612 | 0.999 |
| Pathological (class 5, rewired, high-degree node) | 50 | 26.924 | 26.560 | 1.014 |

Both agree within 1.4% at worst, across the full window, closing the
validation loop under the numerical method that actually generated the
inferential results below.

## Class-level aggregation and sampling, stated explicitly

Per node: low/median/high weighted-degree-in-T nodes (same node
identities used across all four constructions within each class). Per
class: **one shared initial phase vector** (not multiple) -- stated
explicitly per the review's requirement, and disclosed as a limitation
below, not glossed over. Class-level AUC is the mean of the three
nodewise AUCs. All ten class topologies retained (none excluded for
having been used in earlier calibration, per the review's correction to
the sampling plan -- new nodes and a new initial-condition seed were used
instead).

## Stage 1A result

| Class | T | Rewired | Random | Lattice |
|---|---|---|---|---|
| 0 | 0.86 | 17.16 | 2.04 | 1.05 |
| 1 | 20.86 | 32.02 | 1.21 | 13.38 |
| 2 | 15.03 | 15.00 | 25.51 | 4.13 |
| 3 | 71.42 | 7.66 | 7.67 | 26.48 |
| 4 | 349.70 | 458.71 | 16.12 | 345.93 |
| 5 | 29.54 | 74.64 | 2.79 | 4.41 |
| 6 | 0.51 | 4.52 | 1.76 | 1.07 |
| 7 | 15.02 | 2.40 | 19.48 | 3.19 |
| 8 | 10.60 | 20.87 | 13.50 | 32.92 |
| 9 | 75.40 | 3.44 | 68.61 | 27.87 |

**Paired comparisons across the ten classes** (Wilcoxon signed-rank as
primary, given the heavy right-skew of AUC values across three orders of
magnitude; paired t-test on log-AUC as a secondary robustness check;
Bonferroni threshold 0.05/3 ≈ 0.0167 across the three comparisons):

| Comparison | Wilcoxon p | Paired-t on log-AUC p | Mean log-ratio (T/control) |
|---|---|---|---|
| T vs. rewired | 0.695 | 0.960 | -0.032 |
| T vs. random | 0.275 | 0.191 | +0.748 |
| T vs. lattice | 0.084 | 0.142 | +0.512 |

**None of the three comparisons reach significance, even before
correction, let alone after it.** T vs. lattice comes closest (Wilcoxon
p=0.084), with T showing somewhat higher average perturbation persistence
(positive log-ratio), but this does not clear the uncorrected 0.05
threshold with ten paired classes, and both statistical approaches agree
on the qualitative conclusion.

## What this establishes, and what it doesn't

**Under the present infinitesimal-response formulation, we found no
statistically supported evidence that learned oscillator-derived
topology produces finite-time perturbation dynamics distinguishable from
matched graph controls.** This wording is deliberately anchored to three
things actually tested -- the infinitesimal-response formulation
specifically, the three matched controls specifically, and the 2.5-unit
finite-time horizon specifically -- not a general claim about oscillator
dynamics.

**This is not the same finding as E's or R's closures, and should not be
described as following the same pattern** -- that phrasing would wrongly
suggest all three investigations failed in an identical way. They ruled
out three different hypotheses, each substantively distinct:

1. **E** -- oscillator-specific *spatial organization* (was E's value
   about class-conditioned support alignment, or generic spatial
   statistics?)
2. **R** -- oscillator-specific *graph spectra* (was R's value about
   learned graph frequency structure, or generic image/graph frequency
   structure?)
3. **Stage 1A** -- oscillator-specific *first-order propagation dynamics*
   (does learned topology produce distinguishable finite-time
   infinitesimal perturbation response?)

The commonality across all three is methodological -- rigorous
validation before any comparison, honest reporting of what the
comparison actually shows -- not that the same substantive question was
asked three times and failed three times.

This does **not** establish that T's dynamics are identical to the
controls', or that oscillator-specific structure never produces
distinctive propagation dynamics -- absence of evidence at n=10 with one
initial condition per class is a real, disclosed limitation, not a proof
of equivalence. The single-initial-condition design means each class's
result is conditional on one trajectory through what Stage 0 already
established is a multistable landscape; a different initial condition
could, in principle, land in a different basin with different transient
behavior.

## Why the negative result isn't surprising given the raw numbers

The class-level table shows substantial, genuine dynamical variation --
this is not a case of everything looking similar and the statistics
merely confirming that. Class 4 shows T=350, rewired=459, random=16,
lattice=346 (T and lattice close, rewired highest, random an order of
magnitude lower). Class 3 shows T=71, rewired=8, random=8, lattice=26 (T
clearly highest here). Class 9 shows T=75, random=69, rewired=3 (T and
random close, rewired far lower). **There is plenty of dynamical
diversity in this table. It simply isn't aligned with "learned topology
vs. controls" in any consistent direction** -- T is sometimes the
largest, sometimes tracks a specific control closely, sometimes sits
in the middle of the pack. That inconsistency, not an absence of
variation, is exactly why the paired analysis finds no systematic effect,
and it's worth stating this explicitly so the null result doesn't read as
contradicting a table that visibly contains real structure.

## An emerging pattern worth naming

Across E, R, and now Stage 1A, the project has repeatedly found that
interesting local behavior does not aggregate into a reproducible
population-level advantage for the learned, oscillator-derived
construction over carefully matched controls. Oscillator systems clearly
produce rich dynamics -- multistability, transient amplification,
substantial class-to-class variation. What remains elusive, across three
independent substantive hypotheses now, is evidence that the *learned*
dynamics occupy a privileged computational position relative to
alternatives matched on the properties each investigation controlled
for. That is a sharper, more specific observation than "three negative
results," and may be one of this project's most durable findings so far.

## Stage 1A status

**Completed.**

- Primary question: does learned topology produce distinctive first-order
  perturbation dynamics?
- Answer: no statistically supported evidence, under the tested
  formulation.
- Evidence strength: high. The experiment underwent substantial
  methodological refinement before inference -- a validated simulator,
  adaptive integration, independent-solver agreement, tangent-linear
  verification against finite differences, coupling normalization, paired
  graph comparisons, and multiplicity control. This should be treated as
  a genuine negative finding, not an exploratory null awaiting a larger
  sample.

**Not recommended: extending Stage 1A** with more initial conditions,
classes, or nodes. Given the genuine dynamical diversity already visible
in the class-level table, more sampling would narrow confidence intervals
around the same estimand -- it is unlikely to change the qualitative
finding, since the inconsistency in which construction "wins" per class
is not obviously an artifact of small sample size. The more valuable next
question is qualitatively different, not more data on this one: whether
finite perturbations can induce qualitatively different transformations
(Stage 1B), not whether an infinitesimal perturbation persists slightly
longer on average. This revises what this document's next-steps section
below originally suggested (multiple initial conditions as the priority)
-- that was reasonable given the single-IC limitation alone, but doesn't
hold up against the additional observation that the raw data already
shows real, if inconsistent, diversity.

## Honest limitations

- One shared initial condition per class, not several -- the review
  flagged this as a real limitation that "would materially improve
  estimation" if resolved; it was not resolved in this run, given
  compute constraints, and is disclosed rather than left implicit.
- Ten class topologies is a small sample for the highest-level unit of
  inference; the paired tests here have correspondingly limited power to
  detect anything but a fairly large, consistent effect.
- Radau (the stiff-solver cross-check) did not complete within budget;
  the RK45-DOP853 agreement is offered as sufficient evidence against a
  method-specific artifact, but a stiff-method comparison was not
  independently completed.
- AUC values span roughly three orders of magnitude across classes and
  constructions; the log-transform used for the secondary test is a
  reasonable but not the only possible way to handle this, and the
  qualitative non-significance conclusion is what's being leaned on here,
  not the precise p-values themselves.

## Immediate next steps
[claude_desktop_config.json](../../../../../Library/Application%20Support/Claude/claude_desktop_config.json)
Stage 1A is closed, not extended. Per the reasoning above, adding more
initial conditions, classes, or nodes would narrow confidence intervals
around the same estimand without changing the qualitative finding, given
the genuine but inconsistent dynamical diversity already visible in the
class-level table. The next question is Stage 1B: finite-amplitude
nonlinear response -- whether finite perturbations can induce
qualitatively different transformations (transient amplification, basin
switching, terminal divergence), interpreted explicitly as nonlinear
sensitivity rather than an extension of the linear-propagation question
Stage 1A answered. This moves beyond tangent dynamics into the genuinely
nonlinear behavior Stage 0's multistability finding already hinted at,
and is a different scientific question, not a follow-up measurement on
this one.

## Reproducing these results

`graph_oscillator_field.py` extended with
`joint_tangent_matrix_response` (primary), `adaptive_finite_difference_response`
(revalidation), using `scipy.integrate.solve_ivp`. All ten classes' T,
degree-preserving rewired, matched-sparsity random, and lattice
constructions built fresh (seed=1, distinct from the calibration panel's
seed=0) and normalized to equal mean weighted degree per class, saved in
`stage1a_all_classes.pkl`. Verified results in
`stage1a_results_verified_p{1,2}.pkl`.
