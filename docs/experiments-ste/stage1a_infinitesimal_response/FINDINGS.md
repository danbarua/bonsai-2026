Simplified Technical English version of `experiments/stage1a_infinitesimal_response/FINDINGS.md`.

# Diffusion Stage 1A: Tangent Response Check, Fully Verified

## Six Planned Numerical Checks, All Done Before This Run

1. **Joint baseline and tangent integration.** The team combined two ODE
   (differential equation) systems into one. This one system tracks the
   phase values theta(t) and the tangent response together. The tangent
   response is the linearized, first-order approximation of how the
   system responds to a perturbation. The team tracks this response as a
   matrix, Delta(t), with n rows and 3 columns. Each column is for one
   perturbed node. This method cuts the number of separate integrations
   from 120 to 40. There are 10 classes and 4 constructions, so 10 x 4 =
   40. Each integration gives all three node responses from one baseline
   solve.
2. **Fixed-grid AUC evaluation.** AUC means area under the curve. The
   team passed `t_eval` directly to `solve_ivp`. The scipy library uses
   dense output to interpolate values. This gives values at the exact
   planned grid: t = 0 to 2.5, with 51 points. This grid does not depend
   on the solver's own internal step choices. The team never summed AUC
   over the solver's own irregular internal points.
3. **Rotational projection kept.** The team applied the projection
   P = I - (1/n) * ones * ones^T every time it computed S(t) from
   Delta(t). This step removes numerical drift. It keeps the result in
   the zero-mean subspace at the point of measurement.
4. **Max step set to 0.05.** The team set this value directly. This
   stops the error-controlled solver from taking steps larger than the
   resolution used to read the response. The solver can still adapt
   freely within this limit.
5. **Solvers checked against each other.** The team used two solvers:
   RK45 and DOP853. These are different explicit Runge-Kutta solvers,
   from different orders and different families. They agree on the
   earlier problem case. This case was class 5, the rewired construction,
   on the high-degree node. The rewired construction is a graph with the
   same approximate connection pattern as the learned graph, but with
   scrambled edges. The two solvers agree to four decimal places: AUC
   137.2997 versus AUC 137.2998. Both used max_step = 0.05. The team also
   tried the Radau solver. Radau did not finish within the time budget.
   The agreement between RK45 and DOP853 already rules out an artifact
   caused by one specific method, because RK45 and DOP853 are independent
   explicit methods.
6. **Neutral wording used throughout.** The results below use the term
   finite-time infinitesimal perturbation amplification. They do not use
   the words instability, chaos, or basin switching. A time-dependent
   Jacobian (a matrix that shows how a small change in the state grows or
   shrinks over time) can cause growth for a short time along a
   trajectory (one full run of the system from a starting point through
   time). This is expected behavior, not a warning sign, once the team
   confirms it is numerically real (see item 7).

## New Check Against Finite Differences, Using the Real Solver

The team repeated the finite-difference check. A finite-difference check
estimates a response directly from the simulation, without using the
linear (tangent) formula. This new check used the same adaptive solver as
the real analysis: RK45, with the same tolerances and the same max_step.
The team ran this check on two cases: a benign case and the earlier
problem case. The team did this because the earlier check used a
fixed-step solver, and that fixed-step solver was later shown to fail.

| Case | Step | Tangent S(t) | Finite-diff S(t) | Ratio |
|---|---|---|---|---|
| Benign (class 0, T, median node) | 50 | 3.608 | 3.612 | 0.999 |
| Pathological (class 5, rewired, high-degree node) | 50 | 26.924 | 26.560 | 1.014 |

T is the learned topology: the graph learned from a class's own image
population.

The tangent result and the finite-difference result agree within 1.4% in
the worst case, across the whole time window. This closes the validation
loop. The check used the same numerical method that produced the results
below.

## How the Team Grouped and Sampled Results, Stated Clearly

For each class, the team picked three nodes in T: one with low weighted
degree, one with median weighted degree, and one with high weighted
degree. Weighted degree means the total strength of a node's connections.
The team used the same three node identities across all four graph
constructions in a class.

For each class, the team used one shared starting phase vector, not
several. The review asked for this fact to be stated explicitly. The
team lists it below as a limitation. It is not hidden.

The class-level AUC value is the mean of the three node-level AUC
values.

The team kept all ten class topologies. It excluded none of them, even
though some had been used in earlier calibration work. This follows a
correction the review made to the sampling plan. The team used new nodes
and a new starting-condition seed instead of excluding classes.

## Stage 1A Results

Random is a graph with roughly the same edge count as T, but random
connections. Lattice is a graph that connects each pixel only to its
neighbours on the grid, with no learning involved.

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

The team compared T against each control across all ten classes, in
pairs. The main test was the Wilcoxon signed-rank test. The team chose
this test because AUC values are heavily skewed, spanning three orders
of magnitude. As a second check, the team ran a paired t-test on the log
of the AUC values. The team compared three pairs: T vs. rewired, T vs.
random, T vs. lattice. The Bonferroni-corrected significance threshold
for three comparisons is 0.05/3, approximately 0.0167.

| Comparison | Wilcoxon p | Paired-t on log-AUC p | Mean log-ratio (T/control) |
|---|---|---|---|
| T vs. rewired | 0.695 | 0.960 | -0.032 |
| T vs. random | 0.275 | 0.191 | +0.748 |
| T vs. lattice | 0.084 | 0.142 | +0.512 |

None of the three comparisons reach statistical significance. This is
true even before correction for multiple comparisons, so it stays true
after correction too. T vs. lattice comes closest, with Wilcoxon
p = 0.084. Here, T shows somewhat higher average perturbation
persistence, shown by a positive log-ratio. But this result does not
pass the uncorrected 0.05 threshold, with only ten paired classes. Both
statistical methods agree on this conclusion.

## What This Result Shows, and What It Does Not Show

Under this specific test method, the team found no statistically
supported evidence. The learned oscillator-derived topology (T) does not
produce finite-time perturbation dynamics that differ from the matched
graph controls. This statement covers only three specific things: the
infinitesimal-response test method used here, the three matched controls
used here, and the 2.5-unit finite-time window used here. It is not a
general claim about oscillator dynamics.

This finding is not the same as the earlier E and R findings. Do not
describe it as following the same pattern as those. That phrasing would
wrongly suggest all three investigations failed in the same way. In
fact, each investigation ruled out a different hypothesis:

1. **E** tested oscillator-specific spatial organization. The question
   was: did E's value come from class-conditioned support alignment, or
   from generic spatial statistics?
2. **R** tested oscillator-specific graph spectra. The question was: did
   R's value come from the learned graph's frequency structure, or from
   generic image or graph frequency structure?
3. **Stage 1A** tested oscillator-specific first-order propagation
   dynamics. The question was: does the learned topology produce a
   finite-time infinitesimal perturbation response that differs from the
   controls?

The three investigations share a method, not a question. Each one used
rigorous validation before any comparison. Each one reported honestly
what the comparison actually showed. They did not ask the same
substantive question three times, and they did not fail three times in
the same way.

This result does not show that T's dynamics are identical to the
controls' dynamics. It also does not show that oscillator-specific
structure never produces distinctive propagation dynamics. The team had
only n = 10 classes and one starting condition per class. Absence of
evidence is a real, disclosed limitation. It is not proof that T and the
controls behave the same. Because the design uses only one starting
condition per class, each class's result depends on one trajectory
through the system. Stage 0 already showed this system has a
multistable landscape: it can settle into more than one stable pattern.
A different starting condition could, in principle, land in a different
stable pattern with different short-term behavior.

## Why the Null Result Fits the Raw Numbers

The class-level table shows real, large differences between classes.
This is not a case where everything looks the same and the statistics
just confirm that.

For example, in class 4: T=350, rewired=459, random=16, lattice=346.
Here, T and lattice are close, rewired is highest, and random is about
ten times lower than the rest.

In class 3: T=71, rewired=8, random=8, lattice=26. Here, T is clearly
the highest.

In class 9: T=75, random=69, rewired=3. Here, T and random are close,
and rewired is far lower.

There is plenty of variation in this table. But this variation does not
line up consistently in one direction between the learned topology and
the controls. T is sometimes the largest value. T sometimes tracks one
particular control closely. T sometimes sits in the middle. This
inconsistency, not a lack of variation, is why the paired analysis finds
no systematic effect. This point is worth stating clearly. The null
result does not contradict a table that clearly shows real structure.

## A Pattern Emerging Across Investigations

Across E, R, and now Stage 1A, the project keeps finding the same
pattern. Interesting local behavior does not add up to a reproducible,
population-level advantage for the learned, oscillator-derived
construction over carefully matched controls. Oscillator systems clearly
produce rich dynamics. They show multistability, short-term
amplification, and large class-to-class variation. But across three
separate hypotheses now, the project has not found evidence that the
learned dynamics hold a special computational advantage over matched
alternatives. This is a sharper, more specific finding than simply
saying "three negative results." It may be one of the project's most
durable findings so far.

## Stage 1A Status

**Completed.**

- Main question: does the learned topology produce distinctive
  first-order perturbation dynamics?
- Answer: no statistically supported evidence, under the method tested
  here.
- Strength of the evidence: high. The team refined the method carefully
  before drawing any conclusion. This included: a validated simulator,
  adaptive integration, agreement between independent solvers, a
  tangent-response check against finite differences, coupling
  normalization, paired graph comparisons, and correction for multiple
  comparisons.
- Treat this as a genuine negative finding. It is not an exploratory
  null result that just needs a larger sample.

The team does not recommend extending Stage 1A with more starting
conditions, more classes, or more nodes. The class-level table already
shows genuine dynamical diversity. More sampling would only narrow the
confidence interval around the same estimate. It is unlikely to change
the qualitative finding. The inconsistency in which construction "wins"
per class does not look like an artifact of a small sample size.

A more valuable next question is a different kind of question, not more
data on this one. The next question is whether finite perturbations can
cause qualitatively different transformations. This is Stage 1B's
question, not whether an infinitesimal perturbation persists slightly
longer on average.

This revises an earlier suggestion in this document's next-steps section
below. That section originally suggested multiple starting conditions as
the priority. That suggestion was reasonable when based only on the
single-starting-condition limitation. It does not hold up once the team
also considers that the raw data already shows real, though
inconsistent, diversity.

## Known Limitations

- The team used one shared starting condition per class, not several.
  The review flagged this as a real limitation. Resolving it would
  meaningfully improve the estimate. The team did not resolve it in this
  run, due to compute limits. The team discloses this limitation here
  rather than leaving it unstated.
- Ten class topologies is a small sample at the highest level of
  inference. The paired tests here have limited power. They can detect
  only a fairly large, consistent effect.
- The Radau solver, used as a stiff-solver cross-check, did not finish
  within the time budget. The agreement between RK45 and DOP853 is
  offered as sufficient evidence against a method-specific artifact. But
  a stiff-method comparison was not independently completed.
- AUC values span roughly three orders of magnitude across classes and
  constructions. The team used a log transform for the secondary test.
  This is a reasonable choice, but not the only possible one. The team
  relies on the qualitative non-significance conclusion here, not on the
  exact p-values.

## Next Steps

Stage 1A is closed. The team will not extend it. As explained above,
adding more starting conditions, classes, or nodes would only narrow the
confidence interval around the same estimate. It would not change the
qualitative finding. This is because the class-level table already shows
genuine, though inconsistent, dynamical diversity.

The next question is Stage 1B. Stage 1B asks about finite-amplitude
nonlinear response: can finite perturbations cause qualitatively
different transformations? Examples include transient amplification,
basin switching (jumping to a different stable pattern), and terminal
divergence. The team frames this explicitly as a question about
nonlinear sensitivity. It is not simply an extension of the
linear-propagation question that Stage 1A answered. This moves beyond
tangent dynamics into the genuinely nonlinear behavior that Stage 0's
multistability finding already hinted at. It is a different scientific
question, not a follow-up measurement on this same one.

## How to Reproduce These Results

The team extended `graph_oscillator_field.py` with two functions:
`joint_tangent_matrix_response` (the main function) and
`adaptive_finite_difference_response` (used for revalidation). Both use
`scipy.integrate.solve_ivp`.

The team built fresh constructions for all ten classes: T,
degree-preserving rewired, matched-sparsity random, and lattice. These
used seed=1, which is different from the calibration panel's seed=0. The
team normalized each construction to the same mean weighted degree per
class. The team saved these constructions in `stage1a_all_classes.pkl`.
The verified results are saved in `stage1a_results_verified_p{1,2}.pkl`.

## Follow-up Robustness Check

The team ran a follow-up pilot on class 0 only. The team chose not to
extend this pilot to the other nine classes. The pilot tested how
sensitive the T-vs-random comparison is to the random construction's own
randomness. The pilot used this document's exact method:
`joint_tangent_matrix_response`, t=[0,2.5] on a 51-point grid,
low/median/high-degree nodes, and class-level AUC as the mean of the
three node-level AUCs. All constructions were rescaled to T's own mean
weighted degree. The pilot used a fresh starting condition. This is not
the same starting condition as this document's original run, which was
not recorded.

**Fixed constructions do not change across seeds, by design.** Their AUC
values were: T_AUC = 1.83, rewired_AUC = 5.12, lattice_AUC = 3.03.

**The team swept 20 seeds for random_AUC, under both available
definitions of the random control.** Under the historical half-edge
random control (coupling-budget normalized reconstruction), random_AUC
ranged from 0.59 to 198.21, with CV (coefficient of variation) = 2.37.
Under the current edge-count-matched random control, random_AUC ranged
from 1.13 to 41.56, with CV = 1.08.

The sign of log(T/random) shows which of T or random has higher
perturbation persistence. This sign flipped in 7 of 20 seeds under the
historical control. It flipped in 2 of 20 seeds under the current
control.

**This document's original design used exactly one random-construction
seed per class.** This pilot shows that single draw was not a stable
estimate for class 0. A different draw could plausibly have produced the
opposite direction, by one to two orders of magnitude.

**This finding is consistent with this document's original null
conclusion. It reinforces that conclusion. It does not contradict it.**
A control this sensitive to its seed would struggle to give a robust
combined signal, even if classes agreed with each other. The class-level
table above already shows that classes do not agree with each other.
This pilot adds a new, previously undocumented layer: noise within a
single class, underneath the across-class inconsistency this document
already reports.

**What this pilot does not do:** it does not re-run the original
10-class Wilcoxon test above. It cannot directly confirm or disprove
this document's conclusion. It is a robustness observation about one
class's contribution to that test. It is not a replacement for that
test.

**What a properly powered re-verification would require:** multiple
random-construction seeds per class, with explicit aggregation across
seeds. Simply extending the current single-seed-per-class design to all
ten classes would not be enough. That approach would still leave the
within-class instability found here unaddressed. This is a design
question, not a question of building more of the same. The team has
deferred this question, pending a decision on whether it is worth
pursuing.
