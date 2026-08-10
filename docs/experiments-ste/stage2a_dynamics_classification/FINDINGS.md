Simplified Technical English version of `experiments/stage2a_dynamics_classification/FINDINGS.md`

# Stage 2A: Feasibility Stage 1

**Status: this is a mechanical check only, per `DESIGN.md`'s own explicit
framing. This is NOT a scientific result and must not be read as one.**
Its job is to confirm the pipeline runs correctly end to end at a small
scale, before scaling up. The direction of the effect is recorded below
because `DESIGN.md` allows it, but nothing here is confirmatory.

## Scope

1,000 official KMNIST training images (100 per class, with each class
balanced, `SEED=42`), drawn deterministically, only from the official
training split — the official test set was never loaded at this stage.
Three conditions, per `DESIGN.md`'s locked pipeline:

1. Raw pixels (784-dimensional).
2. Encoded, pre-graph-evolution (`_local_converged_phases`, restricted
   to T's 505 active nodes, reference-node centering, 1008-dimensional).
3. Evolved on T (`T_HORIZON=2.5`, same feature method, 1008-dimensional).

## Go/no-go mechanical checks (all passed)

- **Solver failures: 0 of 1000 (0.0%)**, well under the locked 0.1%
  tolerance. Every graph-evolution ODE solve succeeded on its first
  attempt (`RK45`, `max_step=0.05`) — the recovery policy (a smaller
  `max_step`, then `Radau`) was never needed.
- **Non-finite feature values: 0.** Every raw-pixel, pre-evolution, and
  evolved-T feature vector was fully finite.
- **The `R(theta)` check** (order parameter, before and after evolution,
  reported per `DESIGN.md` — never used to change the feature method):
  - Pre-evolution: minimum 0.403, maximum 0.979, mean 0.735, median
    0.743. 0 images below 0.01, 0 above 0.99.
  - Post-evolution: minimum 0.616, maximum 0.992, mean 0.859, median
    0.869. 0 images below 0.01, **1 of 1000 above 0.99** (maximum
    0.9922). This is a single borderline case, not a mass of values at
    the numerical limit — disclosed as required, not flagged as a
    problem.
  - Evolution consistently increases `R` (mean 0.735 to 0.859) across
    this sample — this is plausible, since T's coupling drives phases
    toward local synchrony. It is reported as an observation, not
    interpreted further at this stage.
- **Reference-node feature check**: the two trivially-constant columns
  (`theta_ref`'s own `cos=1, sin=0`) were confirmed dropped for every
  image, using the check built into `reference_node_features()` (also
  covered by `tests/test_stage2a_core.py`) — the effective feature count
  is 1008 throughout, as designed.
- **Classifier convergence**: all three conditions converged in every
  fold and `C` combination of the locked 5-fold cross-validation by
  9-value grid (45 fits per condition, 135 total) — the
  `NonConvergenceError` stop rule was never triggered.

**Overall: GO.** The pipeline runs correctly end to end at this scale.
Nothing here blocks moving to feasibility stage 2.

## The direction of the effect (descriptive only, not a formal result)

Mean 5-fold cross-validated log-loss at each condition's own
cross-validation-selected `C` (each condition selects its own `C`,
independently, per the locked procedure):

| condition | dim | selected C | mean CV log-loss at selected C |
|---|---:|---:|---:|
| raw pixels | 784 | 0.01 | 0.804 |
| encoded, pre-evolution | 1008 | 0.01 | 0.843 |
| evolved on T | 1008 | 0.01 | 0.804 |

At this sample size (1,000 images, 100 per class, with no held-out test
set — this is a training-side cross-validation number, not the locked
primary result), evolved-T's mean cross-validated log-loss is nominally
equal to raw pixels' log-loss, and nominally *lower* than the
pre-evolution condition's log-loss. **This is not evidence of anything.**
`DESIGN.md` states directly that the feasibility ladder may report raw
differences descriptively, with no formal statistical test, and 1,000
images with no dedicated held-out set is far too small, and
methodologically different from the locked confirmatory test (a
20,000-resample paired bootstrap against 10,000 official test images),
to support any claim in either direction. This is recorded here only
because the design permits recording it, not because it means anything
yet.

## Runtime

The 1,000-image pipeline (encode plus restrict plus evolve plus both
feature-method computations, run in parallel across `cpu_count()-1`
worker processes): **65.7s (65.7 ms/image)**. Classifier fitting (3
conditions x 45 fold-and-C fits each): a few seconds, not separately
timed. Extending this linearly to stage 2 (5,000 images) and stage 3
(60,000 images) suggests roughly 5.5 minutes and 66 minutes respectively
for the pipeline stage alone — this is a rough projection to revisit
once stage 2's own measurement exists, not a locked estimate.

## Code

`stage2a_core.py` (encoding, restriction, evolution, and feature
computation), `stage2a_classifier.py` (the locked cross-validation,
standardization, and classifier procedure), `run_feasibility_stage1.py`
(this stage's driver script), `tests/test_stage2a_core.py` (checks the
reference-node's constant-column property and the evolution
recovery-policy order, Tier 1 and Tier 2). The raw results
(`results/stage1_feasibility_results.pkl`) are gitignored, and you can
regenerate them by re-running `run_feasibility_stage1.py`.

## Next step

Feasibility stage 2 (up to 5,000 official-training images: measures
throughput, and runs the encoder-seed robustness check), per
`DESIGN.md`'s locked ladder — see below.

# Stage 2A: Feasibility Stage 2

**Status: complete, after halting once because of a real, investigated,
and resolved non-convergence.** The stage first halted, per `DESIGN.md`'s
own locked stop rule ("any non-converged fit during a required fold and
C combination stops advancement to the next stage, pending
investigation"). The team diagnosed the cause (below), raised `max_iter`
from 1,000 to 10,000 as a disclosed amendment made after locking, and
re-ran the full stage (the primary cross-validation, plus the
encoder-seed robustness check) to completion with no non-convergences
anywhere. This is still a mechanical check only, per `DESIGN.md`'s own
framing — not a scientific result, the same as stage 1.

## What ran before halting (first attempt)

5,000 official KMNIST training images (500 per class, class-balanced,
`SEED=42`) — this is also "the fixed training-derived validation
subset" that `DESIGN.md` specifies is reused for every later development
decision and for the encoder-seed robustness check. The official test
set was never loaded.

**Throughput**: 5,000 images in 328.3s (65.7 ms/image) — the identical
per-image rate to stage 1 (also 65.7 ms/image), confirming linear
scaling, with no unexpected overhead, at 5 times the image count. The
extended stage-3 (60,000-image) pipeline runtime projection:
**3,939s (65.7 minutes)** — this projection must be explicitly approved
before stage 3 launches, per `DESIGN.md`'s locked go/no-go rules; it is
reported here, but not yet approved.

**Go/no-go mechanical checks, at 5,000-image scale**:
- Solver failures: 0 of 5000 (0.0%). The recovery policy was never
  needed.
- Non-finite feature values: 0.
- `R(theta)`: pre-evolution minimum 0.329, maximum 0.984, mean 0.731,
  median 0.737, 0 below 0.01, 0 above 0.99. Post-evolution minimum
  0.532, maximum 0.995, mean 0.857, median 0.866, 0 below 0.01, **4 of
  5000 above 0.99**. This is proportionally consistent with stage 1's
  single boundary case (1 of 1000) — still a small tail, not a mass of
  values, disclosed as required.

## The non-convergence, found and characterized

**The primary cross-validation fitting (seed=0) hit a real
non-convergence**: the `evolved_T` condition (1008-dimensional, n=5000)
failed to converge at `fold=0, C=100.0` (using all 1000 of 1000
iterations, `lbfgs` did not reach `tol=1e-4`). Per the locked stop rule,
this **halted the stage immediately** — the script did not proceed to
the encoder-seed robustness check. (An orchestration bug was found and
fixed in the same session: the driver script initially caught this
error per condition and *did* continue on to the robustness check
regardless, which contradicted `DESIGN.md`'s "stops advancement to the
next stage" rule — this was fixed before the result was reported, not
after.)

**A diagnostic scan (not part of the locked pipeline — its only purpose
is to characterize this failure before deciding how to respond)**: every
one of the 45 (fold, `C`) combinations for all three conditions was fit
without stopping at the first failure. Result:

| condition | non-convergent (fold, C) pairs out of 45 |
|---|---|
| raw pixels | 0 |
| encoded, pre-evolution | 0 |
| evolved on T | **15** -- all 5 folds, at exactly `C in {100, 1000, 10000}` |

**The pattern is clean and unusual**: only `evolved_T` is affected, and
only at the three weakest-regularization values in the 9-value grid (the
top third). Every fold fails at exactly the same three `C` values, and
no fold fails at `C <= 10`. At every `C` value observed so far, across
both stages 1 and 2, the cross-validation-selected `C` has always been
**0.01** for every condition, with mean validation log-loss rising
sharply and steadily for `C >= 1` — the non-convergent region is nowhere
near where `C` would actually ever be selected.

**This does not resolve the stop rule on its own.** `DESIGN.md`'s rule
is deliberately strict, regardless of whether the failing `C` would ever
have been selected, specifically so that "it doesn't matter, that C was
never going to be picked anyway" cannot become a silent, undisclosed
excuse for continuing. Whether to raise `max_iter` for this condition
(or this part of the grid), narrow the grid, or handle it some other way
is a locked-design-parameter decision, not something to change on the
spot mid-implementation — this was left for a separate decision, not
resolved here.

## Why evolved_T specifically fails to converge: ill-conditioning, not separability

A follow-up check (`diagnose_stage2_convergence_hypotheses.py`, also
diagnostic-only) tested two candidate explanations directly, on the
exact failing case (evolved_T, fold=0, C=100, the same 4000-image
training portion the original failure came from), against the other two
conditions at the same fold and `C` values:

| condition | train acc @ C=100 | converged @ C=100 (n_iter) | train logloss @ C=100 | \|\|coef\|\| @ C=0.01 -> C=100 | condition number |
|---|---:|---|---:|---:|---:|
| raw pixels | **1.000000** | yes (49) | 0.0013 | 3.29 -> 65.29 (19.9x) | 1.0e+02 |
| encoded, pre-evolution | **1.000000** | yes (171) | 0.0022 | 3.29 -> 86.68 (26.3x) | 1.2e+03 |
| evolved on T | 0.990750 | **no (1000)** | 0.0555 | 2.93 -> 186.53 (63.7x) | **2.0e+06** |

**The evidence directly supports ill-conditioning (features that are
almost redundant with each other), not near-separability (a clean
dividing line between classes), and actually argues against the
separability explanation.** Both `raw_pixels` and `encoded_pre_evolution`
reach **perfect (100%)** training accuracy on this fold, and still
converge quickly and easily (49 and 171 iterations) — this is the
classic behavior when the data is perfectly separable but well
conditioned. `evolved_T` does **not** reach perfect training accuracy
(99.075%) — it is the *least* separable of the three — yet it is the
one that fails to converge. If separability alone were driving the slow
convergence, the two perfectly-separable conditions should have
struggled more, not less.

What `evolved_T` does have is a **condition number of the standardized
training feature matrix of about 2,000,000** — roughly 4 orders of
magnitude worse than `raw_pixels` (100), and 3 orders of magnitude worse
than `encoded_pre_evolution` (1,164). This comes from a very small
smallest singular value (6.2e-4, against a largest value of 1,228). This
is consistent with the `R(theta)` check already reported above: evolution
measurably increases phase synchronization (mean `R` 0.731 to 0.857
across this same 5,000-image set) — when many nodes' phases move toward
similar values, their `cos`/`sin` reference-node-centered features become
nearly identical to each other, producing near-collinear (redundant)
columns in the standardized feature matrix.

**This link was tested directly, not just left as a plausible story**:
the Pearson correlation between each fold-0 training image's `R(theta)`
and the size of its projection onto the direction with the smallest
singular value (the specific direction responsible for the poor
condition number) gives **r = -0.170, p = 2.3e-27** (n=4000). This
correlation is real, and overwhelmingly statistically significant given
the sample size, but **weak in size** (r^2 is about 0.03 — roughly 3% of
the variance explained), and, notably, **negative** — images with
*higher* `R(theta)` tend to have *smaller* projections onto the
near-null direction, not larger. This does not simply match a naive
"the most-synchronized images are individually the most collinear ones"
story. The honest reading: `R` and the ill-conditioning are related, but
`R(theta)` alone is far from a complete explanation of which images
drive the near-null direction — whatever else determines it is not
simply "high versus low order parameter," and this check does not
resolve what that additional structure is.

The growth in the coefficient's size (`evolved_T`'s 63.7x ratio, versus
19.9x and 26.3x for the other two) is consistent with, and likely
compounds, the ill-conditioning explanation (a stretched-out, poorly
conditioned loss surface leads to larger steps toward a more extreme
solution) — but `evolved_T`'s C=100 fit never actually reached
convergence (still at `max_iter=1000` when it stopped), so this
coefficient size is a mid-optimization snapshot, not its true optimum.
The coefficient-size comparison alone, without the condition-number
evidence, would be much weaker support on its own.

**Verdict, stated plainly**: the ill-conditioning explanation is
supported by direct, substantial evidence (a condition number about
1,700 times worse than the next-worst condition), together with a real
but weak, and directionally counter-to-expectations, statistical link to
the `R(theta)` measurement. The near-separability explanation is not
supported — if anything, the data points the opposite way, since
`evolved_T` is the one condition that does *not* reach perfect
training-set separation. This looks like graph evolution's own
synchronizing effect on the phase state, not incidental separability,
producing a genuinely harder optimization problem for this specific
condition at weak regularization — and specifically a slow-to-converge
problem, not an impossible-to-converge one, once given enough
iterations (below).

## Resolution: max_iter raised to 10,000, and the mechanism confirmed correctable by more iterations

Given the diagnosis (a real but purely slow-optimization problem, not
divergence or a data pathology), `max_iter` was raised from 1,000 to
10,000 — applied uniformly across all three conditions, disclosed as an
amendment made after `DESIGN.md`'s lock (see that document's own note).
This is not "iterate further and hope": the diagnostic script re-ran the
exact previously-failing case (`evolved_T`, fold=0, C=100) at
`max_iter=10000`, and it converged cleanly in **1,185 iterations** —
well inside the new budget, confirming this was a genuine
iteration-count shortfall, not an unbounded divergence that a larger
limit would only delay.

**The full stage-2 run (primary cross-validation across all 9 `C`
values x 5 folds x 3 conditions, plus the encoder-seed robustness check)
then completed end to end with zero non-convergences anywhere.** The
selected `C` per condition (unchanged from before for the two
unaffected conditions, confirming the fix did not disturb anything that
already worked):

| condition | selected C | mean CV log-loss at selected C |
|---|---:|---:|
| raw pixels | 0.01 | 0.681 |
| encoded, pre-evolution | 0.01 | 0.697 |
| evolved on T | **0.1** | 0.661 |

(`evolved_T`'s own selected `C` is 0.1, not 0.01 — its full
cross-validation curve could not be completed at all under the old
`max_iter=1000`, so this is newly-available information, not a changed
earlier result. This is still descriptive only, per this ladder stage's
own framing — not the locked primary result.)

## Encoder-seed robustness check (now run)

Reusing this same 5,000-image subset, per `DESIGN.md`'s locked
procedure: the `encoded_pre_evolution` and `evolved_T` features were
recomputed, with each image's encoder seed set to its own fixed dataset
index (instead of the shared seed 0), then refit on identical folds and
`C`-selection:

| condition | seed=0 mean val loss (own C) | independent-seed mean val loss (own C) | change |
|---|---:|---:|---:|
| encoded, pre-evolution | 0.6973 (C=0.01) | 0.6973 (C=0.01) | -0.0000 |
| evolved on T | 0.6607 (C=0.1) | 0.6606 (C=0.1) | -0.0001 |

**The change is negligible in both conditions** — the shared-seed-0
noise template does not appear to meaningfully affect the classification
result at this scale, for either encoder-derived condition. This is a
descriptive, training-side result (per `DESIGN.md`, it cannot replace or
override the seed-0 primary analysis against the official test set),
but it is a reassuring one: the encoder-random-seed choice, flagged as a
caveat in `DESIGN.md`'s original design, does not appear to be driving
anything important here.

**Overall stage-2 result: GO.** All locked go/no-go rules pass;
throughput (64.6 ms/image, consistent with stage 1's 65.7 ms/image)
projects to about 64.6 minutes for stage 3's full 60,000-image training
set — this projection still needs explicit approval before stage 3
launches, per `DESIGN.md`'s own requirement, which has not yet been
given here.

## A flag for the confirmatory design, not acted on now

If the ill-conditioning mechanism found here is genuinely about
evolution-induced phase synchronization (rather than something specific
to this particular 5,000-image sample), there is real reason to expect
the top of the locked `C` grid (100, 1000, 10000) to stay persistently
uninformative for evolved conditions specifically, **at any sample
size** — multicollinearity is a property of the relationships between
features, not the ratio of samples to features, so scaling up to 60,000
images would not be expected to fix it on its own. This is flagged here
as a candidate for a future, explicitly documented, separately reviewed
`DESIGN.md` change (for example, narrowing the confirmatory grid, or
adding a documented minimum ridge floor) — **only if the pattern is
confirmed to persist at stage 3's scale**, not assumed or acted on now.

## Code

`diagnose_stage2_convergence.py` (a diagnostic-only scan, not part of
the locked pipeline), `stage2a_classifier.py`'s new
`diagnose_convergence_full_grid()` (same caveat),
`diagnose_stage2_convergence_hypotheses.py` (the separability versus
ill-conditioning hypothesis test, the `R(theta)` correlation check, and
the `max_iter=10000` re-check, all diagnostic-only).
`run_feasibility_stage2.py` now halts and saves partial results if a
non-convergence happens again, instead of silently continuing past one;
with `max_iter=10000` it now completes cleanly and saves full results
(`results/stage2_feasibility_results.pkl`, gitignored).

## Next step

Explicit approval of the stage-3 runtime projection (about 64.6
minutes), then feasibility stage 3 (the full 60,000-image official
training set, final feature generation and model selection) —
superseded by the pre-stage-3 investigation below, which revises both
the scope and the runtime projection before any full run was launched.

# Stage 2A: Pre-Stage-3 Investigation (topology synchronization)

**Status: diagnostic only. There is no simulation data from a full run
here — all of this uses either a 400-image timing sub-test, or
zero-simulation graph-spectral analysis. This is not a scientific
result.**

Stage 3 requires evolving every image on **four** topologies (T,
lattice, canonical rewired, canonical curr_random — `DESIGN.md`'s
"Confirmatory expansion"), not just T. A 400-image timing sub-test, run
before committing to the full 60,000-image job, surfaced two things that
needed investigating before launching a multi-hour run.

## Revised runtime projection

400 images across all four topologies (shared encoding, per-topology
evolution): 99.7s (249.3 ms/image) — this projects to **about 249
minutes (about 4.2 hours)** for the full 60,000 images, not the roughly
64.6 minutes discussed earlier (which covered evolving on T alone;
stage 3 needs roughly 4 times the evolution-heavy work). This is still
comfortably feasible on CPU, with no GPU needed — but it is a real
change in scope from what was first approved, disclosed here rather than
launched silently.

## A striking synchronization asymmetry, investigated before launching

At n=400, `R(theta)` after evolution differed sharply by topology:

| topology | R_post mean | R_post > 0.99 |
|---|---:|---:|
| T | 0.860 | 0/400 |
| lattice | 0.867 | 0/400 |
| rewired | 0.997 | 400/400 (100%) |
| curr_random | 0.991 | 241/400 (60%) |

Given stage 2's finding that even T's moderate synchrony (R about 0.86)
produced a condition number of about 2e6, near-total synchronization
under rewired or curr_random risked a much worse version of the same
problem. The team investigated using three checks, all reusing
already-verified code, none requiring the full 60,000-image run:

**1. Laplacian spectrum and algebraic connectivity (zero new simulation
-- purely spectral, on the already-built graphs)**: standard Kuramoto
theory predicts synchronization strength scales with a graph's algebraic
connectivity (the Fiedler value -- the second-smallest Laplacian
eigenvalue; a graph property that measures how well-connected it is).

| topology | Fiedler value | connected components |
|---|---:|---:|
| T | 0.0059 | 1 |
| lattice | 0.0056 | 1 |
| rewired | **0.1322** (22x T's) | 1 |
| curr_random (main component) | **0.2429** (41x T's) | **14** (see correction, below) |

**This directly and cleanly explains the synchronization asymmetry**:
both randomized constructions have dramatically higher algebraic
connectivity than T or lattice — degree-preserving rewiring and
independent resampling both destroy whatever clustered or community
structure T (learned) and lattice (regular) have, and a more "well
mixed" graph synchronizes faster and more completely. This is not a
pipeline artifact; it is a real, spectrally confirmed structural
difference between the learned and regular constructions on one hand,
and the randomized ones on the other.

**A correction, prompted by this check**: computing the Laplacian
spectrum for `curr_random` surfaced 14 near-zero eigenvalues, not the 1
expected for a connected graph — meaning **`curr_random` seed=0 has 13
isolated nodes, not the 1 originally disclosed** in `DESIGN.md`'s graph
statistics table. The original check only tested `nodes_T`'s three fixed
coordinates for isolation (Stage 1D's own limited screening), not all
505 nodes. This was fixed directly in `DESIGN.md` (see its "Confirmatory
expansion" section) rather than left standing. The main 492-node
component's own Fiedler value (0.2429) is what actually explains the
strong synchronization; the 13 isolated single nodes are a separate,
now-correctly-disclosed structural fact about this specific draw.

**2. Information-collapse check**: does a classifier trained on
rewired's or curr_random's evolved features collapse to predicting only
one or two classes (real information loss), or does real per-image
signal survive? At n=400, C=0.01:

| topology | train accuracy | classes actually predicted |
|---|---:|---:|
| T | 0.8425 | 10/10 |
| lattice | 0.8525 | 10/10 |
| rewired | 0.8875 | 10/10 |
| curr_random | 0.9300 | 10/10 |

**No collapse.** Both rewired and curr_random predict across all 10
classes, with roughly balanced distributions, and their training
accuracy is nominally *higher* than T's or lattice's, not lower. Despite
near-total phase synchronization, real per-image discriminative
information survives in the evolved features — the mechanism is severe
optimization difficulty (as stage 2 found for T, likely worse here,
given the much higher Fiedler values), not information destruction.

**3. Multistability check, mirroring Stage 0's own method exactly**
(using `find_equilibrium_lbfgs`, `same_attractor`'s 0.05 duplicate
threshold, 5 seeds numbered 0 through 4, with no image encoding at all —
reused directly from `bonsai.dynamics.graph_oscillator_field`, not
rewritten):

| topology | distinct equilibria (of 5 seeds) |
|---|---|
| T | 5/5 |
| lattice | 1/5 |
| rewired | 1/5 |
| curr_random | 2/5 |

**This answers the question directly**: this is not simply "T and
lattice have multistability that rewired and curr_random lack." T is
uniquely rich in multistability (5/5); lattice and rewired both collapse
to a single basin (1/5 each) — consistent with, though not identical in
specific seeds to, Stage 0's own original table for these two
construction types. `curr_random`'s 2/5 sits in between, for this
specific (13-isolated-node) draw — a different specific realization from
whatever Stage 0's original "matched-sparsity random" check used, so a
different count is expected, not a contradiction (this matches this
project's own established rule that one draw is not a family, from
Stage 1D). This is consistent with the Laplacian finding: reduced
multistability tracks with increased algebraic connectivity, across all
three non-T constructions.

## What this means for stage 3

The synchronization asymmetry is real, well explained by the spectral
analysis, and does not indicate an information-collapse problem —
rewired's and curr_random's evolved features remain genuinely
discriminative, though possibly harder to optimize (an open question the
full stage-3 cross-validation run will answer directly, given
`max_iter=10000` is already in place from stage 2's fix). The revised
about-4.2-hour runtime projection and the corrected 13-isolated-node
fact are both disclosed above. Two things to check honestly once stage
3's full run completes (these are not new requirements, just things to
verify): (1) whether the high-`C` non-convergence pattern from stage 2
persists, worsens, or resolves for rewired and curr_random at 12 times
the sample size; (2) whether the encoder-seed robustness finding
(negligible difference) still holds at full scale.

## Code

`stage2a_topologies.py` (builds all four canonical graphs, reusing
already-verified construction code only), `diagnose_topology_synchronizability.py`
(Laplacian spectrum, zero simulation), `diagnose_rewired_currrandom_synchronization.py`
(the information-collapse and multistability checks, reusing
`bonsai.dynamics.graph_oscillator_field`'s `find_equilibrium_lbfgs`
directly). `stage2a_pipeline.py` was extended with
`run_pipeline_multi_topology()`, `check_go_no_go_multi_topology()`, and
`run_classifier_conditions_multi_topology()` for stage 3's 6-condition
(raw pixels, pre-evolution, plus 4 evolved topologies) design, encoding
each image once and evolving it on all four graphs, rather than
re-encoding it separately per topology.

## Next step

The full feasibility stage 3 (60,000 images, about 4.2-hour projection,
6 conditions) — superseded by the timing-split and GPU-port investigation
below.

# Stage 2A: Encode/Evolve Timing Split, and a Verified JAX Evolution Port

**Status: diagnostic and infrastructure work only. This section has no
stage-3 simulation data — it covers a timing split, a spectral
cross-check, and a from-scratch-verified JAX port, none of which is a
scientific result.**

## Timing split: evolution dominates by about 80x

Encoding (`_local_converged_phases` plus restriction) and graph
evolution were timed separately, on 100 real images, using a single
topology (T), running single-threaded (no parallel processing, to
isolate the two steps cleanly):

| step | total (100 images) | ms/image |
|---|---:|---:|
| encode | 0.458s | 4.58 |
| evolve (T only) | 36.558s | 365.58 |

**Evolution is about 80 times the cost of encoding.** This sets the real
ceiling on what a GPU port can help with: porting encoding would not
meaningfully change throughput; porting evolution addresses essentially
all of the cost.

## curr_random seed=0: where does 13 isolated nodes sit in the family?

Before committing hours of compute to this specific pre-chosen graph
instance, its isolated-node count and its main-component Fiedler value
were checked against all other curr_random realizations already
computed in Stage 1D (3 pilot seeds numbered 0 through 2, and 25
confirmatory seeds numbered 5 through 29 — 28 total, zero new
simulation, just loading and analyzing already-cached graphs):

- **Isolated-node count**: range 3-15 across the 28 realizations, mean
  9.18, median 9.0. Seed=0's 13 sits at the **92.9th percentile** —
  elevated relative to the median, but well within the observed range
  (several other seeds, for example 5 and 1, have 15), not an outlier.
- **Main-component Fiedler value**: seed=0's 0.2429 sits squarely within
  the observed range (about 0.16-0.36) across all 28 realizations.

**Seed=0 is a reasonable, pre-chosen representative** — somewhat
elevated on isolation count, typical on connectivity — not a broken or
unusually extreme draw. There is no reason to redraw the canonical seed.

## JAX/diffrax port of graph evolution

`evolve_on_graph_jax.py` ports `stage2a_core.evolve_on_graph`'s plain
(unperturbed) evolution to JAX/diffrax, reusing
`run_one_trial_jax_faithful.py`'s already-verified plain-evolution `rhs`
function directly (this is Stage 1D's own file — the perturbed-trajectory
half of that function, not the tangent-system half, which Stage 2A never
needed). It returns `(theta_T, success)`, checking
`diffrax.is_successful()` directly rather than assuming success — the
batched, compiled computation cannot raise an error per trial the way
the numpy recovery-policy retry loop does, so the calling code must
check this exactly like the numpy path's `diag['failed']` flag.

**Kernel-level verification** (`verify_evolve_on_graph_jax.py`): 6 real
encoded images (KMNIST, not synthetic random phases), across all 4
topologies, both unbatched and batched (using `vmap`), compared against
the real numpy `evolve_on_graph`, using circular distance (which
accounts for the fact that phase values wrap around at 2*pi).
**Largest absolute circular difference across all 24 (topology, image)
pairs: 5.4e-8** — both the unbatched and batched versions agree with
numpy to this precision, comfortably inside the 1e-4 cross-solver
tolerance this project has used throughout (Stage 1D's own GPU port
matched to 1e-6 to 1e-8).

## End-to-end pipeline equivalence check (not just the kernel)

**This project has been burned before by a kernel that passed
field-by-field verification, while the surrounding batch or calling code
quietly did something different (Stage 1D's GPU episode — see
`experiments/stage1d_topology_specificity_gpu/FINDINGS.md`, and
`CLAUDE.md`'s principle 16).** Per that lesson, the kernel check above
is not treated as enough on its own. `stage2a_pipeline_jax.py` builds a
full JAX-evolution pipeline that matches
`stage2a_pipeline.run_pipeline_multi_topology`'s exact result format
(the same per-image, per-topology dictionary structure, the same index
ordering), reusing `stage2a_core`'s numpy encoding and feature functions
completely unchanged — only the evolution step itself is replaced.

`verify_stage2a_pipeline_equivalence.py` runs both the real numpy
pipeline and this JAX pipeline on the same 40-image batch, drawn from
**all 10 classes** (not a single-class or uniform sample), and compares
the full output:

| check | result |
|---|---|
| row ordering / labels | identical idx 0-39 in both, same label array |
| encoded `theta_0` (via `R_pre`) | **bit-identical** (max diff 0.0) |
| gauge-fixed pre-evolution features (`feat_pre`) | **bit-identical** (max diff 0.0) |
| raw-pixel features | **bit-identical** (max diff 0.0) |
| solver-failure accounting, all 4 topologies | **identical** (0/40 failed, both pipelines agree, every topology) |
| evolved `theta_T` (via `R_post`), all 4 topologies | max diff 3.7e-10 to 3.6e-10 |
| gauge-fixed evolved features (`feat_post`), all 4 topologies | max diff 1.8e-10 to 5.0e-8 |

**All checks pass.** The bit-identical results (encoding, pre-evolution
features, raw pixels) are expected and guaranteed by design — both
pipelines call the exact same numpy functions for those steps, so
anything other than exact equality would indicate a real bug. The
agreement in evolved features (largest difference 5.0e-8, well inside
the 1e-4 tolerance) is the genuine cross-solver check (numpy/scipy RK45
versus JAX/diffrax Tsit5), and the match in solver-failure accounting
confirms the JAX path's `diffrax.is_successful()` check is wired
correctly, not just present in the code but never actually tested.

## What this means for the GPU decision

CPU-only JAX (a single device, no parallel processing) was also timed:
400 images, all 4 topologies, 119.9s (74.9 ms/image/topology) — faster
than numpy's single-threaded rate (365.6 ms/image/topology), but *not*
clearly faster than numpy already run in parallel across about 9 CPU
cores (about 40 ms/image/topology-equivalent). **The JAX port's real
value is unlocking GPU batch parallelism, not CPU JAX by itself** —
consistent with Stage 1D's own experience (that project's 112x speedup
was GPU versus CPU-parallelized numpy, not versus single-threaded
numpy). A GPU session is the natural next step for the full
60,000-image, 4-topology stage-3 run, now that both the kernel and the
full pipeline built around it are verified.

## Code

`evolve_on_graph_jax.py` (the JAX/diffrax evolution kernel),
`verify_evolve_on_graph_jax.py` (kernel-level verification),
`stage2a_pipeline_jax.py` (the full pipeline, with numpy encoding and
feature functions reused unchanged), `verify_stage2a_pipeline_equivalence.py`
(the end-to-end equivalence check against the real numpy pipeline, on
the mixed-class batch).

## Next step

Provision a GPU session, and run the full stage-3 pipeline (60,000
images, 4 topologies, 6 conditions) using the now-verified JAX pipeline —
superseded by the 100-image GPU sanity run below.

# Stage 2A: 100-Image GPU Sanity Run

**Status: this is an infrastructure verification only, not a scientific
result.** It confirms the verified JAX pipeline produces correct results
on real GPU hardware (not just locally on CPU-backend JAX), and gives a
real, measured full-scale runtime projection, before committing to the
full 60,000-image run.

## Setup

A fresh A100 session (`stage2a-gpu-100`, via `mighty-colab`; the team
checked first that no orphaned sessions were left running). Environment:
`jax==0.11.0` (`jax[cuda12]`), `diffrax==0.7.2`, `equinox==0.13.8` — the
same pinned versions already verified working in Stage 1D's own GPU
work, installed, then the session's kernel restarted (a pip install
alone does not take effect in an already-running kernel that had already
imported the older, pre-installed `jax==0.7.2`).

100 images (10 per class, `SEED=42`, using the same `subsample_stratified`
function used throughout) were encoded locally (CPU/numpy, unchanged
code), packaged with all four topologies, and uploaded (10MB). Only the
evolution step ran on GPU.

## Result: a real speedup of about 546x to about 60x, zero failures, correctness confirmed

| topology | GPU time (100 images) | ms/image |
|---|---:|---:|
| T | 0.072s | 0.72 |
| lattice | 0.069s | 0.69 |
| rewired | 0.069s | 0.69 |
| curr_random | 0.058s | 0.58 |

**Total: 0.268s for all 4 topologies, 100 images — 0.67
ms/image/topology.** Compared to numpy's single-threaded rate of 365.58
ms/image/topology, that is a **speedup of about 546 times**; compared
to numpy already run in parallel across about 9 CPU cores (about 40
ms/image/topology-equivalent), it is still **about 60 times faster**.
Zero solver failures were reported (checked using
`diffrax.is_successful()`), across all 400 (image, topology)
combinations.

**Correctness was re-verified on real GPU hardware, not assumed from the
earlier CPU-backend check**: the first 5 images' GPU-produced `theta_T`
values, for all 4 topologies, were compared against the local numpy
`evolve_on_graph` — the largest circular difference was **1.5e-8**,
consistent with, and as tight as, the CPU-backend verification.

**A full go/no-go check on all 100 images, all 4 topologies**: zero
non-finite feature values; the `R_post` distributions match the earlier
CPU findings exactly (T mean 0.865, lattice 0.871, rewired 0.997 with
100 of 100 above 0.99, curr_random 0.991 with 64 of 100 above 0.99) —
this confirms the GPU run reproduces the same dynamics, not just runs
fast and produces different numbers.

## Revised full-scale projection

**Projected full 60,000-image, 4-topology GPU evolution time: about 161s
(about 2.7 minutes)** — down from the earlier about-4.2-hour
CPU-parallel projection. Encoding (CPU-only, about 4.6 ms/image) adds
roughly 4.6 minutes for 60,000 images, unaffected by the GPU port. The
evolution bottleneck that motivated this whole investigation is now
essentially eliminated; the remaining stage-3 runtime will be dominated
by encoding and the 6-condition classifier cross-validation fitting, not
graph evolution.

## Code

Local: `evolve_on_graph_jax.py`, reused unchanged (uploaded as-is). A
remote-only driver script (not committed — this is ephemeral GPU-session
code, per this project's convention; the reusable kernel and pipeline
code is already committed locally).

## Next step

Provision a fresh GPU session (this one to be stopped), and run the full
60,000-image, 4-topology, 6-condition stage-3 pipeline for real — not
done here.

# Stage 2A: Feasibility Stage 3 -- Full 60,000-Image Run

**Status: this is a mechanical and descriptive check at full
training-set scale, per `DESIGN.md`'s own framing throughout the
feasibility ladder. This is NOT the locked confirmatory result.** The
confirmatory claim (a paired bootstrap on official-test-set log-loss,
evolved versus pre-evolution) still requires a separate, final step
against the held-out official test set, not done here. The
training-side cross-validation log-loss is reported below descriptively,
exactly as `DESIGN.md` allows at this stage, and no further than that.

## Scope

All 60,000 official KMNIST training images (no subsampling), all 10
classes, independent of `SEED` (the full set, not a stratified draw).
Six conditions: raw pixels (784-dimensional), encoded pre-evolution
(1008-dimensional), and evolved on each of the 4 confirmatory-expansion
topologies — T, lattice, canonical rewired (seed=0), canonical
curr_random (seed=0), each 1008-dimensional. Encoding used the primary
locked seed (`ENCODER_SEED=0`) for every image.

## Architecture: CPU encode, GPU evolve, CPU classify

Following the explicit instruction to use the verified JAX/GPU pipeline
for full-scale data generation: encoding ran locally (CPU, in parallel,
using unchanged `stage2a_core`/`stage2a_pipeline` code), evolution ran on
a fresh `mighty-colab` A100 session (`stage2a-gpu-stage3`; the team
checked first that no orphaned sessions were left running; using the
same pinned `jax[cuda12]==0.11.0`, `diffrax==0.7.2`, and
`equinox==0.13.8`, confirmed correct on the first `exec` call, with no
kernel restart needed this time), and classifier cross-validation
fitting ran locally afterward (CPU-only, scikit-learn) — the GPU session
was stopped immediately after evolution completed and its results were
downloaded, since classifier fitting gains nothing from GPU, and there
is no reason to keep paying for idle GPU time during that step.

**One real infrastructure snag, worth recording**: the single 250MB
upload package (`theta0_batch` plus all 4 topologies) was rejected by
the `mighty-colab` upload service with a 400 Bad Request error — the
10MB, 100-image package had worked fine, but 250MB went over some
server-side limit. This was worked around by splitting `theta0_batch`
into 12 separate files, each about 20MB (uploaded individually, then
reassembled remotely using `np.concatenate`), plus the small 8MB
topologies file uploaded separately. Both uploaded cleanly. This is an
operational note for any future GPU work at this data scale, not a
scientific finding.

GPU evolution itself was run in chunks (`CHUNK_SIZE=1000`), rather than
as one 60,000-image batch, since `vmap` builds a `(batch, n, n)`
difference array for each right-hand-side evaluation — at `batch=60000,
n=505`, that would need about 120GB (using float64 precision), far
beyond any single GPU's memory, whereas the 100-image full-batch test's
204MB array gave no signal either way about this limit.
`CHUNK_SIZE=1000` keeps the largest temporary array at about 2GB,
conservatively safe. This was verified working, with zero errors, across
all 60 chunks x 4 topologies.

## Result 1: data generation -- fast, clean, ahead of projection

| phase | time | rate |
|---|---:|---:|
| encode (CPU, local, multiprocessing) | 68.1s | 1.135 ms/image |
| GPU evolve, all 4 topologies | 114.0s | 0.475 ms/image/topology |
| R_post/feat_post compute (CPU, local) | 6.4s | -- |
| **total data-gen pipeline** | **188.5s (3.1 min)** | -- |

This is well under the about-7.3-minute (encode plus GPU-evolve)
projection from the 100-image sanity run. The GPU evolution rate
(0.455-0.491 ms/image/topology across the 4 topologies) essentially
matches the 100-image run's 0.67 ms/image/topology, confirming the
chunked approach scales cleanly, with no drop in throughput, at 600
times more images.

**One genuine surprise, stated plainly rather than smoothed over**: the
measured 1.135 ms/image encode rate is roughly 4 times faster than the
about-4.6 ms/image figure quoted in the pre-stage-3 timing investigation
(`evolve_on_graph_jax.py`'s docstring, and the earlier projection of
"about 4.6 minutes for 60,000 images"). That earlier figure appears to
have been a single-threaded measurement. This run's local parallel
processing across about 9 worker processes accounts for most, but
apparently not all, of the gap (a plain 9-times parallel speedup on 4.6
ms/image would predict about 0.51 ms/image, not the observed 1.135
ms/image) — the gap is not fully resolved, and is noted honestly rather
than assumed to be pure parallelism.

There were zero solver failures across all 240,000 (image, topology)
evolutions, and zero non-finite feature values anywhere (raw,
pre-evolution, or evolved, any topology).

## Result 2: R(theta) -- the synchronization asymmetry holds precisely at full scale

| | min | max | mean | median | n < 0.01 | n > 0.99 (of 60,000) |
|---|---:|---:|---:|---:|---:|---:|
| pre-evolution (shared) | 0.321 | 0.991 | 0.733 | 0.739 | 0 | 4 |
| T | 0.532 | 0.998 | 0.858 | 0.866 | 0 | 71 (0.12%) |
| lattice | 0.525 | 0.998 | 0.865 | 0.872 | 0 | 64 (0.11%) |
| rewired | 0.986 | 1.000 | 0.997 | 0.997 | 0 | **59,965 (99.94%)** |
| curr_random | 0.973 | 1.000 | 0.991 | 0.991 | 0 | **36,547 (60.9%)** |

This is not a weakening or drift from the 400-image sub-test's finding —
it is a precise confirmation. That sub-test reported "R_post > 0.99 for
about 100% of images for rewired, and about 60% for curr_random"; at 150
times the sample size, the real figures are 99.94% and 60.9%
respectively, matching almost exactly. Rewired's near-total
synchronization is in fact more extreme than "near-total" suggests: its
*minimum* R_post across all 60,000 images is 0.986 — there is no image,
of any class, that fails to synchronize almost completely under this
topology. T and lattice show no such effect (0.11-0.12% above 0.99, an
order of magnitude smaller than even curr_random's rate).

**A visual supplement** (`results/topology_graph_structure.png`,
`results/phase_state_per_class_per_topology.png`, added later during
the compute-cost follow-on work): the graph-structure plot makes the
mechanism visible directly — `rewired`/`curr_random` are dense, with
long-range edges connecting spatially distant nodes across the whole
image, while `T`/`lattice` are almost entirely short-range and local,
closely tracking the pixel grid's own spatial layout. The per-class
phase-state grid shows the result: for every one of the 10 classes,
`T`'s and `lattice`'s evolved phase fields keep visible spatial
structure echoing the digit's shape, while `rewired`'s and
`curr_random`'s collapse to near-uniform color — near-total
synchronization, visible directly, not just as a summary statistic.

**A real tension this raises, and its resolution**
(`results/phase_state_per_class_per_topology_normalized.png`): if
`rewired`'s and `curr_random`'s columns look nearly uniform by eye, how
does a linear classifier extract enough signal from them to perform
comparably to, or better than, `T` (the confirmatory result's own
finding)? This is resolved by the pipeline itself, not a contradiction:
`StandardScaler` runs before classification, rescaling whatever variance
is actually present — real, but small — up to unit scale before the
classifier ever sees it. Measured directly: the gauge-shifted phase's
raw standard deviation (averaged across the 10 classes, one
representative image each) is about 0.079 radians for `rewired` and
about 0.133 radians for `curr_random`, versus about 0.79 radians for the
pre-evolution baseline and about 0.54-0.55 radians for `T`/`lattice` —
roughly 4 to 10 times smaller in absolute size, genuinely small, but not
zero. Per-panel z-scoring (the same rescaling `StandardScaler` performs)
makes this leftover variation visible directly. **"Looks uniform to the
eye" and "carries zero usable information" are different claims** — this
plot is honest evidence for the first, not the second, consistent with
(not contradicting) the classification result.

**The follow-up observation above was chased down, and the mechanism
behind the speckle pattern is now confirmed directly**
(`results/ink_correlation_decay.png`): the Pearson correlation between
each active pixel's ink intensity *in that specific image* and its
z-scored leftover phase deviation, per class, per condition — not
compared against any population baseline, just "is this pixel inked, in
this image":

| condition | mean r (10 classes) | std |
|---|---:|---:|
| pre_evolution | +0.938 | 0.011 |
| T | +0.725 | 0.053 |
| lattice | +0.722 | 0.049 |
| rewired | +0.373 | 0.039 |
| curr_random | +0.274 | 0.031 |

**Every one of the 10 classes, and every condition, is positive, and
overwhelmingly significant** (the worst case has `p < 2e-7`, and most
have `p < 1e-20` or smaller). This is exactly the mechanism the speckle
pattern suggested by eye: red means inked here, blue means not inked
here, and it holds all the way down to `curr_random`'s r=+0.27 — small,
but a genuine, consistent relationship that decays but never vanishes,
not noise. The order of decay (`pre_evolution > T ≈ lattice > rewired >
curr_random`) tracks the synchronization ordering exactly (Result 2,
above) — more synchronization means more of this signal washed out, but
never all of it. This is a cleaner, more economical finding than
"qualitatively different signal": it is the *same* underlying signal
(local ink presence) at every condition, just weakened by a
condition-dependent, but never total, factor.

**A further, sharper breakdown, also tested rather than assumed**
(`results/ink_correlation_decomposed.png`): "this pixel has ink" splits
exactly into a **population-common part** (the per-pixel mean ink
intensity across the 10 classes — shared, and by construction not
class-specific) and a **class-discriminatory part** (that class's
deviation from the population mean at that pixel — the part that
actually varies by class). Correlating each part against the leftover
phase deviation separately, per condition:

| condition | r(common) | r(discriminatory) | ratio |
|---|---:|---:|---:|
| pre_evolution | +0.370 | +0.854 | 2.3x |
| T | +0.344 | +0.639 | 1.9x |
| lattice | +0.322 | +0.642 | 2.0x |
| rewired | +0.187 | +0.325 | 1.7x |
| curr_random | +0.134 | +0.240 | 1.8x |

**A correction, caught on external review, load-bearing: the paragraph
that previously stood here overclaimed "preferential preservation of the
discriminatory component," and the conclusion was backwards.** The
error: comparing `r(discriminatory)` against `r(common)` *within* each
condition (0.854 versus 0.370 at `pre_evolution`, and so on) only shows
that the discriminatory part is already more strongly correlated than
the common part *before evolution does anything* — that difference is
already present at `pre_evolution` itself, not created by evolution. It
says nothing about which part evolution preserves *better*. The correct
comparison is how much is retained *relative to `pre_evolution`*, for
each part:

| graph | common retained | discriminatory retained |
|---|---:|---:|
| T | 93.0% | 74.8% |
| lattice | 87.0% | 75.2% |
| rewired | 50.5% | 38.1% |
| curr_random | 36.2% | 28.1% |

**At every single condition, the common part is retained at a *higher*
percentage than the discriminatory part — the opposite of "preferential
preservation."** Evolution weakens both, and weakens the discriminatory
correlation proportionally *more*. The representation is more strongly
correlated with the class-discriminatory part at every stage (the
original table, above) — but that difference is inherited from the
encoding step, not produced or increased by evolution.

**This makes the classification result more interesting, not less**: if
evolution simply preserved discriminatory pixel-level ink better than
common ink, that would be a fairly ordinary explanation for the
improvement. It does not. The improvement survives even though evolution
erodes the discriminatory correlation *faster* than the common one —
which means whatever the linear classifier is actually using after
evolution is not well described as "the same pixel-level discriminatory
signal, just a bit fainter." The dynamics may be reorganizing,
decorrelating, or redistributing the surviving information into
directions a linear readout can use more effectively than the raw
per-pixel correlation would suggest — a real, open mechanistic question
this correction leaves standing, not one it answers. (A caveat carried
over from before: the population-common baseline is estimated from only
10 images, one per class — this is enough for the direction and rough
size of this effect, but not a precise population estimate.)

**A further, related caveat, also raised in the same review, worth
naming precisely**: all of the mechanism plots and correlations in this
section are conditional on the class-0-derived 505-node set of active
nodes (`DESIGN.md`'s locked `active_indices`, shared identically by all
four topologies and the pre-evolution condition — see "The class-0
confound `DESIGN.md` flagged," above, for the primary result's own
disclosure of this). Three things follow, stated precisely rather than
left unstated:

- **The primary confirmatory result is unaffected.** `evolved_T` versus
  `pre-evolution` uses the identical mask on both sides — the
  improvement isolates the added effect of edge structure and evolution
  *given* that fixed set of nodes, and cannot be attributed to one
  condition simply keeping more pixels than the other.
- **`active_indices` is not literally an ink outline.** It comes from
  `build_class_topology`'s population-level statistic (an all-pairs,
  Hebbian-style measure across class-0 images), thresholded to suppress
  background-to-background edges, then restricted to nodes that survive
  in at least one sufficiently strong edge — a class-0-derived,
  *correlated* set of nodes, broader than, and not identical to, a
  simple ink-presence mask.
- **The mechanism plots establish survival *within* this set of nodes,
  not a claim about ink outside it.** The local encoder's
  four-neighbor coupling runs on all 784 pixels *before* the 505-node
  restriction is applied — an excluded pixel's own coordinate is
  dropped, but its influence on nearby *retained* pixels during those
  150 coupling steps is not necessarily erased. What these plots show
  is: ink-related information within the class-0-derived set of active
  nodes survives evolution and synchronization, weakened but never to
  zero. They do not show what happens to class-discriminatory ink that
  falls entirely outside that set of nodes — a genuinely open question,
  addressed directly below.

## The class-0-support audit: how much information the projection actually removes

This is external review's own proposed audit, run in full: before asking
what graph evolution does with what remains, quantify what the
class-0-derived 505-node support projection removes in the first place.

**Retained ink fraction, full 60,000-image training set**
(`results/retained_ink_fraction_by_class.png`): `sum(pixel intensity
inside the 505-node support) / sum(pixel intensity over all 784
pixels)`, per image. Class 0 retains a median of **96.3%** of its own
ink (mean 93.9%, a tight distribution) — this is expected, since the
support is derived from class 0. **Every other class retains
substantially less**: medians range from 65.4% (class 1) to 90.0%
(class 5), with real spread — some individual images retain as little as
21-43%. This is not a small effect.

**Where the excluded ink actually falls**
(`results/ink_outside_support_by_class.png`, class-mean heatmaps of ink
lying outside the support): there is a consistent, visible band of
excluded ink at the top-center notch and the bottom margin, present for
classes 1-9 specifically and much fainter for class 0 — exactly the
region the class-0-derived support's own shape
(`results/topology_graph_structure.png`) does not cover. This is real
and systematic, not scattered noise.

**Two baselines, run to separate the projection's cost from evolution's
contribution** (`run_class0_support_audit_classify.py`, `cuml.accel`,
using the same locked cross-validation and fitting procedure as every
other condition -- audit-only, not part of the locked primary or
secondary comparisons):

| condition | dim | C | test accuracy | log-loss |
|---|---:|---:|---:|---:|
| raw pixels, full 784 (known) | 784 | 0.001 | 0.6960 | 0.9848 |
| raw pixels, 505-restricted (new) | 505 | 0.01 | 0.6550 | 1.1527 |
| encoded, 505-restricted = `encoded_pre_evolution` (known) | 1008 | 0.01 | 0.7208 | 0.9558 |
| encoded, full 784, unrestricted (new) | 1566 | 0.01 | 0.7458 | 0.8667 |

The two new rows were fit under `cuml.accel` on a GPU, which does not
give exactly bit-reproducible results. Re-running them on a fresh A100
(2026-08-05, using `make stage2a-class0-classify-gpu`) reproduced the
selected `C` and the 1566-dimensional accuracy exactly, and the
505-dimensional accuracy to within one image out of 10,000 (0.6549
against the 0.6550 above); the per-`C` validation losses move in the
4th-to-6th decimal place, mostly at large, unselected `C` values.
Reproduce these rows by checking the selected `C`, convergence, and
accuracy — not by checking a file hash. See `docs/PROJECT_MEMORY.md`
Part 4.

**The restriction has a real, measurable cost, in both representations**:
raw pixels lose 4.10 percentage points of accuracy (0.6960 to 0.6550)
when restricted to the support; the locally-encoded state loses 2.50
percentage points (0.7458 to 0.7208). Restricting to the class-0-derived
support is not free — it discards real, class-discriminatory signal,
exactly as the retained-ink-fraction and heatmap results above already
suggested directly.

**But evolution's contribution is larger than what the restriction
costs, not merely making up for it**: `evolved_T` (505-restricted,
evolved — the locked primary condition, 0.8058 accuracy, 0.7067
log-loss) beats even the *unrestricted*, un-evolved 784-dimensional
encoded baseline (0.7458 accuracy, 0.8667 log-loss) by 6.00 percentage
points of accuracy and 0.16 log-loss — a larger margin than the 2.50-point
cost the restriction itself imposed. **Stated plainly: even if the
pre-evolution condition were given back every pixel the class-0 support
excludes, graph evolution on the restricted 505-node `T` would still
win.** This is reassuring for how the primary result should be
interpreted, though it does not change the primary result's own numbers,
which were never in question — `evolved_T` versus `pre-evolution`
already used the identical support on both sides, so this audit closes a
question about *interpretation*, not about the comparison's own
validity.

**What this does and does not settle**: it settles that the class-0
support projection has a real, non-trivial, quantified cost, and that
evolution's contribution exceeds that cost rather than merely offsetting
it. It does not settle what a union mask, or a fully class-agnostic
support, would show — that remains a genuinely open, separately scoped
question (raised in this same review thread, not pursued here), since
this audit only compares restricted against unrestricted under the
*existing* class-0-derived mask, not against a different mask
altogether.

## Result 3: classifier CV fitting is NOT negligible -- it dominates everything else

**246.8 minutes (4.1 hours)** for the 6 conditions' full
cross-validation fitting (45 fold-and-C combinations each, 270 total,
plus 6 final refits) — measured explicitly, rather than continuing to
assume it was cheap, per the instruction that prompted this measurement.
This is about **79 times** the entire data-generation pipeline's 188.5
seconds. Classifier fitting, not data generation, is now the dominant
cost of this ladder's full-scale step, and any future re-run of this
stage should budget for that directly, rather than by extending stage
1 and stage 2's much smaller-scale timings.

**All 270 fold-and-C combinations converged, for every condition, with
zero non-convergence events.** This directly resolves the concern
flagged going into this stage: the pre-stage-3 investigation predicted
non-convergence was "most likely to recur … for the higher-Fiedler-value
topologies (rewired, curr_random) at weak regularization," given stage
2's evolved_T non-convergence at 5,000 images. At 60,000 images (12
times stage 2's scale), it did not recur anywhere, for any condition,
including rewired and curr_random, despite their extreme R_post values
above. The `max_iter=10000` amendment made after stage 2's diagnosis
appears to have been not just a workaround for that one instance, but
sufficient headroom at full scale too. No ill-conditioning-versus-
separability diagnosis was needed, since there was no non-convergence to
diagnose.

## Result 4: descriptive training-CV log-loss ranking (NOT the confirmatory result)

| condition | selected C | mean val. log-loss |
|---|---:|---:|
| raw_pixels | 0.001 | 0.6009 |
| encoded_pre_evolution | 0.01 | 0.5609 |
| evolved_lattice | 1000 | 0.4077 |
| evolved_rewired | 10 | 0.3480 |
| evolved_T | 1000 | 0.3415 |
| evolved_curr_random | 1.0 | 0.3255 |

All four evolved conditions beat both raw pixels and encoded
pre-evolution by a wide, consistent margin — this is descriptively
consistent with this whole project's motivating idea, that evolution
adds structure that is linearly decodable. This is training-side
cross-validation log-loss only. It is never the locked confirmatory
statistic (a paired bootstrap on *official-test-set* log-loss, evolved
versus pre-evolution specifically), and must not be read as anything
stronger than descriptive.

**A second genuine surprise, stated plainly**: `curr_random` — a
topology with matched sparsity, but no relationship to T's learned
structure — gives the *lowest* training-cross-validation loss of all
four evolved conditions (0.3255), narrowly beating `evolved_T` (0.3415),
even though T is the topology this whole construction was built around.
And `rewired`, despite the almost-total phase synchronization documented
above (R_post between 0.986 and 1.0 for all 60,000 images), still gives
the second-best evolved loss (0.3480) — clearly better than `lattice`
(0.4077), and far better than either pre-evolution condition. One might
expect synchronization this severe to collapse most of the information
that would distinguish one image's evolved state from another's; it
evidently does not do so here, at least not enough to prevent a linear
reference-node-centered readout from separating classes better than the
un-evolved features. Neither of these findings should be over-read as
bearing on Stage 1D's own "no detectable difference" conclusion (that
used a different metric — a paired bootstrap on the *tangent-departure*
response measure — and a different sample), but they are both real, and
both reported as observed, rather than smoothed toward the expected
story.

## Go/no-go summary

- Solver failures: 0 of 240,000 (image, topology) evolutions.
- Non-finite features: 0, in any condition, any topology.
- Classifier convergence: 270 of 270 fold-and-C fits converged, and 6 of
  6 final refits converged.
- **OVERALL: GO** (this is a mechanical checklist result only — not a
  scientific result).

## What remains open after this stage

1. **The encoder-seed robustness check at full scale** (stage 2's
   negligible-difference finding, re-tested at 60,000 images) was not
   run in this pass. Given the now-measured classifier-fitting cost (4.1
   hours for 6 conditions), a partial re-fit (encoded_pre_evolution plus
   evolved_T only, matching stage 2's original scope) is a bounded but
   real additional cost, plausibly on the order of an hour or more —
   this is flagged here for an explicit go-ahead, rather than run
   automatically, since this stage's own instructions conditioned it on
   "not prohibitively expensive," and the real cost is now known rather
   than assumed.
2. **The locked confirmatory endpoint itself**: a paired bootstrap on
   official-test-set log-loss (evolved versus pre-evolution) — the one
   result this entire ladder has been building toward. Not started.

## Code

`run_feasibility_stage3_encode.py` (the local CPU encode driver script,
for all 60,000 images), `stage2a_pipeline.py`'s new
`run_encode_only_multi_topology` and `_process_one_image_encode_only`
(an encode-only split for the GPU hand-off, reusing
`encode_and_restrict`, `order_parameter`, and `reference_node_features`
unchanged), `analyze_stage3_results.py` (local: rebuilds the full
per-image results structure from the downloaded GPU evolution output
plus the local encode results, using the same `s2a.order_parameter` and
`s2a.reference_node_features` calls used everywhere else, then calls
`check_go_no_go_multi_topology` and `run_classifier_conditions_multi_topology`
unchanged). A remote-only GPU driver script (`stage3_gpu_evolve.py`,
chunked evolution) was not committed, per this project's convention for
ephemeral GPU-session code — it reuses `evolve_on_graph_jax.py`
(uploaded as-is, unchanged).

## Explicit scope decision: full-scale encoder-seed robustness check not re-run

**Decided, reasoned through, and recorded here rather than silently
skipped.** `DESIGN.md`'s robustness check is conditioned on "not
prohibitively expensive" (it is a descriptive, non-blocking check, never
able to override or replace the seed-0 primary analysis). It already ran
once, at feasibility stage 2's 5,000-image scale, and passed cleanly:
the change in mean validation log-loss (comparing independent-per-image
seeds against the shared seed-0 encoding) was **-1.5e-6** for
`encoded_pre_evolution` and **-8.4e-5** for `evolved_T` — both
negligible, well inside noise, and in the *same* direction (independent
seeds giving very slightly *lower* loss, not higher) for both
conditions.

Given classifier cross-validation fitting was just measured at **4.1
hours for 6 conditions at n=60,000** (Result 3, above), a full-scale
re-check (re-encoding all 60,000 images under independent seeds,
re-evolving on GPU, and refitting cross-validation for at least
`encoded_pre_evolution` and `evolved_T`) would plausibly cost **an hour
or more** — to re-test something that already passed once, with no
reason to expect the direction or size to change in kind at 12 times the
sample size. That is not consistent with "not prohibitively expensive"
applied to a check whose only role is descriptive robustness support,
not a locked gate. **Decision: this check was not re-run at full
scale.** The stage-2 result stands as the encoder-seed robustness
evidence for this design.

## Next step

The locked confirmatory run itself: a paired bootstrap, evolved versus
pre-evolution, on the official 10,000-image KMNIST test set — touched
for the first and only time in this project.

# Stage 2A: The Locked Confirmatory Result

**Status: this is it — the one and only locked, pre-registered
official-test-set evaluation this entire design has been building
toward, per `DESIGN.md`'s "Confirmatory endpoint and test" section,
carried out exactly as locked. The official KMNIST test set was touched
here for the first time in this project, and this scikit-learn
evaluation remains the sole confirmatory result under this design —
nothing below changes it retroactively.**

**Amended by external review (see "Post hoc reuse of the test set,"
below): the test set was subsequently reused, after this locked
evaluation, in explicitly post hoc classifier-backend audits (a
JAX/optax port cross-check, and an NVIDIA cuML `accel` cross-backend
replication). Neither one altered this locked analysis, or supplied a
new confirmatory claim, but the original framing here -- "will not be
touched again" -- is no longer accurate, and is corrected here rather
than left standing.**

## Setup: no new model fitting, one refit at an already-selected C

Per `DESIGN.md`, each condition's regularization `C` was already
selected using full-training-set 5-fold cross-validation in feasibility
stage 3 — `{raw_pixels: 0.001, encoded_pre_evolution: 0.01, evolved_T:
1000, evolved_lattice: 1000, evolved_rewired: 10, evolved_curr_random:
1}`. No new hyperparameter search was run. This step does exactly one
thing per condition: fit a fresh scaler and classifier on the *complete*
60,000-image official training set at that already-locked `C`
(`stage2a_classifier.fit_final_at_selected_C`, factored out of the
existing `fit_condition` so no cross-validation-search code path could
be accidentally re-triggered), then apply that fit, unchanged, to the
official test set's 10,000 images — encoded and evolved on GPU exactly
as stage 3's training data was (`run_official_test_encode.py` plus
`evolve_on_graph_jax.py`, zero solver failures across all 4 topologies,
19.1s total GPU evolution for 10,000 images x 4 topologies).

The six final refits took between 8.4s (`raw_pixels`) and 458.3s
(`evolved_T`) — `evolved_T`'s refit was the single slowest, consistent
with it also being the condition whose selected `C=1000` sits closest to
the `max_iter` limit (5,123 of 10,000 iterations used, the most of any
condition).

## Test-set performance, all six conditions

| condition | C | test accuracy | macro-F1 | mean log-loss |
|---|---:|---:|---:|---:|
| raw_pixels | 0.001 | 0.6960 | 0.6976 | 0.9848 |
| encoded_pre_evolution | 0.01 | 0.7208 | 0.7221 | 0.9558 |
| evolved_T | 1000 | 0.8058 | 0.8065 | 0.7067 |
| evolved_lattice | 1000 | 0.7778 | 0.7787 | 0.7815 |
| evolved_rewired | 10 | 0.8183 | 0.8191 | 0.6739 |
| evolved_curr_random | 1 | 0.8221 | 0.8229 | 0.6509 |

## Primary result: T-evolved vs. encoded-pre-evolution -- IMPROVEMENT, stated plainly

`d_i = ell_i(evolved T) - ell_i(pre-evolution)`, using 20,000 paired,
class-stratified bootstrap resamples of the 10,000 official test images:

**Observed mean d_i = -0.2491. 95% percentile interval: [-0.2721,
-0.2266]. The entire interval is below zero.**

Per `DESIGN.md`'s pre-registered success rule, this is unambiguously an
**improvement** — not a result that needs interpretation, or a
borderline call. Graph evolution on T, on top of the already
dynamically-encoded pre-evolution state, reduces mean per-image test
log-loss by a large margin that does not straddle zero. Secondary
confirmation using an exact McNemar's test, on classification
disagreement: of the 1,618 test images where the two conditions'
predictions disagreed, **1,234 were correct only under evolved_T**,
versus 384 correct only under pre-evolution (p = 6.68e-104) — this is
consistent in direction and scale with the log-loss result, not a
separate or conflicting story.

This resolves the question this whole design existed to ask, in the
positive direction, for the first time in this project's history: graph
evolution demonstrably adds classification value beyond the local
encoding dynamics alone, under this task and this linear readout —
"dynamics not useful" (`DESIGN.md`'s named outcome 3) is directly
contradicted by this result, and "topologies equivalent" (named outcome
1) is also inconsistent with it, since T's evolution alone produces
this large, non-straddling improvement.

**Amended by external review**: the primary, locked comparison here is
`evolved_T` versus `encoded_pre_evolution` only, and *that* comparison
is what this bootstrap interval and McNemar test directly support —
"dynamics useful" is established. It does **not**, on its own,
establish outcome 2's stronger second half, "one specific graph wins":
that would need directly testing one evolved graph against another (for
example, `evolved_curr_random` versus `evolved_rewired`), which was
never run. The secondary comparisons below test each of the other three
graphs separately against the same `pre-evolution` baseline —
non-straddling intervals there establish that each graph individually
beats pre-evolution, not that the graphs differ significantly from each
other. See "Secondary comparisons," below, for the corrected framing.

## Secondary comparisons: all three other graphs also improve, none rescuing (or needed to rescue) anything

Using the same direction rule and bootstrap procedure, each compared
against `encoded_pre_evolution`, with no cross-comparison correction
(`DESIGN.md`: none of these is a second chance at the primary claim, so
none needed correcting against the others):

| comparison | observed mean d_i | 95% CI | verdict | McNemar p |
|---|---:|---|---|---:|
| evolved_lattice vs. pre | -0.1743 | [-0.1930, -0.1557] | IMPROVEMENT | 1.55e-56 |
| evolved_rewired vs. pre | -0.2819 | [-0.3074, -0.2570] | IMPROVEMENT | 9.76e-133 |
| evolved_curr_random vs. pre | -0.3049 | [-0.3303, -0.2797] | IMPROVEMENT | 8.42e-138 |

All four prespecified graph instances improved over encoded
pre-evolution, with entirely non-straddling intervals. **Amended by
external review**: the tests performed here compare each graph
separately against the common `pre-evolution` baseline; they do not
directly test one evolved graph against another (for example,
`ell(curr_random) - ell(rewired)`), and separate non-straddling
intervals against a shared baseline do not by themselves establish that
two evolved graphs differ significantly from one another. The
originally stated "the four graphs are not equivalent" claim overstated
what these tests support, and has been removed. What the data do
support, stated descriptively rather than as a confirmatory ranking
claim: their observed test-set effects differed, with `curr_random`
producing the lowest log-loss (-0.305), followed by `rewired` (-0.282),
`T` (-0.249), and `lattice` (-0.174, smallest but still a clear
improvement). Graph-to-graph superiority was not a confirmatory question
under `DESIGN.md`'s locked design, and is reported here descriptively,
not inferentially. A formal task-utility ranking would need direct,
paired graph-to-graph comparisons — these could be computed from the
already-saved per-image losses, but any such comparison would now be
explicitly post hoc, and should use multiplicity correction across
however many pairwise tests it involves.

**A genuine, small, honestly-reported rank swap between the
cross-validation-selection data and the held-out test set**:
feasibility stage 3's training-cross-validation delta-versus-pre-evolution
ranking (most to least improvement) was `curr_random (-0.235) > T
(-0.219) > rewired (-0.213) > lattice (-0.153)` — T ahead of rewired. On
the actual held-out test set, the order swaps exactly those two
positions: `curr_random > rewired > T > lattice`. `curr_random`
(largest) and `lattice` (smallest) are stable across both; only the
middle two trade places. This is a modest, reportable instability in the
*exact* middle-of-the-pack ordering between model-selection data and
truly held-out data — not a reversal of the overall finding, and not
treated as a family-level claim regardless (`DESIGN.md`'s
fixed-prespecified-graph-instances scope), but worth stating rather than
quietly picking whichever ordering looks cleaner. **This swap directly
reinforces the caution above against a graph-to-graph superiority
claim**: if `T` and `rewired`'s relative order is not even stable
between the cross-validation-selection data and the held-out test set,
treating their point-estimate ordering here as a confident ranking
(rather than a descriptive observation) would be reading too much into
noise the data itself already shows is real. The broad pattern -- all
four graphs clearly beat pre-evolution, `curr_random` and `lattice`
anchoring the top and bottom -- is stable; the exact middle ordering is
not.

**On the Stage 1D dissociation, stated precisely rather than loosely**:
Stage 1D found no detectable differences in internal mapping strength
across these topology constructions (a different metric -- a paired
bootstrap on the tangent-departure response measure -- and a different
sample). Stage 2A found visibly different downstream task effects among
these specific graph instances, as the table above shows. These two
findings are not in tension -- "no detectable difference in one metric"
and "a visible difference in another" can both be true of the same
graphs -- but this document does not treat it as a formal, confirmatory
task-utility dissociation, since that would itself need the direct,
paired graph-to-graph comparisons flagged as not yet run above (and, if
computed post hoc from the saved per-image losses, corrected for
multiplicity).

**`rewired`'s near-total phase synchronization (Result 2, above: R_post
between 0.986 and 1.0 for every one of the 60,000 training images) is
now confirmed, on genuinely held-out test data, not to prevent it from
being the second-strongest of the four evolved conditions.** This
extends stage 3's training-cross-validation surprise to the one
evaluation that actually matters: extreme synchronization under this
topology does not erase class information that remains **linearly
decodable under the locked, high-precision, per-feature-standardized
pipeline used throughout this design.** That qualification is precise,
not decorative: when phases become nearly synchronized, the informative
differences between images can be small in absolute magnitude, and
`StandardScaler`'s per-feature normalization can amplify a consistent,
low-variance leftover difference into a coordinate the classifier can
use. That is legitimate predictive computation under this exact
pipeline, not an artifact — but it does not, on its own, establish
robustness to feature quantization, phase noise, lower solver precision,
or small perturbations at inference time. Those are open questions this
result motivates, not ones it answers; a useful follow-up before making
any stronger physical-computation or hardware-robustness claim about
this topology, not a gap in the claim actually made here.

## The class-0 confound `DESIGN.md` flagged: checked directly -- reassuring, not conclusive

`DESIGN.md` warned that the primary comparison, while cleaner than
initially thought (both conditions already share T's class-0-derived
set of active nodes), "still may benefit class 0 differently … under
evolution on T," and locked per-class recall as a required output for
exactly this reason. Checked directly: the per-class recall change
(evolved_T minus pre-evolution), across all 10 classes, is `[0.089,
0.102, 0.038, 0.118, 0.103, 0.089, 0.056, 0.095, 0.072, 0.088]` — **class
0's improvement (+0.089) ranks 5th of 10, squarely in the middle of the
distribution** — not the largest (class 3, +0.118), or the smallest
(class 2, +0.038). No obvious class-0-specific recall advantage was
observed; the improvement is broadly spread across classes, not
concentrated in the one class whose topology happens to be under test.

**Amended by external review**: this supports "no obvious
class-0-specific recall advantage was observed," but does not fully
support the stronger claim originally implied by this section's
heading, "the class-0 confound has been ruled out." A class-0-derived
support or edge structure could still provide generic features useful
across several classes, without producing a class-0-specific *recall*
spike — recall alone does not rule that out. More importantly, this
issue does not threaten the primary causal comparison: the primary
comparison already holds the class-0-derived set of active nodes fixed
across both conditions (`evolved_T` and `encoded_pre_evolution` share
the identical set of nodes), so whatever class-0-derived structure
exists is common to both sides of the comparison, not a confound of it.
What remains open is only the *interpretation* of what makes T's
evolution useful, not the primary result itself. A stronger descriptive
check, not yet run, would report per-class mean log-loss differences
rather than recall alone — this is computable directly from the
already-saved `ell_i` values, with no refitting or any new
model-selection decision.

## Baselines (context only -- never part of the locked primary or secondary comparisons)

| baseline | params | test accuracy | macro-F1 | log-loss |
|---|---:|---:|---:|---:|
| raw pixels (linear) | 7,850 (784x10+10) | 0.6960 | 0.6976 | 0.9848 |
| MLP, H=13 (parameter-matched) | 10,345 | 0.7534 | 0.7544 | 0.8971 |
| MLP, H=128 (competent context) | 101,770 | 0.8863 | 0.8863 | 0.6160 |

**At approximately matched trainable-parameter count** (the oscillator
readout: 10,090 parameters; the `H=13` MLP: 10,345, per `DESIGN.md`'s
own matching), **every one of the four evolved conditions outperforms
this MLP** — even the weakest, `evolved_lattice` (77.78% accuracy), beats
`MLP_H13`'s 75.34% by 2.4 percentage points, and the strongest,
`evolved_curr_random` (82.21%), beats it by 6.9 percentage points. This
is a genuinely favorable comparison for the oscillator representation,
reported descriptively per `DESIGN.md`'s context-only framing for
baselines, not as a locked claim.

**Amended by external review**: `H=13` is matched only on trainable
parameter count in the final linear readout — it is not matched on
frozen graph parameters, or the data-derived structure the graph itself
encodes, or preprocessing capacity, or inference compute, or the
training and hyperparameter-search budget. The wording "equally-sized
ordinary network" overstated how much this comparison actually controls
for, and has been replaced with "an MLP with approximately matched
trainable parameter count" throughout this section. The result remains
favorable contextual evidence for the oscillator representation, but it
is not a complete compute- or model-capacity-matched comparison; the
separate compute-cost design (`COMPUTE_COST_DESIGN.md`) is the right
place to settle that more rigorously.

**Stated plainly, the other direction**: a larger, competently sized MLP
(`H=128`, about 10 times the oscillator readout's parameter count)
reaches 88.63% test accuracy — clearly ahead of every oscillator-evolved
condition (the best is `curr_random` at 82.21%). The oscillator dynamics
improve substantially on the pre-evolution baseline, and beat an MLP
with approximately matched trainable parameter count, but they do not
close the gap to a larger, competently sized one. Both facts are true at
once, and neither one softens the other.

## What this settles, and what it does not

**Settled**: for this task, this linear readout, and these four
prespecified graph instances — graph-level evolution, on top of an
already dynamically-encoded phase state, adds real, statistically
unambiguous classification value on genuinely held-out data. This is the
strongest positive Level 3 result this project has produced, and the
first to survive contact with an official, untouched test set, rather
than training-derived validation data alone.

**Not settled, and explicitly out of scope for this design** (per
`DESIGN.md`'s "What this does not do"): whether this generalizes to a
topology *family* rather than these four specific prespecified
instances; whether the genuinely static `theta_static = pi*x` control
(no local-convergence encoding at all) would show the local encoding
step already carries most of the value; role-matched or per-class
topology selection (circular for a real classifier, rejected by design);
and denoising or generation (Stage 2B, deferred).

This locked scikit-learn evaluation is the one and only *confirmatory*
official-test-set evaluation for this design — no further confirmatory
evaluation against these 10,000 test images is planned or justified
under this locked design. See "Post hoc reuse of the test set," below,
for what has and has not happened to the test set since.

## Post hoc reuse of the test set (amended by external review)

**The original framing above -- that the official test set was touched
once and would never be touched again -- is no longer factually
correct, and is corrected here rather than left standing.** The locked
scikit-learn evaluation documented in this section was the first use of
the official test set, and remains the sole confirmatory result;
nothing below changes that analysis or its numbers. But the test set was
subsequently reused twice, both explicitly post hoc, and both
classifier-*backend* audits rather than new scientific investigations:

1. **The JAX/optax classifier port** (`JAX_CLASSIFIER_PORT_FINDINGS.md`)
   evaluated its from-scratch JAX reimplementation of the classifier fit
   against the cached official test set, at the three real selected `C`
   values from this locked result (`evolved_T=1000`,
   `evolved_rewired=10`, `evolved_curr_random=1`). It found a real,
   unresolved gap from scikit-learn's fitted solution that grows with
   `C` and does not close with recalibration — disclosed there, checked
   at the actual selected `C` values (not just a grid extreme), and
   explicitly **not** used for, or folded into, any result reported in
   this document.
2. **The NVIDIA cuML `accel` cross-check** (`CUML_ACCEL_FINDINGS.md`)
   replicated the complete six-condition, 270-fit cross-validation-grid
   procedure under a different, GPU-native solver backend, and evaluated
   it against the same official test set, reaching the same four
   verdicts (the primary comparison and all three secondary comparisons)
   at closely matching effect sizes.

Neither audit altered the locked scikit-learn analysis above, or
supplied a new confirmatory scientific claim. But as a factual matter,
the test set is no longer untouched for future Stage 2A development, and
this document should not claim otherwise.

**This also changes how the cuML result should be described.** It is a
strong **cross-backend implementation robustness check**: it shows the
positive verdict is not specific to scikit-learn's particular optimizer.
It is **not independent scientific corroboration** of the result,
because it reuses the same training and test samples, the same
oscillator features, the same graph instances, the same folds and `C`
grid, and the same high-level selection and fitting code — only the
numerical classifier backend differs. `CUML_ACCEL_FINDINGS.md` itself
has been amended to use this framing throughout, in place of the
"independent confirmation" or "replication" language it originally
used.

3. **A post hoc, exploratory graph-to-graph pairwise comparison** (below,
   "Post hoc, exploratory: direct graph-to-graph pairwise comparison") —
   the third reuse of the test set, and the first that computes a new
   statistic (a paired bootstrap directly between two evolved graphs)
   rather than auditing an existing one against a different backend. It
   used no new simulation or GPU time: it was computed entirely from
   the per-image test losses this locked evaluation already produced and
   saved.

## Post hoc, exploratory: direct graph-to-graph pairwise comparison

**Explicitly post hoc and exploratory -- not part of, and does not
reopen or alter, the locked confirmatory result above.** This was
prompted directly by this document's own secondary-comparisons section,
which flagged that a graph-to-graph superiority claim "would require
direct, paired graph-to-graph comparisons … these could be computed from
the already-saved per-image losses, but any such comparison would now be
explicitly post hoc and should use multiplicity correction." This
section does exactly that, properly, rather than leaving it as an
unresolved caveat.

**No new simulation, no new GPU time**: `ell_i` for all four evolved
conditions was already computed and saved by
`run_confirmatory_evaluation.py` (`results/stage4_confirmatory_results.pkl`).
This is a new bootstrap computation on existing data only
(`run_posthoc_graph_pairwise.py`).

**Method, amended by external review**: the original version of this
section used one bootstrap procedure for both the descriptive interval
and the `p`-value feeding the Holm correction. External review found
that combination wrong: a `p`-value read off a bootstrap distribution
centered on the *observed* effect (how often it crosses zero) is closely
related to inverting the percentile confidence interval — this is
adequate for the locked primary and secondary tests (where the
pre-registered decision rule *is* "does the 95% CI exclude zero," so the
two agree by construction), but it is not a properly null-calibrated
`p`-value suitable for a family-wise-error claim in this new,
un-pre-registered, six-comparison family. Two statistics are now
computed separately:

1. **Descriptive interval** (unchanged): `d_i = ell_i(graph_A) -
   ell_i(graph_B)`, using 20,000 paired class-stratified bootstrap
   resamples, a two-sided 95% percentile interval on mean `d_i`, with
   the same `seed=42`.
2. **Inferential `p`-value, for Holm correction**: a paired sign-flip
   permutation test (`stage2a_stats.paired_sign_flip_p`). This
   independently flips each image's `d_i` sign, which directly builds a
   null distribution that actually destroys the effect being tested for
   (`CLAUDE.md` principle 10: "a permutation scheme must actually
   destroy the effect it's testing for under the null"), unlike the
   discarded combined procedure above. This was unit-tested on synthetic
   data first (`tests/test_stage2a_stats.py`), per that same principle's
   explicit requirement: identical (zero-difference) input gives `p`
   close to 1; maximally separated (large constant difference) input
   gives `p` at the Monte Carlo floor (the smallest p-value the test can
   report at a given number of permutations).

   **Exactness caveat**: this test's exact validity requires each `d_i`
   to be exchangeable with `-d_i` under the null hypothesis — a symmetry
   condition on `d_i`'s distribution around zero, which is stronger than
   merely "no systematic difference" (`E[d_i]=0`), and which is not
   separately verified here. With `n=10,000` independent test images,
   this is a large-sample-justified approximation for the mean contrast,
   not an exact test by construction alone
   (`stage2a_stats.py`'s `paired_sign_flip_p` docstring carries the same
   caveat). This does not matter for the five comparisons below whose
   `p`-values sit at or near the Monte Carlo floor — approximation error
   is swamped by effect size there — but it is directly relevant to the
   one comparison that sits close to the alpha=0.05 boundary (`rewired`
   versus `curr_random`, below): "significant under this test" there
   should be read as "significant under a large-sample approximation,"
   not as an exact result.

All six pairwise comparisons among the four evolved graphs were run, not
a subset chosen after seeing which looked interesting. Because this is
new, un-pre-registered, multiple-comparison territory (`DESIGN.md` only
locked the four graph-versus-pre-evolution tests, never graph-versus-
graph), **Holm-Bonferroni correction across all six sign-flip-test
`p`-values, treated as one family, is applied** — this is not optional
here, per this project's own standing rule (`CLAUDE.md` principle 3).

| comparison | mean `d_i` | 95% CI (descriptive) | sign-flip raw `p` | Holm-adjusted `p` | survives (alpha=0.05) |
|---|---:|---|---:|---:|---|
| T vs. lattice | -0.0748 | [-0.0935, -0.0564] | 4.9998e-05 | 2.9999e-04 | **yes** |
| T vs. curr_random | +0.0558 | [+0.0307, +0.0803] | 4.9998e-05 | 2.9999e-04 | **yes** |
| lattice vs. rewired | +0.1076 | [+0.0844, +0.1312] | 4.9998e-05 | 2.9999e-04 | **yes** |
| lattice vs. curr_random | +0.1305 | [+0.1070, +0.1543] | 4.9998e-05 | 2.9999e-04 | **yes** |
| T vs. rewired | +0.0328 | [+0.0088, +0.0572] | 7.4496e-03 | 1.4899e-02 | **yes** |
| rewired vs. curr_random | +0.0229 | [+0.0002, +0.0456] | 4.6148e-02 | 4.6148e-02 | **yes, marginal** |

(Sign convention: a positive `d_i` means the first-named graph's
log-loss is higher, meaning the second-named graph wins that pair.
Sign-flip `p` values are computed using 20,000 permutations, using the
Monte Carlo floor convention -- `4.9998e-05` is the floor at `N=20,000`
permutations, not an exact zero.)

**All six of the six comparisons survive Holm correction at
alpha=0.05, under the corrected sign-flip test — which genuinely
destroys the effect under its permutation null (unlike the discarded
bootstrap-`p` approach it replaced), subject to the exactness caveat
noted above — this is the same qualitative conclusion as before the
correction, not a different one, but it now rests on a method that
actually supports the claim.** Five of the six are extremely well
separated (`p` at or near the Monte Carlo floor, surviving correction
by a wide margin — the exactness caveat does not matter at this
margin). The sixth — `rewired` versus `curr_random`, the two closest
performers — remains genuinely marginal, and is exactly where that
caveat matters: `p=0.0461` under the primary seed, and because it is
the largest `p`-value in the family, it receives no Holm penalty
(correction factor 1) and survives at essentially its raw value, just
under the 0.05 threshold. **Checked for stability, not just reported
once**: re-run across 4 additional random seeds, and at 50,000 and
100,000 permutations, `p` ranged 0.0459-0.0497 — consistently just
under 0.05, not flipping sign under the resampling randomness itself,
but with essentially no margin. Worth naming plainly, as before: this
is a significant result, not a robust one — a genuinely different draw
of the underlying test-set images (not just resampling randomness
within this one fixed test set) could plausibly flip it.

**The full pairwise ranking is transitive and internally consistent**:
`curr_random > rewired > T > lattice`, exactly matching the descriptive
point-estimate ordering already reported in "Secondary comparisons,"
above — and now, for the first time, that ordering rests on a properly
powered, multiplicity-corrected, direct pairwise test, not a
point-estimate comparison against a shared baseline. This chain has
three adjacent links, and they do not all carry the same weight:
`T`-versus-`lattice` is at the Monte Carlo floor (well separated);
`T`-versus-`rewired` is Holm-significant but not at the floor (raw
`p=7.4496e-03`); and `curr_random`-versus-`rewired` — the same pair as
the `rewired vs. curr_random` row in the table above — is the one
genuinely marginal link (`p=0.0461`, already discussed). **Restated
precisely, now that it is justified**: `curr_random` (a topology with
matched sparsity but no relationship to `T`'s learned structure)
measurably outperforms `T` (the topology this whole design was built
around) on this held-out test set (`d = +0.0558`, `[+0.0307, +0.0803]`,
Holm-survives comfortably) — a real, corrected, direct result, not the
descriptive observation it was before this check.

**What this does and does not establish, restated for this specific
addendum**: this confirms the four graphs are not equivalent in task
utility under this exact pipeline, on this one test set, for this one
locked feature and classifier procedure — it does not extend to a
topology-*family* claim (still explicitly out of scope, per `DESIGN.md`
and every prior section here), and it does not retroactively change the
primary or secondary locked results above, which stand as reported
regardless of this addendum's outcome.

## Code

`run_official_test_encode.py` (the local CPU encode of the official test
set, mirroring `run_feasibility_stage3_encode.py`),
`run_confirmatory_evaluation.py` (the confirmatory analysis itself:
final refits, the primary and secondary bootstrap, McNemar's test, MLP
baselines), `stage2a_classifier.py`'s new `fit_final_at_selected_C` (the
refit-only half of `fit_condition`, factored out so no new
cross-validation search could run here even by accident). A remote-only
GPU driver script (`stage4_gpu_evolve.py`, using the same chunked
approach as stage 3's) was not committed, per this project's convention
for ephemeral GPU-session code. `run_posthoc_graph_pairwise.py` (the
post hoc graph-to-graph pairwise comparison above — no GPU dependency,
reuses only the already-saved per-image losses).

## Reproducibility gaps (flagged by external review, now closed)

**This was not a blocker to accepting the scientific finding above -- it
was a blocker to describing this branch as fully reproducible from its
public contents.** All five items external review flagged are now
implemented, verified, and committed (not merely planned):

- **Artifact paths parameterized** (`stage2a_paths.py`): every script
  that reads or writes the large scratch files now finds its own
  directory through `train_scratch_dir()`/`test_scratch_dir()`, which
  can be overridden with `STAGE2A_SCRATCH_ROOT`, rather than a hard-coded
  private path. The default location is documented in `README.md`.
- **Both exact GPU evolution driver scripts are committed**:
  `stage4_gpu_evolve.py` (test-set, the one flagged) and
  `stage3_gpu_evolve.py` (training-set — equally important for the same
  reason, committed alongside for the same completeness, not explicitly
  named in the original recommendation but the identical gap). Neither
  one is runnable locally as-is (both run on a remote Colab kernel's
  `/content/...` file layout) — `README.md`'s "Reproducing the
  confirmatory GPU evolution" section documents the exact
  `mighty-colab` upload-and-run sequence, including the chunked-upload
  workaround for the transfer endpoint's size limit.
- **Artifact manifest** (`generate_artifact_manifest.py`, run against
  the real cached files — not a template):
  `results/ARTIFACT_MANIFEST.json` records SHA256 hashes for every
  pickle file the confirmatory result depends on, per-topology
  adjacency and evolved-state hashes, training and test dimensions,
  image-ordering checks (label array hashes, `idx == arange(n)`
  confirmation), the selected `C` values actually consumed
  (`{raw_pixels: 0.001, encoded_pre_evolution: 0.01, evolved_T: 1000,
  evolved_lattice: 1000, evolved_rewired: 10, evolved_curr_random: 1}`),
  and the frozen primary effect itself.
- **Unit tests** (`tests/test_stage2a_stats.py`, 17 tests, all passing):
  Tier 1 synthetic-data coverage for the paired class-stratified
  bootstrap (including a deterministic zero-variance construction that
  directly verifies genuine per-class stratification, not just
  plausible-looking output), per-image log-loss class indexing
  (including a non-default class-ordering case that would silently
  mis-index on a positional bug), exact McNemar's contingency
  construction, the bootstrap-derived p-value, and Holm-Bonferroni
  (including a step-down-stopping edge case that a naive implementation
  could get wrong — this caught a real error in the test's own first
  draft, not the implementation, when first run).
- **Artifact-backed regression test**
  (`test_frozen_primary_effect_matches_findings_md`, Tier 2, skips
  cleanly if the confirmatory pickle is not present locally): this
  recomputes the primary bootstrap directly from the already-saved
  per-image losses and confirms it still lands at `d_i = -0.2491`,
  `CI = [-0.2721, -0.2266]` — it passes now, and it will catch a future
  code change that silently changes the statistic.

The statistics functions themselves were also consolidated during this
work: `run_confirmatory_evaluation.py` originally defined
`paired_class_stratified_bootstrap` and the other statistics itself, and
`run_posthoc_graph_pairwise.py` had copied the bootstrap function
directly, word for word, rather than importing it (this was disclosed as
deliberate at the time, for exact-fidelity reasons) — both now import
from the new `stage2a_stats.py`, so there is exactly one implementation
to trust and test, not two that could silently drift apart. The team
re-ran the post hoc pairwise comparison after the refactor and confirmed
bit-identical output against the pre-refactor numbers reported above.

See `README.md` (new, added alongside this closure) for the full
reproduction sequence, the `mighty-colab`/Colab-A100 GPU-session
pattern, and the public GCS artifact cache
(`gs://bonsai-2026-stage2a-cache`) this closure also made use of.

## Next step

None specified by this design — the locked confirmatory evaluation is
complete, and the reproducibility gaps that were the last open item are
now closed (immediately above). Any further extension (topology-family
generalization, the static-encoding control, Stage 2B denoising) is a
new design decision, not a continuation of this one.

## External review verdict (amendment record)

An external review of this section's original text found the primary
scientific result sound, and recommended the specific corrections
incorporated throughout this document (rather than a rewrite from
scratch) — summarized here as the record of what was reviewed and what
changed.

**Verdict**: the primary Stage 2A result survives review. The locked
comparison, statistical procedure, classifier selection, final refit,
and reported result align correctly with `DESIGN.md`'s pre-registered
specification. The observed primary effect (`d_i = ell(evolved T) -
ell(pre-evolution) = -0.2491`, `95% CI = [-0.2721, -0.2266]`), the
accuracy increase (72.08% to 80.58%), and the concordant McNemar result
together support a bounded Level 3 claim: under this fixed support,
graph instance, encoding, evolution horizon, feature method,
standardization procedure, and linear readout, graph-level evolution
produces a substantially more useful classification representation than
the encoded pre-evolution state alone.

**What review changed**: two overstated inferential claims were
corrected (the "one specific graph wins" and "the four graphs are not
equivalent" claims, and the "test set touched only once" claim), and one
reproducibility gap was documented as open, rather than implicitly
assumed closed. Three smaller wording precisions were also made: the
class-0 diagnostic is described as reassuring rather than conclusive,
the MLP baseline comparison is described as approximately
parameter-matched rather than "equally-sized," and the
locked-pipeline-specific decodability finding under near-total
synchronization is stated with its precision qualifier, rather than as
an unqualified claim about the underlying representation.

**What review did not change**: the primary confirmatory result itself,
the JAX/optax investigation's handling (already meeting this project's
evidentiary standard — the unresolved performance gap is disclosed,
checked at the real selected `C` values rather than only a grid
extreme, and explicitly excluded from any reported result), or the
overall accept decision. Side investigations are, per this review,
appropriately honest, with the cuML cross-check now correctly framed as
cross-backend implementation robustness rather than independent
scientific replication.

**Strongest defensible conclusion, as amended**: Stage 2A establishes
external task utility for runtime graph-oscillator evolution, under a
bounded classification design. On the official KMNIST test set,
evolution on the prespecified class-0-learned graph `T` reduced mean
per-image log-loss by 0.2491, relative to the same dynamically encoded
phase state before graph evolution, with a paired class-stratified 95%
bootstrap interval of `[-0.2721, -0.2266]`. Accuracy increased from
72.08% to 80.58%, with concordant McNemar evidence. All four tested
graph instances improved descriptively over pre-evolution, but
graph-to-graph superiority and topology-family generality were not
confirmatory claims under this design. The oscillator representation
beats an MLP with approximately matched trainable parameter count, while
remaining clearly below a larger, competently sized MLP.
