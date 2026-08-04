# Stage 2A: Feasibility Stage 1

**Status: mechanical validation only, per `DESIGN.md`'s own explicit
framing. This is NOT a scientific result and must not be read as one.**
Its job is to confirm the pipeline runs correctly end-to-end at a small
scale before scaling up -- effect direction is recorded below because
`DESIGN.md` permits it, but nothing here is confirmatory.

## Scope

1,000 official KMNIST training images (100/class, class-stratified,
`SEED=42`), drawn deterministically from the official training split
only -- the official test set was never loaded in this stage. Three
conditions, per `DESIGN.md`'s locked pipeline:

1. Raw pixels (784-dim).
2. Encoded, pre-graph-evolution (`_local_converged_phases`, restricted
   to T's 505 active nodes, reference-node gauge, 1008-dim).
3. Evolved on T (`T_HORIZON=2.5`, same gauge, 1008-dim).

## Go/no-go mechanical checks (all passed)

- **Solver failures: 0/1000 (0.0%)**, well under the locked 0.1%
  tolerance. Every graph-evolution ODE solve succeeded on its primary
  attempt (`RK45`, `max_step=0.05`) -- the recovery policy (smaller
  `max_step`, then `Radau`) was never invoked.
- **Non-finite feature vectors: 0.** Every raw-pixel, pre-evolution, and
  evolved-T feature vector was fully finite.
- **`R(theta)` diagnostic** (order parameter, pre- and post-evolution,
  reported per `DESIGN.md` -- never used to alter the gauge):
  - Pre-evolution: min 0.403, max 0.979, mean 0.735, median 0.743. 0
    images below 0.01, 0 above 0.99.
  - Post-evolution: min 0.616, max 0.992, mean 0.859, median 0.869. 0
    images below 0.01, **1 of 1000 above 0.99** (max 0.9922). A single
    boundary case, not a mass concentration at the numerical limit --
    disclosed as required, not flagged as a pathology.
  - Evolution consistently increases `R` (mean 0.735 -> 0.859) across
    this sample -- plausible given T's coupling drives phases toward
    local synchrony, reported as an observation, not interpreted further
    at this stage.
- **Reference-node gauge**: the two trivially-constant columns
  (`theta_ref`'s own `cos=1, sin=0`) were confirmed dropped on every
  image via the assertion inside `reference_node_features()` (also
  covered by `tests/test_stage2a_core.py`) -- effective feature
  dimension 1008 throughout, as designed.
- **Classifier convergence**: all three conditions converged in every
  fold/`C` combination of the locked 5-fold-CV x 9-value grid (45 fits
  per condition, 135 total) -- the `NonConvergenceError` stop-gate was
  never triggered.

**Overall: GO.** Pipeline runs correctly end-to-end at this scale;
nothing here blocks proceeding to feasibility stage 2.

## Effect direction (descriptive only, not inferential)

Mean 5-fold cross-validated log-loss at each condition's own
CV-selected `C` (selected independently per condition, per the locked
procedure):

| condition | dim | selected C | mean CV log-loss at selected C |
|---|---:|---:|---:|
| raw pixels | 784 | 0.01 | 0.804 |
| encoded, pre-evolution | 1008 | 0.01 | 0.843 |
| evolved on T | 1008 | 0.01 | 0.804 |

At this sample size (1,000 images, 100/class, no held-out test set --
this is a training-side CV number, not the locked primary endpoint),
evolved-T's mean CV log-loss is nominally equal to raw pixels' and
nominally *lower* than the pre-evolution condition's. **This is not
evidence of anything** -- `DESIGN.md` is explicit that the feasibility
ladder may report raw differences descriptively without formal
inference, and 1,000 images with no dedicated held-out set is far too
small and methodologically different from the locked confirmatory test
(20,000-resample paired bootstrap against 10,000 official test images)
to support any claim in either direction. Recorded here only because
the design permits recording it, not because it means anything yet.

## Runtime

1,000-image pipeline (encode + restrict + evolve + both gauge
computations, parallelized across `cpu_count()-1` workers): **65.7s
(65.7 ms/image)**. Classifier fitting (3 conditions x 45 fold/C fits
each): a few seconds, not separately timed. Extrapolating linearly to
stage 2 (5,000 images) and stage 3 (60,000 images) suggests roughly
5.5 minutes and 66 minutes respectively for the pipeline stage alone --
a rough projection to revisit with stage 2's own measurement, not a
locked estimate.

## Code

`stage2a_core.py` (encoding/restriction/evolution/gauge features),
`stage2a_classifier.py` (locked CV/standardization/classifier
procedure), `run_feasibility_stage1.py` (this stage's driver),
`tests/test_stage2a_core.py` (reference-node constant-column property
and evolution-recovery-policy ordering, Tier 1 + Tier 2). Raw results
(`results/stage1_feasibility_results.pkl`) are gitignored, regenerable
by re-running `run_feasibility_stage1.py`.

## Next step

Feasibility stage 2 (up to 5,000 official-training images, throughput
measurement, and the encoder-seed robustness check), per `DESIGN.md`'s
locked ladder -- see below.

# Stage 2A: Feasibility Stage 2

**Status: complete (after halting once on a real, investigated, and
resolved non-convergence).** Initially halted per `DESIGN.md`'s own
locked stop-gate ("any non-converged fit during a required fold/C
combination stops advancement to the next stage, pending
investigation"); the cause was diagnosed (below), `max_iter` raised from
1,000 to 10,000 as a disclosed post-lock amendment, and the full stage
(primary CV + encoder-seed robustness check) re-run and completed with
no non-convergences anywhere. Still mechanical validation only, per
`DESIGN.md`'s own framing -- not a scientific result, same as stage 1.

## What ran before halting (first attempt)

5,000 official KMNIST training images (500/class, class-stratified,
`SEED=42`) -- this is also "the fixed training-derived validation
subset" `DESIGN.md` specifies is reused for every subsequent
development decision and the encoder-seed robustness check. The
official test set was never loaded.

**Throughput**: 5,000 images in 328.3s (65.7 ms/image) -- identical
per-image rate to stage 1 (also 65.7 ms/image), confirming linear
scaling with no unexpected overhead at 5x the image count. Extrapolated
stage-3 (60,000-image) pipeline runtime: **3,939s (65.7 minutes)** --
this projection must be explicitly approved before stage 3 launches,
per `DESIGN.md`'s locked go/no-go criteria; it is reported here, not yet
approved.

**Go/no-go mechanical checks, at 5,000-image scale**:
- Solver failures: 0/5000 (0.0%). No recovery-policy step was ever
  invoked.
- Non-finite feature vectors: 0.
- `R(theta)`: pre-evolution min 0.329, max 0.984, mean 0.731, median
  0.737, 0 below 0.01, 0 above 0.99. Post-evolution min 0.532, max
  0.995, mean 0.857, median 0.866, 0 below 0.01, **4 of 5000 above
  0.99**. Proportionally consistent with stage 1's single boundary case
  (1/1000) -- still a small tail, not a mass concentration, disclosed as
  required.

## The non-convergence, found and characterized

**Primary CV fitting (seed=0) hit a real non-convergence**: the
`evolved_T` condition (1008-dim, n=5000) failed to converge at
`fold=0, C=100.0` (1000/1000 iterations used, `lbfgs` did not reach
`tol=1e-4`). Per the locked stop-gate, this **halted the stage
immediately** -- the script did not proceed to the encoder-seed
robustness check. (An orchestration bug was caught and fixed in the
same session: the driver script initially caught this exception
per-condition and *did* continue on to the robustness check regardless,
contradicting `DESIGN.md`'s "stops advancement to the next stage"
requirement -- fixed before this was reported, not after.)

**Diagnostic scan (not part of the locked pipeline -- exists solely to
characterize this failure before deciding how to respond)**: every one
of the 45 (fold, `C`) combinations for all three conditions was fit
without stopping on the first failure. Result:

| condition | non-convergent (fold, C) pairs out of 45 |
|---|---|
| raw pixels | 0 |
| encoded, pre-evolution | 0 |
| evolved on T | **15** -- all 5 folds, at exactly `C in {100, 1000, 10000}` |

**The pattern is clean and unusual**: only `evolved_T` is affected, and
only at the three weakest-regularization values in the 9-value grid
(the top third) -- every fold fails at exactly the same three `C`
values, and no fold fails at `C <= 10`. At every `C` value observed so
far across both stages (1 and 2), the CV-selected `C` has always been
**0.01** for every condition, with mean validation log-loss rising
sharply and monotonically for `C >= 1` -- the non-convergent region is
nowhere near where `C` would ever actually be selected.

**This does not resolve the stop-gate on its own.** `DESIGN.md`'s rule
is deliberately strict regardless of whether the failing `C` would have
been selected, specifically so "it doesn't matter, that C was never
going to be picked anyway" cannot become a silent, undisclosed
justification for continuing. Whether to raise `max_iter` for this
condition (or region of the grid), narrow the grid, or handle it some
other way is a locked-design-parameter decision, not something to
change unilaterally mid-implementation -- left for a separate decision,
not resolved here.

## Why evolved_T specifically fails to converge: ill-conditioning, not separability

A follow-up diagnostic (`diagnose_stage2_convergence_hypotheses.py`,
also diagnostic-only) tested two candidate explanations directly, on the
exact failing case (evolved_T, fold=0, C=100, same 4000-image training
partition the original failure came from) against the other two
conditions at the same fold and `C` values:

| condition | train acc @ C=100 | converged @ C=100 (n_iter) | train logloss @ C=100 | \|\|coef\|\| @ C=0.01 -> C=100 | condition number |
|---|---:|---|---:|---:|---:|
| raw pixels | **1.000000** | yes (49) | 0.0013 | 3.29 -> 65.29 (19.9x) | 1.0e+02 |
| encoded, pre-evolution | **1.000000** | yes (171) | 0.0022 | 3.29 -> 86.68 (26.3x) | 1.2e+03 |
| evolved on T | 0.990750 | **no (1000)** | 0.0555 | 2.93 -> 186.53 (63.7x) | **2.0e+06** |

**The evidence directly favors ill-conditioning, not near-separability,
and actually argues against the separability hypothesis specifically.**
Both `raw_pixels` and `encoded_pre_evolution` reach **perfect (100%)**
training accuracy on this fold and still converge quickly and easily
(49 and 171 iterations) -- the classic behavior when data is perfectly
separable but well-conditioned. `evolved_T` does **not** reach perfect
training accuracy (99.075%) -- it is the *least* separable of the three
-- yet it is the one that fails to converge. If separability alone
were driving the slow convergence, the two perfectly-separable
conditions should have struggled more, not less.

What `evolved_T` does have is a **condition number of the standardized
training feature matrix ~2,000,000** -- roughly 4 orders of magnitude
worse than `raw_pixels` (100) and 3 orders of magnitude worse than
`encoded_pre_evolution` (1,164), driven by a very small minimum singular
value (6.2e-4 against a max of 1,228). This is consistent with the
`R(theta)` diagnostic already reported above: evolution measurably
increases phase synchronization (mean `R` 0.731 -> 0.857 across this
same 5,000-image set) -- when many nodes' phases move toward similar
values, their `cos`/`sin` reference-node-gauge features become nearly
identical to each other, producing near-collinear (redundant) columns
in the standardized feature matrix.

**This link was tested directly, not left as a plausible story**:
Pearson correlation between each fold-0 training image's `R(theta)` and
its |projection| onto the smallest-singular-value direction (the
specific direction responsible for the poor condition number) gives
**r = -0.170, p = 2.3e-27** (n=4000). The correlation is real and
overwhelmingly statistically significant given the sample size, but
**weak in magnitude** (r^2 ~ 0.03 -- roughly 3% of variance explained)
and, notably, **negative** -- images with *higher* `R(theta)` tend to
have *smaller* projections onto the near-null direction, not larger.
This does not trivially match a naive "the most-synchronized images are
individually the most collinear ones" story. The honest reading: `R`
and the ill-conditioning are related, but `R(theta)` alone is far from a
complete explanation of which images drive the near-null direction --
whatever else determines it is not simply "high vs. low order
parameter," and this diagnostic does not resolve what that additional
structure is.

The coefficient-norm growth (`evolved_T`'s 63.7x ratio vs. 19.9x and
26.3x for the other two) is consistent with, and likely compounds, the
ill-conditioning explanation (an elongated, poorly-conditioned loss
surface makes for larger steps toward a more extreme optimum) -- but
`evolved_T`'s C=100 fit never actually reached convergence (still at
`max_iter=1000` when stopped), so this norm is a mid-optimization
snapshot, not its true optimum; the coefficient-norm comparison alone,
without the condition-number evidence, would be far weaker support on
its own.

**Verdict, stated plainly**: the ill-conditioning hypothesis is
supported by direct, substantial evidence (condition number ~1,700x
worse than the next-worst condition), with a real but weak, and
directionally counter-naive, statistical link to the `R(theta)`
diagnostic. The near-separability hypothesis is not supported -- if
anything, the data points the opposite direction, since `evolved_T` is
the one condition that does *not* achieve perfect training-set
separation. This looks like graph evolution's own synchronizing effect
on the phase state, not incidental separability, producing a genuinely
harder optimization problem for this specific condition at weak
regularization -- and specifically a slow-to-converge one, not an
unconverging one, once given enough iterations (below).

## Resolution: max_iter raised to 10,000, and the mechanism confirmed correctable by more iterations

Given the diagnosis (a real but purely slow-optimization problem, not
divergence or a data pathology), `max_iter` was raised from 1,000 to
10,000 -- uniformly across all three conditions, disclosed as a
post-lock `DESIGN.md` amendment (see that document's own note). This is
not "iterate further and hope": the diagnostic script re-ran the exact
previously-failing case (`evolved_T`, fold=0, C=100) at `max_iter=10000`
and it converged cleanly in **1,185 iterations** -- well inside the new
budget, confirming this was a genuine iteration-count shortfall, not an
unbounded divergence that a larger cap would only postpone.

**The full stage-2 run (primary CV across all 9 `C` values x 5 folds x
3 conditions, plus the encoder-seed robustness check) then completed
end-to-end with zero non-convergences anywhere.** Selected `C` per
condition (unchanged from before for the two unaffected conditions,
confirming the fix didn't perturb anything that already worked):

| condition | selected C | mean CV log-loss at selected C |
|---|---:|---:|
| raw pixels | 0.01 | 0.681 |
| encoded, pre-evolution | 0.01 | 0.697 |
| evolved on T | **0.1** | 0.661 |

(`evolved_T`'s own selected `C` is 0.1, not 0.01 -- its full CV curve
could not be completed at all under the old `max_iter=1000`, so this is
newly-available information, not a changed prior result. Still
descriptive only, per this ladder stage's own framing -- not the locked
primary endpoint.)

## Encoder-seed robustness check (now run)

Reusing this same 5,000-image subset, per `DESIGN.md`'s locked
protocol: recomputed `encoded_pre_evolution` and `evolved_T` features
with each image's encoder seed set to its immutable dataset index
(rather than the shared seed 0), refit on identical folds/`C`-selection:

| condition | seed=0 mean val loss (own C) | independent-seed mean val loss (own C) | change |
|---|---:|---:|---:|
| encoded, pre-evolution | 0.6973 (C=0.01) | 0.6973 (C=0.01) | -0.0000 |
| evolved on T | 0.6607 (C=0.1) | 0.6606 (C=0.1) | -0.0001 |

**Negligible change in both conditions** -- the shared-seed-0 noise
template does not appear to materially affect the classification result
at this scale, for either encoder-derived condition. This is a
descriptive, training-side result (per `DESIGN.md`, cannot replace or
override the seed-0 primary analysis against the official test set),
but it is a reassuring one: the encoder-RNG choice flagged as a caveat
in `DESIGN.md`'s original design does not appear to be driving anything
material here.

**Overall stage-2 result: GO.** All locked go/no-go criteria pass;
throughput (64.6 ms/image, consistent with stage 1's 65.7 ms/image)
projects to ~64.6 minutes for stage 3's full 60,000-image training set
-- this projection still needs explicit approval before stage 3
launches, per `DESIGN.md`'s own requirement, not yet given here.

## A flag for the confirmatory design, not acted on now

If the ill-conditioning mechanism found here is genuinely about
evolution-induced phase synchronization (rather than an artifact of
this particular 5,000-image draw), there is real reason to expect the
top of the locked `C` grid (100, 1000, 10000) to remain persistently
uninformative for evolved conditions specifically, **at any sample
size** -- multicollinearity is a property of relationships between
features, not the ratio of samples to features, so scaling to 60,000
images would not be expected to fix it on its own. This is flagged here
as a candidate for a future, explicitly-documented, separately-reviewed
`DESIGN.md` change (e.g. narrowing the confirmatory grid, or adding a
documented minimum ridge floor) -- **only if the pattern is confirmed to
persist at stage 3's scale**, not assumed or acted on now.

## Code

`diagnose_stage2_convergence.py` (diagnostic-only scan, not part of the
locked pipeline), `stage2a_classifier.py`'s new
`diagnose_convergence_full_grid()` (same caveat),
`diagnose_stage2_convergence_hypotheses.py` (the separability/
ill-conditioning hypothesis test, the `R(theta)` correlation check, and
the `max_iter=10000` re-verification, all diagnostic-only).
`run_feasibility_stage2.py` now halts and saves partial results if a
non-convergence recurs, rather than silently continuing past one; with
`max_iter=10000` it now completes cleanly and saves full results
(`results/stage2_feasibility_results.pkl`, gitignored).

## Next step

Explicit approval of the stage-3 runtime projection (~64.6 minutes),
then feasibility stage 3 (the full 60,000-image official training set,
final feature generation and model selection) -- superseded by the
pre-stage-3 investigation below, which revises both the scope and the
runtime projection before any full run was launched.

# Stage 2A: Pre-Stage-3 Investigation (topology synchronization)

**Status: diagnostic only, no simulation data from a full run -- all of
this uses either a 400-image timing sub-test or zero-simulation
graph-spectral analysis. Not a scientific result.**

Stage 3 requires evolving every image on **four** topologies (T,
lattice, canonical rewired, canonical curr_random -- `DESIGN.md`'s
"Confirmatory expansion"), not just T. A 400-image timing sub-test,
run before committing to the full 60,000-image job, surfaced two things
that needed investigating before launching a multi-hour run.

## Revised runtime projection

400 images across all four topologies (shared encoding, per-topology
evolution): 99.7s (249.3 ms/image) -- projecting to **~249 minutes
(~4.2 hours)** for the full 60,000 images, not the ~64.6 minutes
previously discussed (which covered evolving on T alone; stage 3
requires roughly 4x the evolution-heavy work). Still comfortably
CPU-feasible, no GPU needed -- but a real scope difference from what was
first approved, disclosed here rather than launched silently.

## A striking synchronization asymmetry, investigated before launching

At n=400, `R(theta)` post-evolution differed sharply by topology:

| topology | R_post mean | R_post > 0.99 |
|---|---:|---:|
| T | 0.860 | 0/400 |
| lattice | 0.867 | 0/400 |
| rewired | 0.997 | 400/400 (100%) |
| curr_random | 0.991 | 241/400 (60%) |

Given stage 2's finding that even T's moderate synchrony (R~0.86)
produced a condition number of ~2e6, near-total synchronization under
rewired/curr_random risked a much worse version of the same problem.
Investigated via three checks, all using already-verified code, none
requiring the full 60,000-image run:

**1. Laplacian spectrum / algebraic connectivity (zero new simulation --
purely spectral, on the already-built graphs)**: standard Kuramoto
theory predicts synchronization strength scales with a graph's
algebraic connectivity (the Fiedler value -- second-smallest Laplacian
eigenvalue).

| topology | Fiedler value | connected components |
|---|---:|---:|
| T | 0.0059 | 1 |
| lattice | 0.0056 | 1 |
| rewired | **0.1322** (22x T's) | 1 |
| curr_random (main component) | **0.2429** (41x T's) | **14** (see correction, below) |

**This directly and cleanly explains the synchronization asymmetry**:
both randomized constructions have dramatically higher algebraic
connectivity than T or lattice -- degree-preserving rewiring and
independent resampling both destroy whatever clustered/community
structure T (learned) and lattice (regular) have, and a more
"well-mixed" graph synchronizes faster and more completely. Not a
pipeline artifact; a real, spectrally-confirmed structural difference
between the learned/regular constructions and the randomized ones.

**Correction, prompted by this check**: computing the Laplacian
spectrum for `curr_random` surfaced 14 near-zero eigenvalues, not the 1
expected for a connected graph -- meaning **`curr_random` seed=0 has 13
isolated nodes, not the 1 originally disclosed** in `DESIGN.md`'s graph-
statistics table. The original check only tested `nodes_T`'s three
fixed coordinates for isolation (Stage 1D's own pre-screening scope),
not all 505 nodes. Fixed directly in `DESIGN.md` (see its "Confirmatory
expansion" section) rather than left standing. The main 492-node
component's own Fiedler value (0.2429) is what actually explains the
strong synchronization; the 13 isolated singletons are a separate,
now-correctly-disclosed structural fact about this specific draw.

**2. Information-collapse check**: does a classifier trained on
rewired/curr_random's evolved features collapse to predicting one or
two classes (genuine information loss), or does real per-image signal
survive? At n=400, C=0.01:

| topology | train accuracy | classes actually predicted |
|---|---:|---:|
| T | 0.8425 | 10/10 |
| lattice | 0.8525 | 10/10 |
| rewired | 0.8875 | 10/10 |
| curr_random | 0.9300 | 10/10 |

**No collapse.** Both rewired and curr_random predict across all 10
classes with roughly balanced distributions, and their training
accuracy is nominally *higher* than T's or lattice's, not lower. Despite
near-total phase synchronization, real per-image discriminative
information survives in the evolved features -- the mechanism is
severe optimization difficulty (as stage 2 found for T, likely worse
here given the much higher Fiedler values), not information destruction.

**3. Multistability check, mirroring Stage 0's own method exactly**
(`find_equilibrium_lbfgs`, `same_attractor`'s 0.05 dedup threshold, 5
seeds 0-4, no image encoding at all -- reused directly from
`bonsai.dynamics.graph_oscillator_field`, not reimplemented):

| topology | distinct equilibria (of 5 seeds) |
|---|---|
| T | 5/5 |
| lattice | 1/5 |
| rewired | 1/5 |
| curr_random | 2/5 |

**Answers the question directly**: this is not simply "T/lattice have
multistability that rewired/curr_random lack." T is uniquely richly
multistable (5/5); lattice and rewired both collapse to a single basin
(1/5 each) -- consistent with, though not identical in specific seeds
to, Stage 0's own original table for these two construction types.
`curr_random`'s 2/5 sits in between, for this specific (13-isolated-node)
draw -- a different specific realization from whatever Stage 0's
original "matched-sparsity random" check used, so a different count is
expected, not a contradiction (this project's own established
one-draw-is-not-a-family caveat, Stage 1D). Consistent with the
Laplacian finding: reduced multistability tracks with increased
algebraic connectivity across all three non-T constructions.

## What this means for stage 3

The synchronization asymmetry is real, spectrally well-explained, and
does not indicate an information-collapse problem -- rewired and
curr_random's evolved features remain genuinely discriminative, if
possibly harder to optimize (an open question the full stage-3 CV run
will answer directly, given `max_iter=10000` is already in place from
stage 2's fix). The revised ~4.2-hour runtime projection and the
corrected 13-isolated-node fact are both disclosed above. Two things to
check honestly once stage 3's full run completes (not new requirements,
just what to verify): (1) whether the high-`C` non-convergence pattern
from stage 2 persists, worsens, or resolves for rewired/curr_random at
12x the sample size; (2) whether the encoder-seed robustness finding
(negligible difference) still holds at full scale.

## Code

`stage2a_topologies.py` (builds all four canonical graphs, reused
verified construction code only), `diagnose_topology_synchronizability.py`
(Laplacian spectrum, zero simulation), `diagnose_rewired_currrandom_synchronization.py`
(information-collapse + multistability checks, reuses
`bonsai.dynamics.graph_oscillator_field`'s `find_equilibrium_lbfgs`
directly). `stage2a_pipeline.py` extended with
`run_pipeline_multi_topology()`/`check_go_no_go_multi_topology()`/
`run_classifier_conditions_multi_topology()` for stage 3's 6-condition
(raw pixels, pre-evolution, 4 evolved topologies) design, encoding each
image once and evolving on all four graphs rather than re-encoding per
topology.

## Next step

Full feasibility stage 3 (60,000 images, ~4.2-hour projection, 6
conditions) -- superseded by the timing-split and GPU-port investigation
below.

# Stage 2A: Encode/Evolve Timing Split, and a Verified JAX Evolution Port

**Status: diagnostic and infrastructure only. No stage-3 simulation data
in this section -- a timing split, a spectral cross-check, and a
from-scratch-verified JAX port, none of which is a scientific result.**

## Timing split: evolution dominates by ~80x

Encoding (`_local_converged_phases` + restriction) and graph evolution
timed separately, 100 real images, single topology (T), single-threaded
(no multiprocessing, to isolate the two steps cleanly):

| step | total (100 images) | ms/image |
|---|---:|---:|
| encode | 0.458s | 4.58 |
| evolve (T only) | 36.558s | 365.58 |

**Evolution is ~80x the cost of encoding.** This sets the real ceiling
on what a GPU port can help with: porting encoding would not
meaningfully change throughput; porting evolution addresses essentially
all of the cost.

## curr_random seed=0: where does 13 isolated nodes sit in the family?

Before committing hours of compute to this specific prespecified graph
instance, checked its isolated-node count and main-component Fiedler
value against all other curr_random realizations already computed in
Stage 1D (3 pilot seeds 0-2, 25 confirmatory seeds 5-29 -- 28 total,
zero new simulation, just loading and analyzing already-cached
constructions):

- **Isolated-node count**: range 3-15 across 28 realizations, mean 9.18,
  median 9.0. Seed=0's 13 sits at the **92.9th percentile** -- elevated
  relative to the median, but well within the observed range (several
  other seeds, e.g. 5 and 1, have 15), not an outlier.
- **Main-component Fiedler value**: seed=0's 0.2429 sits squarely within
  the observed range (~0.16-0.36) across all 28 realizations.

**Seed=0 is a reasonable prespecified representative** -- somewhat
elevated on isolation count, typical on connectivity -- not a broken or
unusually extreme draw. No reason to redraw the canonical seed.

## JAX/diffrax port of graph evolution

`evolve_on_graph_jax.py` ports `stage2a_core.evolve_on_graph`'s plain
(unperturbed) evolution to JAX/diffrax, reusing
`run_one_trial_jax_faithful.py`'s already-verified plain-evolution `rhs`
directly (Stage 1D's own file, the perturbed-trajectory half of that
function -- not the tangent-system half, which Stage 2A never needed).
Returns `(theta_T, success)`, checking `diffrax.is_successful()`
explicitly rather than assuming success -- the batched/jitted
computation can't raise per-trial the way the numpy recovery-policy
retry loop does, so the caller must gate on this exactly like the numpy
path's `diag['failed']`.

**Kernel-level verification** (`verify_evolve_on_graph_jax.py`): 6 real
encoded images (KMNIST, not synthetic random phases) x all 4 topologies,
unbatched and batched (vmap), compared against the real numpy
`evolve_on_graph` via circular distance (accounting for mod-2pi
wraparound). **Max absolute circular difference across all 24
(topology, image) pairs: 5.4e-8** -- both unbatched and batched versions
agree with numpy to this precision, comfortably inside the 1e-4
cross-solver tolerance this project has used throughout (Stage 1D's own
GPU port matched to 1e-6 to 1e-8).

## End-to-end pipeline equivalence check (not just the kernel)

**This project has been burned before by a kernel that passed
field-by-field verification while the surrounding batch/caller code
quietly did something different (Stage 1D's GPU episode -- see
`experiments/stage1d_topology_specificity_gpu/FINDINGS.md`, and
`CLAUDE.md`'s principle 16).** Per that lesson, the kernel check above
is not treated as sufficient on its own. `stage2a_pipeline_jax.py`
builds a full JAX-evolution pipeline matching
`stage2a_pipeline.run_pipeline_multi_topology`'s exact result contract
(same per-image/per-topology dict structure, same idx ordering),
reusing `stage2a_core`'s numpy encoding and gauge-feature functions
completely unchanged -- only the evolution step itself is replaced.

`verify_stage2a_pipeline_equivalence.py` runs both the real numpy
pipeline and this JAX pipeline on the same 40-image, **all-10-classes**
mixed batch (not a homogeneous or single-class sample) and diffs the
full output:

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
features, raw pixels) are expected and mechanically guaranteed -- both
pipelines call the exact same numpy functions for those steps, so
anything other than exact equality would indicate a real bug. The
evolved-feature agreement (max 5.0e-8, well inside the 1e-4 tolerance)
is the genuine cross-solver check (numpy/scipy RK45 vs. JAX/diffrax
Tsit5), and the solver-failure-accounting match confirms the JAX path's
`diffrax.is_successful()` gating is wired correctly, not just present
in the code but untested.

## What this means for the GPU decision

CPU-only JAX (single device, no multiprocessing) was also timed: 400
images, all 4 topologies, 119.9s (74.9 ms/image/topology) -- faster
than numpy's single-threaded rate (365.6 ms/image/topology) but *not*
clearly faster than numpy already parallelized via multiprocessing
across ~9 cores (~40 ms/image/topology-equivalent). **The JAX port's
real value is unlocking GPU batch parallelism, not CPU JAX by itself**
-- consistent with Stage 1D's own experience (112x speedup was GPU vs.
CPU-multiprocessing baseline, not vs. single-threaded numpy). A GPU
session is the natural next step for the full 60,000-image, 4-topology
stage-3 run, now that both the kernel and the full pipeline built
around it are verified.

## Code

`evolve_on_graph_jax.py` (the JAX/diffrax evolution kernel),
`verify_evolve_on_graph_jax.py` (kernel-level verification),
`stage2a_pipeline_jax.py` (full pipeline, numpy encoding/gauge features
reused unchanged), `verify_stage2a_pipeline_equivalence.py` (end-to-end
equivalence check against the real numpy pipeline, mixed-class batch).

## Next step

Provision a GPU session and run the full stage-3 pipeline (60,000
images, 4 topologies, 6 conditions) using the now-verified JAX pipeline
-- superseded by the 100-image GPU sanity run below.

# Stage 2A: 100-Image GPU Sanity Run

**Status: infrastructure verification only, not a scientific result.**
Confirms the verified JAX pipeline produces correct results on real GPU
hardware (not just locally on CPU-backend JAX) and gives a real,
measured full-scale runtime projection, before committing to the full
60,000-image run.

## Setup

Fresh A100 session (`stage2a-gpu-100`, via `mighty-colab`; no orphaned
sessions found first). Environment: `jax==0.11.0` (`jax[cuda12]`),
`diffrax==0.7.2`, `equinox==0.13.8` -- same pinned versions verified
working in Stage 1D's own GPU work, installed then the kernel restarted
(a pip install alone doesn't take effect in an already-running kernel
that already imported the older pre-installed `jax==0.7.2`).

100 images (10/class, `SEED=42`, same `subsample_stratified` used
throughout) encoded locally (CPU/numpy, unchanged code), packaged with
all four topologies, and uploaded (10MB). Only the evolution step ran
on GPU.

## Result: real ~546x-to-~60x speedup, zero failures, correctness confirmed

| topology | GPU time (100 images) | ms/image |
|---|---:|---:|
| T | 0.072s | 0.72 |
| lattice | 0.069s | 0.69 |
| rewired | 0.069s | 0.69 |
| curr_random | 0.058s | 0.58 |

**Total: 0.268s for all 4 topologies, 100 images -- 0.67 ms/image/topology.**
Against numpy's single-threaded 365.58 ms/image/topology, that is a
**~546x speedup**; against numpy already parallelized via
multiprocessing (~40 ms/image/topology-equivalent across ~9 cores),
still **~60x**. Zero solver failures reported (`diffrax.is_successful()`
gate), across all 400 (image, topology) combinations.

**Correctness re-verified on real GPU hardware, not assumed from the
earlier CPU-backend check**: the first 5 images' GPU-produced `theta_T`
for all 4 topologies compared against local numpy `evolve_on_graph` --
max circular difference **1.5e-8**, consistent with (and as tight as)
the CPU-backend verification.

**Full go/no-go check on all 100 images, all 4 topologies**: zero
non-finite features; `R_post` distributions match the earlier CPU
findings exactly (T mean 0.865, lattice 0.871, rewired 0.997 with
100/100 above 0.99, curr_random 0.991 with 64/100 above 0.99) --
confirms the GPU run reproduces the same dynamics, not just fast,
different numbers.

## Revised full-scale projection

**Projected full 60,000-image, 4-topology GPU evolution time: ~161s
(~2.7 minutes)** -- down from the ~4.2-hour CPU-multiprocessing
projection. Encoding (CPU-only, ~4.6 ms/image) adds roughly 4.6 minutes
for 60,000 images, unaffected by the GPU port. The evolution bottleneck
that motivated this whole investigation is now essentially eliminated;
remaining stage-3 runtime will be dominated by encoding and the 6-
condition classifier CV fitting, not graph evolution.

## Code

Local: reused `evolve_on_graph_jax.py` unchanged (uploaded as-is).
Remote-only driver script (not committed -- ephemeral GPU-session code,
per this project's convention; the reusable kernel/pipeline code is
already committed locally).

## Next step

Provision a fresh GPU session (this one to be stopped) and run the full
60,000-image, 4-topology, 6-condition stage-3 pipeline for real -- not
done here.

# Stage 2A: Feasibility Stage 3 -- Full 60,000-Image Run

**Status: mechanical/descriptive validation at full training-set scale,
per `DESIGN.md`'s own framing throughout the feasibility ladder. This is
NOT the locked confirmatory result.** The confirmatory claim (paired
bootstrap on official-test-set log-loss, evolved vs. pre-evolution)
still requires a separate, final step against the held-out official
test set, not done here. Training-side CV log-loss is reported below
descriptively, exactly as `DESIGN.md` permits at this stage, and no
further than that.

## Scope

All 60,000 official KMNIST training images (no subsampling), all 10
classes, `SEED`-independent (the full set, not a stratified draw). Six
conditions: raw pixels (784-dim), encoded pre-evolution (1008-dim), and
evolved on each of the 4 confirmatory-expansion topologies -- T,
lattice, canonical rewired (seed=0), canonical curr_random (seed=0),
each 1008-dim. Encoding used the primary locked seed (`ENCODER_SEED=0`)
for every image.

## Architecture: CPU encode, GPU evolve, CPU classify

Per the explicit instruction to use the verified JAX/GPU pipeline for
full-scale data generation: encoding ran locally (CPU, multiprocessing,
unchanged `stage2a_core`/`stage2a_pipeline` code), evolution ran on a
fresh `mighty-colab` A100 session (`stage2a-gpu-stage3`; no orphaned
sessions found first; same pinned `jax[cuda12]==0.11.0` /
`diffrax==0.7.2` / `equinox==0.13.8`, confirmed correct on first `exec`
with no kernel restart needed this time), and classifier CV fitting ran
locally afterward (CPU-only, sklearn) -- the GPU session was stopped
immediately after evolution completed and its results were downloaded,
since classifier fitting gains nothing from GPU and there is no reason
to keep paying for idle-relative-to-task GPU time during it.

**One real infrastructure snag, worth recording**: the single 250MB
upload package (`theta0_batch` + all 4 topologies) was rejected by the
`mighty-colab` upload endpoint with a 400 Bad Request -- the 10MB
100-image package had worked fine, but 250MB exceeded some server-side
limit. Worked around by splitting `theta0_batch` into 12 separate
~20MB `.npy` chunks (uploaded individually, reassembled remotely via
`np.concatenate`) plus the small 8MB topologies pickle uploaded
separately. Both uploaded cleanly. This is an operational note for any
future GPU work at this data scale, not a scientific finding.

GPU evolution itself was chunked (`CHUNK_SIZE=1000`) rather than run as
one 60,000-image batch, since `vmap` materializes a `(batch, n, n)`
diff tensor per RHS evaluation -- at `batch=60000, n=505` that would be
~120GB (float64), far beyond any single GPU's memory, whereas the
100-image full-batch test's 204MB tensor gave no signal either way about
this limit. `CHUNK_SIZE=1000` keeps the largest transient tensor at
~2GB, conservatively safe; verified working with zero errors across all
60 chunks x 4 topologies.

## Result 1: data generation -- fast, clean, ahead of projection

| phase | time | rate |
|---|---:|---:|
| encode (CPU, local, multiprocessing) | 68.1s | 1.135 ms/image |
| GPU evolve, all 4 topologies | 114.0s | 0.475 ms/image/topology |
| R_post/feat_post compute (CPU, local) | 6.4s | -- |
| **total data-gen pipeline** | **188.5s (3.1 min)** | -- |

Well under the ~7.3-minute (encode + GPU-evolve) projection from the
100-image sanity run. GPU evolution rate (0.455-0.491 ms/image/topology
across the 4 topologies) essentially matches the 100-image run's 0.67
ms/image/topology, confirming the chunked approach scales cleanly with
no throughput degradation at 600x more images.

**One genuine surprise, stated plainly rather than smoothed over**: the
measured 1.135 ms/image encode rate is roughly 4x faster than the ~4.6
ms/image figure quoted in the pre-stage-3 timing investigation
(`evolve_on_graph_jax.py`'s docstring, and the earlier projection
"~4.6 minutes for 60,000 images"). That earlier figure appears to have
been a single-threaded measurement; this run's local multiprocessing
across ~9 workers accounts for most, but apparently not all, of the
gap (a naive 9x parallel speedup on 4.6 ms/image would predict ~0.51
ms/image, not the observed 1.135 ms/image) -- the discrepancy is not
fully resolved and is noted honestly rather than assumed to be pure
parallelism.

Zero solver failures across all 240,000 (image, topology) evolutions;
zero non-finite features anywhere (raw, pre-evolution, or evolved, any
topology).

## Result 2: R(theta) -- the synchronization asymmetry holds precisely at full scale

| | min | max | mean | median | n < 0.01 | n > 0.99 (of 60,000) |
|---|---:|---:|---:|---:|---:|---:|
| pre-evolution (shared) | 0.321 | 0.991 | 0.733 | 0.739 | 0 | 4 |
| T | 0.532 | 0.998 | 0.858 | 0.866 | 0 | 71 (0.12%) |
| lattice | 0.525 | 0.998 | 0.865 | 0.872 | 0 | 64 (0.11%) |
| rewired | 0.986 | 1.000 | 0.997 | 0.997 | 0 | **59,965 (99.94%)** |
| curr_random | 0.973 | 1.000 | 0.991 | 0.991 | 0 | **36,547 (60.9%)** |

This is not a weakening or drift of the 400-image sub-test's finding --
it is a precise confirmation. That sub-test reported "R_post > 0.99 for
~100%/~60% of images" for rewired/curr_random; at 150x the sample size,
the real figures are 99.94% and 60.9% respectively, matching almost
exactly. Rewired's near-total synchronization is in fact more extreme
than "near-total" suggests: its *minimum* R_post across all 60,000
images is 0.986 -- there is no image, of any class, that fails to
synchronize almost completely under this topology. T and lattice show
no such effect (0.11-0.12% above 0.99, an order of magnitude smaller
than even curr_random's rate).

**Visual supplement** (`results/topology_graph_structure.png`,
`results/phase_state_per_class_per_topology.png`, added later during
the compute-cost follow-on): the graph-structure plot makes the
mechanism visible directly -- `rewired`/`curr_random` are dense with
long-range edges connecting spatially distant nodes across the whole
image, while `T`/`lattice` are almost entirely short-range/local,
closely tracking the pixel grid's own spatial layout. The per-class
phase-state grid shows the consequence: for every one of the 10
classes, `T`/`lattice`'s evolved phase fields retain visible spatial
structure echoing the digit's shape, while `rewired`/`curr_random`
collapse to near-uniform color -- near-total synchronization, visible
directly rather than only as a summary statistic.

**A real tension this raises, and its resolution** (`results/
phase_state_per_class_per_topology_normalized.png`): if `rewired`/
`curr_random`'s columns look nearly uniform by eye, how does a linear
classifier extract enough signal from them to perform comparably to or
better than `T` (the confirmatory result's own finding)? Resolved by
the pipeline, not a contradiction: `StandardScaler` runs before
classification, rescaling whatever variance is actually present --
real but small -- up to unit scale before the classifier ever sees it.
Measured directly: the gauge-shifted phase's raw standard deviation
(mean across the 10 classes, one representative image each) is ~0.079
rad for `rewired` and ~0.133 rad for `curr_random`, versus ~0.79 rad
for the pre-evolution baseline and ~0.54-0.55 rad for `T`/`lattice` --
roughly 4-10x smaller in absolute magnitude, genuinely small, but not
zero. Per-panel z-scoring (the same rescaling `StandardScaler` performs)
makes this residual visible directly. **"Looks uniform to the eye" and
"carries zero exploitable information" are different claims** -- this
plot is honest evidence for the first, not the second, consistent with
(not contradicting) the classification result.

**The follow-up observation above was chased, and the mechanism behind
the speckle pattern is now confirmed directly** (`results/
ink_correlation_decay.png`): Pearson correlation between each active
pixel's ink intensity *in that specific image* and its z-scored
residual phase deviation, per class, per condition -- not a comparison
against any population baseline, just "is this pixel inked, in this
image":

| condition | mean r (10 classes) | std |
|---|---:|---:|
| pre_evolution | +0.938 | 0.011 |
| T | +0.725 | 0.053 |
| lattice | +0.722 | 0.049 |
| rewired | +0.373 | 0.039 |
| curr_random | +0.274 | 0.031 |

**Every one of the 10 classes, every condition: positive, and
overwhelmingly significant** (worst case `p < 2e-7`, most `p < 1e-20`
or smaller). This is exactly the mechanism the speckle pattern
suggested by eye: red = inked here, blue = not inked here, and it holds
all the way down to `curr_random`'s r=+0.27 -- small, but a genuine,
consistent, monotonically-decaying-not-vanishing relationship, not
noise. The decay order (`pre_evolution > T ≈ lattice > rewired >
curr_random`) tracks the synchronization ordering exactly (Result 2,
above) -- more synchronization, more of this signal washed out, but
never all of it. This is a cleaner, more parsimonious finding than
"qualitatively different signal": it is the *same* underlying signal
(local ink presence) at every condition, just attenuated by a
condition-dependent, but never total, factor.

**A further, sharper decomposition, also tested rather than assumed**
(`results/ink_correlation_decomposed.png`): "this pixel has ink" splits
exactly into a **population-common component** (the per-pixel mean ink
intensity across the 10 classes -- shared, not class-discriminatory by
construction) and a **class-discriminatory component** (that class's
deviation from the population mean at that pixel -- the part that
actually varies by class). Correlating each sub-component against the
residual separately, per condition:

| condition | r(common) | r(discriminatory) | ratio |
|---|---:|---:|---:|
| pre_evolution | +0.370 | +0.854 | 2.3x |
| T | +0.344 | +0.639 | 1.9x |
| lattice | +0.322 | +0.642 | 2.0x |
| rewired | +0.187 | +0.325 | 1.7x |
| curr_random | +0.134 | +0.240 | 1.8x |

**Correction, caught on external review, load-bearing: the paragraph
that previously stood here overclaimed "preferential preservation of
the discriminatory component," and the conclusion was backwards.** The
error: comparing `r(discriminatory)` against `r(common)` *within* each
condition (0.854 vs. 0.370 at `pre_evolution`, etc.) only shows the
discriminatory component is already more strongly correlated than the
common component *before evolution does anything* -- that asymmetry is
present at `pre_evolution` itself, not created by evolution. It says
nothing about which component evolution preserves *better*. The correct
comparison is retention *relative to `pre_evolution`*, per component:

| graph | common retained | discriminatory retained |
|---|---:|---:|
| T | 93.0% | 74.8% |
| lattice | 87.0% | 75.2% |
| rewired | 50.5% | 38.1% |
| curr_random | 36.2% | 28.1% |

**At every single condition, the common component is retained at a
*higher* percentage than the discriminatory component -- the opposite
of "preferential preservation."** Evolution attenuates both, and
attenuates the discriminatory correlation proportionally *more*. The
representation is more strongly correlated with the class-discriminatory
component at every stage (the original table, above) -- but that
asymmetry is inherited from the encoding, not produced or amplified by
evolution.

**This makes the classification result more interesting, not less**:
if evolution simply preserved discriminatory pixel-level ink better
than common ink, that would be a fairly mundane explanation for the
improvement. It does not. The improvement survives despite evolution
eroding the discriminatory correlation *faster* than the common one --
which means whatever the linear classifier is actually exploiting after
evolution is not well described as "the same pixel-level discriminatory
signal, just a bit fainter." The dynamics may be reorganizing,
decorrelating, or redistributing the surviving information into
directions a linear readout can use more effectively than the raw
per-pixel correlation would suggest -- a real, open mechanistic
question this correction leaves standing, not one it answers. (Caveat
carried over: the population-common baseline is estimated from only 10
images, one per class -- sufficient for the direction and rough size of
this effect, not a precise population estimate.)

**A further, related caveat, also raised on the same review, worth
naming precisely**: all of the mechanistic plots and correlations in
this section are conditional on the class-0-derived 505-node active
support (`DESIGN.md`'s locked `active_indices`, shared identically by
all four topologies and the pre-evolution condition -- see "The class-0
confound `DESIGN.md` flagged," above, for the primary result's own
disclosure of this). Three things follow, stated precisely rather than
left implicit:

- **The primary confirmatory result is unaffected.** `evolved_T` vs.
  `pre-evolution` uses the identical mask on both sides -- the
  improvement isolates the incremental effect of edge structure and
  evolution *given* that fixed support, and cannot be attributed to one
  condition simply retaining more pixels than the other.
- **`active_indices` is not literally an ink silhouette.** It comes from
  `build_class_topology`'s population-developmental statistic (an
  all-pairs Hebbian-style measure across class-0 images), thresholded to
  suppress background-background edges, then restricted to nodes
  surviving in at least one sufficiently strong edge -- a class-0-derived
  *correlated* support, broader than and not identical to a simple
  ink-presence mask.
- **The mechanistic plots establish survival *within* this support, not
  a claim about ink outside it.** The local encoder's four-neighbor
  coupling runs on all 784 pixels *before* the 505-node restriction is
  applied -- an excluded pixel's own coordinate is discarded, but its
  influence on nearby *retained* pixels during those 150 coupling steps
  is not necessarily erased. What these plots show is: ink-related
  information within the class-0-derived active support survives
  evolution and synchronization, attenuated but never to zero. They do
  not show what happens to class-discriminatory ink that falls entirely
  outside that support -- a genuinely open question, addressed directly
  below.

## The class-0-support audit: how much information the projection actually removes

External review's own proposed audit, run in full: before asking what
graph evolution does with what remains, quantify what the class-0-
derived 505-node support projection removes in the first place.

**Retained ink fraction, full 60,000-image training set**
(`results/retained_ink_fraction_by_class.png`): `sum(pixel intensity
inside the 505-node support) / sum(pixel intensity over all 784
pixels)`, per image. Class 0 retains a median of **96.3%** of its own
ink (mean 93.9%, tight distribution) -- expected, since the support is
derived from class 0. **Every other class retains substantially less**:
medians range from 65.4% (class 1) to 90.0% (class 5), with real spread
-- some individual images retain as little as 21-43%. This is not a
small effect.

**Where the excluded ink actually falls**
(`results/ink_outside_support_by_class.png`, class-mean heatmaps of ink
lying outside the support): a consistent, visible band of excluded ink
at the top-center notch and the bottom margin, present for classes 1-9
specifically and much fainter for class 0 -- exactly the region the
class-0-derived support's own shape (`results/
topology_graph_structure.png`) doesn't cover. Real, systematic, not
scattered noise.

**Two baselines, run to separate the projection's cost from evolution's
contribution** (`run_class0_support_audit_classify.py`, `cuml.accel`,
same locked CV/fit procedure as every other condition -- audit-only,
not part of the locked primary/secondary comparisons):

| condition | dim | C | test accuracy | log-loss |
|---|---:|---:|---:|---:|
| raw pixels, full 784 (known) | 784 | 0.001 | 0.6960 | 0.9848 |
| raw pixels, 505-restricted (new) | 505 | 0.01 | 0.6550 | 1.1527 |
| encoded, 505-restricted = `encoded_pre_evolution` (known) | 1008 | 0.01 | 0.7208 | 0.9558 |
| encoded, full 784, unrestricted (new) | 1566 | 0.01 | 0.7458 | 0.8667 |

**The restriction has a real, measurable cost, in both representations**:
raw pixels lose 4.10 points of accuracy (0.6960 -> 0.6550) when
restricted to the support; the locally-encoded state loses 2.50 points
(0.7458 -> 0.7208). Restricting to the class-0-derived support is not
free -- it discards real, class-discriminatory signal, exactly as the
retained-ink-fraction and heatmap results above already suggested
directly.

**But evolution's contribution is larger than what the restriction
costs, not merely compensating for it**: `evolved_T` (505-restricted,
evolved -- the locked primary condition, 0.8058 accuracy, 0.7067
log-loss) beats even the *unrestricted*, un-evolved 784-dim encoded
baseline (0.7458, 0.8667) by 6.00 points of accuracy and 0.16 log-loss
-- a larger margin than the 2.50-point cost restriction itself imposed.
**Stated plainly: even if the pre-evolution condition were given back
every pixel the class-0 support excludes, graph evolution on the
restricted 505-node `T` would still win.** This is reassuring for the
primary result's interpretation, though it does not change the primary
result's own numbers, which were never in question -- `evolved_T` vs.
`pre-evolution` already used the identical support on both sides, so
this audit closes a question about *interpretation*, not about the
comparison's own validity.

**What this does and does not settle**: it settles that the class-0
support projection has a real, non-trivial, quantified cost, and that
evolution's contribution exceeds that cost rather than merely offsetting
it. It does not settle what a union mask or a fully class-agnostic
support would show -- that remains a genuinely open, separately-scoped
question (raised in this same review thread, not pursued here), since
this audit only compares restricted-vs-unrestricted under the *existing*
class-0-derived mask, not against an alternative mask altogether.

## Result 3: classifier CV fitting is NOT negligible -- it dominates everything else

**246.8 minutes (4.1 hours)** for the 6 conditions' full CV fitting
(45 fold/C combinations each, 270 total, plus 6 final refits) --
measured explicitly rather than continuing to assume it was cheap, per
the instruction that motivated this measurement. This is **~79x** the
entire data-generation pipeline's 188.5 seconds. Classifier fitting,
not data generation, is now the dominant cost of this ladder's full-scale
step, and any future re-run of this stage should budget for that
directly rather than by extrapolating stage 1/2's much smaller-scale
timings.

**All 270 fold/C combinations converged, for every condition, with zero
non-convergence events.** This directly resolves the concern flagged
going into this stage: the pre-stage-3 investigation predicted
non-convergence was "most likely to recur... for the higher-Fiedler-
value topologies (rewired, curr_random) at weak regularization," given
stage 2's evolved_T non-convergence at 5,000 images. At 60,000 images
(12x stage 2's scale), it did not recur anywhere, for any condition,
including rewired and curr_random despite their extreme R_post values
above. The `max_iter=10000` amendment made after stage 2's diagnosis
appears to have been not just a workaround for that one instance, but
sufficient headroom at full scale too. No ill-conditioning-vs-
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
pre-evolution by a wide, consistent margin -- descriptively consistent
with this whole programme's motivating premise that evolution adds
linearly-decodable structure. This is training-side CV log-loss only,
never the locked confirmatory statistic (paired bootstrap on
*official-test-set* log-loss, evolved vs. pre-evolution specifically),
and must not be read as anything stronger than descriptive.

**A second genuine surprise, stated plainly**: `curr_random` -- a
topology with matched sparsity but no relationship to T's learned
structure -- gives the *lowest* training-CV loss of all four evolved
conditions (0.3255), edging out `evolved_T` (0.3415) despite T being
the topology this whole construction was built around. And `rewired`,
despite the almost-total phase synchronization documented above (R_post
∈ [0.986, 1.0] for all 60,000 images), still gives the second-best
evolved loss (0.3480) -- clearly better than `lattice` (0.4077) and far
better than either pre-evolution condition. Naively, synchronization
this severe should collapse most of the information that would
distinguish one image's evolved state from another's; it evidently does
not do so here, at least not enough to prevent a linear reference-node-
gauge readout from separating classes better than the un-evolved
features. Neither of these findings should be over-read as bearing on
Stage 1D's own "no detectable difference" closure (a different metric
-- paired bootstrap on the *tangent-departure* response measure -- and a
different sample) but they are both real, and both reported as observed
rather than smoothed toward the expected story.

## Go/no-go summary

- Solver failures: 0/240,000 (image, topology) evolutions.
- Non-finite features: 0, any condition, any topology.
- Classifier convergence: 270/270 fold/C fits converged, 6/6 final
  refits converged.
- **OVERALL: GO** (mechanical criteria only -- not a scientific result).

## What remains open after this stage

1. **The encoder-seed robustness check at full scale** (stage 2's
   negligible-difference finding, re-tested at 60,000 images) was not
   run in this pass. Given the now-measured classifier-fitting cost
   (4.1 hours for 6 conditions), a partial re-fit (encoded_pre_evolution
   + evolved_T only, matching stage 2's original scope) is a bounded but
   real additional cost, plausibly on the order of an hour or more --
   flagged for an explicit go-ahead rather than run automatically, given
   that this stage's own instructions conditioned it on "not
   prohibitively expensive," and the real cost is now known rather than
   assumed.
2. **The locked confirmatory endpoint itself**: paired bootstrap on
   official-test-set log-loss (evolved vs. pre-evolution), the one
   result this entire ladder has been building toward. Not started.

## Code

`run_feasibility_stage3_encode.py` (local CPU encode driver, all 60,000
images), `stage2a_pipeline.py`'s new `run_encode_only_multi_topology` /
`_process_one_image_encode_only` (encode-only split for the GPU
hand-off, reusing `encode_and_restrict`/`order_parameter`/
`reference_node_features` unchanged), `analyze_stage3_results.py`
(local: reconstructs the full per-image results structure from the
downloaded GPU evolution output + local encode results, via the same
`s2a.order_parameter`/`s2a.reference_node_features` calls used
everywhere else, then calls `check_go_no_go_multi_topology` and
`run_classifier_conditions_multi_topology` unchanged). Remote-only GPU
driver script (`stage3_gpu_evolve.py`, chunked evolution) not committed,
per this project's convention for ephemeral GPU-session code -- reuses
`evolve_on_graph_jax.py` (uploaded as-is, unchanged).

## Explicit scope decision: full-scale encoder-seed robustness check not re-run

**Decided, reasoned, and recorded here rather than silently skipped.**
`DESIGN.md`'s robustness check is conditioned on "not prohibitively
expensive" (it is a descriptive, non-blocking check, never able to
override or replace the seed-0 primary analysis). It already ran once,
at feasibility stage 2's 5,000-image scale, and passed cleanly: the
change in mean validation log-loss (independent-per-image-seed vs. the
shared seed-0 encoding) was **-1.5e-6** for `encoded_pre_evolution` and
**-8.4e-5** for `evolved_T` -- both negligible, well inside noise, and
in the *same* direction (independent seeds very slightly *lower* loss,
not higher) for both conditions.

Given classifier CV fitting was just measured at **4.1 hours for 6
conditions at n=60,000** (Result 3, above), a full-scale re-check
(re-encoding all 60,000 images under independent seeds, re-evolving on
GPU, and refitting CV for at least `encoded_pre_evolution` and
`evolved_T`) would plausibly cost **an hour or more** -- to re-test
something that already passed once, with no reason to expect the
direction or magnitude to change qualitatively at 12x the sample size.
That is not consistent with "not prohibitively expensive" applied to a
check whose only role is descriptive robustness support, not a locked
gate. **Decision: not re-run at full scale.** The stage-2 result stands
as the encoder-seed robustness evidence for this design.

## Next step

The locked confirmatory run itself: paired bootstrap, evolved vs.
pre-evolution, on the official 10,000-image KMNIST test set -- touched
for the first and only time in this project.

# Stage 2A: The Locked Confirmatory Result

**Status: this is it -- the one and only locked, pre-registered
official-test-set evaluation this entire design has been building
toward, per `DESIGN.md`'s "Confirmatory endpoint and test" section,
executed exactly as locked. The official KMNIST test set was touched
here for the first time in this project, and this sklearn evaluation
remains the sole confirmatory result under this design -- nothing below
retroactively alters it.**

**Amended by external review (see "Post hoc reuse of the test set,"
below): the test set was subsequently reused, after this locked
evaluation, in explicitly post hoc classifier-backend audits (a JAX/optax
port cross-check and an NVIDIA cuML `accel` cross-backend replication).
Neither altered this locked analysis or supplied a new confirmatory
claim, but the original framing here -- "will not be touched again" --
is no longer accurate and has been corrected rather than left standing.**

## Setup: no new model fitting, one refit at an already-selected C

Per `DESIGN.md`, each condition's regularization `C` was already
selected via full-training-set 5-fold CV in feasibility stage 3 --
`{raw_pixels: 0.001, encoded_pre_evolution: 0.01, evolved_T: 1000,
evolved_lattice: 1000, evolved_rewired: 10, evolved_curr_random: 1}`.
No new hyperparameter search was run. This step does exactly one thing
per condition: fit a fresh scaler and classifier on the *complete*
60,000-image official training set at that already-locked `C`
(`stage2a_classifier.fit_final_at_selected_C`, factored out of the
existing `fit_condition` so no CV-search code path could be
accidentally re-triggered), then apply that fit, unchanged, to the
official test set's 10,000 images -- encoded and evolved on GPU exactly
as stage 3's training data was (`run_official_test_encode.py` +
`evolve_on_graph_jax.py`, zero solver failures across all 4 topologies,
19.1s total GPU evolution for 10,000 images x 4 topologies).

The six final refits took between 8.4s (`raw_pixels`) and 458.3s
(`evolved_T`) -- `evolved_T`'s refit was the single slowest, consistent
with it also being the condition whose selected `C=1000` sits closest
to `max_iter`'s ceiling (5,123 of 10,000 iterations used, the most of
any condition).

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

`d_i = ell_i(evolved T) - ell_i(pre-evolution)`, 20,000 paired
class-stratified bootstrap resamples of the 10,000 official test images:

**Observed mean d_i = -0.2491. 95% percentile interval: [-0.2721,
-0.2266]. The entire interval is below zero.**

Per `DESIGN.md`'s pre-registered success criterion, this is
unambiguously an **improvement** -- not a result requiring interpretation
or a borderline call. Graph evolution on T, on top of the already-
dynamically-encoded pre-evolution state, reduces mean per-image test
log-loss by a large, non-straddling margin. Secondary confirmation via
exact McNemar's test on classification disagreement: of the 1,618 test
images where the two conditions' predictions disagreed, **1,234 were
correct only under evolved_T** versus 384 correct only under
pre-evolution (p = 6.68e-104) -- consistent in direction and scale with
the log-loss result, not a separate or conflicting story.

This resolves the question this whole design existed to ask, in the
positive direction, for the first time in this project's history:
graph evolution demonstrably adds classification value beyond the local
encoding dynamics alone, under this task and this linear readout --
"dynamics not useful" (`DESIGN.md`'s named outcome 3) is directly
contradicted by this result, and "topologies equivalent" (named outcome
1) is also inconsistent with it, since T's evolution alone produces this
large, non-straddling improvement.

**Amended by external review**: the primary, locked comparison here is
`evolved_T` vs. `encoded_pre_evolution` only, and *that* comparison is
what this bootstrap interval and McNemar test directly support --
"dynamics useful" is established. It does **not**, on its own, establish
named outcome 2's stronger second half, "one specific graph wins": that
would require directly testing one evolved graph against another (e.g.
`evolved_curr_random` vs. `evolved_rewired`), which was never run. The
secondary comparisons below test each of the other three graphs
separately against the same `pre-evolution` baseline -- non-straddling
intervals there establish that each graph individually beats
pre-evolution, not that the graphs differ significantly from each
other. See "Secondary comparisons," below, for the corrected framing.

## Secondary comparisons: all three other graphs also improve, none rescuing (or needed to rescue) anything

Same direction convention and bootstrap procedure, each vs.
`encoded_pre_evolution`, no cross-comparison correction (`DESIGN.md`:
none of these is a second chance at the primary claim, so none needed
correcting against the others):

| comparison | observed mean d_i | 95% CI | verdict | McNemar p |
|---|---:|---|---|---:|
| evolved_lattice vs. pre | -0.1743 | [-0.1930, -0.1557] | IMPROVEMENT | 1.55e-56 |
| evolved_rewired vs. pre | -0.2819 | [-0.3074, -0.2570] | IMPROVEMENT | 9.76e-133 |
| evolved_curr_random vs. pre | -0.3049 | [-0.3303, -0.2797] | IMPROVEMENT | 8.42e-138 |

All four prespecified graph instances improved over encoded
pre-evolution, with entirely non-straddling intervals. **Amended by
external review**: the tests performed here compare each graph
separately against the common `pre-evolution` baseline; they do not
directly test one evolved graph against another (e.g.
`ell(curr_random) - ell(rewired)`), and separate non-straddling
intervals against a shared baseline do not themselves establish that two
evolved graphs differ significantly from one another. The originally
stated "the four graphs are not equivalent" overclaimed what these tests
support and has been removed. What the data do support, stated
descriptively rather than as a confirmatory ranking claim: their
observed test-set effects differed, with `curr_random` producing the
lowest log-loss (-0.305), followed by `rewired` (-0.282), `T` (-0.249),
and `lattice` (-0.174, smallest but still a clear improvement).
Graph-to-graph superiority was not a confirmatory estimand under
`DESIGN.md`'s locked design and is reported here descriptively, not
inferentially. A formal task-utility ranking would require direct,
paired graph-to-graph comparisons -- these could be computed from the
already-saved per-image losses, but any such comparison would now be
explicitly post hoc and should use multiplicity correction across
however many pairwise tests it involved.

**A genuine, small, honestly-reported rank swap between the CV-selection
data and the held-out test set**: feasibility stage 3's training-CV
delta-vs-pre-evolution ranking (most-to-least improvement) was
`curr_random (-0.235) > T (-0.219) > rewired (-0.213) > lattice
(-0.153)` -- T ahead of rewired. On the actual held-out test set, the
order of exactly those two swaps: `curr_random > rewired > T > lattice`.
`curr_random` (largest) and `lattice` (smallest) are stable across both;
only the middle two trade places. This is a modest, reportable
instability in the *exact* middle-of-the-pack ordering between
model-selection data and truly held-out data -- not a reversal of the
overall finding, and not treated as a family-level claim regardless
(`DESIGN.md`'s fixed-prespecified-graph-instances scope), but worth
stating rather than quietly picking whichever ordering looks cleaner.
**This swap directly reinforces the caution above against a
graph-to-graph superiority claim**: if `T` and `rewired`'s relative
order isn't even stable between the CV-selection data and the held-out
test set, treating their point-estimate ordering here as a confident
ranking (rather than a descriptive observation) would be
overinterpreting noise the data itself already shows is real. The broad
pattern -- all four graphs clearly beat pre-evolution, `curr_random` and
`lattice` anchoring the top and bottom -- is stable; the exact middle
ordering is not.

**On the Stage 1D dissociation, stated precisely rather than loosely**:
Stage 1D found no detectable differences in internal mapping strength
across these topology constructions (a different metric -- paired
bootstrap on the tangent-departure response measure -- and a different
sample). Stage 2A produced visibly different downstream task effects
among these specific graph instances, as the table above shows. These
two findings are not in tension -- "no detectable difference in one
metric" and "visible difference in another" can both be true of the
same graphs -- but this document does not treat it as a formal,
confirmatory task-utility dissociation, since that would itself require
the direct paired graph-to-graph comparisons flagged as not yet run
above (and, if computed post hoc from the saved per-image losses,
multiplicity-corrected).

**`rewired`'s near-total phase synchronization (Result 2, above: R_post
in [0.986, 1.0] for every one of the 60,000 training images) is now
confirmed, on genuinely held-out test data, not to prevent it from being
the second-strongest of the four evolved conditions.** This extends
stage 3's training-CV surprise to the one evaluation that actually
matters: extreme synchronization under this topology does not erase
class information that remains **linearly decodable under the locked
high-precision, per-feature-standardized pipeline used throughout this
design**. That qualification is precise, not decorative: when phases
become nearly synchronized, the informative differences between images
can be small in absolute magnitude, and `StandardScaler`'s per-feature
normalization can amplify a consistent, low-variance residual difference
into a classifier-usable coordinate. That is legitimate predictive
computation under this exact pipeline, not an artifact -- but it does
not, on its own, establish robustness to feature quantization, phase
noise, lower solver precision, or small perturbations at inference time.
Those are open questions this result motivates, not ones it answers; a
useful follow-up before making any stronger physical-computation or
hardware-robustness claim about this topology, not a gap in the claim
actually made here.

## The class-0 confound `DESIGN.md` flagged: checked directly -- reassuring, not dispositive

`DESIGN.md` warned that the primary comparison, while cleaner than
initially thought (both conditions already share T's class-0-derived
active-node support), "still may benefit class 0 differently... under
evolution on T," and locked per-class recall as a required output for
exactly this reason. Checked directly: the per-class recall delta
(evolved_T minus pre-evolution) across all 10 classes is `[0.089, 0.102,
0.038, 0.118, 0.103, 0.089, 0.056, 0.095, 0.072, 0.088]` -- **class 0's
improvement (+0.089) ranks 5th of 10, squarely in the middle of the
distribution**, not the largest (class 3, +0.118) or smallest (class 2,
+0.038). No obvious class-0-specific recall advantage was observed; the
improvement is broadly distributed across classes, not concentrated in
the one class whose topology happens to be under test.

**Amended by external review**: this supports "no obvious
class-0-specific recall advantage was observed," but does not fully
support the stronger claim originally implied by this section's
heading, "the class-0 confound has been ruled out." A class-0-derived
support or edge structure could still provide generic features useful
across several classes without producing a class-0-specific *recall*
spike -- recall alone doesn't rule that out. More importantly, this
issue does not threaten the primary causal contrast: the primary
comparison already holds the class-0-derived active support fixed
across both conditions (`evolved_T` and `encoded_pre_evolution` share
the identical support), so whatever class-0-derived structure exists is
common to both sides of the comparison, not a confound of it. What
remains open is only the *interpretation* of what makes T's evolution
useful, not the primary result itself. A stronger descriptive check,
not yet run, would report per-class mean log-loss differences rather
than recall alone -- computable directly from the already-saved `ell_i`
values, without refitting or any new model-selection decision.

## Baselines (context only -- never part of the locked primary/secondary comparisons)

| baseline | params | test accuracy | macro-F1 | log-loss |
|---|---:|---:|---:|---:|
| raw pixels (linear) | 7,850 (784x10+10) | 0.6960 | 0.6976 | 0.9848 |
| MLP, H=13 (parameter-matched) | 10,345 | 0.7534 | 0.7544 | 0.8971 |
| MLP, H=128 (competent context) | 101,770 | 0.8863 | 0.8863 | 0.6160 |

**At approximately matched trainable-parameter count** (oscillator
readout: 10,090 params; `H=13` MLP: 10,345, `DESIGN.md`'s own matching),
**every one of the four evolved conditions outperforms this MLP** --
even the weakest, `evolved_lattice` (77.78% accuracy), beats
`MLP_H13`'s 75.34% by 2.4 points, and the strongest,
`evolved_curr_random` (82.21%), beats it by 6.9 points. This is a
genuinely favorable comparison for the oscillator representation,
reported descriptively per `DESIGN.md`'s context-only framing for
baselines, not as a locked claim.

**Amended by external review**: `H=13` is matched only on trainable
parameter count in the final linear readout -- it is not matched on
frozen graph parameters or the data-derived structure the graph itself
encodes, preprocessing capacity, inference compute, or training/
hyperparameter-search budget. The wording "equally-sized ordinary
network" overstated how much this comparison actually controls for,
and has been replaced with "an MLP with approximately matched trainable
parameter count" throughout this section. The result remains favorable
contextual evidence for the oscillator representation, but it is not a
complete compute- or model-capacity-matched comparison; the separate
compute-cost design (`COMPUTE_COST_DESIGN.md`) is the right place to
settle that more rigorously.

**Stated plainly, the other direction**: a larger, competently-sized
MLP (`H=128`, ~10x the oscillator readout's parameter count) reaches
88.63% test accuracy -- clearly ahead of every oscillator-evolved
condition (best: `curr_random` at 82.21%). The oscillator dynamics
improve on the pre-evolution baseline substantially, and beat an MLP
with approximately matched trainable parameter count, but do not close
the gap to a larger, competently-sized one. Both facts are true at once
and neither is softened by the other.

## What this settles, and what it does not

**Settled**: for this task, this linear readout, and these four
prespecified graph instances -- graph-level evolution on top of an
already-dynamically-encoded phase state adds real, statistically
unambiguous classification value on genuinely held-out data. This is
the strongest positive Level 3 result this project has produced, and
the first to survive contact with an official, untouched test set
rather than training-derived validation data alone.

**Not settled, and explicitly out of scope for this design** (per
`DESIGN.md`'s "What this does not do"): whether this generalizes to a
topology *family* rather than these four specific prespecified
instances; whether the genuinely static `theta_static = pi*x` control
(no local-convergence encoding at all) would show the local encoding
step already carries most of the value; role-matched or per-class
topology selection (circular for a real classifier, rejected by
design); and denoising or generation (Stage 2B, deferred).

This locked sklearn evaluation is the one and only *confirmatory*
official-test-set evaluation for this design -- no further confirmatory
evaluation against these 10,000 test images is planned or justified
under this locked design. See "Post hoc reuse of the test set," below,
for what has and has not happened to the test set since.

## Post hoc reuse of the test set (amended by external review)

**The original framing above -- that the official test set was touched
once and would never be touched again -- is no longer factually
correct, and is corrected here rather than left standing.** The locked
sklearn evaluation documented in this section was the first use of the
official test set and remains the sole confirmatory result; nothing
below alters that analysis or its numbers. But the test set was
subsequently reused twice, both explicitly post hoc and both
classifier-*backend* audits rather than new scientific investigations:

1. **The JAX/optax classifier port** (`JAX_CLASSIFIER_PORT_FINDINGS.md`)
   evaluated its from-scratch JAX reimplementation of the classifier
   fit against the cached official test set, at the three real selected
   `C` values from this locked result (`evolved_T=1000`,
   `evolved_rewired=10`, `evolved_curr_random=1`). It found a real,
   unresolved divergence from sklearn's fitted solution that grows with
   `C` and does not close with recalibration -- disclosed there,
   verified at the actual selected `C` values (not just a grid
   extreme), and explicitly **not** used for, or folded into, any
   result reported in this document.
2. **The NVIDIA cuML `accel` cross-check** (`CUML_ACCEL_FINDINGS.md`)
   replicated the complete six-condition, 270-fit CV-grid procedure
   under a different GPU-native solver backend and evaluated it against
   the same official test set, reaching the same four verdicts (primary
   and all three secondary comparisons) at closely matching effect
   sizes.

Neither audit altered the locked sklearn analysis above or supplied a
new confirmatory scientific claim. But as a factual matter, the test set
is no longer untouched for future Stage 2A development, and this
document should not claim otherwise.

**This also changes how the cuML result should be described.** It is a
strong **cross-backend implementation robustness check**: it shows the
positive verdict is not peculiar to sklearn's particular optimizer. It
is **not independent scientific corroboration** of the result, because
it reuses the same training and test samples, the same oscillator
features, the same graph instances, the same folds and `C` grid, and
the same high-level selection and fitting code -- only the numerical
classifier backend differs. `CUML_ACCEL_FINDINGS.md` itself has been
amended to use this framing throughout, in place of the "independent
confirmation/replication" language it originally used.

3. **A post hoc, exploratory graph-to-graph pairwise comparison** (below,
   "Post hoc, exploratory: direct graph-to-graph pairwise comparison")
   -- the third reuse of the test set, and the first that computes a new
   statistic (paired bootstrap directly between two evolved graphs)
   rather than auditing an existing one against a different backend. No
   new simulation or GPU time: computed entirely from the per-image test
   losses this locked evaluation already produced and saved.

## Post hoc, exploratory: direct graph-to-graph pairwise comparison

**Explicitly post hoc and exploratory -- not part of, and does not
reopen or alter, the locked confirmatory result above.** Prompted
directly by this document's own secondary-comparisons section flagging
that a graph-to-graph superiority claim "would require direct, paired
graph-to-graph comparisons... these could be computed from the
already-saved per-image losses, but any such comparison would now be
explicitly post hoc and should use multiplicity correction." This
section does exactly that, properly, rather than leaving it as an
unresolved caveat.

**No new simulation, no new GPU time**: `ell_i` for all four evolved
conditions was already computed and saved by `run_confirmatory_
evaluation.py` (`results/stage4_confirmatory_results.pkl`). This is a
new bootstrap computation on existing data only
(`run_posthoc_graph_pairwise.py`).

**Method**: identical to the locked primary/secondary procedure --
`d_i = ell_i(graph_A) - ell_i(graph_B)`, 20,000 paired class-stratified
bootstrap resamples of the test set, two-sided 95% percentile interval
on mean `d_i`, same `seed=42`. All six pairwise comparisons among the
four evolved graphs, not a subset chosen after seeing which looked
interesting. Because this is new, un-pre-registered multiple-comparison
territory (`DESIGN.md` only locked the four graph-vs-pre-evolution
tests, never graph-vs-graph), **Holm-Bonferroni correction across all
six comparisons, as one family, is applied** -- not optional here, per
this project's own standing rule (`CLAUDE.md` principle 3).

| comparison | mean `d_i` | 95% CI | raw `p` | Holm-adjusted `p` | survives (alpha=0.05) |
|---|---:|---|---:|---:|---|
| T vs. lattice | -0.0748 | [-0.0935, -0.0564] | 9.9995e-05 | 5.9997e-04 | **yes** |
| T vs. curr_random | +0.0558 | [+0.0307, +0.0803] | 9.9995e-05 | 5.9997e-04 | **yes** |
| lattice vs. rewired | +0.1076 | [+0.0844, +0.1312] | 9.9995e-05 | 5.9997e-04 | **yes** |
| lattice vs. curr_random | +0.1305 | [+0.1070, +0.1543] | 9.9995e-05 | 5.9997e-04 | **yes** |
| T vs. rewired | +0.0328 | [+0.0088, +0.0572] | 8.1996e-03 | 1.6399e-02 | **yes** |
| rewired vs. curr_random | +0.0229 | [+0.0002, +0.0456] | 4.8398e-02 | 4.8398e-02 | **yes, barely** |

(Sign convention: positive `d_i` means the first-named graph's log-loss
is higher, i.e. the second-named graph wins that pair. Raw `p` computed
from each bootstrap's own resampled-mean distribution via the
double-the-smaller-tail method, with the Monte Carlo floor convention
applied per tail -- `9.9995e-05` is the floor at `N=20,000` resamples,
not an exact zero.)

**All six of six comparisons survive Holm correction at alpha=0.05.**
This is the strongest of the outcomes this check could have produced,
and it is reported exactly as measured, not softened or oversold. Five
of the six are extremely well-separated (raw `p` at or near the Monte
Carlo floor, surviving correction by a wide margin). The sixth --
`rewired` vs. `curr_random`, the two closest performers -- is real but
genuinely marginal: raw `p=0.048`, and because it is the largest
`p`-value in the family it receives no Holm penalty (correction factor
1) and survives at essentially its raw value, just under the 0.05
threshold. Worth naming plainly: this is a significant result, not a
robust one -- a small change in resampling seed or test-set composition
could plausibly flip it.

**The full pairwise ranking is transitive and internally consistent**:
`curr_random > rewired > T > lattice`, exactly matching the descriptive
point-estimate ordering already reported in "Secondary comparisons,"
above -- and now, for the first time, that ordering rests on a properly
powered, multiplicity-corrected, direct pairwise test, not a point-
estimate comparison against a shared baseline. **Restated precisely,
now that it is justified**: `curr_random` (a topology with matched
sparsity but no relationship to `T`'s learned structure) measurably
outperforms `T` (the topology this whole design was built around) on
this held-out test set (`d = +0.0558`, `[+0.0307, +0.0803]`, Holm-
survives) -- a real, corrected, direct result, not the descriptive
observation it was before this check.

**What this does and does not establish, restated for this specific
addendum**: this confirms the four graphs are not equivalent in task
utility under this exact pipeline, on this one test set, for this one
locked feature/classifier procedure -- it does not extend to a
topology-*family* claim (still explicitly out of scope, per `DESIGN.md`
and every prior section here), and it does not retroactively change the
primary/secondary locked results above, which stand as reported
regardless of this addendum's outcome.

## Code

`run_official_test_encode.py` (local CPU encode of the official test
set, mirrors `run_feasibility_stage3_encode.py`), `run_confirmatory_evaluation.py`
(the confirmatory analysis itself: final refits, primary/secondary
bootstrap, McNemar, MLP baselines), `stage2a_classifier.py`'s new
`fit_final_at_selected_C` (the refit-only half of `fit_condition`,
factored out so no new CV search could run here even by accident).
Remote-only GPU driver (`stage4_gpu_evolve.py`, same chunked approach as
stage 3's) not committed, per this project's convention for ephemeral
GPU-session code. `run_posthoc_graph_pairwise.py` (the post hoc
graph-to-graph pairwise comparison, above -- no GPU dependency, reuses
the already-saved per-image losses only).

## Reproducibility gaps (flagged by external review, now closed)

**Not a blocker to accepting the scientific finding above -- it was a
blocker to describing this branch as fully reproducible from its public
contents.** All five items external review flagged are now implemented,
verified, and committed (not merely planned):

- **Artifact paths parameterized** (`stage2a_paths.py`): every script
  that reads or writes the large scratch artifacts now resolves its
  directory through `train_scratch_dir()`/`test_scratch_dir()`,
  overridable via `STAGE2A_SCRATCH_ROOT`, rather than a hard-coded
  private path. Default location documented in `README.md`.
- **Both exact GPU evolution drivers committed**: `stage4_gpu_evolve.py`
  (test-set, the one flagged) and `stage3_gpu_evolve.py` (training-set
  -- equally load-bearing for the same reason, committed alongside for
  the same completeness, not explicitly named in the original
  recommendation but the identical gap). Neither is runnable locally
  as-is (both execute on a remote Colab kernel's `/content/...`
  filesystem) -- `README.md`'s "Reproducing the confirmatory GPU
  evolution" documents the exact `mighty-colab` upload/exec sequence,
  including the chunked-upload workaround for the transfer endpoint's
  size limit.
- **Artifact manifest** (`generate_artifact_manifest.py`, run against
  the real cached artifacts -- not a template): `results/ARTIFACT_
  MANIFEST.json` records SHA256 hashes for every pkl the confirmatory
  result depends on, per-topology adjacency and evolved-state hashes,
  training/test dimensions, image-ordering checks (label array hashes,
  `idx == arange(n)` confirmation), the selected `C` values actually
  consumed (`{raw_pixels: 0.001, encoded_pre_evolution: 0.01,
  evolved_T: 1000, evolved_lattice: 1000, evolved_rewired: 10,
  evolved_curr_random: 1}`), and the frozen primary effect itself.
- **Unit tests** (`tests/test_stage2a_stats.py`, 17 tests, all passing):
  Tier 1 synthetic-data coverage for the paired class-stratified
  bootstrap (including a deterministic zero-variance construction that
  directly verifies genuine per-class stratification, not just
  plausible-looking output), per-image log-loss class indexing
  (including a non-default class-ordering case that would silently
  mis-index on a positional bug), exact McNemar's contingency
  construction, the bootstrap-derived p-value, and Holm-Bonferroni
  (including a step-down-stopping edge case a naive implementation
  could get wrong -- caught a real error in the test's own first draft,
  not the implementation, when first run).
- **Artifact-backed regression test**
  (`test_frozen_primary_effect_matches_findings_md`, Tier 2, skips
  cleanly if the confirmatory pkl isn't present locally): recomputes
  the primary bootstrap directly from the already-saved per-image
  losses and asserts it still lands at `d_i = -0.2491`,
  `CI = [-0.2721, -0.2266]` -- passing now, and will catch a future
  refactor that silently changes the statistic.

The stats functions themselves were also consolidated during this work:
`run_confirmatory_evaluation.py` originally defined
`paired_class_stratified_bootstrap` and the other statistics itself,
and `run_posthoc_graph_pairwise.py` had copied the bootstrap function
verbatim rather than importing it (disclosed as deliberate at the time,
for exact-fidelity reasons) -- both now import from the new
`stage2a_stats.py`, so there is exactly one implementation to trust and
test, not two that could silently drift apart. Re-ran the post hoc
pairwise comparison after the refactor and confirmed bit-identical
output against the pre-refactor numbers reported above.

See `README.md` (new, added alongside this closure) for the full
reproduction sequence, the `mighty-colab`/Colab-A100 GPU-session
pattern, and the public GCS artifact cache
(`gs://bonsai-2026-stage2a-cache`) this closure also made use of.

## Next step

None specified by this design -- the locked confirmatory evaluation is
complete, and the reproducibility gaps that were the last open item are
now closed (immediately above). Any further extension (topology-family
generalization, the static-encoding control, Stage 2B denoising) is a
new design decision, not a continuation of this one.

## External review verdict (amendment record)

An external review of this section's original text found the primary
scientific result sound and recommended the specific corrections
incorporated throughout this document (rather than a rewrite from
scratch) -- summarized here as the record of what was reviewed and what
changed.

**Verdict**: the primary Stage 2A result survives review. The locked
contrast, statistical procedure, classifier selection, final refit, and
reported result align correctly with `DESIGN.md`'s pre-registered
specification. The observed primary effect
(`d_i = ell(evolved T) - ell(pre-evolution) = -0.2491`,
`95% CI = [-0.2721, -0.2266]`), the accuracy increase (72.08% to
80.58%), and the concordant McNemar result together support a bounded
Level 3 claim: under this fixed support, graph instance, encoding,
evolution horizon, gauge, standardization procedure, and linear
readout, graph-level evolution produces a substantially more useful
classification representation than the encoded pre-evolution state
alone.

**What review changed**: two inferential overstatements were corrected
(the "one specific graph wins" / "the four graphs are not equivalent"
claims, and the "test set touched only once" claims), and one
reproducibility gap was documented as open rather than implicitly
assumed closed. Three smaller wording precisions were also made: the
class-0 diagnostic is described as reassuring rather than dispositive,
the MLP baseline comparison is described as approximately
parameter-matched rather than "equally-sized," and the
locked-pipeline-specific decodability finding under near-total
synchronization is stated with its precision qualifier rather than as
an unqualified claim about the underlying representation.

**What review did not change**: the primary confirmatory result itself,
the JAX/optax investigation's handling (already meeting this project's
evidentiary standard -- the unresolved performance gap is disclosed,
verified at the real selected `C` values rather than only a grid
extreme, and explicitly excluded from any reported result), or the
overall accept decision. Side investigations are, per this review,
appropriately honest, with the cuML cross-check now correctly framed as
cross-backend implementation robustness rather than independent
scientific replication.

**Strongest defensible conclusion, as amended**: Stage 2A establishes
external task utility for runtime graph-oscillator evolution under a
bounded classification design. On the official KMNIST test set,
evolution on the prespecified class-0-learned graph `T` reduced mean
per-image log-loss by 0.2491 relative to the same dynamically encoded
phase state before graph evolution, with a paired class-stratified 95%
bootstrap interval of `[-0.2721, -0.2266]`. Accuracy increased from
72.08% to 80.58%, with concordant McNemar evidence. All four tested
graph instances improved descriptively over pre-evolution, but
graph-to-graph superiority and topology-family generality were not
confirmatory claims under this design. The oscillator representation
beats an MLP with approximately matched trainable parameter count,
while remaining clearly below a larger, competently-sized MLP.
