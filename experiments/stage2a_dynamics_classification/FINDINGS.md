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
final feature generation and model selection) -- not done here.
