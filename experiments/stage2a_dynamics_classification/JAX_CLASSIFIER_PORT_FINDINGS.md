# JAX Classifier Port: Investigation and Current Status

**Status: investigative, not a verified drop-in replacement for
`stage2a_classifier.py`, and not used to produce any reported Stage 3
result.** This document is self-contained and covers only the JAX-port
investigation triggered by `analyze_stage3_results.py`'s multi-hour
runtime. It does not restate `FINDINGS.md`'s scientific results. The
real 60,000-image `analyze_stage3_results.py` CPU run this investigation
was triggered by was never touched, stopped, or written to by anything
described here -- it was independently monitored elsewhere and finished
on its own.

## Motivation

`experiments/stage2a_dynamics_classification/analyze_stage3_results.py`
(the phase-2 combine-and-classify step for feasibility stage 3, 60,000
images x 4 topologies) had been running for hours. A JAX-vectorized
port of it existed (`analyze_stage_3_results_jax.py`) and the question
was whether running it on GPU would resolve the runtime.

## Finding 1: the existing JAX port speeds up the wrong step

`analyze_stage_3_results_jax.py` batches `R_post`/`feat_post`
computation (`stage2a_core.order_parameter` /
`reference_node_features`) via `jax.vmap`, replacing a Python loop over
240,000 (image x topology) pairs. This was verified correct:

- `order_parameter`/`reference_node_features`, compared against the
  `stage2a_core` numpy reference on synthetic 505-node states (20
  trials plus explicit `ref_idx in {0, n-1}` edge cases): max
  `|R_ref - R_jax|` = 3.469e-17, max `|feat_ref - feat_jax|` = 0.0.
- Full `build_results_structure` output (including solver-failure
  handling across all four topologies), compared field-by-field between
  `analyze_stage3_results.py` and `analyze_stage_3_results_jax.py` on
  synthetic mock data with injected solver failures: 0 mismatches, max
  numeric diff 2.220e-16.

But this is not what was taking hours. Both scripts call the identical,
unchanged `stage2a_pipeline.run_classifier_conditions_multi_topology`,
which fits 6 conditions x 5 folds x 9 `C` values = 270 sklearn
`LogisticRegression` fits (`stage2a_classifier.py`) -- plain CPU
sklearn, untouched by either script's JAX path. The original script's
own inline comments already identify this as "the expensive step ...
timed explicitly," and the session independently monitoring the real
CPU run confirmed it: ~236+ minutes and climbing, consistent with
`stage2a_classifier.py`'s own documented history of needing
`max_iter=10000` (raised from 1000) because of "severe feature
multicollinearity (condition number ~2e6)" in `evolved_T` specifically.
So porting `R_post`/`feat_post` to GPU, however correct, could not have
fixed the multi-hour runtime.

## Finding 2: most of a "port everything to JAX" ask was already done

Before building anything new, a directory survey found that the actual
expensive physics step -- the graph-evolution ODE solve
(`stage2a_core.evolve_on_graph`) -- was already ported to JAX/`diffrax`
in a prior session:

- `evolve_on_graph_jax.py`: `diffrax.Tsit5` port of the plain
  (unperturbed) evolution ODE, reusing the already-verified `rhs` from
  Stage 1D's `run_one_trial_jax_faithful.py`.
- `stage2a_pipeline_jax.py`: wraps it with the exact
  `run_pipeline_multi_topology` result-dict contract, per this project's
  own established lesson (`CLAUDE.md` principle 16 / Stage 1D) that a
  verified kernel can still feed a wrong result if the calling code
  around it differs from the real pipeline.
- Per `FINDINGS.md` / git history: verified, ~546x speedup measured on
  a 100-image GPU sanity run.

So the only genuinely unported piece of the classification pipeline was
the classifier CV fit itself -- `stage2a_classifier.select_C_via_cv`.

## The classifier port: `stage_2a_classifier_jax.py`

New file, `experiments/stage2a_dynamics_classification/stage_2a_classifier_jax.py`.
Reimplements `stage2a_classifier.select_C_via_cv`'s multinomial
logistic regression (L2-regularized, 5-fold stratified CV over the
locked 9-value `C` grid) with the model fit itself done in JAX:

- Fold splitting (`StratifiedKFold`) and per-fold standardization
  (`StandardScaler`) are reused **unchanged** from scikit-learn --
  cheap, not the bottleneck, and this keeps fold assignment and scaling
  byte-identical to the reference pipeline.
- The fit itself: softmax cross-entropy + L2 on the weight matrix only
  (unregularized intercept, matching sklearn's
  `LogisticRegression(solver="lbfgs")` objective:
  `C * sum_i NLL_i(W,b) + 0.5*sum(W**2)`), minimized with
  `optax.lbfgs` (added as a project dependency, `optax>=0.2.8`).
- The 9 `C`-grid problems within a fold are solved as **one batched
  optimization via `jax.vmap`**, instead of 9 serial sklearn fits --
  this is the actual target for GPU parallelism, since sklearn's CV
  loop is single-threaded with no `n_jobs` in the locked procedure.
- Convergence is judged by `||grad|| <= GRAD_NORM_TOL` (module default
  `1e-6`) after at most `MAX_ITER` steps (module default `2000`,
  overridable). **This is a different criterion from sklearn's internal
  `n_iter_[0] < max_iter` stopping rule and is not directly
  comparable** -- see "Open gaps" below.

### Bug found and fixed: flaky NaN under `vmap` + `lax.while_loop`

Initial version used `optax.value_and_grad_from_state` (the
library-recommended caching pattern to avoid redundant value/grad
recomputation). This produced NaN on the very first L-BFGS step under
`jax.vmap` + `jax.lax.while_loop`, reproduced even at a `vmap` batch
size of exactly 1 -- ruling out cross-lane contamination between `C`
values. Fix: switched to plain `jax.value_and_grad`, recomputed
explicitly every step (a discarded reuse optimization, not a
correctness change).

That fix reduced but did not eliminate the problem: the same
"`n_iter=0`, `gnorm=nan`" signature recurred intermittently, run to run,
with identical code and identical data (isolated via: batch-of-1
reproduction: ruled out cross-lane contamination; fold-order
permutation: ruled out call-order dependence; sklearn-then-JAX
interleaving: ruled out cross-library interference). Traced to the
**unguarded initial gradient computation**, evaluated once outside the
`while_loop` to gate entry -- every in-loop step already had a
finite-check guard, but that first evaluation didn't. Most likely
explanation: CPU-backend thread-scheduling nondeterminism in the
large-reduction matmuls feeding value/grad, occasionally perturbing an
L-BFGS curvature pair (`s_k`, `y_k`) enough that `1/(y_k . s_k)`
blows up -- classic L-BFGS curvature degeneracy, plausible exactly where
`C` is small enough that the objective is nearly pure-quadratic and a
single step lands very close to the optimum (the failure was first
isolated at `C=1e-4`, the grid's smallest value).

Final fix: restructured so **every** value/grad evaluation, including
the first, goes through one uniform guarded path (no special-cased
pre-loop computation) -- `jax.lax.cond` per iteration: if the freshly
computed value/grad is non-finite, leave params/state untouched and
retry on the next `while_loop` iteration (a fresh XLA dispatch of the
identical computation, which empirically recovers); if a step computes
a finite value/grad but then produces a non-finite optimizer update,
discard it and reset the L-BFGS curvature memory at the last-good point
(standard L-BFGS robustification) rather than letting a corrupted
curvature pair persist.

Stress-test results after the fix, on the synthetic case that had
originally exposed the bug (`n=3000, d=200, k=10`, 5-fold x 9-`C`
select_C_via_cv_jax call):

| backend | runs | failures |
|---|---:|---:|
| CPU (this machine) | 13 | 0 |
| GPU (A100, Colab) | 20 | 0 |

(Before the final fix: roughly 1-in-2 failure rate on CPU immediately
after the `value_and_grad_from_state` fix alone, dropping to roughly
1-in-6 after the body-loop guard alone -- neither sufficient on its own.)

### Correctness verification against sklearn (synthetic data)

Three synthetic cases (`sklearn.datasets.make_classification`), each
compared against `stage2a_classifier.select_C_via_cv` on identical data:

| case | n | d | k | `best_C` match | max \|val_loss diff\| across grid |
|---|---:|---:|---:|---|---:|
| small_2class | 400 | 20 | 2 | yes (1.0) | 7.50e-02 |
| small_10class | 1200 | 40 | 10 | yes (1.0) | 1.20e-01 |
| medium_10class_wide | 3000 | 200 | 10 | yes (0.1) | 4.18e-01 (at C=10000) |

`best_C` selection matched sklearn's in all three cases. **The full
per-`C` validation-loss curve did not** -- divergence grows with `C`
(weakest regularization), reaching a 0.42 absolute log-loss gap at
`C=10000` in the widest case. Since `stage2a_pipeline`'s
`run_classifier_conditions_multi_topology` reports the full
`mean_val_loss_per_C` curve as a diagnostic (not just the argmin), this
gap is an open item, not a passed check -- see "Open gaps."

### Timing (A100, Colab, synthetic data)

| case | n | d | sklearn (CPU) | JAX (A100) | ratio |
|---|---:|---:|---:|---:|---:|
| small_2class | 400 | 20 | 0.25s | 0.68s | 0.4x (GPU slower -- dispatch overhead not amortized) |
| small_10class | 1200 | 40 | 0.83s | 2.10s | 0.4x |
| medium_10class_wide | 3000 | 200 | 17.57s | 1.45s | **12.1x** |
| production_scale (synthetic, n=60000, d=1008, k=10, ill-conditioned via `n_redundant=800`) | 60000 | 1008 | not run (would take comparable time to the real CPU job) | 35.0s | -- |

The production-scale case's loss curve was nearly flat across `C`
(1.6706 from `C=0.01` to `C=10000`), suggesting `make_classification`'s
`n_redundant` (exact linear combinations) is a cruder, more degenerate
form of collinearity than the real data's condition-number-~2e6 case --
**the 35.0s figure is a real, hardware-measured number for a
similarly-shaped but not necessarily similarly-hard problem, not a
validated stand-in for the real 6-condition production runtime.**

### Real data test

Read-only against the actual Stage-3 artifacts
(`stage3_encode_local.pkl`, `stage3_gpu_results.pkl`; 60,000 images,
`ref_idx=363`, 0 solver failures across all 4 topologies at full scale)
via the already-verified `analyze_stage_3_results_jax.build_results_structure`.
Nothing was written to `experiments/stage2a_dynamics_classification/results/`
or the shared scratch directory.

The full 60,000-image per-condition arrays (up to ~242MB each) could
not be uploaded to the Colab session -- repeated HTTP 500s from an
undocumented backend size limit somewhere between 6MB and 15.7MB (a
`/content/data/` subdirectory left in a bad state by the earlier failed
large uploads was a compounding, separate issue, resolved by uploading
flat under `/content/`). Used a **stratified random subsample of 6,000
of the 60,000 real images** instead (seed 0, stratified by label,
`np.random.default_rng`), sharded to ~8MB per file to stay under the
limit.

`select_C_via_cv_jax` result, all 6 conditions, `max_iter=10000`
(sklearn's own budget) after `max_iter=2000` (this module's original
default) produced 6/6 non-convergence:

| condition | n | d | result | `best_C` | elapsed |
|---|---:|---:|---|---:|---:|
| raw_pixels | 6000 | 784 | converged | 0.01 | 55.8s |
| encoded_pre_evolution | 6000 | 1008 | converged | 0.01 | 116.8s |
| evolved_T | 6000 | 1008 | **non-converged** (final \|\|grad\|\|=2.60e-02) | -- | 32.8s |
| evolved_lattice | 6000 | 1008 | **non-converged** (final \|\|grad\|\|=2.23e-01) | -- | 32.5s |
| evolved_rewired | 6000 | 1008 | **non-converged** (final \|\|grad\|\|=6.88e-05, close) | -- | 32.9s |
| evolved_curr_random | 6000 | 1008 | converged | 0.1 | 122.3s |

Total: 393.1s (6.55 min) for all 6 attempted.

`evolved_T`'s non-convergence here is consistent with -- not
contradicted by -- `stage2a_classifier.py`'s own documented history for
this exact condition (the `max_iter` 1000->10000 change, condition
number ~2e6). It is evidence this module's `GRAD_NORM_TOL=1e-6`
criterion is uncalibrated against genuinely ill-conditioned real data,
not evidence of a new bug distinct from the earlier large-`C` gap --
both point at the same underlying issue: this port's stopping criterion
does not yet track sklearn's effective precision across conditioning
regimes.

## Follow-up: convergence-criterion recalibration (evolved_T)

Direct follow-up to the open gap above, done as an isolated diagnostic
-- still independent of, and never touching, the real CPU
`analyze_stage3_results.py` run (which finished on its own, using the
already-trusted sklearn path, during this follow-up).

### Step 1: what gradient norm does sklearn's own converged solution achieve?

`stage2a_classifier._fit_one`'s fitted `coef_`/`intercept_`, for each
`C` in the locked grid, on real `evolved_T` data (6000-image stratified
subsample, fold 0, `n_train=4800`) -- with `||grad||` recomputed using
`stage_2a_classifier_jax._make_loss_fn` directly (not reimplemented):

| C | sklearn `n_iter` | converged | `\|\|grad\|\|` at sklearn's solution |
|---:|---:|---|---:|
| 1e-4 | 68 | True | 6.995e-04 |
| 1e-3 | 106 | True | 6.423e-03 |
| 1e-2 | 165 | True | 1.141e-01 |
| 1e-1 | 222 | True | 9.428e-01 |
| 1 | 473 | True | 1.285e+01 |
| 10 | 996 | True | 1.101e+02 |
| 100 | 1460 | True | 1.134e+03 |
| 1000 | 1378 | True | 1.237e+04 |
| 10000 | 1415 | True | 1.330e+05 |

Sklearn's own achieved `||grad||` ranges over **eight orders of
magnitude** (7.0e-4 to 1.33e5) and is three to eleven orders of
magnitude looser than `GRAD_NORM_TOL=1e-6` at every single grid point.
Not sklearn being imprecise -- the objective
`C * sum_i NLL_i(W,b) + 0.5*sum(W**2)` is an unweighted **sum** (not
mean) over `n_train` samples, so `||grad||` at any fixed fit quality
scales near-linearly with both `C` and `n_train` by construction. A
fixed absolute threshold cannot be right across a grid spanning
`C=1e-4` to `C=1e4`.

The normalized quantity `||grad|| / (C * n_train)` is tight over the
same measurement: **[1.338e-3, 2.771e-3]**, a ~2.07x spread (vs.
`||grad||` alone's ~1.9e8x spread):

| C | `\|\|grad\|\|/C` | `\|\|grad\|\|/(C*n_train)` |
|---:|---:|---:|
| 1e-4 | 6.995 | 1.457e-03 |
| 1e-3 | 6.423 | 1.338e-03 |
| 1e-2 | 11.405 | 2.376e-03 |
| 1e-1 | 9.428 | 1.964e-03 |
| 1 | 12.855 | 2.678e-03 |
| 10 | 11.006 | 2.293e-03 |
| 100 | 11.338 | 2.362e-03 |
| 1000 | 12.366 | 2.576e-03 |
| 10000 | 13.302 | 2.771e-03 |

**Systematic trend vs. noise, checked directly (not chased further, per
scope):** both are present. There is a real, roughly 2x systematic
upward trend in the ratio from small `C` (~1.3-1.5e-3) to large `C`
(~2.4-2.8e-3), but the middle of the grid (`C=0.01` to `C=10`) zigzags
non-monotonically on top of it (e.g. `C=0.01`'s ratio exceeds
`C=0.1`'s; `C=1`'s exceeds `C=10`'s) -- plausibly per-fit noise from
sklearn's own relative `tol=1e-4` stopping point landing at a slightly
different absolute precision on each independent solve. The tail
(`C>=100`) is cleanly monotonic. A single fixed tolerance with a margin
absorbs both the trend and the noise reasonably; a tighter
`C`-dependent correction is possible but not pursued.

### Implementation

`stage_2a_classifier_jax.py` changed:

- `GRAD_NORM_TOL=1e-6` (fixed absolute) replaced with
  `GRAD_NORM_REL=6e-3` (~2x the max observed ratio, 2.771e-3), applied
  as `grad_tol = GRAD_NORM_REL * C * n_train` -- computed inside
  `_solve_one` from the actual fold's `X.shape[0]`, so `n_train` is
  genuinely per-fold (`StratifiedKFold`'s unequal fold sizes are
  handled correctly, never assumed constant).
- `MAX_ITER` default raised `2000 -> 10000`, matching sklearn's own
  `CLASSIFIER_KWARGS max_iter` exactly, for the same documented reason
  `stage2a_classifier.py` itself raised it: real non-convergence on
  `evolved_T`'s condition number ~2e6, not a number picked for this
  module in isolation.
- Full derivation, with this same measurement, is now in the module's
  own docstring (matching the documentation standard already used there
  for the `max_iter` 1000->10000 precedent).

### Synthetic re-verification (sanity check, not the real bar)

Re-ran the original three synthetic cases after recalibration:

| case | `best_C` match | max \|val_loss diff\| (before -> after) |
|---|---|---:|
| small_2class | yes | 7.50e-02 -> **2.680e-01** (worse) |
| small_10class | yes | 1.20e-01 -> **4.108e-01** (worse) |
| medium_10class_wide | yes | 4.18e-01 -> **1.580e-01** (better, ~62% reduction) |

Mixed, and not surprising: `GRAD_NORM_REL` was derived from real
ill-conditioned data, which is a *looser* bar than what easy,
well-conditioned synthetic problems can actually achieve -- so the new
threshold now stops the easy cases *earlier* (less precisely converged)
than the old absolute `1e-6` did. `best_C` selection still matched
sklearn in all three cases regardless.

### Step 2: real-data re-verification (evolved_T only)

Full sklearn-vs-JAX per-`C` validation-loss curve, real `evolved_T`
data (6000-image subsample), the same condition `GRAD_NORM_REL` was
derived from -- the one case that has to verify cleanly for the
recalibration to mean anything. (An initial attempt covering all three
hard conditions -- evolved_T, evolved_lattice, evolved_rewired -- was
killed after 21 minutes of wall-clock time with zero output, i.e. the
*first* condition's sklearn CV alone hadn't finished; narrowed to
evolved_T alone per real-time discipline, not a pre-run estimate --
this is the concrete incident behind the new `CLAUDE.md` principle 18,
"different computations need their own timing checks, even within the
same pipeline.")

Real, measured elapsed times (CPU only -- this run was local, not GPU):
sklearn 249.7s (4.2 min), JAX 510.1s (8.5 min, **slower than sklearn
here** -- the recalibrated tolerance now makes JAX actually grind
through many more iterations at large `C` instead of declaring early
non-convergence).

`best_C` matched (0.1, both). Non-convergence is fixed (0/9, vs. always
failing before recalibration). The full curve:

| C | sklearn | jax | diff |
|---:|---:|---:|---:|
| 1e-4 | 1.283012 | 1.282900 | 0.0001 |
| 1e-3 | 0.876343 | 0.876250 | 0.0001 |
| 1e-2 | 0.698169 | 0.698034 | 0.0001 |
| 1e-1 | 0.654089 | 0.655044 | 0.0010 |
| 1 | 0.697745 | 0.702978 | 0.0052 |
| 10 | 0.892122 | 0.928273 | 0.0361 |
| 100 | 1.635789 | 1.733410 | 0.0976 |
| 1000 | 3.239479 | 2.733631 | **0.5058** |
| 10000 | 4.347592 | 3.258859 | **1.0887** |

**Does not verify cleanly.** Small-to-moderate `C` (<=10) match tightly
(hundredths or better). Large `C` (100-10000) diverges substantially --
1.09 absolute log-loss at `C=10000`, *worse* than the earlier synthetic
gap (0.158) at the same grid point. Per the pre-committed gating for
this follow-up: this result is the important finding to report now,
before spending more time on `evolved_lattice`/`evolved_rewired` or any
full-scale timing run -- neither was attempted.

**What this does and doesn't establish:** the `C * n_train` normalized
criterion is a real, measured improvement -- it correctly diagnosed why
the old fixed threshold was wrong by many orders of magnitude, and
fixed both the spurious non-convergence and `best_C` selection on real
ill-conditioned data. It does **not** establish that this module
reproduces sklearn's actual fitted solution at large `C` -- something
beyond convergence-threshold calibration is still causing the two
solvers to land on meaningfully different loss values there. Candidate
explanations not yet investigated: the weak-regularization regime may
have a wide, flat set of near-equally-good solutions where sklearn's
and this module's different optimizer trajectories (different
quasi-Newton implementations entirely) land in different places despite
both being "converged" by their own respective criteria; or the
normalized criterion, while a large improvement, may still not be tight
enough specifically at the grid's largest `C` values.

## Cross-session feedback: real-scale context from the confirmatory run

The other session's `analyze_stage3_results.py` run (the real,
official, 60,000-image sklearn fit that this whole investigation was
independent of and never touched) finished and produced its own locked
confirmatory result. That session reviewed this document and returned
two points worth recording here directly, since both change how the
open gap above should be read.

**1. The C values that actually matter aren't the grid extreme this
document tested at.** The real confirmatory result's CV-selected `C`
per condition: `evolved_T` -> **C=1000**, `evolved_rewired` -> **C=10**,
`evolved_curr_random` -> **C=1**. Step 2 above measured the divergence
*curve shape* across the whole grid (correctly, for characterizing the
port generally), but the single number that would actually matter if
this port were ever used to reproduce or double-check the locked
result is the divergence *at the selected C*, on the *full 60,000-image
fit* -- not at `C=10000` on a 6,000-image subsample. At `C=1000`
specifically (evolved_T's selected value), Step 2's subsample measured
a 0.506 absolute log-loss divergence -- still substantial, and still
the right thing to resolve, but it's worth being precise that this is
the relevant number, not the 1.09 at `C=10000` quoted as the headline
gap above.

**2. Iteration count does not extrapolate from subsample to full scale
-- confirmed by a real second data point, not assumed.** The other
session handed over one concrete number: sklearn's own full-scale
(60,000-image) `evolved_T` refit at `C=1000` took **n_iter=5123** (of
`max_iter=10000`) -- vs. this document's 6,000-image subsample's
n_iter=1378 at the same `C` (see Step 1's table above). Roughly 3.7x
more iterations at 10x the data, not the same iteration count a naive
"small-scale timing extrapolates linearly" assumption would predict.
This is exactly the failure mode `CLAUDE.md` principle 18 (added
earlier in this investigation, prompted by this same document's own
Step 2 timing misestimate) names directly -- and here it's the
*iteration count*, not just wall-clock time, that doesn't extrapolate.
Any future full-scale verification of this port needs its own,
directly-measured convergence behavior at full scale, not an
extrapolation from the 6,000-image subsample used throughout this
document.

**3. The CPU-slower-than-sklearn result (510s vs. 250s, Step 2) may not
be the right comparison to conclude from.** The other session's own
`evolve_on_graph_jax.py` port showed the same pattern at first: faster
than single-threaded numpy, but not clearly faster than numpy already
parallelized across this machine's ~9 CPU cores -- the real win only
appeared once that port ran on GPU. sklearn's CPU baseline likely
benefits from the same multi-core BLAS parallelism. The CPU-only
`optax` comparison in Step 2 is a real, honestly-reported number, but
per this precedent it probably isn't where a genuine speedup would show
up even if the accuracy gap were resolved -- the A100 path (already
demonstrated fast at 12.1x on synthetic `medium_10class_wide` data, and
35.0s on the synthetic production-scale case) is the comparison that
would actually answer whether this port helps, not the CPU-only run.
Not yet re-tested on GPU with the recalibrated `GRAD_NORM_REL` on real
data -- an open item, not a conclusion drawn here.

## Current status

- The `vmap`/`lax.while_loop` NaN instability is fixed and
  stress-tested clean (13/13 CPU, 20/20 GPU) on the case that originally
  exposed it.
- `best_C` selection matches sklearn on easy synthetic data (3/3 cases)
  and on the one real ill-conditioned condition checked in depth
  (evolved_T).
- The convergence-criterion recalibration (`GRAD_NORM_REL=6e-3`,
  `C * n_train`-normalized, derived from a direct measurement of
  sklearn's own converged gradient norm on real data) fixed the
  real-data non-convergence problem and got `best_C` right on
  `evolved_T` -- a genuine, measured improvement, not a guess.
- It did **not** make the full validation-loss curve trustworthy at
  large `C`: 1.09 absolute log-loss divergence from sklearn at
  `C=10000` on real `evolved_T` data, worse than the pre-recalibration
  synthetic gap at the same point.
- `evolved_lattice` and `evolved_rewired` were never re-checked after
  recalibration (deliberately -- evolved_T was gated as the condition
  that had to verify cleanly first).
- No full-scale (60,000-image, all-6-condition) real timing exists.
- **Still not a trustworthy replacement for `stage2a_classifier.py`.**
  Not used for, and should not be used for, any reported Stage 3 result
  as-is.

## Open gaps (for whoever picks this up next)

1. **Large-`C` divergence from sklearn, on real data, is unresolved --
   and the number that actually matters is at the selected `C`, on the
   full 60,000-image fit, not at the grid extreme on the subsample.**
   The `C * n_train`-normalized convergence criterion was a real fix for
   the non-convergence problem but did not close this gap. Per the
   cross-session feedback above, `evolved_T`'s real, locked result
   selected `C=1000` (0.506 divergence measured on the subsample at that
   `C`, not the 1.09 at `C=10000` this document otherwise headlines);
   `evolved_rewired` selected `C=10`; `evolved_curr_random` selected
   `C=1`. Whoever picks this up should verify at those specific values,
   at full scale, not assume the grid-extreme number is representative.
   Not an assumption that a further-tightened tolerance will fix it --
   see the candidate explanations at the end of the Step 2 section
   above.
2. **Convergence behavior does not extrapolate from the 6,000-image
   subsample to full scale -- confirmed, not assumed.** Real data point
   from the other session: `evolved_T` at `C=1000` took n_iter=5123 (of
   max_iter=10000) at full 60,000-image scale, vs. n_iter=1378 on this
   document's 6,000-image subsample at the same `C` -- ~3.7x more
   iterations at 10x the data. Any full-scale verification needs its
   own directly-measured convergence numbers, not a scaled-up guess from
   this document's subsample results (`CLAUDE.md` principle 18).
3. **`evolved_lattice` and `evolved_rewired`**, re-checked against
   sklearn on real data with the recalibrated criterion -- not done,
   deliberately deferred until (1) is understood, since evolved_T was
   the one case that had to verify cleanly first.
4. **Full 60,000-image real-data run**, blocked on both (1)/(2) and a
   practical way to get full-scale data onto a GPU session (the Colab
   upload path hit hard size limits well under 20MB per file; options
   include finer sharding at a size confirmed safe, or a different
   transfer path).
5. **The CPU-only timing comparison (510s vs 250s, JAX slower) likely
   isn't the right basis for concluding whether this port helps.** Per
   the other session's `evolve_on_graph_jax.py` precedent -- faster than
   single-threaded numpy but not clearly faster than numpy already
   parallelized across ~9 CPU cores, with the real win only appearing on
   GPU -- sklearn's CPU baseline here likely benefits from the same
   multi-core BLAS parallelism. The recalibrated module has not yet been
   re-tested on GPU against real data; that, not the CPU comparison, is
   the number that would actually answer whether this port is worth
   using. The A100 numbers elsewhere in this document (12.1x on
   synthetic `medium_10class_wide`, 35.0s on synthetic production-scale)
   are real but predate the `GRAD_NORM_REL` recalibration and are not
   yet confirmed to hold on real, recalibrated, GPU-run data.

## Files

- `stage_2a_classifier_jax.py` -- the port (new, this investigation;
  updated during the follow-up with the `GRAD_NORM_REL` recalibration).
- `analyze_stage_3_results_jax.py` -- R_post/feat_post JAX port (existed
  before this investigation; verified here as a side effect of building
  the real-data test).
- `evolve_on_graph_jax.py`, `stage2a_pipeline_jax.py` -- ODE-evolution
  JAX port (prior session, not modified here).
- `pyproject.toml` -- added `optax>=0.2.8`.
- `CLAUDE.md` -- added methodological principle 18 (a pipeline stage's
  timing does not extrapolate to other stages), prompted directly by
  this investigation's own initial classifier-CV timing misestimate.
