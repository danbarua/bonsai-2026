Simplified Technical English version of `experiments/stage2a_dynamics_classification/JAX_CLASSIFIER_PORT_FINDINGS.md`

# JAX Classifier Port: Investigation and Current Status

**Status: this is investigative work. It is not a verified drop-in
replacement for `stage2a_classifier.py`, and it is not used to produce
any reported Stage 3 result.** This document stands on its own. It
covers only the JAX-port investigation triggered by
`analyze_stage3_results.py`'s multi-hour runtime. It does not restate
`FINDINGS.md`'s scientific results. The real 60,000-image
`analyze_stage3_results.py` CPU run that triggered this investigation
was never touched, stopped, or written to by anything described here —
it was independently watched elsewhere, and it finished on its own.

## Motivation

`experiments/stage2a_dynamics_classification/analyze_stage3_results.py`
(the phase-2 step that combines data and runs the classifier, for
feasibility stage 3: 60,000 images across 4 topologies) had been running
for hours. A JAX-vectorized (batch-processed) port of it already existed
(`analyze_stage3_results_jax.py`), and the question was whether running
it on GPU would fix the runtime.

## Finding 1: the existing JAX port speeds up the wrong step

`analyze_stage3_results_jax.py` batches the `R_post` and `feat_post`
computation (`stage2a_core.order_parameter` and
`reference_node_features`) using `jax.vmap`, replacing a Python loop
over 240,000 (image, topology) pairs. This was verified correct:

- `order_parameter` and `reference_node_features`, compared against the
  `stage2a_core` plain numpy version, on synthetic (made-up) 505-node
  states (20 trials, plus explicit edge cases where `ref_idx` is 0 or
  `n-1`): the largest difference `|R_ref - R_jax|` was 3.469e-17, and
  the largest difference `|feat_ref - feat_jax|` was 0.0.
- The full `build_results_structure` output (including solver-failure
  handling across all four topologies), compared field by field between
  `analyze_stage3_results.py` and `analyze_stage3_results_jax.py`, on
  synthetic mock data with deliberately inserted solver failures: 0
  mismatches, largest numeric difference 2.220e-16.

But this step was not what was taking hours. Both scripts call the
identical, unchanged
`stage2a_pipeline.run_classifier_conditions_multi_topology`, which fits
6 conditions x 5 folds x 9 `C` values = 270 scikit-learn
`LogisticRegression` fits (`stage2a_classifier.py`) — this is plain CPU
scikit-learn, untouched by either script's JAX path. The original
script's own inline comments already identify this as "the expensive
step … timed explicitly," and the session independently watching the
real CPU run confirmed it: about 236 minutes and still running, matching
`stage2a_classifier.py`'s own documented history of needing
`max_iter=10000` (raised from 1000) because of "severe feature
multicollinearity (condition number ~2e6)" specific to `evolved_T`. So
porting `R_post` and `feat_post` to GPU, however correct, could not have
fixed the multi-hour runtime.

## Finding 2: most of a "port everything to JAX" request was already done

Before building anything new, a search of the directory found that the
actual expensive physics step — the graph-evolution ODE solve
(`stage2a_core.evolve_on_graph`) — was already ported to JAX/`diffrax`
in an earlier session:

- `evolve_on_graph_jax.py`: a `diffrax.Tsit5` port of the plain
  (unperturbed) evolution ODE, reusing the already-verified `rhs`
  function from Stage 1D's `run_one_trial_jax_faithful.py`.
- `stage2a_pipeline_jax.py`: wraps it with the exact
  `run_pipeline_multi_topology` result-dictionary format, following this
  project's own established lesson (`CLAUDE.md` principle 16, from Stage
  1D) that a verified kernel can still feed a wrong result if the
  calling code around it differs from the real pipeline.
- Per `FINDINGS.md` and the git history: verified, with a measured
  speedup of about 546 times on a 100-image GPU sanity run.

So the only genuinely unported piece of the classification pipeline was
the classifier cross-validation fit itself —
`stage2a_classifier.select_C_via_cv`.

## The classifier port: `stage2a_classifier_jax.py`

A new file, `experiments/stage2a_dynamics_classification/stage2a_classifier_jax.py`.
It reimplements `stage2a_classifier.select_C_via_cv`'s multinomial
logistic regression (L2-regularized, 5-fold class-balanced
cross-validation over the locked 9-value `C` grid), with the model fit
itself done in JAX:

- Fold splitting (`StratifiedKFold`) and per-fold standardization
  (`StandardScaler`) are reused **unchanged** from scikit-learn — they
  are cheap, not the bottleneck, and this keeps fold assignment and
  scaling byte-identical to the reference pipeline.
- The fit itself: softmax cross-entropy loss plus L2 regularization on
  the weight matrix only (the intercept is not regularized, matching
  scikit-learn's `LogisticRegression(solver="lbfgs")` objective:
  `C * sum_i NLL_i(W,b) + 0.5*sum(W**2)`), minimized using `optax.lbfgs`
  (added as a project dependency, `optax>=0.2.8`).
- The 9 `C`-grid problems within a fold are solved as **one batched
  optimization, using `jax.vmap`**, instead of 9 separate scikit-learn
  fits — this is the real target for GPU parallelism, since
  scikit-learn's cross-validation loop is single-threaded, with no
  `n_jobs` setting in the locked procedure.
- Convergence is judged by `||grad|| <= GRAD_NORM_TOL` (the gradient's
  size falls below a tolerance; module default `1e-6`) after at most
  `MAX_ITER` steps (module default `2000`, can be overridden). **This is
  a different rule from scikit-learn's internal `n_iter_[0] < max_iter`
  stopping rule, and the two are not directly comparable** — see "Open
  gaps" below.

### Bug found and fixed: unstable NaN values under `vmap` and `lax.while_loop`

The initial version used `optax.value_and_grad_from_state` (the
library's recommended pattern for caching, to avoid recomputing the same
value and gradient). This produced a NaN (not-a-number) value on the
very first L-BFGS step, under `jax.vmap` combined with
`jax.lax.while_loop`. This was reproduced even at a `vmap` batch size of
exactly 1 — ruling out cross-lane contamination between different `C`
values as the cause. Fix: switched to plain `jax.value_and_grad`,
recomputed explicitly at every step (this gives up a caching
optimization; it is not a correctness change).

That fix reduced, but did not eliminate, the problem. The same
"`n_iter=0`, `gnorm=nan`" pattern recurred sometimes, run to run, with
identical code and identical data. The team ruled out several causes in
turn: a batch-of-1 test ruled out cross-lane contamination; a
fold-order-permutation test ruled out any dependence on call order; a
test interleaving scikit-learn and JAX calls ruled out cross-library
interference. The cause was traced to the **unguarded initial gradient
computation**, evaluated once outside the `while_loop`, to decide
whether to enter the loop. Every step *inside* the loop already had a
check for finite values, but this first evaluation did not. The most
likely explanation: CPU-backend thread-scheduling randomness in the
large-reduction matrix multiplications feeding the value and gradient
computation, occasionally disturbing an L-BFGS curvature pair (`s_k`,
`y_k`) enough that `1/(y_k . s_k)` becomes very large — a classic
L-BFGS curvature problem, plausible exactly where `C` is small enough
that the objective is nearly a pure quadratic, and a single step lands
very close to the optimum (the failure was first isolated at `C=1e-4`,
the grid's smallest value).

Final fix: the code was restructured so **every** value-and-gradient
evaluation, including the first one, goes through one uniform, guarded
path (there is no special pre-loop computation any more) — using
`jax.lax.cond` at every iteration: if the freshly computed value and
gradient are not finite, the step leaves the parameters and optimizer
state untouched, and retries on the next `while_loop` iteration (a fresh
computation dispatch of the identical calculation, which in practice
recovers); if a step computes a finite value and gradient but then
produces a non-finite optimizer update, that update is discarded, and
the L-BFGS curvature memory is reset to the last good point (a standard
L-BFGS robustness technique), rather than letting a corrupted curvature
pair persist.

Stress-test results after the fix, on the synthetic case that had
originally exposed the bug (`n=3000, d=200, k=10`, 5-fold x 9-`C`
`select_C_via_cv_jax` call):

| backend | runs | failures |
|---|---:|---:|
| CPU (this machine) | 13 | 0 |
| GPU (A100, Colab) | 20 | 0 |

(Before the final fix: roughly 1-in-2 failure rate on CPU immediately
after the `value_and_grad_from_state` fix alone, dropping to roughly
1-in-6 after the loop-body guard alone — neither fix was enough on its
own.)

### Correctness check against scikit-learn (synthetic data)

Three synthetic (made-up) cases, built with
`sklearn.datasets.make_classification`, each compared against
`stage2a_classifier.select_C_via_cv` on identical data:

| case | n | d | k | `best_C` match | max \|val_loss diff\| across grid |
|---|---:|---:|---:|---|---:|
| small_2class | 400 | 20 | 2 | yes (1.0) | 7.50e-02 |
| small_10class | 1200 | 40 | 10 | yes (1.0) | 1.20e-01 |
| medium_10class_wide | 3000 | 200 | 10 | yes (0.1) | 4.18e-01 (at C=10000) |

`best_C` selection matched scikit-learn's in all three cases. **The
full per-`C` validation-loss curve did not match as well** — the
difference grows with `C` (weaker regularization), reaching a 0.42
absolute log-loss gap at `C=10000` in the widest case. Since
`stage2a_pipeline`'s `run_classifier_conditions_multi_topology` reports
the full `mean_val_loss_per_C` curve as a diagnostic, not just the best
value, this gap is an open item, not a passed check — see "Open gaps."

### Timing (A100, Colab, synthetic data)

| case | n | d | sklearn (CPU) | JAX (A100) | ratio |
|---|---:|---:|---:|---:|---:|
| small_2class | 400 | 20 | 0.25s | 0.68s | 0.4x (GPU slower — dispatch overhead not paid off yet) |
| small_10class | 1200 | 40 | 0.83s | 2.10s | 0.4x |
| medium_10class_wide | 3000 | 200 | 17.57s | 1.45s | **12.1x** |
| production_scale (synthetic, n=60000, d=1008, k=10, ill-conditioned via `n_redundant=800`) | 60000 | 1008 | not run (would take about as long as the real CPU job) | 35.0s | -- |

The production-scale case's loss curve was almost flat across `C`
(1.6706 from `C=0.01` to `C=10000`). This suggests `make_classification`'s
`n_redundant` setting (exact linear combinations of other features) is
a cruder, more extreme form of feature collinearity than the real
data's condition-number-of-about-2e6 case — **the 35.0s figure is a
real, hardware-measured number for a similarly-shaped, but not
necessarily similarly hard, problem. It is not a validated stand-in for
the real 6-condition production runtime.**

### Real data test

This was tested read-only against the actual Stage 3 data files
(`stage3_encode_local.pkl`, `stage3_gpu_results.pkl`; 60,000 images,
`ref_idx=363`, 0 solver failures across all 4 topologies at full scale),
using the already-verified
`analyze_stage3_results_jax.build_results_structure`. Nothing was
written to `experiments/stage2a_dynamics_classification/results/` or the
shared scratch directory.

The full 60,000-image per-condition arrays (up to about 242MB each)
could not be uploaded to the Colab session — repeated HTTP 500 errors
came from an undocumented backend size limit somewhere between 6MB and
15.7MB (a `/content/data/` subdirectory left in a bad state by earlier
failed large uploads was a compounding, separate problem, resolved by
uploading files flat, directly under `/content/`). A **stratified
random sample of 6,000 of the 60,000 real images** was used instead
(random seed 0, stratified by label, using `np.random.default_rng`),
split into files of about 8MB each to stay under the limit.

`select_C_via_cv_jax` result, all 6 conditions, run at `max_iter=10000`
(scikit-learn's own budget) after `max_iter=2000` (this module's
original default) produced non-convergence in 6 of 6 conditions:

| condition | n | d | result | `best_C` | elapsed |
|---|---:|---:|---|---:|---:|
| raw_pixels | 6000 | 784 | converged | 0.01 | 55.8s |
| encoded_pre_evolution | 6000 | 1008 | converged | 0.01 | 116.8s |
| evolved_T | 6000 | 1008 | **non-converged** (final \|\|grad\|\|=2.60e-02) | -- | 32.8s |
| evolved_lattice | 6000 | 1008 | **non-converged** (final \|\|grad\|\|=2.23e-01) | -- | 32.5s |
| evolved_rewired | 6000 | 1008 | **non-converged** (final \|\|grad\|\|=6.88e-05, close) | -- | 32.9s |
| evolved_curr_random | 6000 | 1008 | converged | 0.1 | 122.3s |

Total: 393.1s (6.55 min) for all 6 conditions attempted.

`evolved_T`'s non-convergence here matches — it does not contradict —
`stage2a_classifier.py`'s own documented history for this exact
condition (the `max_iter` 1000-to-10000 change, condition number about
2e6). This is evidence that this module's `GRAD_NORM_TOL=1e-6` rule is
not calibrated for genuinely ill-conditioned real data. It is not
evidence of a new, separate bug — both findings point at the same
underlying issue: this port's stopping rule does not yet track
scikit-learn's effective precision across different conditioning
levels.

## Follow-up: recalibrating the convergence rule (evolved_T)

This is a direct follow-up to the open gap above, done as a separate
check. It never touched the real CPU `analyze_stage3_results.py` run,
which finished on its own, using the already-trusted scikit-learn path,
during this follow-up.

### Step 1: what gradient size does scikit-learn's own converged solution reach?

Using `stage2a_classifier._fit_one`'s fitted `coef_` and `intercept_`,
for each `C` in the locked grid, on real `evolved_T` data (a 6,000-image
stratified sample, fold 0, `n_train=4800`) — with `||grad||`
recomputed directly using `stage2a_classifier_jax._make_loss_fn` (not a
reimplementation):

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

Scikit-learn's own achieved `||grad||` value ranges over **eight orders
of magnitude** (7.0e-4 to 1.33e5), and is three to eleven orders of
magnitude looser than `GRAD_NORM_TOL=1e-6` at every single grid point.
This is not scikit-learn being imprecise — the objective function
`C * sum_i NLL_i(W,b) + 0.5*sum(W**2)` is an unweighted **sum**, not a
mean, over `n_train` samples. So `||grad||`, at any fixed fit quality,
scales close to linearly with both `C` and `n_train`, by construction. A
single fixed threshold cannot be right across a grid spanning `C=1e-4`
to `C=1e4`.

The normalized quantity `||grad|| / (C * n_train)` is tight over the
same measurements: **[1.338e-3, 2.771e-3]**, a spread of about 2.07x
(compared to `||grad||` alone's spread of about 1.9e8x):

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

**A systematic trend versus noise was checked directly (not pursued
further, to keep scope narrow):** both are present. There is a real,
roughly 2x systematic increase in the ratio from small `C` (about
1.3-1.5e-3) to large `C` (about 2.4-2.8e-3), but the middle of the grid
(`C=0.01` to `C=10`) moves up and down on top of this trend (for
example, `C=0.01`'s ratio is higher than `C=0.1`'s; `C=1`'s is higher
than `C=10`'s) — this is plausibly per-fit noise, from scikit-learn's
own relative `tol=1e-4` stopping rule landing at a slightly different
absolute precision on each independent solve. The upper part of the
grid (`C>=100`) increases cleanly and steadily. A single fixed tolerance
with a safety margin absorbs both the trend and the noise reasonably
well; a tighter, `C`-dependent correction is possible, but was not
pursued.

### Implementation

`stage2a_classifier_jax.py` changed in the following ways:

- `GRAD_NORM_TOL=1e-6` (a fixed value) was replaced with
  `GRAD_NORM_REL=6e-3` (about 2 times the largest observed ratio,
  2.771e-3), applied as `grad_tol = GRAD_NORM_REL * C * n_train`,
  computed inside `_solve_one` from the actual fold's `X.shape[0]`. So
  `n_train` is genuinely computed per fold (`StratifiedKFold`'s
  possibly-unequal fold sizes are handled correctly, never assumed to
  be constant).
- `MAX_ITER`'s default was raised from `2000` to `10000`, matching
  scikit-learn's own `CLASSIFIER_KWARGS max_iter` exactly, for the same
  documented reason `stage2a_classifier.py` itself raised it: real
  non-convergence on `evolved_T`'s condition number of about 2e6, not a
  number chosen separately for this module.
- The full derivation, with this same measurement, is now written into
  the module's own docstring, matching the documentation standard
  already used there for the `max_iter` 1000-to-10000 precedent.

### Synthetic re-check (a sanity check, not the real bar)

The original three synthetic cases were re-run after recalibration:

| case | `best_C` match | max \|val_loss diff\| (before -> after) |
|---|---|---:|
| small_2class | yes | 7.50e-02 -> **2.680e-01** (worse) |
| small_10class | yes | 1.20e-01 -> **4.108e-01** (worse) |
| medium_10class_wide | yes | 4.18e-01 -> **1.580e-01** (better, about 62% smaller) |

This is a mixed result, and it is not surprising: `GRAD_NORM_REL` was
derived from real ill-conditioned data, which is a *looser* bar than
what easy, well-conditioned synthetic problems can actually reach. So
the new threshold now stops the easy cases *earlier* (at a less precise
fit) than the old fixed `1e-6` threshold did. `best_C` selection still
matched scikit-learn in all three cases regardless.

### Step 2: re-checking against real data (evolved_T only)

This compares the full scikit-learn-versus-JAX per-`C` validation-loss
curve, on real `evolved_T` data (the 6,000-image sample) — the same
condition `GRAD_NORM_REL` was derived from. This is the one case that
has to pass cleanly for the recalibration to mean anything. (An initial
attempt to cover all three hard conditions at once — evolved_T,
evolved_lattice, evolved_rewired — was stopped after 21 minutes of
wall-clock time with zero output, meaning even the *first* condition's
scikit-learn cross-validation had not finished yet. The check was
narrowed to evolved_T alone, based on real-time observation, not on a
pre-run time estimate — this is the concrete incident behind the new
`CLAUDE.md` principle 18, "different computations need their own timing
checks, even within the same pipeline.")

Real, measured elapsed times (CPU only — this run was local, not GPU):
scikit-learn 249.7s (4.2 min), JAX 510.1s (8.5 min, **slower than
scikit-learn here** — the recalibrated tolerance now makes JAX actually
work through many more iterations at large `C`, instead of declaring
early non-convergence).

`best_C` matched (0.1, both). Non-convergence is fixed (0 of 9, versus
always failing before recalibration). The full curve:

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

**This does not pass cleanly.** Small-to-moderate `C` (10 or below)
match tightly (to a hundredth or better). Large `C` (100 to 10000)
diverges substantially — 1.09 absolute log-loss at `C=10000`, *worse*
than the earlier synthetic gap (0.158) at the same grid point. Per the
plan set before this check ran: this result is the important finding to
report now, before spending more time on `evolved_lattice` or
`evolved_rewired`, or any full-scale timing run — neither of those was
attempted.

**What this does and does not establish:** the `C * n_train` normalized
rule is a real, measured improvement — it correctly explains why the
old fixed threshold was wrong by many orders of magnitude, and it fixed
both the spurious non-convergence and the `best_C` selection on real
ill-conditioned data. It does **not** establish that this module
reproduces scikit-learn's actual fitted solution at large `C` —
something beyond just calibrating the convergence threshold is still
causing the two solvers to land on meaningfully different loss values
there. Candidate explanations not yet investigated: at weak
regularization, there may be a wide, flat set of near-equally-good
solutions, where scikit-learn's optimizer and this module's optimizer
(entirely different quasi-Newton implementations) land in different
places despite both being "converged" by their own separate rules; or
the normalized rule, while a large improvement, may still not be tight
enough specifically at the largest `C` values in the grid.

## Cross-session feedback: real-scale context from the confirmatory run

The other session's `analyze_stage3_results.py` run — the real,
official, 60,000-image scikit-learn fit that this whole investigation
ran independently of and never touched — finished, and produced its own
locked confirmatory result. That session reviewed this document and sent
back two points worth recording here directly, since both change how
the open gap above should be read.

**1. The C values that actually matter are not the grid extreme this
document tested at.** The real confirmatory result's cross-validation-
selected `C`, per condition: `evolved_T` -> **C=1000**,
`evolved_rewired` -> **C=10**, `evolved_curr_random` -> **C=1**. Step 2
above measured the gap's *shape* across the whole grid (correctly, for
characterizing the port in general), but the one number that would
actually matter, if this port were ever used to reproduce or
double-check the locked result, is the gap *at the selected C*, on the
*full 60,000-image fit* — not at `C=10000` on a 6,000-image sample. At
`C=1000` specifically (evolved_T's selected value), Step 2's sample
measured a 0.506 absolute log-loss gap — still substantial, and still
the right thing to resolve, but it is worth being precise that this is
the relevant number, not the 1.09 at `C=10000` quoted as the headline
gap above.

**2. Iteration count does not scale up from a sample to the full
dataset — confirmed by a real second data point, not assumed.** The
other session provided one concrete number: scikit-learn's own
full-scale (60,000-image) `evolved_T` refit at `C=1000` took
**n_iter=5123** (out of `max_iter=10000`), versus this document's
6,000-image sample's n_iter=1378 at the same `C` (see Step 1's table
above). That is roughly 3.7 times more iterations at 10 times the data,
not the same iteration count a naive "small-scale timing scales up
linearly" assumption would predict. This is exactly the failure pattern
`CLAUDE.md` principle 18 (added earlier in this investigation, prompted
by this same document's own Step 2 timing misjudgment) names directly —
and here it is the *iteration count*, not just wall-clock time, that
does not scale up as expected. Any future full-scale check of this port
needs its own, directly-measured convergence behavior at full scale, not
a scaled-up guess from this document's sample results.

**3. The CPU-slower-than-scikit-learn result (510s versus 250s, Step 2)
was not the right comparison to draw conclusions from — confirmed by
immediately re-running on GPU.** The other session's own
`evolve_on_graph_jax.py` port showed the same pattern at first (faster
than single-threaded numpy, but not clearly faster than numpy already
parallelized across about 9 CPU cores; the real win only showed up on
GPU). The same pattern held here too. Using the same real `evolved_T`
data (the 6,000-image sample), the same recalibrated module, but an A100
GPU instead of CPU:

| | elapsed | vs. sklearn (250s) | vs. CPU JAX (510s) |
|---|---:|---:|---:|
| sklearn (CPU) | 249.7s | -- | -- |
| JAX (CPU) | 510.1s | 0.5x (slower) | -- |
| **JAX (A100)** | **25.9s** | **9.6x** | **19.7x** |

`best_C` still matched (0.1). The per-`C` values matched the CPU-JAX
run closely (small hardware-level floating-point differences, not a
different result — consistent with the randomness already characterized
earlier in this investigation). Importantly, this also answers a
question the accuracy-gap section above left open: **the large-`C` gap
from scikit-learn persists on GPU, at essentially the same size** (0.610
at `C=1000`, 0.942 at `C=10000`, versus 0.506 and 1.09 on CPU) — ruling
out CPU-versus-GPU backend differences as the explanation, and
confirming this is a genuine algorithm or convergence-path issue, not a
hardware artifact.

## Current status

- The `vmap`/`lax.while_loop` NaN instability is fixed, and
  stress-tested clean (13 of 13 on CPU, 20 of 20 on GPU) on the case
  that originally exposed it.
- `best_C` selection matches scikit-learn on easy synthetic data (3 of 3
  cases), and on the one real ill-conditioned condition checked in depth
  (evolved_T).
- The convergence-rule recalibration (`GRAD_NORM_REL=6e-3`, normalized
  by `C * n_train`, derived from a direct measurement of scikit-learn's
  own converged gradient size on real data) fixed the real-data
  non-convergence problem, and got `best_C` right on `evolved_T` — a
  genuine, measured improvement, not a guess.
- It did **not** make the full validation-loss curve trustworthy at
  large `C`: there is a 1.09 absolute log-loss gap from scikit-learn at
  `C=10000` on real `evolved_T` data, worse than the pre-recalibration
  synthetic gap at the same point — and this gap is confirmed present on
  GPU too (0.94 at the same point), which rules out CPU-versus-GPU
  backend differences as the cause.
- **Confirmed at full scale, at all three real selected `C` values
  (1000, 10, and 1), against the real confirmatory result's known
  test-set numbers**: this port's held-out test accuracy and log-loss
  are measurably worse than scikit-learn's, in every case (1.87
  percentage points and 0.061 log-loss at `C=1000`, down to 0.55
  percentage points and 0.026 log-loss at `C=1`) — the gap shrinks with
  `C`, as predicted, but never disappears. This is not a subsample or
  grid-extreme artifact.
- GPU **is** where this port's real speed advantage shows up: 25.9s on
  A100, versus 249.7s scikit-learn-CPU (9.6 times faster), versus
  510.1s CPU-JAX (19.7 times faster), on the same real `evolved_T` data,
  with the same recalibrated module. The earlier CPU-only "JAX is
  slower" result was real, but it was not representative of GPU
  performance — consistent with the other session's
  `evolve_on_graph_jax.py` precedent, confirmed directly here rather
  than assumed.
- `evolved_lattice` and `evolved_rewired` were never re-checked after
  recalibration (deliberately — evolved_T was chosen as the condition
  that had to verify cleanly first).
- No full-scale (60,000-image, all-6-condition) real timing exists yet.
- **This is still not a trustworthy replacement for
  `stage2a_classifier.py`.** It is not used for, and should not be used
  for, any reported Stage 3 result as it stands — the large-`C` accuracy
  gap is the blocker, separate from the now-resolved speed question.

## Cross-session follow-up: item 1 checked directly, at full scale, all three real selected `C` values

This was done by the other session, using the already-cached full
60,000-image training features and the already-cached official
10,000-image test set from the real confirmatory run (no refitting of
scikit-learn was needed — its test-set numbers were already known from
that run). This module's `_solve_one` and `_predict_proba` functions
were called directly (not reimplemented), once per condition, at exactly
the `C` value each condition actually selected — not the grid extremes
this document otherwise focuses on, and not a 6,000-image sample:

| condition | `C` | sklearn acc / log-loss (known, confirmatory run) | this port's acc / log-loss | `\|\|grad\|\|` at convergence | n_iter |
|---|---:|---|---|---:|---:|
| evolved_T | 1000 | 0.8058 / 0.7067 | 0.7871 / 0.7679 | 3.567e5 (tol 3.6e5) | 1847 |
| evolved_rewired | 10 | 0.8183 / 0.6739 | 0.8105 / 0.6998 | 3.582e3 (tol 3.6e3) | 976 |
| evolved_curr_random | 1 | 0.8221 / 0.6509 | 0.8166 / 0.6774 | 3.557e2 (tol 3.6e2) | 333 |

**The gap is real at every one of the three values that actually
matter, not just at the grid extremes.** It shrinks with `C`, exactly in
the direction this document's own trend analysis predicted (accuracy
gap 1.87 percentage points at `C=1000`, down to 0.55 percentage points
at `C=1`; log-loss gap 0.061 down to 0.026), but it does not disappear
even at `C=1`, the mildest of the three. This confirms item 1 was not an
artifact of the sample or the grid extremes — **this module's held-out
test-set performance would have been measurably, consistently worse
than scikit-learn's, on all three real comparisons, had it been used
for the actual confirmatory result.** Item 1 (the accuracy gap) is
answered, in the sense that "yes, it is real, and it matters at the
values that count." Its root cause (a flat-loss-surface with different
optimizer paths, versus a genuinely different optimum) remains open, per
the candidate explanations already recorded in the Step 2 section above.

GPU throughput at this full scale, as a side note: 9.4s, 3.0s, and 1.1s
for `evolved_T`, `evolved_rewired`, and `evolved_curr_random`
respectively (all three faster than scikit-learn's corresponding
final-refit times of 458.3s, 190.9s, and 49.5s from the confirmatory
run) — the speed advantage holds at full scale too, on top of the
now-confirmed accuracy gap. Speed and correctness are separate axes:
this result settles neither one in this port's favor on its own — being
fast and measurably wrong is not a replacement for being slow and
right.

## Open gaps (for whoever picks this up next)

1. ~~The large-`C` gap from scikit-learn, on real data, is unresolved
   — and the number that actually matters is at the selected `C`, on
   the full 60,000-image fit, not at the grid extreme on the sample.~~
   **Checked directly, above: confirmed real at all three actual
   selected `C` values, at full scale, against the real confirmatory
   test-set numbers.** The root cause is still open — see the candidate
   explanations at the end of the Step 2 section above. This module
   remains not usable for any reported result until that is resolved,
   separate from its now-twice-confirmed speed advantage.
2. **Convergence behavior does not scale up from the 6,000-image sample
   to full scale — confirmed, not assumed.** Real data point from the
   other session: `evolved_T` at `C=1000` took n_iter=5123 (out of
   max_iter=10000) at full 60,000-image scale, versus n_iter=1378 on
   this document's 6,000-image sample at the same `C` — about 3.7 times
   more iterations at 10 times the data. Any full-scale check needs its
   own directly-measured convergence numbers, not a scaled-up guess from
   this document's sample results (`CLAUDE.md` principle 18).
3. **`evolved_lattice` and `evolved_rewired`**, re-checked against
   scikit-learn on real data with the recalibrated rule — not done,
   deliberately deferred until item 1 is understood, since evolved_T was
   the one case that had to verify cleanly first.
4. **A full 60,000-image real-data run**, blocked on items 1 and 2, and
   on a practical way to get full-scale data onto a GPU session (the
   Colab upload path hit hard size limits well under 20MB per file;
   options include splitting into smaller files at a confirmed-safe
   size, or using a different transfer path).
5. ~~The CPU-only timing comparison likely is not the right basis for
   deciding whether this port helps.~~ **Resolved.** The same real
   `evolved_T` data was re-run on A100: 25.9s versus scikit-learn-CPU's
   249.7s (9.6 times faster), and CPU-JAX's 510.1s (19.7 times faster).
   The speed question is answered. The accuracy question (item 1) is
   not, and it is now the sole blocker to this port being trustworthy,
   separate from speed.

## A reproducibility gap in this investigation itself, found and closed

**A self-review finding, not part of external review's original pass**:
`stage2a_classifier_jax.py`'s own module docstring twice refers to
`verify_stage2a_classifier_jax.py` by name ("see
verify_stage2a_classifier_jax.py", "See
verify_stage2a_classifier_jax.py's debug history") — but that file was
never actually committed. It only ever existed in an interactive
session's local scratch directory. This is a reference in committed code
that points to a file that does not exist — the same category of
problem external review flagged for the confirmatory pipeline (already
fixed there — see `FINDINGS.md`'s "Reproducibility gaps" section), just
found here in this investigation's own follow-on code instead. It was
fixed the same way: by committing two scripts.

- `verify_stage2a_classifier_jax.py` — the file the docstring already
  pointed at. It is synthetic-data-only (fast, with no dependency on
  cached real data files): it checks correctness (the `best_C` match,
  and the reported loss-curve gap) across three synthetic cases, plus a
  10-repetition stress test of the `vmap`/`lax.while_loop` NaN-guard
  fix (a smaller, always-runnable version of the 13-run and 20-run
  CPU/GPU stress test reported earlier in this document).
- `diagnose_classifier_jax_grad_norm_calibration.py` — reproduces both
  real-data measurements behind the `GRAD_NORM_REL` recalibration
  (scikit-learn's own converged `||grad||` table, and the full real-data
  scikit-learn-versus-JAX curve comparison on `evolved_T`), using the
  real cached Stage 3 files, through the already-verified
  `build_results_structure` function, instead of the makeshift npz-file
  splitting workaround the original interactive session used to get
  data onto a Colab session (that workaround was only needed for GPU
  upload-size limits, not for this script's local-only CPU
  computation). This script is slow (about 13 minutes — a real 5-fold
  scikit-learn cross-validation fit at `max_iter=10000` on
  ill-conditioned data) and diagnostic-only. It does not run
  automatically. It reproduces the *qualitative* finding exactly
  (`GRAD_NORM_REL=6e-3` keeps its margin; `best_C` matches), but not the
  byte-identical numbers — the original interactive run passed features
  through float32 files, as part of that same now-unnecessary
  Colab-upload workaround; this script keeps full float64 precision
  throughout. See the script's own docstring for the measured
  difference.

**A second instance of the same gap, found during a later organization
and reproducibility check of the whole directory**:
`analyze_stage3_results_jax.py` was the only JAX port in the directory
with no committed `verify_*.py` companion script. `Finding 1` above
documents that its `R_post` and `feat_post` computation *was* verified
(largest differences 3.469e-17 and 0.0, and 0 mismatches, on synthetic
mock data), but only as a side effect of building the real-data test in
an interactive session, with no standalone script committed to record
that check. This gap is now closed with `verify_analyze_stage3_results_jax.py`,
which records exactly the two checks already documented in `Finding 1`
— no new or expanded check, and it was confirmed to actually pass
before being committed.

## Files

- `stage2a_classifier_jax.py` — the port itself (new in this
  investigation; updated during the follow-up with the `GRAD_NORM_REL`
  recalibration).
- `verify_stage2a_classifier_jax.py`,
  `diagnose_classifier_jax_grad_norm_calibration.py` — verification and
  diagnostic scripts for the above, committed to close the
  reproducibility gap noted directly above (they previously existed only
  in local scratch space).
- `analyze_stage3_results_jax.py` — the `R_post`/`feat_post` JAX port
  (existed before this investigation). `verify_analyze_stage3_results_jax.py`
  — a standalone verification script (committed later, closing the
  reproducibility gap noted above, where this check was previously only
  done ad hoc).
- `evolve_on_graph_jax.py`, `stage2a_pipeline_jax.py` — the ODE-evolution
  JAX port (from a prior session, not changed here).
- `pyproject.toml` — added `optax>=0.2.8`.
- `CLAUDE.md` — added methodological principle 18 (a pipeline stage's
  timing does not scale up the same way to other stages), prompted
  directly by this investigation's own initial classifier
  cross-validation timing misjudgment.

**Reproducing this document's checks**: the two verify scripts are fast,
synthetic-data-only, and always runnable (no dependency on a cached real
data file):
```bash
uv run python experiments/stage2a_dynamics_classification/verify_stage2a_classifier_jax.py
uv run python experiments/stage2a_dynamics_classification/verify_analyze_stage3_results_jax.py
```
`diagnose_classifier_jax_grad_norm_calibration.py` reproduces the
real-data measurements (Step 1 and Step 2 above), but it needs the real
cached Stage 3 training data locally
(`stage2a_paths.train_scratch_dir()`, or pulled from the public GCS
bucket — see `README.md`'s "Public artifact cache (GCS)" section), and
it is slow (about 13 minutes):
```bash
uv run python experiments/stage2a_dynamics_classification/diagnose_classifier_jax_grad_norm_calibration.py
```
