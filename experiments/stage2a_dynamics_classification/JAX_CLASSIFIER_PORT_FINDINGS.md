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

## Current status

- The `vmap`/`lax.while_loop` NaN instability is fixed and
  stress-tested clean (13/13 CPU, 20/20 GPU) on the case that originally
  exposed it.
- `best_C` selection matches sklearn on easy synthetic data (3/3 cases).
- Real data, subsampled: 3 of 6 conditions converge and plausibly match
  sklearn's regime; the 3 hardest (evolved_T, evolved_lattice,
  evolved_rewired -- the scientifically interesting topologies) do not
  converge under this module's current stopping criterion, even at
  sklearn's own `max_iter=10000` budget.
- **Not yet a trustworthy replacement for `stage2a_classifier.py`** on
  the conditions that matter most. Not used for, and should not be used
  for, any reported Stage 3 result as-is.

## Open gaps (for whoever picks this up next)

1. **Convergence criterion recalibration.** `GRAD_NORM_TOL=1e-6` is
   this module's own invention, not derived from sklearn's actual
   effective precision. The large-`C` loss-curve gap (synthetic) and the
   real-data non-convergence on ill-conditioned topologies both trace to
   this. Needs either a criterion demonstrably equivalent to sklearn's,
   or a documented, justified alternative -- not a looser tolerance
   picked to make the symptom disappear.
2. **Full 60,000-image real-data run**, once (1) is resolved and once
   there's a practical way to get the full-scale data onto a GPU session
   (the Colab upload path hit hard size limits well under 20MB per file;
   options include finer sharding at a size confirmed safe, or a
   different transfer path).
3. **Timing number for the real, full-scale, all-6-condition run** is
   still not established -- every number in this document is either a
   synthetic proxy or a partial (6,000/60,000-image, 3/6-converged) real
   run.

## Files

- `stage_2a_classifier_jax.py` -- the port (new, this investigation).
- `analyze_stage_3_results_jax.py` -- R_post/feat_post JAX port (existed
  before this investigation; verified here as a side effect of building
  the real-data test).
- `evolve_on_graph_jax.py`, `stage2a_pipeline_jax.py` -- ODE-evolution
  JAX port (prior session, not modified here).
- `pyproject.toml` -- added `optax>=0.2.8`.
