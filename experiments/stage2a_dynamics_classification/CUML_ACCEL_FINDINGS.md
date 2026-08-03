# NVIDIA cuML `accel`: zero-code-change GPU acceleration check

**Status: the one caveat found is resolved, and a full 6-condition,
270-fit CV-grid replication independently reproduces every verdict of
the locked confirmatory result (14.9x faster). Still not adopted for
any reported result -- no reported result has needed it, and this
project's own sklearn-based numbers remain what's reported.** Prompted
by the same question the JAX/optax classifier
port was built to answer (is the slow, sklearn-based classifier CV
fitting acceleratable?), tested here via a different route: NVIDIA
RAPIDS' `cuml.accel`, which monkey-patches `sklearn.linear_model.
LogisticRegression` (among other estimators) to dispatch to a GPU-native
implementation while keeping the exact same class name and `fit`/
`predict_proba` API. Unlike the JAX port, this is **our actual,
unmodified `stage2a_classifier._fit_one`** -- no reimplementation, no
new module to maintain or independently verify from scratch.

## Setup

Fresh A100 session (`cuml-crosscheck`, via `mighty-colab`), `cuml-cu12`
installed via `mighty-colab reinstall` (single atomic install+restart
call, v0.1.21) with `--extra-index-url=https://pypi.nvidia.com` in a
requirements file -- installed cleanly on first attempt, `cuml==26.02.000`.
`cuml.accel.install()` called before importing `stage2a_classifier` (and
therefore before `sklearn` itself is imported), per `cuml.accel`'s own
activation contract. Same already-cached, already-standardized full
60,000-image training features and official 10,000-image test features
used for the JAX-port cross-check (same `StandardScaler`-fit-on-full-
train preprocessing, byte-comparable to what the real confirmatory run
used).

Tested at exactly the three real selected `C` values from the locked
confirmatory result -- `evolved_T=1000`, `evolved_rewired=10`,
`evolved_curr_random=1` -- not a grid extreme, not a subsample.

## Result: real speedups, essentially-matching (often marginally better) predictive quality, one real caveat

| condition | `C` | sklearn acc / log-loss / n_iter / time | `cuml.accel` acc / log-loss / n_iter / time / converged |
|---|---:|---|---|
| evolved_T | 1000 | 0.8058 / 0.7067 / 5123 / 458.3s | 0.8059 / 0.6860 / 10000 / 13.9s / **False*** |
| evolved_rewired | 10 | 0.8183 / 0.6739 / 2054 / 190.9s | 0.8197 / 0.6659 / 3514 / 5.0s / True |
| evolved_curr_random | 1 | 0.8221 / 0.6509 / -- / 49.5s | 0.8223 / 0.6462 / 1020 / 2.7s / True |

*`evolved_T`'s `converged=False` at `max_iter=10000` is resolved below
(genuinely converges at `max_iter=15000`, with even better predictive
quality) -- not a lingering caveat.

**Speedups**: 33x, 38x, 18x. **Accuracy**: matches within 0.14
percentage points at every condition (0.01/0.14/0.02pp), never worse.
**Log-loss**: `cuml.accel`'s value is actually *lower* (better) than
sklearn's at all three conditions (by 0.021/0.008/0.005) -- a real,
consistent pattern, not noise in one direction only. This is a
materially different, more favorable outcome than the JAX/optax port's
cross-check, which found a real, one-directional divergence (sklearn
always better) that scaled with `C` and never closed.

**The one real caveat**: `evolved_T` at `C=1000` -- the condition with
the worst-conditioned feature matrix in this whole design -- hits
`cuml.accel`'s internal solver's `max_iter=10000` ceiling and reports
`converged=False` via the exact same `clf.n_iter_[0] < max_iter` check
`_fit_one` already uses. **This would trip this project's own locked
`NonConvergenceError` stop-gate immediately**, exactly as designed,
despite the resulting fit being predictively fine (arguably the best of
the three log-loss margins). `cuml.accel`'s underlying solver evidently
needs a different iteration budget than sklearn's own lbfgs to reach a
comparable-quality solution on this specific ill-conditioned problem --
not a correctness gap, but a real mismatch between this project's
`max_iter=10000` (calibrated specifically to sklearn's own convergence
behavior, per `stage2a_classifier.py`'s own documented history) and
`cuml.accel`'s different convergence footprint on the same data.

## Caveat resolved: `max_iter` sweep, same session's follow-up

Given how cheap each fit is (13.9s even at the `max_iter=10000` ceiling),
a direct sweep was the obvious next check rather than assuming a fix:
`evolved_T`/`C=1000`, `max_iter` in `{15000, 20000, 30000, 50000,
100000}`, stopping at the first value that reaches genuine convergence.

**Converged cleanly at `max_iter=15000`** (`n_iter=11621`, still only
18.7s -- a 24.5x speedup over sklearn's 458.3s, not meaningfully worse
than the earlier non-converged 13.9s run): **accuracy=0.8115,
log_loss=0.6762** -- now clearly *better* than sklearn's own converged
solution (accuracy +0.57pp, log-loss -0.031), not just comparable. No
sweep beyond 15000 was needed.

**This resolves the one open caveat cleanly, not just cheaply.** The
fix is a single, cheap, documented number (`max_iter=15000` for
`cuml.accel`, vs. sklearn's own `10000`) -- not a deeper unresolved
algorithmic issue like the JAX/optax port's persistent large-`C`
divergence. All three real selected-`C` conditions now converge
genuinely under `cuml.accel`, with predictive quality matching or
exceeding sklearn's at every one, and speedups of 18-38x intact.

## What this does and does not establish

**Does establish**: `cuml.accel`, on our own unmodified code, gives
real, large speedups (18-38x, 25x for `evolved_T` once genuinely
converged) with predictive quality that matches or clearly beats
sklearn's own converged solution, at all three real selected `C` values
checked, once each condition uses a `max_iter` calibrated to
`cuml.accel`'s own convergence behavior rather than assumed to inherit
sklearn's. A materially more favorable result than the from-scratch
JAX/optax port's, which showed a real, unresolved, `C`-scaling
divergence in sklearn's favor that did not close with more iterations.

**Does not establish**: that `cuml.accel` is a drop-in replacement
*using sklearn's own `max_iter=10000` unchanged*. `cuml.accel`'s solver
needs a different iteration budget than sklearn's own lbfgs on this
specific ill-conditioned problem -- a real, measured mismatch, resolved
here by measuring and using the right number for this backend
(`max_iter=15000` for `evolved_T`'s `C=1000`), not by loosening the
stop-gate or trusting an unconverged fit. Only `evolved_T`'s worst-`C`
case was checked against a higher `max_iter`; `evolved_rewired` and
`evolved_curr_random` already converged cleanly at `10000` and were not
swept further.

Not used for, and should not be used for, any reported Stage 2A result
as-is -- a real adoption would still need this `max_iter` finding
written into a `cuml.accel`-specific `CLASSIFIER_KWARGS`-equivalent
(disclosed the same way `stage2a_classifier.py`'s own `1000->10000`
amendment was), and a decision about whether the confirmatory result
should ever be re-run under a different backend at all -- not something
to decide implicitly by swapping the solver in.

## On sweeping `max_iter` for the other two conditions

Not re-run as a separate sweep. Unlike `evolved_T` (which hit the
`max_iter=10000` ceiling, leaving its true iteration requirement
unknown until raised), `evolved_rewired` (`n_iter=3514`) and
`evolved_curr_random` (`n_iter=1020`) already converged *below* the
ceiling in the first cross-check -- with a deterministic solver on
fixed data, that `n_iter` already *is* the exact margin (35% and 90%
of the `10000` budget respectively), not an estimate a further sweep
would refine.

## Full 6-condition CV-grid replication: independent cross-check of the entire locked confirmatory result

Not "does one fit match" but "does a completely different, GPU-native
implementation reach the same scientific conclusion" -- run at the
user's request as a cheap piece of extra confidence, not because the
locked sklearn result needed rechecking.

**Method**: this project's own real, unmodified `select_C_via_cv` and
`fit_final_at_selected_C` (no reimplementation), called under
`cuml.accel`, for **all six conditions**, full 9-value `C` grid,
5-fold CV, on the complete 60,000-image training set, evaluated once
against the official 10,000-image test set -- the entire procedure
`analyze_stage3_results.py` and `run_confirmatory_evaluation.py` ran
under sklearn, replicated end to end under a different backend. Data
(`stage3_encode_local.pkl`, `stage3_gpu_results.pkl`, and the test-set
equivalents) was pulled directly from the public GCS bucket
(`gs://bonsai-2026-stage2a-cache`) into the Colab session via plain
HTTPS -- ~1.2GB in under 2 minutes, sidestepping the slow local-upload
path entirely. `CLASSIFIER_KWARGS["max_iter"]` overridden to `20000`
for this run only (disclosed, not silent -- generously above
`evolved_T`'s known `cuml.accel` requirement of 11,621 iterations, to
reduce the chance of hitting the stop-gate elsewhere in the grid; the
gate itself was left untouched and would have stopped the run on any
real non-convergence).

**Result: 991.6s (16.5 min) for the entire 6-condition, 270-fit CV
grid** -- a **14.9x** speedup over sklearn's real, measured 14,808.8s
(246.8 min) for the identical procedure. Zero non-convergence anywhere
in the grid at `max_iter=20000`.

**`best_C` selection: 5 of 6 conditions matched sklearn's real locked
selection exactly:**

| condition | sklearn `best_C` | `cuml.accel` `best_C` |
|---|---:|---:|
| raw_pixels | 0.001 | 0.001 |
| encoded_pre_evolution | 0.01 | 0.01 |
| evolved_T | 1000 | 1000 |
| evolved_lattice | 1000 | **10000** |
| evolved_rewired | 10 | 10 |
| evolved_curr_random | 1 | 1 |

`evolved_lattice`'s selection differs -- but its mean validation
log-loss at `C=1000` (0.40207) and `C=10000` (0.40178) are separated by
0.00028, a near-tie at the flattest, weakest-regularization tail of the
grid (`evolved_lattice` was already the least-informative evolved
condition in every prior result). Not a real disagreement about which
regularization strength is right, just two backends landing on
opposite sides of a near-degenerate flat spot.

**Primary and all three secondary comparisons reach the identical
verdict as the locked confirmatory result, with closely matching
magnitudes:**

| comparison | `cuml.accel` mean d_i [95% CI] | sklearn (locked) mean d_i [95% CI] | verdict (both) |
|---|---|---|---|
| evolved_T vs. pre-evolution (**primary**) | -0.2761 [-0.2989, -0.2538] | -0.2491 [-0.2721, -0.2266] | **IMPROVEMENT** |
| evolved_lattice vs. pre-evolution | -0.1737 [-0.1915, -0.1560] | -0.1743 [-0.1930, -0.1557] | **IMPROVEMENT** |
| evolved_rewired vs. pre-evolution | -0.2894 [-0.3151, -0.2642] | -0.2819 [-0.3074, -0.2570] | **IMPROVEMENT** |
| evolved_curr_random vs. pre-evolution | -0.3094 [-0.3349, -0.2843] | -0.3049 [-0.3303, -0.2797] | **IMPROVEMENT** |

`evolved_lattice`'s test-set result is nearly identical between the two
backends (-0.1737 vs -0.1743) *despite* the different selected `C` --
direct confirmation that the near-tie above really is a near-tie in
practice, not a hidden divergence papered over by the CI.

**Total wall-clock for the entire independent replication (CV grid +
final refits + bootstrap): ~17 minutes**, against sklearn's 4.1 hours
for the CV grid alone.

## What this establishes, stated plainly

A completely independent, GPU-native reimplementation of the classifier
backend -- different solver, different hardware, different numerical
library end to end -- reproduces every one of the locked confirmatory
result's four verdicts (primary and all three secondary), at closely
matching effect sizes, using this project's own real, unmodified
selection and fitting code. This is exactly the kind of cheap,
independent confirmation the request was for: not a replacement for the
sklearn-based locked result (which remains what's reported, and remains
untouched by any of this), but real evidence that the scientific
conclusion is not an artifact of one specific numerical implementation.

## Next step, if pursued

A real adoption decision (not attempted here) would still need: the
`max_iter=20000` override written into a disclosed, `cuml.accel`-
specific configuration (not the committed `CLASSIFIER_KWARGS` itself,
which stays calibrated to sklearn); a decision on `evolved_lattice`'s
near-tie `C` selection if `cuml.accel` were ever used as the primary
fitting path rather than a cross-check; and an explicit choice about
whether any future re-run of this pipeline should use `cuml.accel` for
speed or keep sklearn for continuity with this project's own established
numerical history.
