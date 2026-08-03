# NVIDIA cuML `accel`: zero-code-change GPU acceleration check

**Status: promising, one real caveat, not yet adopted for anything
reported.** Prompted by the same question the JAX/optax classifier
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
| evolved_T | 1000 | 0.8058 / 0.7067 / 5123 / 458.3s | 0.8059 / 0.6860 / 10000 / 13.9s / **False** |
| evolved_rewired | 10 | 0.8183 / 0.6739 / 2054 / 190.9s | 0.8197 / 0.6659 / 3514 / 5.0s / True |
| evolved_curr_random | 1 | 0.8221 / 0.6509 / -- / 49.5s | 0.8223 / 0.6462 / 1020 / 2.7s / True |

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

## What this does and does not establish

**Does establish**: `cuml.accel`, on our own unmodified code, gives
real, large speedups (18-38x) with predictive quality that matches or
slightly beats sklearn's own converged solution, at all three real
selected `C` values checked -- a materially more favorable result than
the from-scratch JAX/optax port's, which showed a real, unresolved,
`C`-scaling divergence in sklearn's favor.

**Does not establish**: that `cuml.accel` is a drop-in replacement as
currently configured. The `evolved_T`/`C=1000` non-convergence flag
means naive adoption would halt at exactly the stop-gate this project
built for exactly this purpose -- appropriately, since the gate doesn't
know the fit is actually fine, only that `n_iter_` hit `max_iter`. A
real adoption would need either a `cuml.accel`-specific `max_iter`
(recalibrated the same way `stage2a_classifier.py`'s own `10000` was --
measured against `cuml.accel`'s actual convergence behavior, not
assumed to transfer from sklearn's), or a documented, deliberate
decision to trust `cuml.accel`'s output at this condition despite the
flag -- not something to do silently.

Not used for, and should not be used for, any reported Stage 2A result
as-is -- exactly the same standard already applied to the JAX/optax
port.

## Next step, if pursued

Recalibrate `max_iter` (or investigate a `cuml.accel`-specific
convergence criterion) specifically for `evolved_T`'s `C=1000` case,
re-verify `converged=True` is achievable without materially changing
the already-matching predictions, then re-run this same three-condition
check before considering `cuml.accel` for any full classifier-CV
re-run. Not started here -- this was a bounded feasibility check, not
an adoption decision.
