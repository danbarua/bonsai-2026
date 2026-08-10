Simplified Technical English version of `experiments/stage2a_dynamics_classification/CUML_ACCEL_FINDINGS.md`

# NVIDIA cuML `accel`: a zero-code-change GPU speedup check

**Status: the one caveat found is resolved. A full 6-condition, 270-fit
cross-validation-grid replication, run under a different numerical
backend, reproduces every verdict of the locked confirmatory result, and
runs 14.9 times faster. This is a cross-backend implementation
robustness check, not independent scientific corroboration — see "What
this does and does not establish," below, for why that distinction
matters. It is still not adopted for any reported result — no reported
result has needed it, and this project's own scikit-learn-based numbers
remain what is reported.** This check was prompted by the same question
the JAX/optax classifier port was built to answer (can the slow,
scikit-learn-based classifier cross-validation fitting be sped up?), but
tested here through a different route: NVIDIA RAPIDS' `cuml.accel`.
`cuml.accel` monkey-patches (replaces at runtime)
`sklearn.linear_model.LogisticRegression`, among other models, so it
runs on a GPU-native implementation while keeping the exact same class
name and `fit`/`predict_proba` interface. Unlike the JAX port, this uses
**our actual, unmodified `stage2a_classifier._fit_one`** function — no
reimplementation, and no new code module to maintain or independently
verify from scratch.

## Setup

A fresh A100 GPU session (`cuml-crosscheck`, via `mighty-colab`). The
`cuml-cu12` package was installed via `mighty-colab reinstall` (a single
combined install-and-restart step, tool version v0.1.21), using
`--extra-index-url=https://pypi.nvidia.com` in a requirements file — it
installed cleanly on the first attempt, giving `cuml==26.02.000`.
`cuml.accel.install()` was called before importing `stage2a_classifier`
(and therefore before `sklearn` itself is imported), following
`cuml.accel`'s own required activation order. This used the same
already-cached, already-standardized full 60,000-image training features
and official 10,000-image test features used for the JAX-port
cross-check (the same `StandardScaler`, fit on the full training set,
byte-for-byte comparable to what the real confirmatory run used).

This was tested at exactly the three real, selected `C` values from the
locked confirmatory result — `evolved_T=1000`, `evolved_rewired=10`,
`evolved_curr_random=1` — not a grid extreme, and not a subsample.

## Result: real speedups, closely matching (often marginally better) prediction quality, with one real caveat

| condition | `C` | sklearn acc / log-loss / n_iter / time | `cuml.accel` acc / log-loss / n_iter / time / converged |
|---|---:|---|---|
| evolved_T | 1000 | 0.8058 / 0.7067 / 5123 / 458.3s | 0.8059 / 0.6860 / 10000 / 13.9s / **False*** |
| evolved_rewired | 10 | 0.8183 / 0.6739 / 2054 / 190.9s | 0.8197 / 0.6659 / 3514 / 5.0s / True |
| evolved_curr_random | 1 | 0.8221 / 0.6509 / -- / 49.5s | 0.8223 / 0.6462 / 1020 / 2.7s / True |

*`evolved_T`'s `converged=False` at `max_iter=10000` is resolved below
(it genuinely converges at `max_iter=15000`, with even better prediction
quality) — this is not a lingering caveat.

**Speedups**: 33x, 38x, and 18x. **Accuracy**: matches within 0.14
percentage points at every condition (0.01/0.14/0.02 percentage points),
and is never worse. **Log-loss**: `cuml.accel`'s value is actually
*lower* (better) than scikit-learn's at all three conditions (by
0.021/0.008/0.005) — this is a real, consistent pattern, not noise in
one direction only. This is a materially different, more favorable
outcome than the JAX/optax port's cross-check, which found a real,
one-directional gap (scikit-learn always better) that grew with `C` and
never closed.

**The one real caveat**: `evolved_T` at `C=1000` — the condition with
the worst-conditioned feature matrix in this whole design — reaches
`cuml.accel`'s internal solver's `max_iter=10000` limit and reports
`converged=False`. This is measured using the exact same
`clf.n_iter_[0] < max_iter` check `_fit_one` already uses. **This would
trip this project's own locked `NonConvergenceError` stop rule
immediately**, exactly as designed, even though the resulting fit turns
out to be fine for prediction (arguably the best log-loss margin of the
three). `cuml.accel`'s underlying solver evidently needs a different
iteration budget than scikit-learn's own lbfgs to reach a
comparable-quality solution on this specific ill-conditioned problem.
This is not a correctness gap. It is a real mismatch between this
project's `max_iter=10000` (calibrated specifically to scikit-learn's
own convergence behavior, per `stage2a_classifier.py`'s own documented
history) and `cuml.accel`'s different convergence behavior on the same
data.

## Caveat resolved: a `max_iter` sweep, in the same session's follow-up

Given how cheap each fit is (13.9s, even at the `max_iter=10000` limit),
running a direct sweep was the obvious next check, rather than assuming
a fix would work: `evolved_T` at `C=1000`, with `max_iter` set to
`{15000, 20000, 30000, 50000, 100000}` in turn, stopping at the first
value that reaches real convergence.

**Converged cleanly at `max_iter=15000`** (`n_iter=11621`, still only
18.7s — a 24.5 times speedup over scikit-learn's 458.3s, and not
meaningfully worse than the earlier non-converged 13.9s run):
**accuracy=0.8115, log_loss=0.6762** — now clearly *better* than
scikit-learn's own converged solution (accuracy up 0.57 percentage
points, log-loss down 0.031), not just comparable. No sweep beyond 15000
was needed.

**This resolves the one open caveat cleanly, not just cheaply.** The fix
is a single, cheap, documented number (`max_iter=15000` for
`cuml.accel`, versus scikit-learn's own `10000`) — not a deeper,
unresolved algorithmic issue like the JAX/optax port's persistent gap at
large `C`. All three real, selected-`C` conditions now converge properly
under `cuml.accel`, with prediction quality matching or exceeding
scikit-learn's at every one, and speedups of 18-38x intact.

## What this does and does not establish

**Does establish**: `cuml.accel`, on our own unmodified code, gives
real, large speedups (18-38 times, or 25 times for `evolved_T` once
genuinely converged) with prediction quality that matches or clearly
beats scikit-learn's own converged solution, at all three real selected
`C` values checked — once each condition uses a `max_iter` calibrated
to `cuml.accel`'s own convergence behavior, rather than assumed to
inherit scikit-learn's. This is a materially more favorable result than
the from-scratch JAX/optax port's, which showed a real, unresolved gap
that grew with `C`, favoring scikit-learn, and did not close with more
iterations.

**Does not establish**: that `cuml.accel` is a drop-in replacement
*using scikit-learn's own `max_iter=10000` unchanged*. `cuml.accel`'s
solver needs a different iteration budget than scikit-learn's own lbfgs
on this specific ill-conditioned problem — a real, measured mismatch,
resolved here by measuring and using the right number for this backend
(`max_iter=15000` for `evolved_T`'s `C=1000`), not by loosening the stop
rule or trusting an unconverged fit. Only `evolved_T`'s worst-`C` case
was checked against a higher `max_iter`. `evolved_rewired` and
`evolved_curr_random` already converged cleanly at `10000` and were not
swept further.

This is not used for, and should not be used for, any reported Stage 2A
result as it stands. A real adoption would still need this `max_iter`
finding written into a `cuml.accel`-specific version of
`CLASSIFIER_KWARGS`, disclosed the same way `stage2a_classifier.py`'s
own `1000 -> 10000` change was disclosed, and a decision about whether
the confirmatory result should ever be re-run under a different backend
at all. This should not be decided implicitly, just by swapping the
solver in.

## On sweeping `max_iter` for the other two conditions

This was not re-run as a separate sweep. Unlike `evolved_T` (which hit
the `max_iter=10000` limit, leaving its true iteration requirement
unknown until raised), `evolved_rewired` (`n_iter=3514`) and
`evolved_curr_random` (`n_iter=1020`) already converged *below* the
limit in the first cross-check. With a deterministic solver on fixed
data, that `n_iter` value already *is* the exact margin (35% and 90% of
the `10000` budget respectively), not an estimate that a further sweep
would refine.

## Full 6-condition cross-validation-grid replication: a cross-backend robustness check on the entire locked confirmatory result

This checks not "does one fit match," but "does a completely different,
GPU-native implementation reach the same scientific conclusion." This
was run at the user's request, as a cheap piece of extra confidence, not
because the locked scikit-learn result needed rechecking. This tests
implementation robustness, not independent scientific replication — see
"What this does and does not establish," below.

**Method**: this project's own real, unmodified `select_C_via_cv` and
`fit_final_at_selected_C` functions (no reimplementation), called under
`cuml.accel`, for **all six conditions**, using the full 9-value `C`
grid, 5-fold cross-validation, on the complete 60,000-image training
set, evaluated once against the official 10,000-image test set — the
entire procedure `analyze_stage3_results.py` and
`run_confirmatory_evaluation.py` ran under scikit-learn, replicated end
to end under a different backend. Data
(`stage3_encode_local.pkl`, `stage3_gpu_results.pkl`, and the matching
test-set files) was pulled directly from the public GCS bucket
(`gs://bonsai-2026-stage2a-cache`) into the Colab session over plain
HTTPS — about 1.2GB in under 2 minutes, avoiding the slow local-upload
path entirely. `CLASSIFIER_KWARGS["max_iter"]` was set to `20000` for
this run only (disclosed, not hidden — well above `evolved_T`'s known
`cuml.accel` requirement of 11,621 iterations, to reduce the chance of
hitting the stop rule elsewhere in the grid; the stop rule itself was
left unchanged, and would have stopped the run on any real
non-convergence).

**Result: 991.6s (16.5 min) for the entire 6-condition, 270-fit
cross-validation grid** — a **14.9 times** speedup over scikit-learn's
real, measured 14,808.8s (246.8 min) for the identical procedure. There
was zero non-convergence anywhere in the grid at `max_iter=20000`.

**`best_C` selection: 5 of the 6 conditions matched scikit-learn's real,
locked selection exactly:**

| condition | sklearn `best_C` | `cuml.accel` `best_C` |
|---|---:|---:|
| raw_pixels | 0.001 | 0.001 |
| encoded_pre_evolution | 0.01 | 0.01 |
| evolved_T | 1000 | 1000 |
| evolved_lattice | 1000 | **10000** |
| evolved_rewired | 10 | 10 |
| evolved_curr_random | 1 | 1 |

`evolved_lattice`'s selection differs — but its mean validation log-loss
at `C=1000` (0.40207) and `C=10000` (0.40178) are separated by only
0.00028, an almost exact tie at the flattest, weakest-regularization end
of the grid (`evolved_lattice` was already the least informative
evolved condition in every earlier result). This is not a real
disagreement about which regularization strength is right — just two
backends landing on opposite sides of an almost-flat, near-tied region.

**The primary comparison and all three secondary comparisons reach the
identical verdict as the locked confirmatory result, with closely
matching effect sizes:**

| comparison | `cuml.accel` mean d_i [95% CI] | sklearn (locked) mean d_i [95% CI] | verdict (both) |
|---|---|---|---|
| evolved_T vs. pre-evolution (**primary**) | -0.2761 [-0.2989, -0.2538] | -0.2491 [-0.2721, -0.2266] | **IMPROVEMENT** |
| evolved_lattice vs. pre-evolution | -0.1737 [-0.1915, -0.1560] | -0.1743 [-0.1930, -0.1557] | **IMPROVEMENT** |
| evolved_rewired vs. pre-evolution | -0.2894 [-0.3151, -0.2642] | -0.2819 [-0.3074, -0.2570] | **IMPROVEMENT** |
| evolved_curr_random vs. pre-evolution | -0.3094 [-0.3349, -0.2843] | -0.3049 [-0.3303, -0.2797] | **IMPROVEMENT** |

`evolved_lattice`'s test-set result is nearly identical between the two
backends (-0.1737 versus -0.1743), *despite* the different selected `C`
— this directly confirms the near-tie described above really is a
near-tie in practice, and not a hidden gap that the confidence interval
happens to paper over.

**Total wall-clock time for the entire cross-backend robustness check
(cross-validation grid, plus final refits, plus bootstrap): about 17
minutes**, against scikit-learn's 4.1 hours for the cross-validation
grid alone.

## What this establishes, stated plainly

**Amended by external review**: earlier drafts of this section described
this as "independent confirmation" or "independent scientific
corroboration." That overstated what was actually tested. This run
reuses the same training and test samples, the same oscillator features,
the same graph instances, the same folds and `C` grid, and the same
high-level selection and fitting code as the locked scikit-learn result
— only the numerical classifier backend (a different solver, different
hardware, different numerical library) differs. This makes it a
**cross-backend implementation robustness check**: it reproduces every
one of the locked confirmatory result's four verdicts (the primary and
all three secondary comparisons), at closely matching effect sizes,
using this project's own real, unmodified selection and fitting code,
and shows the positive verdict is not specific to scikit-learn's
particular optimizer. It is **not** independent scientific
corroboration — true independence would require, at minimum, varying
something the two runs currently share (a different draw of the data, a
different feature pipeline, or a genuinely separate implementation of
the selection and fitting logic, not just a different backend under the
same code). This is not a replacement for the scikit-learn-based locked
result, which remains what is reported and is untouched by any of this,
but it is real, correctly-scoped evidence that the scientific conclusion
is not an artifact of one specific numerical implementation.

## Code

There is no dedicated driver script for the cross-validation-grid and
confirmatory replication above — it reuses this project's own unmodified
`analyze_stage3_results.py` and `run_confirmatory_evaluation.py`, and
the whole point is that no reimplementation was needed. Reproducing
those numbers means running those same scripts on a `mighty-colab` A100
session: install `cuml-cu12` using `mighty-colab reinstall -s <session>
--requirement cuml_requirements.txt` (the committed requirements file),
then call `cuml.accel.install()` before `sklearn` (or anything that
imports it) is loaded — `cuml.accel` replaces sklearn's model classes at
import time. This is verified to speed up `LogisticRegression`, and
verified **not** to speed up `MLPClassifier` (a silent CPU fallback, with
no error) — see `COMPUTE_COST_FINDINGS.md`'s "Check 0" section for that
finding. Check any new model type directly before trusting it, rather
than assuming it is covered.

The class-0 support audit's GPU classifier fits (the
`raw_pixels_505restricted` and `encoded_784_unrestricted` numbers
reported in `FINDINGS.md`'s "class-0 support audit" section) are a
partial exception: that specific run *does* have a dedicated, committed,
actually-tested driver script, `class0_support_audit_classify_gpu.py` —
wired into `make stage2a-class0-classify-gpu`. It still reuses
`stage2a_classifier.py` and `stage2a_stats.py` unmodified (uploaded
alongside it, unchanged), but it is its own separate file, rather than
the local `run_class0_support_audit_classify.py --cuml` path, because it
downloads its six input `.npy` files directly from the public GCS
mirror (`gs://bonsai-2026-stage2a-cache/class0_support_audit/`), instead
of reading them through `stage2a_paths.scratch_root()` — the same
`/content`-based convention that `stage3_gpu_evolve.py` and
`stage4_gpu_evolve.py` already use for remote-only driver scripts. The
committed `run_class0_support_audit_classify.py --cuml` combination,
with `STAGE2A_SCRATCH_ROOT` pointed at `/content`, has never actually
been run remotely — do not assume it works untested.

## Next step, if pursued

A real decision to adopt this for the *whole confirmatory pipeline* has
not been made, and this section's three original open items are only
partly addressed by what has happened since:

- **The `max_iter` override, written into a disclosed,
  `cuml.accel`-specific configuration, not into the committed
  `CLASSIFIER_KWARGS` itself** — done, but only narrowly:
  `run_class0_support_audit_classify.py` and
  `class0_support_audit_classify_gpu.py` both use a local
  `CUML_MAX_ITER = 20000` constant, applied by overriding
  `CLASSIFIER_KWARGS["max_iter"]` at runtime, rather than editing the
  committed scikit-learn-calibrated default — exactly the
  disclosed-not-silent pattern this item asked for, but scoped only to
  the class-0-audit thread. No general-purpose `cuml.accel`-specific
  configuration exists for the cross-validation-grid and confirmatory
  replication itself.
- **`evolved_lattice`'s near-tie `C` selection** — still entirely open.
  No decision has been made or needed, since `cuml.accel` has not been
  adopted as a primary fitting path anywhere.
- **Whether any future re-run should use `cuml.accel` or keep
  scikit-learn** — already answered, for now: this document's own
  opening line states plainly that no reported result has adopted
  `cuml.accel`, and this project's scikit-learn-based numbers remain
  what is reported. That is a continuity decision already made, not a
  gap — but it is a "for now" answer, not a permanent one. Revisit it
  explicitly if a future stage makes the roughly 4-hour scikit-learn
  cross-validation cost a real blocker, rather than an annoyance.
