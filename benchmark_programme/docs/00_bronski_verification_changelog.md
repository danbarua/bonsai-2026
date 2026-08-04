# Bonsai — Bronski Verification Changelog (branch `2026-07`)

## Context

While reviewing `tests/test_hebian_kuramoto.py`, Dan suspected some tests made
incorrect assumptions about the baseline `HebbianKuramotoOperator` model's
behavior. Rather than assume the tests were wrong and the implementation
correct, we did the reverse: dug into the actual paper the model claims to
implement (Bronski, He, Li, Liu, Sponseller, Wolbert (2017), "The stability of
fixed points for a Kuramoto model with Hebbian interactions," Chaos 27,
053110, arXiv:1611.09941), checked the implementation against its exact
equations, and found a real, previously undiscovered bug in the model itself.

## 1. `maths/graphs.py` — `GraphLaplacian` fixed and documented

**Bug fixed:** `__post_init__` validated that all Laplacian matrices have
rows summing to zero — but this property only holds for the *unnormalized*
Laplacian (`L = D - A`). The symmetric-normalized Laplacian
(`D^-1/2 L D^-1/2`) genuinely does not have zero row sums for irregular
graphs (confirmed with a plain 3-node path graph: row sums come out to
`[0.293, -0.414, 0.293]`, a real mathematical property, not numerical noise).
Fixed by only applying the check when `is_normalized=False`.

**Documentation added:** `GraphLaplacian` was previously unused anywhere in
the real model code (only a demo script touched it) — it turns out to be
exactly the right structure for implementing Bronski et al.'s central
theorem. The paper proves that a Hebbian-Kuramoto fixed point's stability
reduces (via a Haynsworth/Schur-complement argument) to the sign structure of
a matrix that takes the form of a graph Laplacian, with signed edge weights
`kappa_ij = cos(2*(theta_i - theta_j)) / alpha`. Added extensive docstring
comments explaining this connection, plus:

- `GraphLaplacian.from_bronski_stability_matrix(phases, alpha)` — builds this
  stability matrix directly from a candidate fixed point's phases.
- `GraphLaplacian.unstable_dimension()` — count of negative eigenvalues.
- `GraphLaplacian.is_bronski_stable` — convenience boolean (no negative
  eigenvalues means the fixed point is stable per Theorem 2.3).

Sanity-checked against the trivial case (all oscillators synchronized):
one eigenvalue ~0 (rotational symmetry), rest positive → correctly predicted
stable.

## 2. Coupling sign bug — the real find

**The paper's coupling term** (equation 2): `dtheta_i/dt = omega_i + sum_j
gamma_ij * sin(theta_j - theta_i)` — note the order, "other minus self."
This is the standard *attractive* Kuramoto coupling for `gamma_ij > 0`.

**The code** (identical bug in `models/hebbian/minimalist.py` and
`HebbianKuramotoOperator`, and also present in `PredictiveHebbianOperator`'s
within-layer term, not yet fixed — see Open Items): computed `phase_diffs[i,j]
= theta_i - theta_j` ("self minus other"), then coupled via
`sin(phase_diffs)` — the *negative* of the paper's term (sin is odd). This
silently turned attractive coupling into **repulsive** coupling for positive
weights.

**Why this was invisible for so long:** at exact synchrony (identical
phases, zero relative frequency), `sin(0) = 0` regardless of sign — the bug
has zero effect at the degenerate fixed point every existing "stability"
test happened to use. It only manifests once real phase spread exists (e.g.
from frequency differences), which is exactly the condition Dan's suspected
test hit.

**Empirical confirmation, not just theoretical argument:** ran the exact
scenario from the previously-failing
`test_fixed_point_stability_with_small_frequency_differences` test with both sign conventions. Correct sign (paper):
coherence held at exactly 1.0 for the whole run. Bug's sign: coherence
collapsed to near-zero within ~10 steps. Also explains why *increasing*
coupling strength made the old bugged test's failure worse, not better —
stronger repulsion, not stronger attraction.

**Fixed in:** `models/hebbian/minimalist.py::update_hebbian_kuramoto`,
`models/hebbian/hebbian_kumaroto.py::HebbianKuramotoOperator.apply`. Both now
have comments explaining the fix and referencing the paper directly.

## 3. New test class: `tests/test_hebbian_kuramoto_bronski.py` (8 tests)

Verifies the model against the paper directly, rather than against ad hoc
coherence thresholds:

- `TestMinimalistHebbianKuramoto` (2 tests) — the `minimalist.py` reference
  implementation had **zero test coverage** before this session. Checks its
  phase/weight updates match the paper's equations exactly (finite-step
  comparison), and that a known stable fixed point is stationary.
- `TestHebbianKuramotoBronskiEquations` (2 tests) — same direct
  equation-matching check for the full `HebbianKuramotoOperator`.
- `TestHebbianKuramotoBronskiStability` (4 tests) — uses an
  analytically-derived pair of fixed points for two oscillators with
  slightly detuned frequencies (the classical 2-oscillator Kuramoto
  fixed-point pair, mapped through the paper's theta -> theta/2
  correspondence): one branch is stable, one is the classical saddle
  (unstable). Verifies `GraphLaplacian`'s prediction against **actual
  simulated recovery/divergence** from a small perturbation — not just
  theory checking theory. Also includes the trivial fully-synchronized
  sanity case.

**Confirmed these tests actually catch the bug**: temporarily reverted the
sign fix and re-ran — 4 of 8 tests failed, including the direct
equation-matching test firing immediately. Restored the fix afterward; all
8 pass.

## 4. Two more bugs found and fixed in `models/hebbian/minimalist.py` along the way

- **Type-hint typo:** `decay: np.float16 = 0.1` — should be `float`. The
  default value itself (`0.1`) is a plain Python float, not an actual
  `np.float16` instance, so `beartype` rejected any legitimate call. Fixed.
- **Missing diagonal zeroing:** unlike `HebbianKuramotoOperator` (which
  explicitly zeros the weight matrix diagonal), `update_hebbian_kuramoto`
  let self-coupling weights drift toward `mu/alpha` over time (since
  `cos(theta_i - theta_i) = cos(0) = 1`). Fixed by zeroing the diagonal
  after each weight update, matching the full operator's behavior.

## Test suite impact

Comparing against the last known full-suite baseline (25 failed, 140 passed,
1 skipped, 5 errors), after all fixes above: 27 failed, 146 passed, 1 skipped,
5 errors. Net changes:

- **Fixed:** `test_graph_laplacian_creation_normalized` (Laplacian bug).
- **Flipped to passing:** `test_occlusion_handling` in
  `test_predictive_hebbian_character.py` — this test compares
  `PredictiveHebbianOperator` against `HebbianKuramotoOperator` as a
  baseline; fixing the Hebbian baseline's dynamics changed which side of
  the comparison assertion holds. Not something we touched directly.
- **Two new failures**, both `TestHebbianKuramotoCharacterProcessing` tests
  (duplicated across `tests/test_character_processing.py` and
  `tests/learning/hebbian/test_character_processing.py`, which appear to be
  near-duplicate files — same pattern as the predictive `_backup.py` file
  removed earlier this project):
  - `test_frequency_vs_perturbation`: expects a uniform (same for every
    oscillator) frequency to significantly change absolute phase after 100
    steps. With correctly-attractive coupling, a uniform frequency shift
    shouldn't disrupt relative/synchronized structure at all — this is
    standard Kuramoto theory (the paper itself invokes exactly this: "by
    working in the co-rotating frame we can assume sum(omega_i) = 0"). The
    test's premise looks questionable independent of the bug fix.
  - `test_processing_stability`: a coefficient-of-variation check across 3
    random-seed runs, marginally over its own already-loosened threshold
    (0.569 vs 0.5). Same "old threshold tuned against old behavior" category
    as several tests fixed earlier this project.

Neither looks like a new bug — both are in the same "test needs recalibrating
against corrected behavior" bucket as the earlier `PredictiveHebbianOperator`
perturbations fix from earlier in this project.

## Open items / next steps (per agreed plan: foundation first, then up)

1. **Revisit the existing `tests/test_hebian_kuramoto.py` test suite** now
   that the model's actual correct behavior is understood — likely deleting
   some tests and rewriting others (e.g. the two newly-failing tests above,
   and the two pre-existing failures:
   `test_fixed_point_stability_with_small_frequency_differences`, which
   still fails because it *also* has its own separate bug — a "theoretical
   fixed point" weight formula missing the `mu` factor — and
   `test_perturbation_response`, a boundary-exact assertion artifact).
2. **`PredictiveHebbianOperator`'s within-layer Hebbian-Kuramoto term has the
   identical sign bug** (same `phase_diffs[i,j] = theta_i - theta_j`
   pattern, same file/line structure) — not yet fixed. This may retroactively
   change the interpretation of the limit-cycle behavior characterized
   earlier in this project (period ~123 iterations, sustained oscillation):
   that could be genuine emergent tension between predictive-coding
   correction and within-layer coupling, or partly an artifact of the same
   repulsion bug. Needs the same fix-then-reverify treatment once the
   Hebbian model's test suite is settled.
3. Also worth checking whether `models/akorn/` inherits the same coupling
   pattern, given how easy this bug was to miss.
4. `test_akorne_deluxe.py` and stub `NotImplementedError` failures remain
   unrelated pre-existing issues, unaffected by this work.
