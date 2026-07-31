# Diffusion Experiment, Stage 0: Candidate Design After First Cross-Construction Calibration

## Corrections applied per review, in order

**1. Cross-construction linearity check, not T alone.** The earlier
window (steps 1-50) was checked only on T. Repeating the identical check
-- same paired initial condition, same perturbed node, same impulse --
across T, degree-preserving rewiring, matched-sparsity random, and the
regular lattice (all under equal coupling normalization, below) gives a
materially different answer.

**2. Coupling normalization applied before any comparison.** Mean
weighted degree varied substantially before correction (T: 3.881,
rewired: 3.881 -- identical by construction, matched-sparsity random:
2.038, lattice: 3.703). All four were rescaled to a common budget
(C = T's own mean weighted degree, 3.881) by A_tilde = A * (C / mean
weighted degree). Rewiring's exact match to T pre-normalization confirms
it preserves weight assignment as closely as possible, as required.

**3. Multistability table language corrected.** Five sampled
initializations cannot establish absence of multistability -- "none" is
too strong a label.

| Construction | Observed across five sampled initializations |
|---|---|
| T | Five distinct stable solutions recovered |
| Degree-preserving rewiring | One deduplicated solution recovered |
| Matched-sparsity random | Five distinct stable solutions recovered |
| Lattice | One deduplicated solution recovered |

## The cross-construction linearity result

| Step | T ratio | Rewired ratio | Random ratio | Lattice ratio | All within 5%? |
|---|---|---|---|---|---|
| 1-8 | 1.002-1.004 | 0.998-1.000 | 1.0000 (exact) | 0.953-0.999 | Yes |
| 9 | 1.0021 | 0.9987 | 1.0000 | **0.9421** | **No** |
| 10-15 | ~1.00 | ~0.99-1.00 | 1.0000 | 0.86-0.93 | No |

**The lattice construction leaves the local-response regime much earlier
than the other three** -- T, rewired, and random all stay within 1% of
exact quadratic scaling through step 15 and beyond, while lattice's
scaling ratio degrades steadily from step 9 onward, reaching 0.86 by step
15. Per the prespecified rule (use the earliest common cutoff, not a
construction-specific window), **the locked common window is steps 1
through 8** -- shorter than the originally proposed 1-50, and this is
reported as the honest result of applying the rule, not adjusted to
recover a longer window.

Random's exact 1.0000 ratio at every checkpoint is notable and not fully
explained here -- plausibly a property of this specific random draw's
structure, but this has not been independently investigated and should
not be read as a general property of matched-sparsity constructions
without further checking.

## Calibration panel: testing the epsilon grid properly, not on one node

Per the review's correction, the earlier single-node check (class 0, node
0, epsilon=0.2) was too fragile to define the global window. A proper
calibration panel was built: 3 prespecified classes (0, 3, 7 -- chosen for
index spread, not results), 3 nodes per class stratified by weighted
degree in T (approximately 10th/50th/90th percentile), one fixed initial
phase vector per class, identical node and state pairing across all four
constructions -- 36 calibration cases per epsilon tested.

**The prespecified descending grid {0.2, 0.1, 0.05, 0.025} was tested
against the strict rule (every calibration case, every construction, must
pass the 5% quadratic-scaling tolerance through the target window of step
50):**

| Epsilon | Passes through step 50? | First failure |
|---|---|---|
| 0.2 | No | Step 3 (class 3, low-degree node, lattice) |
| 0.1 | No | Step 5 (class 3, high-degree node, rewired) |
| 0.05 | No | Step 10 (class 7, low-degree node, lattice) |
| 0.025 | No | Step 14 (class 0, high-degree node, random) |

**None of the four prespecified epsilon values reach the full 50-step
target window under the strict rule.** This is itself the honest,
decision-relevant Stage 0 finding: the calibration panel, precisely
because it is less fragile than a single-node check, reveals that the
four graph families have genuinely different finite-time linearization
scales, not just that one particular node/epsilon combination happened to
be unlucky.

**Numerical signal check at epsilon=0.025** (the smallest tested, per the
review's required check before accepting any impulse): D(0) values across
all 36 calibration cases range 1.10x10^-6 to 1.24x10^-6 -- comfortably
above float64 machine precision (~2.2x10^-16) by roughly ten orders of
magnitude. Numerical noise is not the limiting factor here.

**The longest common validated window, at the smallest prespecified
epsilon (0.025), per the review's explicit fallback rule**: steps 1
through 13 (max deviation across all 36 cases stays under 5% through step
13; step 14 is the first failure at 5.35% deviation, and deviation grows
steadily worse -- 7.2% at step 15, 16.3% by step 20).

## Candidate design, after proper calibration

- Impulse magnitude: **epsilon = 0.025** -- not 0.2. This is the smallest
  value in the prespecified descending grid, selected because none of the
  four tested values reached the full 50-step target window, per the
  review's explicit fallback: select the longest common validated window
  using the smallest impulse, provided numerical signal remains adequate
  (confirmed above).
- Primary response: gauge-corrected squared separation between paired
  baseline and perturbed trajectories, launched from the identical
  initial condition
- Normalization: S(t) = D(t) / D(0)
- Primary endpoint: normalized early-response AUC, sum of S(t) for
  t = 1 to 13 (steps; dt = 0.05, so t = 0.05 to 0.65 in time units) --
  the longest window validated by the full 36-case calibration panel,
  not the 8-step estimate from the earlier single-node check, and still
  short of the originally-hoped-for 50-step target
- Graph normalization: equal mean weighted degree (C = 3.881, T's own
  value, applied identically to all four constructions) -- this controls
  the global coupling budget, not edge-level coupling strength; the
  lattice's fewer edges under equal total budget necessarily means
  stronger per-edge coupling, which is a documented property of this
  normalization choice, not a flaw to correct now
- No-coupling condition: simulator sanity check only, not part of the
  inferential comparison family
- Late-time behavior beyond step 13 (including any basin-switching):
  descriptive secondary observation, not promoted to a primary endpoint
- Calibration panel cases (classes 0, 3, 7; their specific stratified
  nodes and fixed initial states) will NOT be reused in the Stage 1
  inferential sample, per the review's explicit instruction -- Stage 1
  will draw new classes, nodes, and/or initial conditions

## Honest assessment before proceeding

A 13-step window, reached only at the smallest prespecified impulse, is
short -- meaningfully longer than the single-node estimate of 8 steps at
epsilon=0.2, but still far from the originally-proposed 50-step target.
This is the result of properly calibrating against 36 cases rather than
one, not a limitation to correct by loosening tolerance or extending the
epsilon grid after seeing this outcome. The reviewer's own anticipated
third branch has materialized: the four graph families have measurably
different finite-time linearization scales even at the smallest tested
impulse, which raises a legitimate question about whether a common
linear-response comparison across all four is the right scientific
object, or whether it should proceed at this shorter, honestly-earned
window with that limitation stated plainly. Extending the epsilon grid
below 0.025 was not part of the original prespecification and would need
its own justification, not a quiet extension chosen because it might give
a longer window.

## What has not yet been done

- The full paired comparison (T vs. each control) at the locked window,
  across multiple paired node locations and multiple class topologies --
  this document stops before final design lock, per the explicit discipline of
  validating before running the inferential comparison.
- Any claim about what the AUC differences would show, since they have
  not yet been computed under the corrected design.

## Reproducing these results

`graph_oscillator_field.py` extended with `gauge_corrected_distance` and
`paired_trajectory_response`. All four constructions built from KMNIST
class 0's topology and its active-node set, normalized to equal mean
weighted degree, saved to `kmnist_c0_controls_normalized.npz`. Linearity
check results in `kmnist_c0_linearity_check.npz`.

## Independent verification (post-consolidation)

This stage's original driver code was consolidated into
`src/bonsai/dynamics/graph_oscillator_field.py` (see `NOTE.md`), confirmed
via diff at the time to not have lost any code -- but that check never
confirmed the consolidated code still produces this stage's own
quantitative claims. It has now been verified directly, using only
`bonsai.dynamics.graph_oscillator_field`'s importable functions and KMNIST
class 0's T topology (the same construction described above; no
stage-0-specific topology cache exists in the current checkout, so this
reused `class0_constructions.pkl`'s `T` matrix, generated for Stage
1B.2). Verification code: `tests/test_stage0_simulator_calibration.py`.

**Multistability, reproduced.** Five seeds (0-4 -- chosen for no reason
beyond being the first five non-negative integers; the original run's
seeds are not recorded anywhere in this document or in
`docs/PROJECT_MEMORY.md`) via `find_equilibrium_lbfgs`, deduplicated with
the same residual-after-rotational-alignment rule used throughout this
project (`< 0.05`, the threshold `stage1b_taxonomy.py`'s
`same_attractor()` cites as matching this stage's rule): **5 of 5
distinct**. This reproduces the finding (5 distinct equilibria from 5
initializations of T) but not necessarily the same 5 equilibria as the
original run, since the original seeds are unrecorded -- the same class
of caveat already noted for Stage 1B's topology-cache substitution.

**Stability confirmed, and the spectral gap now has an actual number.**
Every one of the 5 recovered equilibria was confirmed via
`GraphOscillatorField.jacobian_at`: exactly one near-zero eigenvalue (the
global rotation mode), zero negative eigenvalues -- genuine stable
equilibria, not artifacts, matching the qualitative description already
given in `docs/PROJECT_MEMORY.md`. The spectral gap itself (smallest
positive eigenvalue -- distance from the zero mode) had never been given
a number anywhere in this project before this check: **5.3x10^-3 to
5.8x10^-3 across the five equilibria**, against a largest eigenvalue of
~13.27 -- genuinely small in relative terms (roughly three orders of
magnitude below the spectrum's upper end), replacing the previously
qualitative-only "small spectral gap" language (e.g.
`find_equilibrium_lbfgs`'s own docstring, and the
`FORCE_CONVERGED_THRESHOLD` justification in `stage1b_taxonomy.py`) with
a measured range.

**RK45 vs. DOP853, reproduced -- and a gap closed in how it was checked.**
`joint_tangent_matrix_response`'s `method=` parameter has carried a
docstring since it was written stating it is "exposed so an independent
solver family (e.g. DOP853) can be substituted for cross-validation
without duplicating this function" -- but a repository-wide search
confirms `method='DOP853'` was never actually passed anywhere in this
codebase before this verification. The claim itself was true and is now
genuinely exercised, not just asserted: running the identical
tangent-matrix response (KMNIST class-0 T, seed=2000 initial condition,
node 0, t in [0, 2.5]) under both `method='RK45'` and `method='DOP853'`
(rtol=1e-8, atol=1e-10) gives a maximum absolute difference in S(t) of
1.4x10^-9 -- agreement to roughly 9 decimal places, comfortably exceeding
the originally claimed 4.
