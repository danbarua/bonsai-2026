Simplified Technical English version of `experiments/stage0_simulator_calibration/FINDINGS.md`.

# Stage 0 Diffusion Experiment: Candidate Design After the First Calibration Across Graph Constructions

## Corrections From the Review, In Order

**1. Check linearity across four graph constructions, not just T.**
The team first checked the early time window (steps 1 to 50) only on T. T is the learned topology, the graph built from real image data.
The team repeated the same check on four graph constructions: T, degree-preserving rewiring, matched-sparsity random, and the regular lattice.

- Degree-preserving rewiring is a version of T with the same connection counts per node, but the connections are shuffled.
- Matched-sparsity random is a random graph with a similar number of connections to T.
- The regular lattice is a simple grid graph. Each point connects only to its close neighbors.

Each repeat used the same paired starting condition, the same perturbed node, and the same impulse. An impulse is the small kick used to start the response.
All four constructions used equal coupling normalization (see point 2, below).
This repeat gave a different result from the first check on T alone.

**2. The team normalized coupling strength before any comparison.**
Mean weighted degree is a number that shows the average connection strength per node.
Before correction, mean weighted degree varied a lot between constructions:

- T: 3.881
- Rewired: 3.881 (identical to T, because of how rewiring works)
- Matched-sparsity random: 2.038
- Lattice: 3.703

The team rescaled all four constructions to a common budget. The common budget, C, was T's own mean weighted degree, 3.881.
The team used this formula: A_tilde = A * (C / mean weighted degree).
Rewiring matched T exactly before normalization. This confirms that rewiring keeps the same weight assignment as T, as it should.

**3. The team corrected the language in the multistability table.**
Multistability means a system can settle into more than one stable state, depending on where it starts.
Five sampled starting conditions cannot prove that multistability is absent. The word "none" was too strong. The team removed it.

| Construction | Result from five starting conditions |
|---|---|
| T | Five distinct stable solutions found |
| Degree-preserving rewiring | Only one distinct stable solution found, from five runs |
| Matched-sparsity random | Five distinct stable solutions found |
| Lattice | Only one distinct stable solution found, from five runs |

## The Result: Linearity Across Graph Constructions

Each ratio below compares the actual response size to the size expected under simple quadratic scaling. A ratio of 1.000 is a perfect match.

| Step | T ratio | Rewired ratio | Random ratio | Lattice ratio | All within 5%? |
|---|---|---|---|---|---|
| 1-8 | 1.002-1.004 | 0.998-1.000 | 1.0000 (exact) | 0.953-0.999 | Yes |
| 9 | 1.0021 | 0.9987 | 1.0000 | **0.9421** | **No** |
| 10-15 | ~1.00 | ~0.99-1.00 | 1.0000 | 0.86-0.93 | No |

The lattice construction leaves the simple quadratic-scaling pattern much earlier than the other three constructions.
T, rewired, and random all stay within 1% of exact quadratic scaling through step 15 and beyond.
Lattice's ratio gets steadily worse from step 9 onward. By step 15, its ratio is 0.86.

The team followed a rule set before the test: use the earliest cutoff that works for all constructions, not a cutoff chosen for just one construction.
Under this rule, **the locked common window is steps 1 through 8**.
This window is shorter than the originally proposed window of steps 1 through 50.
The team reports this shorter window as the honest result of the rule. The team did not adjust the rule to get a longer window.

The random construction's ratio was exactly 1.0000 at every checkpoint. This is notable, and this document does not fully explain it.
One possible reason is a property of this specific random draw's structure. The team has not checked this separately.
Readers should not treat this exact match as a general property of matched-sparsity random constructions. Further checking is needed.

## The Calibration Panel: Testing the Impulse Sizes Properly, Not on One Node

The review found a problem. The earlier check used only one node (class 0, node 0, epsilon = 0.2, where epsilon is the impulse size). This single check was too fragile to set the window for all cases.

The team built a proper calibration panel instead.

- The panel used 3 classes, chosen in advance: 0, 3, and 7. The team chose these classes for spread across the numbering, not for their results.
- The panel used 3 nodes per class. The team picked these nodes by weighted degree in T, at about the 10th, 50th, and 90th percentile.
- The panel used one fixed starting phase vector per class.
- The panel used the same node and state pairing across all four constructions.

This gave 36 calibration cases for each epsilon value tested.

The team tested a preset list of epsilon values, from largest to smallest: 0.2, 0.1, 0.05, and 0.025.
The team applied a strict rule: every calibration case, in every construction, must stay within 5% of exact quadratic scaling through step 50.

| Epsilon | Reaches step 50? | First failure |
|---|---|---|
| 0.2 | No | Step 3 (class 3, low-degree node, lattice) |
| 0.1 | No | Step 5 (class 3, high-degree node, rewired) |
| 0.05 | No | Step 10 (class 7, low-degree node, lattice) |
| 0.025 | No | Step 14 (class 0, high-degree node, random) |

**None of the four preset epsilon values reach the full 50-step target window under the strict rule.**
This result is itself an honest and important Stage 0 finding.
The calibration panel is less fragile than a single-node check.
Because of this, the panel shows that the four graph families have genuinely different linearization scales over time. A linearization scale is how long the simple linear approximation stays accurate.
This is not just bad luck from one node-and-epsilon combination.

**Numerical signal check at epsilon = 0.025.**
This is the smallest epsilon value tested. The review required this check before the team could accept any impulse size.
D(0) values (the squared separation measure at the moment of the kick) across all 36 calibration cases range from 1.10x10^-6 to 1.24x10^-6.
This range sits comfortably above float64 machine precision (about 2.2x10^-16). It is roughly ten orders of magnitude above that precision limit.
Numerical noise is not the limiting factor here.

**The longest common validated window, at epsilon = 0.025.**
The review set a fallback rule for this case. Under that rule, the longest common validated window is steps 1 through 13.
The maximum deviation across all 36 cases stays under 5% through step 13.
Step 14 is the first failure, at 5.35% deviation.
Deviation keeps growing after that: 7.2% at step 15, and 16.3% by step 20.

## Candidate Design, After Proper Calibration

- **Impulse size: epsilon = 0.025, not 0.2.**
  This is the smallest value in the preset list.
  The team picked it because none of the four tested values reached the full 50-step target window.
  The review's fallback rule applies: pick the longest common validated window, using the smallest impulse, as long as the numerical signal stays adequate. The check above confirmed the signal is adequate.
- **Primary response measure:** gauge-corrected squared separation between the paired baseline trajectory and the perturbed trajectory, both launched from the identical starting condition. Gauge-corrected means the measure is adjusted to remove irrelevant differences, such as an overall phase shift, that would otherwise inflate it.
- **Normalization:** S(t) = D(t) / D(0).
- **Primary endpoint:** normalized early-response AUC. AUC (area under the curve) is the sum of S(t) for t = 1 to 13 (steps; dt = 0.05, so t = 0.05 to 0.65 in time units).
  This is the longest window validated by the full 36-case calibration panel.
  It is not the 8-step estimate from the earlier single-node check.
  It is still short of the originally hoped-for 50-step target.
- **Graph normalization:** equal mean weighted degree (C = 3.881, T's own value), applied identically to all four constructions.
  This controls the total coupling budget, not the strength of each single connection.
  The lattice has fewer edges. Under an equal total budget, this means each of its edges must carry stronger coupling.
  This is a documented property of the normalization choice. It is not a flaw that needs a fix now.
- **No-coupling condition:** used only as a simulator sanity check. It is not part of the comparison family used for conclusions.
- **Late-time behavior beyond step 13** (including any basin-switching): a descriptive, secondary observation, not a primary endpoint. Basin-switching means the system moves from one stable state to a different one.
- **Calibration panel cases will not be reused in Stage 1.**
  The panel used classes 0, 3, and 7, with their specific stratified nodes and fixed starting states.
  The review gave this instruction directly: do not reuse these cases in Stage 1.
  Stage 1 will draw new classes, new nodes, new starting conditions, or some combination of these.

## Honest Assessment Before Moving Forward

The 13-step window is short. The team reached it only at the smallest preset impulse size.
It is meaningfully longer than the single-node estimate of 8 steps at epsilon = 0.2.
It is still far short of the originally proposed 50-step target.

This short window is the honest result of calibrating against 36 cases instead of just one.
The team will not fix this by loosening the tolerance or by extending the epsilon grid after seeing this outcome.

The reviewer had predicted this exact possibility in advance, as a third possible outcome. That outcome has now happened.
The four graph families have measurably different linearization scales over time, even at the smallest tested impulse.
This raises a real question. Is a single, shared linear-response comparison across all four constructions still the right scientific approach? Or should the team proceed with this shorter window and state the limitation plainly?

Extending the epsilon grid below 0.025 was not part of the original plan.
Any such extension would need its own justification. The team will not add it quietly just because it might give a longer window.

## What Has Not Yet Been Done

- The team has not yet run the full paired comparison of T against each control construction, at the locked window, across multiple paired node locations and multiple class topologies. This document stops before the final design lock. The project's discipline requires validating the design before running the comparison used for conclusions.
- The team makes no claim about what the AUC differences would show. The team has not yet computed them under the corrected design.

## How to Reproduce These Results

The team extended `graph_oscillator_field.py` with two new functions: `gauge_corrected_distance` and `paired_trajectory_response`.
The team built all four constructions from KMNIST class 0's topology and its active-node set.
The team normalized all four to equal mean weighted degree.
The team saved the result to `kmnist_c0_controls_normalized.npz`.
The linearity check results are in `kmnist_c0_linearity_check.npz`.

## Independent Verification (After Code Consolidation)

The team moved this stage's original driver code into `src/bonsai/dynamics/graph_oscillator_field.py`. See `NOTE.md` for details.
At the time, a file comparison (a diff) confirmed that this move did not lose any code.
That diff check confirmed completeness. It did not confirm that the moved code still produces this stage's own numeric claims.

The team has now verified this directly.
The verification used only the functions available to import from `bonsai.dynamics.graph_oscillator_field`.
The verification used KMNIST class 0's T topology, the same construction described above.
No stage-0-specific topology cache exists in the current checkout. So the verification reused the `T` matrix from `class0_constructions.pkl`, a file generated for Stage 1B.2.
Verification code: `tests/test_stage0_simulator_calibration.py`.

**Multistability, reproduced.**
The team used 5 seeds (0 through 4) with `find_equilibrium_lbfgs`. An equilibrium is a stable state the system settles into and does not move away from.
The team chose seeds 0 through 4 for no special reason. These are simply the first five non-negative integers.
The original run's seeds are not recorded anywhere, not in this document and not in `docs/PROJECT_MEMORY.md`.
The team removed duplicate equilibria using the same rule used throughout this project: residual after rotational alignment, threshold < 0.05. This is the same threshold that `stage1b_taxonomy.py`'s `same_attractor()` function cites as matching this stage's rule.
Result: 5 of 5 equilibria were distinct.
This reproduces the original finding: 5 distinct equilibria from 5 starting conditions of T.
It does not necessarily reproduce the same 5 equilibria as the original run. The original seeds are unrecorded, so an exact match cannot be confirmed.
This is the same kind of caveat already noted for Stage 1B's topology-cache substitution.

**Stability confirmed, and the spectral gap now has a real number.**
The team checked all 5 recovered equilibria using `GraphOscillatorField.jacobian_at`. A Jacobian is a matrix that describes how small changes near a point grow or shrink over time.
Each equilibrium had exactly one near-zero eigenvalue. An eigenvalue here is a number that shows whether a small change grows, shrinks, or stays the same over time. The near-zero eigenvalue reflects the global rotation mode, a direction of change that does not affect stability.
Each equilibrium had zero negative eigenvalues.
This confirms genuine stable equilibria, not artifacts. It matches the qualitative description already given in `docs/PROJECT_MEMORY.md`.

The spectral gap is the distance between the smallest positive eigenvalue and zero.
No one had given the spectral gap an actual number anywhere in this project before this check.
The measured range is 5.3x10^-3 to 5.8x10^-3, across the five equilibria.
The largest eigenvalue is about 13.27.
So the spectral gap is genuinely small in relative terms, roughly three orders of magnitude below the top of the spectrum.
This measured range replaces earlier language that only called the gap "small" without a number. That earlier language appeared in `find_equilibrium_lbfgs`'s own docstring, and in the `FORCE_CONVERGED_THRESHOLD` justification in `stage1b_taxonomy.py`.

**RK45 vs. DOP853, reproduced, and a gap closed in how it was checked.**
`joint_tangent_matrix_response` has a `method=` parameter. RK45 and DOP853 are two different numerical solver methods for stepping the simulation forward in time.
The docstring for this parameter has always said the parameter exists so a second solver, such as DOP853, can cross-check the first, without duplicating the function.
A search across the whole codebase confirms that `method='DOP853'` was never actually used anywhere before this verification.
The claim itself was true. It is now genuinely tested, not just stated.
The team ran the identical tangent-matrix response under both methods: KMNIST class-0 T, seed = 2000 starting condition, node 0, t in [0, 2.5].
The team ran `method='RK45'` and `method='DOP853'`, with rtol=1e-8 and atol=1e-10.
The maximum absolute difference in S(t) between the two methods was 1.4x10^-9.
This is agreement to roughly 9 decimal places. It comfortably exceeds the originally claimed agreement of 4 decimal places.
