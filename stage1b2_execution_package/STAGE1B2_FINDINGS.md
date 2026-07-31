# Stage 1B.2 Findings: Structured Internal Transformation Established (Locally)

## Design, in brief

432 finite-response trials: 3 nodes (low/median/high weighted-degree in
T) x 2 signs x 3 amplitudes (0.025 tangent-consistent, 0.2 intermediate,
0.8 nonlinear) x 4 perturbation times along one baseline trajectory
(KMNIST class 0, T topology, seed=3000) x 6 fixed nearby-state replicas
per perturbation time (scale=0.1, verified local at every t_p). Primary
statistic: Delta_map = B - W, where W is mean output-space distance
(d_q = sqrt(JSD)) between same-input outputs across replicas, and B is
the balanced mean distance between different-input outputs, also across
replicas. One-sided Monte Carlo permutation test, 10,000 permutations,
independent per-replica label shuffling (the only null that actually
destroys input identity while preserving each replica's own output
geometry -- an earlier draft using a common cross-replica relabeling was
caught as degenerate before any inferential result was drawn from it).

## Primary result

| Response representation | Delta_map | p |
|---|---:|---:|
| Finite response | 0.3505 | 0.00010 |
| Stimulated node excluded | 0.3418 | 0.00010 |
| Tangent-only response | 0.3248 | 0.00010 |
| Nonlinear residual (finite minus tangent) | 0.3896 | 0.00010 |

All four hit the permutation floor (1/10,001, the minimum attainable
value at this permutation count).

## Factor-specific results, Holm-corrected

| Input factor | Delta | Raw p | Holm p |
|---|---:|---:|---:|
| Node | 0.8185 | 0.00010 | 0.00030 |
| Sign | 0.1058 | 0.00010 | 0.00030 |
| Amplitude | 0.1273 | 0.00010 | 0.00030 |

Node is clearly the dominant factor by effect size, but sign and
amplitude discrimination are independently established, not absorbed
into noise once node is accounted for. All three input dimensions are
associated with reproducible spatial outputs.

## What this establishes, precisely

**The mapping is not reducible to the directly stimulated coordinate.**
Excluding the perturbed node from the output representation entirely
barely moves the effect (0.3505 -> 0.3418), with identical
floor-level significance. If the result were merely "the poked node
still shows the biggest signal," removing that node should have gutted
it. It didn't.

**Source energy genuinely redistributes over the response window.**
Fraction of total displacement energy at the source node: ~99.8%
immediately after the impulse (near-definitional at tau=0) -> ~52% at
the event-aligned nonlinear-departure time -> median ~7.5% by the end of
the response horizon. The response moves substantially beyond the
initially perturbed node, not just decays in place.

**Both linear and nonlinear structure independently carry the mapping.**
The tangent-only response already implements a structured, significant
mapping (Delta=0.3248) -- first-order graph propagation is not
information-free. The nonlinear residual, z_eps(tau) = P*Delta_theta_eps(tau)
- eps*P*delta(tau), *also* carries a significant, independent mapping
(Delta=0.3896, p=0.00010). This means the nonlinear correction is not
unstructured error riding on top of an otherwise-informative linear
signal -- its own spatial organization is reproducibly associated with
input identity across nearby-state replicas.

**A necessary correction on comparing the four Delta_map values directly.**
0.3896 (residual) is numerically larger than 0.3505 (finite) and 0.3248
(tangent), but this does not mean the nonlinear correction is physically
"stronger" than the linear propagation it corrects. Each response space
(finite, tangent, residual) is separately normalized (each q-distribution
sums to 1 within its own space), so Delta_map in one space measures
within-space discrimination relative to that space's own replica
dispersion -- it is not commensurate with Delta_map in a different space,
and comparing raw magnitudes across spaces conflates "significant within
its own geometry" with "dominant in absolute physical contribution." The
correct, careful statement: both tangent-linear propagation and
finite-amplitude nonlinear corrections contribute structured,
input-sensitive spatial organization. Neither is described as stronger
than the other from this comparison alone.

## Capability hierarchy, updated

- **Level 1 (nonlinear behavior): established** -- since Stage 1B.
- **Level 2 (structured internal transformation): established, locally.**
  The three conditions previously reserved for this stronger claim are
  now all met: the mapping survives source-node exclusion; it appears
  independently in the nonlinear residual, not merely inherited from the
  tangent signal; and all three input factors (not just node) survive
  multiplicity correction.
- **Level 3 (useful computation): not established.** No external task or
  information-processing objective has been defined or tested.

## Scope, stated explicitly (the "locally" qualifier is essential)

This result is conditional on: one baseline trajectory (seed=3000); one
class (KMNIST class 0); four repeated states along that single
trajectory, not four independent trajectories; T only, no graph controls
(rewired/random/lattice) yet compared; no external task or
information-processing objective.

## Strongest defensible conclusion

Stage 1B.2 establishes a locally robust structured internal
transformation along the prespecified seed-3000 class-0 trajectory. The
transformation preserves and redistributes information about
perturbation location, sign, and amplitude across nearby dynamical
states. It survives removal of the directly stimulated node and is
present both in tangent propagation and in the specifically nonlinear
finite-minus-tangent residual. Generality across trajectories,
topology-specificity, and external usefulness remain untested.

## What remains open

The question is no longer whether a structured internal mapping exists
-- it does, under this trajectory-conditioned design. The next questions:

1. **Generalization across trajectories**: does this reproduce with
   independent baseline trajectories (different seeds), or is it
   specific to this one?
2. **Topology specificity**: does learned topology T produce this
   mapping more strongly, more efficiently, or differently structured
   than the matched controls (degree-preserving rewiring, matched-
   sparsity random, regular lattice) established throughout the earlier
   E/R/Stage-1A investigations? This has not yet been tested for Stage
   1B.2 specifically.
3. **External usefulness (Level 3)**: can this structured mapping be
   linked to an externally defined task or information-processing
   objective, rather than characterized only in terms of its own
   internal reproducibility?

## Reproducing these results

`stage1b2_core.py` (per-trial computation, extended to save q_tangent,
q_residual, q_excl_node, source-energy fraction alongside the original
q/r/J_tan diagnostics), `run_stage1b2.py` (432-trial driver, checkpointed,
parallelized), `analyze_stage1b2.py` (primary omnibus test, corrected
permutation scheme, parallelized), `analyze_stage1b2_diagnostics.py`
(the four diagnostic decompositions plus Holm-corrected factor-specific
tests, parallelized). Raw results in `stage1b2_results.pkl`; primary
analysis in `stage1b2_final_analysis.pkl`; diagnostic decomposition in
`stage1b2_diagnostics.pkl`.
