# Stage 1B2 Findings: Structured Internal Transformation Established (Locally)

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

| Response representation | Delta_map | p_MC |
|---|---:|---:|
| Finite response | 0.3505 | 1/10,001 ~ 0.00010 |
| Stimulated node excluded (common-support mask across all trials) | 0.3418 | 1/10,001 ~ 0.00010 |
| Tangent-only response | 0.3248 | 1/10,001 ~ 0.00010 |
| Nonlinear residual (finite minus tangent) | 0.3896 | 1/10,001 ~ 0.00010 |

All four sit at the Monte Carlo floor: zero of 10,000 permuted statistics
equalled or exceeded the observed statistic in each case, giving
p_MC = (0+1)/(10,000+1) ~ 0.00010. This is the attainable floor at this
permutation count, not necessarily the exact tail probability -- more
permutations would be needed to resolve p below this floor, though
nothing in this analysis requires that resolution.

## Factor-specific results, Holm-corrected

| Input factor | Delta | Raw p | Holm p |
|---|---:|---:|---:|
| Node | 0.8185 | 0.00010 | 0.00030 |
| Sign | 0.1058 | 0.00010 | 0.00030 |
| Amplitude | 0.1273 | 0.00010 | 0.00030 |

Node is clearly the dominant factor by effect size. Sign and amplitude
discrimination are separately significant under factor-restricted
permutation tests (which preserve the matched cells of the other two
factors while testing the selected one) -- not absorbed into noise once
node is accounted for, but also not established as probabilistically or
causally independent of node, only as separately significant under their
respective restricted tests. All three input dimensions are associated
with reproducible spatial outputs. (Node and amplitude have three levels
in this design while sign has two, and the factors differ intrinsically
in intervention geometry -- node's much larger Delta supports descriptive
dominance in this specific design, but should not be read as a universal
information-capacity ranking among the factors.)

## Residual validity and magnitude: is the structured residual mapping supported by numerically resolved nonlinear departures?

The permutation test on q_residual establishes that the residual's
*normalized* spatial geometry is reproducibly associated with input
identity. Because q_residual is normalized (q_res_j = z_j^2 / sum_k
z_k^2), a small residual can in principle still produce a sharply
organized normalized pattern -- so this does not by itself establish that
every residual is physically substantial. Checked directly:

**By amplitude:**

| Amplitude | Median \|\|z(tau*)\|\| | IQR | Median E(tau*) | n >= Q-thr | n >= E_min | Undefined | Total |
|---|---:|---|---:|---:|---:|---:|---:|
| 0.025 | 0.00002 | [0.00001, 0.00003] | 0.00204 | 144 | 144 | 0 | 144 |
| 0.2 | 0.00098 | [0.00061, 0.00166] | 0.01667 | 144 | 144 | 0 | 144 |
| 0.8 | 0.02092 | [0.01432, 0.03928] | 0.07355 | 144 | 144 | 0 | 144 |

**By perturbation time:**

| t_p | Median \|\|z(tau*)\|\| | IQR | Median E(tau*) | n >= Q-thr | n >= E_min | Undefined | Total |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.00814 | [0.00030, 0.05093] | 0.05267 | 108 | 108 | 0 | 108 |
| 0.833 | 0.00098 | [0.00003, 0.01772] | 0.01970 | 108 | 108 | 0 | 108 |
| 1.667 | 0.00095 | [0.00002, 0.01330] | 0.01196 | 108 | 108 | 0 | 108 |
| 2.5 | 0.00055 | [0.00001, 0.01254] | 0.00637 | 108 | 108 | 0 | 108 |

**Overall: 432/432 trials (100%) clear both the numerical-validity
threshold (Q_NORM_THRESHOLD=1e-6) and the nonlinear-departure threshold
(E_MIN=1e-4). Zero trials produced an undefined residual map that had to
be excluded from the residual analysis.**

An important qualification on what "exceeds E_min" establishes. E_min was
calibrated from the duplicate-solve numerical error envelope (Stage
1B.2's numerical calibration), not as a threshold for scientific or
physical materiality. Crossing it establishes that the observed departure
is reliably above solver noise -- it does not establish that every
residual is materially large in a physical sense. The amplitude breakdown
makes the distinction concrete: at amplitude 0.025, the median departure
(E~0.002) and residual norm (~0.00002) are numerically resolved but
genuinely small; at amplitude 0.8, the departure (E~0.074) and residual
norm (~0.021) are substantial. The correct statement is: **all 432
residual maps were numerically valid and all residual departures
exceeded the prespecified numerical-departure threshold; this does not
mean every residual was materially large. The normalized residual mapping
is not an artifact of values at or below numerical precision, but the
physical magnitude of the nonlinear correction remains strongly
amplitude-dependent, as expected.**

Residual magnitude also does *not* scale with perturbation time the way
it scales with amplitude -- median residual magnitude is largest at t_p=0
and steadily shrinks at later perturbation times, the opposite direction
from the residual's *discriminability* (Delta_map), which grew with
perturbation time (0.34 -> 0.34 -> 0.42 -> 0.45 across t_p=0->2.5,
reported earlier). This is a non-obvious pattern worth keeping distinct
from the amplitude-scaling result above: the residual mapping is not
becoming more discriminable merely because the residual is getting larger
over the trajectory -- if anything, it becomes more discriminable while
getting smaller in absolute terms.

**The residual mapping is therefore supported by numerically resolved
departures in all 432 trials, with nonlinear magnitude increasing
strongly with perturbation amplitude.** The mapping is not an artifact of
undefined or solver-noise-scale residuals, although the smallest-amplitude
residuals remain physically small.

## What this establishes, precisely

**The mapping is not reducible to the directly stimulated coordinate.**

An earlier version of this exclusion test zeroed only the *actually
stimulated* coordinate in each trial -- which, even with correct
coordinate alignment (the previous correction), leaked node identity
through a different channel: the *position* of the forced zero was
itself a deterministic, input-specific signature (a low-node trial's
zero always sits at index 17, a median-node trial's at 363, a high-node
trial's at 129), detectable by JSD without any genuine propagated
response elsewhere in the graph. This was caught before being reported
as a clean result.

The corrected construction uses a **common exclusion mask, identical
across every trial regardless of which node was actually stimulated**:
all three candidate source coordinates {i_low, i_median, i_high} = {17,
363, 129} are zeroed in every trial's output vector, and the remainder
renormalized over the shared support. No trial's exclusion pattern
differs from any other's, so the output cannot reveal which node was
stimulated merely through which coordinates are missing -- only through
the actual response pattern over the common remaining nodes. This is a
deterministic post-processing correction on the already-saved
event_aligned_q vectors (verified directly: all three source coordinates
exactly zero in every checked trial, renormalization sums to 1
correctly) -- no re-integration of the 432 trajectories was needed.

**Result: Delta_map = 0.3418, p_MC = 1/10,001 ~ 0.00010** -- still at the
permutation floor. This omnibus result (combining node, sign, and
amplitude) establishes that the *balanced input mapping* survives on the
remaining graph -- it does not, by itself, establish that node identity
*specifically* remains discriminable after common-support exclusion,
since the omnibus statistic combines all three factors. That narrower
claim was tested directly, not inferred: a factor-restricted node
permutation test (node labels permuted within matched sign-amplitude
cells, preserving sign and amplitude assignment) run on the common-support
representation gives **Delta_node^(-S) = 0.8074, p_MC ~ 0.00010** -- also
at the floor, and close to the original full-space node effect (0.8185).
Node identity specifically, not merely the combined input, remains
reproducibly encoded in the response pattern over the common remaining
support.

**Audit trail confirming the corrected object is genuinely different**,
not merely rounding to the same statistic as an earlier, leakier
construction: checked directly across all 432 trials, the common-support
q differs elementwise from the single-source-zeroed version in every
single trial (432/432 changed, maximum elementwise difference
1.9x10^-4), and all three candidate source coordinates are confirmed
exactly zero in all 432 outputs. The rounded omnibus Delta_map matching
the earlier value to four decimal places was a property of this
particular dataset (the two additional excluded coordinates evidently
carried little energy on average), not evidence that the correction made
no difference to the underlying computation.

**The source-retention objection is now resolved cleanly**, with both
the omnibus and the node-specific claims directly tested: displacement
energy redistributes substantially away from the stimulated node (see
below), and the resulting spatial pattern over nodes other than any of
the three candidate sources reproducibly encodes both the combined input
and node identity specifically.

**Source energy genuinely redistributes over the response window**, though
mean and median diverge substantially at the two later time points (the
distribution is skewed), so both are reported rather than collapsed to
one figure:

| Time | Mean source fraction | Median source fraction |
|---|---:|---:|
| tau=0 (immediately after impulse) | 99.8% | 99.8% (no divergence here) |
| tau=tau* (event-aligned) | 52.4% | 68.6% |
| tau=T (fixed-time, end of horizon) | 15.9% | 7.5% |

Whichever statistic is used, the pattern is the same direction and the
same qualitative conclusion holds: the response moves substantially
beyond the initially perturbed node over the observation window, not
just decays in place at the source.

**Both linear and nonlinear structure separately carry the mapping.**
The tangent-only response already implements a structured, significant
mapping (Delta=0.3248) -- first-order graph propagation is not
information-free. The nonlinear residual, z_eps(tau) = P*Delta_theta_eps(tau)
- eps*P*delta(tau), *also* carries a separately significant mapping
(Delta=0.3896, p_MC~0.00010), now confirmed to be supported by
numerically resolved departures, not an artifact of normalizing a
vanishingly small quantity (see validity table above). Tangent-linear
propagation and the specifically nonlinear finite-minus-tangent residual
each carry separately significant, input-sensitive spatial organization.
Because x_finite = x_tangent + z by construction, the tangent and
residual representations are mathematically related, not independent
quantities -- the residual test establishes that z itself carries a
significant structured mapping, not that the tangent and nonlinear
contributions are statistically or causally independent of one another.

**A necessary correction on comparing the four Delta_map values directly.**
0.3896 (residual) is numerically larger than 0.3505 (finite) and 0.3248
(tangent), but this does not mean the nonlinear correction is physically
"stronger" than the linear propagation it corrects. Each response space
(finite, tangent, residual) is separately normalized (each q-distribution
sums to 1 within its own space), so Delta_map in one space measures
within-space discrimination relative to that space's own replica
dispersion -- it is not commensurate with Delta_map in a different space,
and comparing raw magnitudes across spaces conflates "significant within
its own geometry" with "dominant in absolute physical contribution."
Both tangent-linear propagation and finite-amplitude nonlinear
corrections contribute structured, input-sensitive spatial organization;
neither is described as stronger than the other from this comparison
alone.

## Capability hierarchy, updated

- **Level 1 (nonlinear behavior): established** -- since Stage 1B.
- **Level 2 (structured internal transformation): established, locally.**
  The conditions previously reserved for this stronger claim are now all
  met: the mapping survives source-node exclusion; it appears separately
  in the nonlinear residual, with that residual now shown to be
  supported by numerically resolved departures rather than merely a
  normalized artifact of tiny quantities; and all three input factors
  (not just node) are separately significant under multiplicity-corrected
  restricted tests.
- **Level 3 (useful computation): not established.** No external task or
  information-processing objective has been defined or tested.

## Scope, stated explicitly (the "locally" qualifier is essential)

This result is conditional on: one baseline trajectory (seed=3000); one
class (KMNIST class 0); four repeated states along that single
trajectory, not four independent trajectories; T only, no graph controls
(rewired/random/lattice) yet compared; no external task or
information-processing objective.

## Strongest defensible conclusion

Stage 1B2 establishes a locally robust structured internal transformation
along the prespecified seed-3000 class-0 trajectory. Nearby dynamical
states map perturbation location, sign, and amplitude into reproducible
spatial outputs. The balanced mapping survives common-support removal of
all three candidate stimulated-node coordinates (not merely removal of
whichever node was actually stimulated in a given trial), and
displacement energy redistributes substantially away from its source.
Tangent-linear propagation and the specifically nonlinear
finite-minus-tangent residual each carry separately significant
input-sensitive spatial organization. All residual maps are numerically
resolved, while the magnitude of the nonlinear correction depends
strongly on perturbation amplitude. The result establishes Level 2
capability locally, but not generality across trajectories,
topology-specific advantage, or external task usefulness.

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
   1B2 specifically.
3. **External usefulness (Level 3)**: can this structured mapping be
   linked to an externally defined task or information-processing
   objective, rather than characterized only in terms of its own
   internal reproducibility?

## Reproducing these results

`stage1b2_core.py` (per-trial computation -- saves q_tangent, q_residual,
q_excl_node, source-energy fraction, and raw residual norm alongside the
original q/r/J_tan diagnostics), `run_stage1b2.py` (432-trial driver,
checkpointed, parallelized), `analyze_stage1b2.py` (primary omnibus test,
corrected permutation scheme, parallelized), `analyze_stage1b2_diagnostics.py`
(the four diagnostic decompositions plus Holm-corrected factor-specific
tests, parallelized), `analyze_stage1b2_residual_materiality.py`
(residual validity and magnitude summary by amplitude and perturbation
time), `analyze_stage1b2_common_support_exclusion.py` (the corrected
source-exclusion diagnostic -- a common exclusion mask across all three
candidate source nodes, applied identically to every trial regardless of
which node was actually stimulated, run as deterministic post-processing
on the already-saved event_aligned_q vectors, no re-integration needed),
`analyze_stage1b2_common_support_node_test.py` (the node-specific
factor-restricted test on the common-support representation, plus the
audit trail confirming the corrected q genuinely differs from the
earlier, leakier construction). Raw results in `results/stage1b2_results.pkl`;
primary analysis in `results/stage1b2_final_analysis.pkl`; diagnostic
decomposition in `results/stage1b2_diagnostics.pkl`; common-support exclusion
result in `results/stage1b2_common_support_exclusion.pkl`; node-specific
common-support test in `results/stage1b2_common_support_node_test.pkl`.
All code now lives in `experiments/stage1b2_structured_transformation/`,
with the shared, reusable dynamics/stats modules consolidated into
`src/bonsai/` as part of the project's broader restructuring.
