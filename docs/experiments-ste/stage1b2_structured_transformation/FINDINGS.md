Simplified Technical English version of `experiments/stage1b2_structured_transformation/FINDINGS.md`.

# Stage 1B2 Findings: The Network Performs a Structured Internal Transformation (Local Result)

## The Design, in Short

The team ran 432 finite-response trials. A finite response is the real, nonlinear difference between a perturbed run of the network and the baseline (unperturbed) run.

The design has these parts:

- 3 stimulated nodes: low, median, and high weighted-degree in T (the learned topology, the graph built from the image data).
- 2 signs (positive and negative).
- 3 amplitudes: 0.025 (tangent-consistent), 0.2 (intermediate), and 0.8 (nonlinear).
- 4 perturbation times (t_p), along one baseline trajectory. t_p is the time when the team sends the perturbation into the network. A baseline trajectory is one full, unperturbed run of the network from a starting point.
- 6 fixed nearby-state replicas per perturbation time. A replica is a slightly nudged version of the same moment on the same path. The team uses replicas to check if a result still holds when a run starts almost, but not exactly, in the same place.

The baseline trajectory uses KMNIST class 0, the T topology, and seed=3000. The replica scale is 0.1. The team verified the replicas stay local (close to the baseline) at every t_p.

The primary statistic is Delta_map = B - W.

- W is the mean output-space distance between same-input outputs, across replicas.
- B is the balanced mean distance between different-input outputs, also across replicas.
- The output-space distance is d_q = sqrt(JSD), where JSD is a statistical distance measure between two distributions.

The team used a one-sided Monte Carlo permutation test, with 10,000 permutations. The test shuffles labels independently per replica. This is the only shuffling method that actually destroys input identity, while it keeps each replica's own output pattern intact. An earlier draft used a different method: a shared relabeling scheme across all replicas. The team found this scheme did not change the test statistic under permutation. It failed to destroy the input-identity signal. The team caught this problem and did not use this scheme for any reported result.

## Primary Result

| Response representation | Delta_map | p_MC |
|---|---:|---:|
| Finite response | 0.3505 | 1/10,001 ~ 0.00010 |
| Stimulated node excluded (common-support mask across all trials) | 0.3418 | 1/10,001 ~ 0.00010 |
| Tangent-only response | 0.3248 | 1/10,001 ~ 0.00010 |
| Nonlinear residual (finite minus tangent) | 0.3896 | 1/10,001 ~ 0.00010 |

For all four response types, the result sits at the Monte Carlo floor. Zero of the 10,000 permuted statistics equalled or went above the observed statistic. This gives p_MC = (0+1)/(10,000+1), about 0.00010.

This value is the lowest p-value this test can produce with 10,000 permutations. It is not necessarily the exact tail probability. To find a smaller p-value, the team would need more permutations. This analysis does not need a smaller p-value.

## Factor-Specific Results, Holm-Corrected

| Input factor | Delta | Raw p | Holm p |
|---|---:|---:|---:|
| Node | 0.8185 | 0.00010 | 0.00030 |
| Sign | 0.1058 | 0.00010 | 0.00030 |
| Amplitude | 0.1273 | 0.00010 | 0.00030 |

Node has the largest effect size. It is clearly the dominant factor.

The team also tested sign and amplitude separately, using factor-restricted permutation tests. Each of these tests keeps the matched cells of the other two factors fixed, and tests only the one selected factor.

Sign and amplitude are each significant on their own, in these restricted tests. Their effect does not disappear once the team accounts for node. But this result does not show that sign, amplitude, and node are independent of each other, in a probability sense or a causal sense. It shows only that each factor is significant on its own, under its own restricted test.

All three input factors (node, sign, amplitude) link to reproducible spatial output patterns.

Note on the design: node and amplitude each have three levels in this design. Sign has two levels. The three factors also differ in how the team applies them (their intervention geometry). Node's much larger Delta value supports the claim that node dominates in this specific design. It should not be read as a general ranking of how much information each factor can carry.

## Residual Validity and Size: Is the Residual Mapping Backed by Real, Measurable Nonlinear Change?

The permutation test on q_residual shows that the residual's normalized spatial pattern links reliably to input identity. The residual is the purely nonlinear part of the response: the finite response minus the tangent response.

q_residual is a normalized value: q_res_j = z_j^2 / sum_k z_k^2. Because it is normalized, even a very small residual can still produce a sharply organized pattern. So this test alone does not show that every residual is physically large. The team checked this directly.

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

Overall, all 432 trials (100%) pass two threshold checks:

- The numerical-validity threshold, Q_NORM_THRESHOLD = 1e-6.
- The nonlinear-departure threshold, E_MIN = 1e-4.

Zero trials produced an undefined residual map. The team did not need to exclude any trial from the residual analysis.

One qualification matters here: what does "exceeds E_min" actually show?

The team calibrated E_min from the duplicate-solve numerical error range, during Stage 1B.2's own numerical calibration. E_min is not a threshold for scientific or physical importance. Crossing E_min shows only that the observed departure is reliably above solver noise. It does not show that every residual is physically large.

The amplitude breakdown makes this clear:

- At amplitude 0.025, the median departure (E about 0.002) and residual size (about 0.00002) are numerically real, but genuinely small.
- At amplitude 0.8, the departure (E about 0.074) and residual size (about 0.021) are substantial.

The correct statement is this: all 432 residual maps were numerically valid. All residual departures crossed the prespecified numerical-departure threshold. This does not mean every residual was physically large. The normalized residual mapping is not an artifact of values at or below numerical precision. But the physical size of the nonlinear correction still depends strongly on amplitude, as expected.

Residual size does not scale with perturbation time the same way it scales with amplitude. The median residual size is largest at t_p=0. It steadily shrinks at later perturbation times.

This is the opposite direction from the residual's discriminability, measured by Delta_map. Delta_map grows with perturbation time: 0.34, then 0.34, then 0.42, then 0.45, across t_p=0 to t_p=2.5.

This pattern is not obvious, and it is separate from the amplitude-scaling result above. The residual mapping is not becoming easier to discriminate simply because the residual grows larger over the trajectory. In fact, the mapping becomes more discriminable while the residual gets smaller in absolute size.

In summary, numerically real departures support the residual mapping in all 432 trials. The nonlinear magnitude increases strongly as perturbation amplitude increases. The mapping is not an artifact of undefined values or solver-noise-scale residuals. But the residuals at the smallest amplitude remain physically small.

## What This Result Shows, Precisely

The mapping does not reduce to the single directly-stimulated node.

An earlier version of this test zeroed only the node that was actually stimulated, in each trial. This version used correct coordinate alignment, from an earlier fix. But it still leaked node identity, through a different channel.

The position of the forced zero was itself a fixed, input-specific signature:

- A low-node trial's zero always sits at index 17.
- A median-node trial's zero always sits at index 363.
- A high-node trial's zero always sits at index 129.

A statistical distance measure (JSD) could detect this signature. It did not need any real response elsewhere in the graph to do so. The team caught this problem before they reported it as a clean result.

The corrected version uses one common exclusion mask. The team applies this same mask to every trial, no matter which node was actually stimulated. The team zeroes all three candidate source coordinates in every trial's output vector: {i_low, i_median, i_high} = {17, 363, 129}. The team then renormalizes the remainder over the shared support.

No trial's exclusion pattern differs from any other trial's. So the output cannot reveal which node was stimulated just from which coordinates are missing. It can only reveal this through the actual response pattern over the remaining, common nodes.

This correction is a deterministic step, applied after the fact, on the already-saved event_aligned_q vectors. The team did not need to re-run the 432 trajectories. The team verified the correction directly: all three source coordinates are exactly zero in every checked trial, and the renormalization sums to 1 correctly in every case.

Result: Delta_map = 0.3418, p_MC = 1/10,001, about 0.00010. This is still at the permutation floor.

This result combines node, sign, and amplitude into one omnibus test. It shows that the balanced input mapping survives on the remaining graph, after the common exclusion. But it does not, by itself, show that node identity specifically stays discriminable after this exclusion, because the omnibus statistic combines all three factors.

The team tested this narrower claim directly, rather than inferring it. They ran a factor-restricted node permutation test on the common-support representation. This test permutes node labels within matched sign-amplitude cells. It keeps sign and amplitude assignment fixed. The result: Delta_node^(-S) = 0.8074, p_MC about 0.00010. This is also at the floor, and close to the original full-space node effect of 0.8185.

So node identity specifically, not just the combined input, stays reliably encoded in the response pattern over the common remaining support.

The team confirmed the corrected object is genuinely different, not just rounded to the same value as the earlier, leakier construction. They checked all 432 trials directly, element by element. The common-support q differs from the single-source-zeroed version in every single trial (432 of 432 changed). The largest single-element difference is 1.9 x 10^-4. All three candidate source coordinates are confirmed exactly zero in all 432 outputs.

The rounded omnibus Delta_map matched the earlier value to four decimal places. This match is a feature of this particular dataset. The two extra excluded coordinates evidently carried little energy on average. It is not evidence that the correction made no real difference to the underlying computation.

This resolves the source-retention objection cleanly. The team tested both the omnibus claim and the node-specific claim directly. Displacement energy moves substantially away from the stimulated node, as shown below. The resulting spatial pattern, over nodes other than the three candidate sources, reliably encodes both the combined input and node identity specifically.

Source energy genuinely moves away from the source node over the response window. At the two later time points, the mean and the median diverge substantially, because the distribution is skewed. The team reports both values, instead of collapsing them into one figure:

| Time | Mean source fraction | Median source fraction |
|---|---:|---:|
| tau=0 (immediately after impulse) | 99.8% | 99.8% (no divergence here) |
| tau=tau* (event-aligned) | 52.4% | 68.6% |
| tau=T (fixed-time, end of horizon) | 15.9% | 7.5% |

Whichever statistic the team uses, the pattern points the same direction. The same conclusion holds: the response moves substantially beyond the originally-perturbed node, over the observation window. It does not just decay in place at the source.

Both the linear part and the nonlinear part of the response carry the mapping, each on its own.

The tangent-only response, the linear estimate of the response, already shows a structured, significant mapping (Delta = 0.3248). First-order graph propagation is not information-free.

The nonlinear residual, z_eps(tau) = P*Delta_theta_eps(tau) - eps*P*delta(tau), also carries a separately significant mapping (Delta = 0.3896, p_MC about 0.00010). The team confirmed this result is backed by numerically real departures. It is not an artifact of normalizing a very small quantity (see the validity table above).

So the tangent-linear propagation, and the specifically nonlinear finite-minus-tangent residual, each carry separately significant, input-sensitive spatial patterns.

By construction, x_finite = x_tangent + z. So the tangent and residual values are mathematically related, not independent quantities. The residual test shows that z itself carries a significant structured mapping. It does not show that the tangent part and the nonlinear part are statistically or causally independent of each other.

One correction is needed here, about comparing the four Delta_map values directly.

The residual's Delta_map (0.3896) is numerically larger than the finite response's (0.3505) and the tangent response's (0.3248). This does not mean the nonlinear correction is physically stronger than the linear propagation it corrects.

Each response space (finite, tangent, residual) is normalized on its own. Each q-distribution sums to 1, within its own space. So Delta_map in one space measures discrimination within that space, relative to that space's own replica spread. It is not directly comparable to Delta_map in a different space. Comparing raw numbers across spaces confuses two different things: "significant within its own space" and "the largest physical contribution overall."

Both the tangent-linear propagation and the finite-amplitude nonlinear correction add structured, input-sensitive spatial patterns. This comparison alone does not show that one is stronger than the other.

## Capability Levels, Updated

- Level 1 (nonlinear behavior): established. This has held since Stage 1B.
- Level 2 (structured internal transformation): established, locally. The team has now met every condition set earlier for this stronger claim: the mapping survives exclusion of the source node; the mapping also appears separately in the nonlinear residual, and the team has shown this residual is backed by numerically real departures, not just a normalized artifact of tiny values; and all three input factors (not just node) are each significant on their own, under multiplicity-corrected restricted tests.
- Level 3 (useful computation): not established. The team has not defined or tested any external task or information-processing goal.

## Scope: Why "Locally" Matters

This result depends on several limits:

- One baseline trajectory only (seed=3000).
- One class only (KMNIST class 0).
- Four repeated states along that single trajectory. These are not four independent trajectories.
- The T topology only. The team has not yet compared graph controls such as rewired, random, or lattice graphs.
- No external task or information-processing goal.

## The Strongest Conclusion the Data Supports

Stage 1B2 establishes a locally robust, structured internal transformation, along the prespecified seed-3000, class-0 trajectory.

Nearby dynamical states map perturbation location, sign, and amplitude into reproducible spatial output patterns. The balanced mapping survives even when the team removes all three candidate stimulated-node coordinates together, not just whichever node was actually stimulated in a given trial. Displacement energy moves substantially away from its source.

The tangent-linear propagation, and the specifically nonlinear finite-minus-tangent residual, each carry separately significant, input-sensitive spatial patterns. All residual maps are numerically real. The size of the nonlinear correction depends strongly on perturbation amplitude.

This result establishes Level 2 capability, locally. It does not establish that this result generalizes across trajectories. It does not establish any advantage for the learned topology specifically. It does not establish usefulness for an external task.

## What Remains Open

The question is no longer whether a structured internal mapping exists. It does exist, under this trajectory-conditioned design. These questions remain open:

1. **Generalization across trajectories**: does this result reproduce with independent baseline trajectories, using different seeds? Or is it specific to this one trajectory?
2. **Topology specificity**: does the learned topology T produce this mapping more strongly, more efficiently, or with a different structure, than the matched control graphs? These controls are degree-preserving rewiring, matched-sparsity random graphs, and regular lattice graphs, from the earlier E/R and Stage 1A work. The team has not yet tested this question for Stage 1B2 specifically.
3. **External usefulness (Level 3)**: can the team link this structured mapping to an externally defined task or information-processing goal? So far, the team has only characterized it in terms of its own internal reproducibility.

## How to Reproduce These Results

The code for these results:

- `stage1b2_core.py`: runs the per-trial computation. It saves q_tangent, q_residual, q_excl_node, the source-energy fraction, and the raw residual size. It also saves the original q/r/J_tan diagnostics.
- `run_stage1b2.py`: drives all 432 trials. It checkpoints progress and runs in parallel.
- `analyze_stage1b2.py`: runs the primary omnibus test, using the corrected permutation scheme, in parallel.
- `analyze_stage1b2_diagnostics.py`: runs the four diagnostic decompositions, plus the Holm-corrected factor-specific tests, in parallel.
- `analyze_stage1b2_residual_materiality.py`: summarizes residual validity and size, by amplitude and by perturbation time.
- `analyze_stage1b2_common_support_exclusion.py`: runs the corrected source-exclusion diagnostic. It applies one common exclusion mask, across all three candidate source nodes, identically to every trial, no matter which node was actually stimulated. It runs as a deterministic step after the fact, on the already-saved event_aligned_q vectors. It does not need to re-run the 432 trajectories.
- `analyze_stage1b2_common_support_node_test.py`: runs the node-specific, factor-restricted test on the common-support representation. It also produces the audit trail that confirms the corrected q genuinely differs from the earlier, leakier construction.

The result files:

- Raw results: `results/stage1b2_results.pkl`.
- Primary analysis: `results/stage1b2_final_analysis.pkl`.
- Diagnostic decomposition: `results/stage1b2_diagnostics.pkl`.
- Common-support exclusion result: `results/stage1b2_common_support_exclusion.pkl`.
- Node-specific common-support test: `results/stage1b2_common_support_node_test.pkl`.

All code now lives in `experiments/stage1b2_structured_transformation/`. The project moved the shared, reusable dynamics and statistics modules into `src/bonsai/`, as part of the project's broader restructuring.

## Addendum

A [follow-up addendum](CONCENTRATION_REGIME_NOTE.md) describes a deterministic routing pattern. This pattern depends on sign and amplitude. It appears specifically for the highest-degree node at t_p=0. The team confirmed this pattern is first-order linear routing: the tangent response alone (q_tangent) reproduces it. It is not nonlinear attractor-switching. This addendum does not change the frozen finding above.
