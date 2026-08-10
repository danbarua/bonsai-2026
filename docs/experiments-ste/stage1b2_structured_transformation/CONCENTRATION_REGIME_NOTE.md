Simplified Technical English version of `experiments/stage1b2_structured_transformation/CONCENTRATION_REGIME_NOTE.md`.

# Follow-up Note: When Does the Response Move to One Other Node, and When Does It Spread?

**Status: this is a narrow follow-up look at data from Stage 1B.2. The team already froze (locked) Stage 1B.2's data before this note. This note does not revise the [Stage 1B.2 findings](FINDINGS.md).**

`FINDINGS.md` and Stage 1B.2's own conclusions do not change because of this note. Read this document as an addendum. It is not an update to the original findings.

Parts 1, 2, and 5, and the "Further Follow-Up" section, only re-analyze data that the team already collected and saved (cached). They do not use new simulation runs. Parts 3 and 4 are the exceptions. The note flags each of these as new simulation, at the point where it introduces them.

## Where This Note Came From

The team was building a plain-language report visual (`docs/report_visuals/generate_report_visuals.py`). During this work, they looked at two trial choices:

- amplitude=0.8 at t_p=0.
- amplitude=0.2 at t_p=0.

In both trials, the response energy at the final time point moved almost entirely onto one single node. This node was not the stimulated node. The energy did not spread broadly across many nodes.

This differs from the trial the team eventually used for the visual: t_p=2.5, amplitude=0.2. That trial does spread broadly.

This raised an obvious question. Is the concentration onto one node a real, describable pattern? Or are these two trials just unusual, unrepresentative examples?

## Method

**This is a pure re-analysis of the 432 trials already computed in `results/stage1b2_results.pkl`. It uses no new simulation.**

For each trial, the team used `fixed_time_q`. This is the energy distribution across nodes, already saved at tau=T=2.5 (the end of the observation window). From this, the team computed three concentration measures:

- `top1`: the fraction of the total response energy held by the single largest node.
- `top2`: the fraction held by the two largest nodes combined.
- `effective_n`: the effective number of nodes meaningfully involved. `effective_n = 1 / sum(q_i^2)`. A larger effective_n means the energy spreads more broadly.
- `argmax_node`: which node holds the largest energy share.
- `argmax_is_source`: whether that node is the same node the team stimulated.

The team checked whether these measures vary in a systematic way, with amplitude, t_p, sign, or the stimulated node. They compared this against the case where these measures stay roughly uniform across all 432 trials (see below).

The team then found one specific (node, t_p) combination where this concentration effect occurs. They ran two focused follow-ups on this combination:

- Part 1: a precise, trial-by-trial breakdown of exactly which trials concentrate, not just an aggregate count.
- Part 2: a check of whether the same concentration-and-destination pattern already exists in the tangent-only response (the linear part), or whether it is specific to the nonlinear part.

Code: `analyze_stage1b2_concentration_regime.py`.

This is a descriptive analysis of an already-collected dataset. It is not a new confirmatory hypothesis test. The team applied no multiplicity correction. The p-values below describe how unlikely the observed group differences are, under random label shuffling. They are not a pre-registered claim.

## Result: Real and Sharply Localized, Not What the First Two Examples Suggested

**Amplitude has no detectable effect.** The team ran a Kruskal-Wallis test across the three amplitude levels: H=0.154, p=0.926. They also computed the Spearman rank correlation between amplitude and `top1`: rho=-0.019, p=0.696.

The median `top1` value is 0.181 to 0.182, at every amplitude level.

The original idea was that the amplitude=0.8 trial's concentration came from an amplitude effect. The full data does not support this idea. That trial's concentration had nothing to do with its amplitude.

**Sign has no detectable effect.** The median `top1` value is 0.182, for both signs.

**t_p has a real, strong effect on its own.** t_p is the perturbation time. Kruskal-Wallis test: H=43.05, p<0.00001. Spearman rank correlation: rho=-0.236 for `top1`, and rho=+0.284 for `effective_n`, both p<0.00001.

Trials at t_p=0 are much more concentrated than trials at other times. Median `top1` at t_p=0 is 0.337. Median `top1` at t_p=0.833, 1.667, and 2.5 is about 0.17 to 0.18, at each of those times.

**The stimulated node also has a strong effect on its own.** Median `top1` is:

- 0.338 for the low-degree node.
- 0.182 for the high-degree node.
- 0.082 for the median-degree node.

This order is the opposite of what the word "degree" might suggest. Each node level's own effect is much larger than amplitude's effect, which is null (has no detectable effect).

**The real pattern is an interaction between t_p and the stimulated node. It is not two separate, independent effects.** The table below breaks `top1` down by t_p and stimulated node, together:

| t_p | low-degree node | median-degree node | high-degree node |
|---|---|---|---|
| 0 | 0.350 | 0.090 | **0.622** |
| 0.833 | 0.331 | 0.082 | 0.190 |
| 1.667 | 0.337 | 0.081 | 0.182 |
| 2.5 | 0.345 | 0.082 | 0.174 |

The low-degree node and the median-degree node show almost no dependence on t_p at all. Each column in the table stays flat across rows. The values differ from each other by only about 0.01 to 0.02.

The high-degree node is the exception. At t_p=0 specifically, its value jumps to 0.622. This is about 3.3 times its own value at any other t_p. This single cell (high-degree node, t_p=0) is the only place in the entire design where this jump happens.

**This cell separates cleanly from everything else. It is not a weak or fuzzy tendency.**

The team looked only at the (t_p=0, high-degree-node) cell. This cell has 36 trials, covering all combinations of sign, amplitude, and replica.

In this cell, `top1` ranges from 0.181 to 0.976. In 24 of these 36 trials, `top1` exceeds 0.5. This means a single node holds more than half of the total response energy, in those 24 trials.

Across the other 396 trials in the entire design, `top1` never exceeds 0.415. There is a hard gap between this one cell and every other combination of factors.

## Part 1: A Detailed Look at the (High-Degree Node, t_p=0) Cell

The "24 of 36 trials" figure understates how structured this cell really is. These 24 trials are not scattered at random among the 36. Instead, the pattern is exact:

- 4 of the 6 (sign, amplitude) combinations concentrate in all 6 of their replicas.
- The other 2 combinations concentrate in none of their 6 replicas.

No combination is mixed. No (sign, amplitude) pair has some replicas concentrate and others not.

Replica identity plays almost no role in whether a trial concentrates. (Replica identity is which nearby-state direction the team adds the perturbation to.) Only sign and amplitude together decide this.

| sign | amplitude | destination node(s) | `top1` range | concentrated in |
|---|---|---|---|---|
| −1 | 0.025 | {103} | 0.616–0.670 | 6/6 replicas |
| −1 | 0.2 | {103} | 0.387–0.446 | 0/6 replicas |
| −1 | 0.8 | **{152}** | 0.181–0.188 | 0/6 replicas |
| +1 | 0.025 | {103} | 0.671–0.720 | 6/6 replicas |
| +1 | 0.2 | {103} | 0.818–0.851 | 6/6 replicas |
| +1 | 0.8 | {103} | 0.976–0.976 | 6/6 replicas |

Across all 36 trials, the dominant destination node is:

- Node 103, in 30 trials.
- Node 152, in the remaining 6 trials. These 6 trials are exactly the sign=−1, amplitude=0.8 condition, and this holds consistently across all 6 of its replicas.

The dominant node is never the stimulated node itself, in any of the 36 trials.

So five of the six (sign, amplitude) conditions route to the same destination, node 103. The sixth condition (sign=−1, amplitude=0.8) routes consistently to a different node, node 152, instead. It never routes to the source node, and it never varies across its own replicas.

"24 of 36 trials concentrate" is true, but the more correct description is this: 4 of 6 (sign, amplitude) conditions concentrate deterministically. This is not a 67% probability.

## Part 2: Is This Linear Routing, or a Nonlinear Effect?

The team computed the same measures on two other values, for the same 36-trial cell:

- `q_tangent`: the tangent response, the pure linear estimate of the response. Already saved per trial.
- `q_residual`: the normalized energy of the residual vector, z = x_finite - x_tangent. Also already saved per trial.

In other words, `q_residual = normalized_energy(z)`. It is not `q_finite` minus `q_tangent`.

The tangent response, `q_tangent`, alone already reproduces the destination node (node 103) in all 36 of 36 trials. Its `top1` value stays in a narrow band, 0.644 to 0.696, and does not vary with sign or amplitude at all. This is expected: `q_tangent` comes from `normalized_energy(epsilon * tangent_direction)`. Normalizing by total energy removes epsilon's scale and sign entirely.

The residual, `q_residual`, also favors node 103 in all 36 of 36 trials, and even more strongly. Its `top1` value is 0.945 to 0.961.

The team set a decision rule before this analysis. Per that rule: since `q_tangent` alone already reproduces the core concentration-and-destination pattern, in every trial of this cell, the core phenomenon is first-order, state-dependent graph routing. It is not evidence of anything nonlinear.

The purely linear response already "points toward" node 103, with moderate concentration (about 0.65 to 0.70), before the team adds any nonlinear term.

Per the decision rule, this result does NOT support looking further into the attractor-redirection hypothesis, or comparing against Stage 0's known equilibria. The team did not run either follow-up.

One real, secondary pattern sits on top of this main result. It is worth naming, but it does not meet the bar for the further test the team held in reserve.

The nonlinear residual changes the strength of the linear routing's concentration, depending on sign:

- For sign=+1, as amplitude increases, concentration increases: 0.71, then 0.85, then 0.98.
- For sign=−1, as amplitude increases, concentration decreases: 0.65, then 0.42, then 0.18.

At the single most extreme combination, sign=−1 and amplitude=0.8, this decrease is strong enough to flip the actual (finite) response's destination. It flips away from node 103, to node 152. This happens even though `q_tangent` and `q_residual`, individually, each still favor node 103 at that same combination.

This means a partial cancellation happens between the linear and nonlinear terms, specifically in that one corner of the design. It is not a case where the nonlinear part independently "chooses" a different target on its own.

This is a real, secondary effect. But it changes the strength of an already-linear routing tendency. It does not create a new tendency. On its own, it does not meet the bar the decision rule set, for pursuing attractor redirection further.

## Part 3: Does the Winning Node Lead the Whole Time, or Take Over Late? (Time-Resolved Tangent Energy Share)

**This part uses genuine new simulation. It is not a pure re-analysis of the frozen cache. This note states that plainly, unlike Parts 1 and 2 and everything above.**

`stage1b2_results.pkl` only ever saved `fixed_time_q` at the single tau=T=2.5 end point, plus one event-aligned snapshot. It never kept the in-between time points of the tangent solution.

To answer the question "does the eventual winner build up steadily, or does it overshoot and reverse along the way," the team needs the actual time series. So this part re-runs the same tangent equation, `run_one_trial` (`stage1b2_core.py`), already solved for these trials. But this time, the team keeps all 51 evaluated time points in [0, 2.5], instead of discarding everything but the end point.

This part does not re-solve the separate nonlinear perturbed-theta system. `q_tangent(t)` depends only on the tangent solution, `delta_tau(t)`. See the header note in `analyze_stage1b2_time_resolved_propagator.py` for why this also means the result does not depend on sign or amplitude. It depends only on (baseline_seed, node, replica).

This part does not touch, change, or regenerate `results/stage1b2_results.pkl` or any Stage 1C cache. Its output lives in new files: `results/stage1b2_time_resolved_propagator.pkl` and `results/stage1b2_time_resolved_propagator.png`.

This part follows directly from the notebook's Sections 10 and 11, about dynamical geometry.

Section 10 found that J(0) alone plausibly explains, for three trajectories, which of two candidate pathways opens first. J(t) is the phase-dependent Jacobian: how strongly two nodes are effectively coupled at a given moment, as opposed to the fixed graph weight W, which never changes over time. The two candidate pathways are: a relay through node 105 into node 103, or a direct edge into node 152.

Section 11 found that checking the full time interval complicates this story. Several Jacobian entries flip sign across the interval [0, 2.5]. A simple time-integral of individual entries also does not cleanly predict the winner. For example, seed=3000's direct source-to-152 edge has the largest integrated exposure of its three candidate edges, yet seed=3000 still concentrates onto node 103.

This part goes one level further. Instead of the edge-level Jacobian, it tracks the actual per-node energy share, q_i(tau), throughout the interval. q_i(tau) is the fraction of the total response energy sitting at node i, at time tau; it always sums to 1 across all nodes at a given tau. Formally, q_i(tau) = delta_tau(tau)_i^2 / sum_j delta_tau(tau)_j^2.

The team tracks q_i(tau) for the three nodes Sections 8-10 already identified as relevant: 103, 105, and 152. It does this across the same three trajectories: seed=3000, 3030, and 3090. It uses the same concrete trial already listed in Part 1: node=high, t_p=0, sign=+1, amplitude=0.025, replica=0.

**Result: the three trajectories tell three genuinely different stories. This does not resolve into one tidy rule.**

- **Seed=3000 (final winner: node 103): does NOT lead the whole time. Node 152 overtakes it early, then node 103 takes back the lead partway through.**

  Node 152's share rises first and faster. It reaches q=0.353 at tau=0.95. This is well above node 103's share at that time, q=0.180. Node 152's share then declines for the rest of the interval.

  Node 103's share rises more slowly at first, but it never declines. It overtakes node 152 between tau=1.35 (q_103=0.282 vs q_152=0.304) and tau=1.40 (q_103=0.305 vs q_152=0.293). It then keeps climbing, to q=0.682 by tau=2.5.

  So the actual final winner spends roughly the first half of the interval behind the eventual loser, not ahead of it. This is a genuine overtake, not a steady buildup.

  Node 105 (the relay node identified in Section 8) shows a separate, small, early bump. It peaks at q=0.095 at tau=0.4, then decays to near zero by tau=1.0, and stays there. This is consistent with node 105 being a transit point that energy passes through early, not a node that builds up its own share.

- **Seed=3090 (final winner: node 152): leads essentially the whole time, with no reversal.**

  Node 152's share rises from the start. It never gives up ground to either of the other two nodes; it stays flat or increases across all 51 time points. It reaches q=0.751 by tau=2.5.

  There is a visible change in growth rate around tau=1.5 to 2.0: the curve speeds up instead of leveling off. But there is no dip, and no competing rise from node 103 or node 105 at any point. Node 103 never exceeds q=0.0025 across the entire interval. This rules out anything like seed=3000's crossover, for this trajectory.

  Node 105 shows the same kind of small, early, decaying bump seen in the other trajectories. It peaks at q=0.044 at tau=0.55. This is again consistent with node 105 being a transit point, not a competing destination.

- **Seed=3030 (no concentration): shows a real, but small and late-arriving, trend in the direction Section 11's Jacobian result suggested. It is not nothing.**

  Section 11 found this trajectory's relay edge, J(105,103), starts weak (+0.074 at t=0, the reason it was called "bottlenecked"). This edge grows to +0.905 by t=2.5.

  Consistent with that growth, node 103's share does rise late here, from near zero to q=0.086 by tau=2.4. But node 105 itself does not show a late buildup. It peaks early, at q=0.101 at tau=0.5, then decays to a low plateau. This is the same early-bump shape seen in the other two trajectories.

  So the more precise way to state this is: "the relay strengthens late" shows up as a late rise in the downstream node (103), not in the relay node's (105) own energy share.

  Either way, the size never approaches concentration. Among the three tracked nodes, node 103 has the largest final share, at 0.085. Using the actual final-time-point largest node over ALL nodes, not just these three, the largest is node 153, at 0.176. This is still far below the 0.5 threshold used throughout this note.

  Node 152, in this trajectory, stays essentially flat and small throughout. It never exceeds 0.029.

**What this changes about the overall picture.**

Section 10's account of initial pathway gating still stands. Section 11's finding, that neither a single snapshot nor a simple time-integral fully explains the winner, also still stands. This part adds a third angle: the actual time-resolved energy share, rather than an edge-level proxy. This new angle does not converge on a simpler story either.

If anything, it makes the picture more complex. The two trajectories that genuinely concentrate (3000 and 3090) reach the same kind of end point (one node winning decisively), but by visibly different routes. One shows a clean, steady buildup. The other shows a lead change roughly halfway through the window.

Nothing here supports "the eventual winner is ahead from early on" as a general rule. Seed=3000 is a direct counterexample to that rule.

The team reports this as a genuine, still-open complication. It does not smooth this into one single mechanism. This is consistent with Section 11's own conclusion: the full time-ordered propagator does something that these simpler diagnostics, now including this one, cannot fully capture.

**Scope of this part specifically**: three trajectories, one fixed trial selection (node=high, t_p=0, replica=0). Sign and amplitude do not affect `q_tangent(t)`, as explained above. Three tracked nodes: 103, 105, and 152. The team chose these because Sections 8-10 already identified them as the relevant candidates, for this specific (node, t_p) cell.

This is not a re-run of the full 432-trial grid. It is not evidence about any other cell of the design.

Code: `analyze_stage1b2_time_resolved_propagator.py`. Plot: `results/stage1b2_time_resolved_propagator.png`.

## Part 4: Does the Early-Leader Failure Happen Beyond Seed=3000?

Part 3 found one striking fact about a single trajectory. For seed=3000, the early tangent leader (node 152, dominant by tau=0.95) is NOT the eventual finite winner (node 103). This is a genuine overtake, not a steady buildup.

This raised an obvious follow-up question: is this just one example, or does it happen for other seeds that concentrate too? This part checks "early leader vs. final winner" across ALL 5 baseline seeds that concentrate anywhere in the (node=high, t_p=0) cell, not just seed=3000.

**This part needed additional new simulation.** The team discloses this plainly, as with Part 3.

`generate_frontier_visuals_data.py` (a companion data-generation step in `docs/report_visuals/`) computes the tangent solution at every (seed, replica) combination where at least one trial actually concentrates. The team found these combinations by reading the already-cached result files, using `find_concentrating_non_zero_replicas()`. They did not guess these combinations.

This gives full coverage of all 87 concentrated trials, across the 5 concentrating seeds:

- Seed 3000: 24 trials.
- Seed 3010: 21 trials.
- Seed 3020: 2 trials.
- Seed 3080: 5 trials.
- Seed 3090: 35 trials.

An earlier pass covered only the 14 trials that happen to fall at replica=0. That earlier pass silently dropped seeds 3020 and 3080 entirely, because neither of those seeds ever concentrates at replica=0.

**Result: seed=3000's overtake is not a one-off example. It is the norm. Seed=3090's clean, steady buildup is the outlier.**

The team compared the early tangent leader (the node with the largest `q_tangent` at tau=0.95) against the actual final winner (the node with the largest `fixed_time_q`), for every concentrated trial in each seed:

| Seed | Concentrated trials | Early leader | Final winner(s) | Match? |
|---|---|---|---|---|
| 3000 | 24 | node 152 | node 103 | **0/24 -- never** |
| 3010 | 21 | node 129 | node 130 | **0/21 -- never** |
| 3020 | 2 | node 152 | node 35 | **0/2 -- never** |
| 3080 | 5 | node 154 | node 55 | **0/5 -- never** |
| 3090 | 35 | node 152 | node 152 | **35/35 -- always** |

This result is not a mixed 35/87 (about 40%) probability, spread evenly across trials. It is a clean, deterministic split, per seed. Within every single seed, either 100% of its concentrated trials match, or 0% do. No seed shows a mix.

4 of the 5 concentrating seeds show total early-leader failure. Only seed=3090 shows total early-leader success. Seed=3090 is the same trajectory that Part 3 already found builds up steadily, with no reversal.

The early tangent leader is a poor predictor of the eventual winner, for the large majority of trajectories that concentrate at all. This is not true only for the one trajectory (seed=3000) that the team happened to pick as an example.

**What this result shows, and what it does not show.**

This result extends Part 3's "overtake, not buildup" finding from one trajectory to 4 of 5 trajectories. This is a real strengthening of the case that a single tau=0.95 snapshot, or any single early-time read of the tangent solution, cannot predict the eventual winner. This is consistent with Section 11's conclusion: only the full time-ordered propagator determines the outcome.

This result does NOT explain why seed=3090 is the exception. It does not explain what makes a trajectory's early tangent leader hold up, versus get overtaken. That question remains open, the same way Part 3 and Section 11 already left it.

This result also does not extend to the 5 non-concentrating seeds (3030, and 3040 through 3070). These seeds contribute no rows here, by construction. No trial in their cell exceeds the top1 > 0.5 threshold.

**Scope**: this covers the (node=high, t_p=0) cell, across all replicas, for the 5 seeds that concentrate anywhere in it. It is not a re-run of the full 432-trial grid, and it is not evidence about any other cell.

Code: `plot10_early_leader_vs_final_winner`, in `docs/report_visuals/generate_report_visuals.py`. Plot: `docs/report_visuals/10_early_leader_vs_final_winner.png`.

**Why the simple time-integral, from Section 11, was never guaranteed to work.**

The full propagator is Phi(T,0), a time-ordered exponential of the integral of J(t) from 0 to T.

In general, J(t1) times J(t2) does not equal J(t2) times J(t1). The Jacobian at different times does not commute with itself. Because of this, the propagator cannot reduce to a simple function of each entry's own scalar time-integral, even in principle.

So Section 11's finding, that the simple integral did not track the outcome, was not a surprising empirical failure. It is the expected result of this noncommutativity. The order in which pathways open and close determines the outcome, not just their total accumulated exposure.

**Closing this investigation.**

The static topology determines which pathways can structurally exist: the 105-relay, and the direct edge to node 152. The evolving phase configuration continuously opens and closes these pathways, through the state-dependent Jacobian. The final routed response comes from their time-ordered interaction. This interaction can include temporary leaders, pathway bottlenecks, relay activity, and late overtaking, as Part 3 showed concretely for these three trajectories.

No compact predictor reduces this to a simple rule. This includes the static adjacency (the graph structure alone), the J(0) snapshot, the integrated exposure, and early leadership. This is the substantive finding of this investigation, not a gap in it.

This investigation does not break Phi(T,0) down into rigorous, pathway-by-pathway contributions. The node-105 transient bump is consistent with relay transmission, but it is not formal proof of it.

This thread is closed. It does not change Stage 1B.2's frozen `FINDINGS.md`.

## Part 5: Splitting the Early-Leader Failure Into Its Two Parts

Part 4 found that the early tangent leader (the node with the largest `q_tangent` at tau=0.95) fails to predict the final winner, in 4 of 5 concentrating seeds. The aggregate match was 35/87.

That single comparison actually mixes together two genuinely different transitions:

(a) Time evolution within the linear tangent system itself. This is the change from the early tangent leader at tau=0.95, to the final tangent leader at tau=T, using `fixed_time_q_tangent`.

(b) The nonlinear step, at the same fixed time point. This is the change from the final tangent leader at tau=T, to the final finite leader at tau=T, using `fixed_time_q`.

Part 3 already showed genuine overtaking within the tangent system itself, directly for seed=3000. There, node 152 leads until roughly tau=1.35 to 1.40, before node 103 overtakes it.

This part checks whether that same linear-overtaking pattern holds for all 4 mismatching seeds. It also checks whether some of Part 4's mismatch instead comes from the nonlinear step.

**This is a pure re-analysis of already-cached data. It uses no new simulation.**

For the early tangent leader, this part reuses `q_tangent_full`, from `stage1b2_frontier_visuals_data.pkl`. The team already computed this for Part 4, in plot10 of `docs/report_visuals/generate_report_visuals.py`.

For the two final states, this part reuses each trial's own already-cached `fixed_time_q_tangent` and `fixed_time_q`. Both values are already saved per trial, in `stage1b2_results.pkl` and the Stage 1C result files. Plot9 already reads `fixed_time_q_tangent` the same way.

Code: `analyze_stage1b2_early_leader_decomposition.py`. This part covers the identical 87 concentrated trials, across the same 5 concentrating seeds as Part 4.

**Transition (a): early tangent leader (tau=0.95) versus final tangent leader (tau=T). This tests LINEAR overtaking.**

| Seed | Concentrated trials | Match |
|---|---|---|
| 3000 | 24 | 0/24 |
| 3010 | 21 | 0/21 |
| 3020 | 2 | 0/2 |
| 3080 | 5 | 0/5 |
| 3090 | 35 | 35/35 |

The aggregate match is 35/87. This is an identical, clean per-seed split to Part 4's headline result.

This confirms that the mismatch Part 4 reported is entirely a linear phenomenon. The change in which node leads happens within the tangent solution's own time evolution, evaluated between tau=0.95 and tau=T.

**Transition (b): final tangent leader (tau=T) versus final finite leader (tau=T). This tests NONLINEAR destination change.**

| Seed | Concentrated trials | Match |
|---|---|---|
| 3000 | 24 | 24/24 |
| 3010 | 21 | 21/21 |
| 3020 | 2 | 2/2 |
| 3080 | 5 | 5/5 |
| 3090 | 35 | 35/35 |

The aggregate match is 87/87. This is a perfect match, with zero mismatches. The final tangent leader and the final finite leader agree, in all 87 selected trials.

**Conclusion: within these 87 selected trials, every change in the final leading node, relative to the early tangent leader, is already present in the tangent solution.**

Whatever mechanism makes the early tangent leader a poor predictor, it is fully explained by the tangent system's own evolution between tau=0.95 and tau=T. This mechanism is the time-ordered overtaking that Part 3 showed concretely for seed=3000.

At tau=T, the tangent response and the finite response pick the same leading node, in every one of these 87 trials.

**Scope, stated precisely, not generalized.**

This result shows that nonlinearity does not change which node leads at the endpoint, in these 87 trials. It does not show that nonlinearity leaves the full routing behavior untouched.

The comparison above only checks the leading node of `q_tangent` against the leading node of `q_finite`. It says nothing about whether their full spatial patterns agree. It says nothing about whether concentration strength (`top1`) agrees. It says nothing about how much the residual redistributes the energy that did not go to the leading node.

This analysis does not support a broader claim like "nonlinearity does not alter routing."

A low-cost extension is possible, using data already available for these same 87 trials. It would report `top1(q_tangent)`, `top1(q_finite)`, and `sqrt(JSD(q_tangent, q_finite))`, side by side. The team has not done this. It is a natural next step, if the full routing behavior, not just the destination identity, becomes the question.

**One clarification, not a contradiction.**

Part 1 reported a destination flip for seed=3000, sign=−1, amplitude=0.8: the destination flips from node 103 to node 152. This trial does not appear among this section's 87 trials.

That trial's finite `top1` is 0.181 to 0.188 (see Part 1's table). This never crosses the 0.5 concentration threshold that defines the 87-trial set used throughout Part 4 and this section. The team excludes this trial by construction, because it is not a "concentrated" trial under this note's own definition. It is not excluded because the flip stopped happening.

Part 2 already established that this flip comes from a partial linear and nonlinear cancellation, specific to that one (sign, amplitude) corner. Nothing here changes that finding. This section's transition (b) only covers the separate set of trials that do cross the concentration threshold.

**Scope**: identical to Part 4. This covers the (node=high, t_p=0) cell, across all replicas, the same 5 concentrating seeds, and the same 87-trial set. This is not a re-run of the full 432-trial grid.

Code: `analyze_stage1b2_early_leader_decomposition.py`. The corresponding plot, plot10 in `docs/report_visuals/generate_report_visuals.py`, now shows both transitions side by side. It originally showed a single comparison that mixed the two transitions together.

## What This Note Shows, and What It Does Not Show

**What this note shows.**

For this class-0 learned topology, under Stage 1B.2's exact design: perturbing the highest-weighted-degree node, at the very start of the baseline trajectory (t_p=0), produces a different kind of final-time-point response. The energy relocates onto one other specific node, almost always node 103, deterministically depending on sign and amplitude, and never onto the source node. It does not spread broadly.

This does not happen for the low-degree node or the median-degree node, at any t_p. It also does not happen for the high-degree node, at any t_p other than 0.

The routing itself is already present in the linear (tangent) response. Sign and amplitude change its strength. In one specific combination, they also change its destination. But sign and amplitude do not create the routing.

**What this note does not show.**

This note does not show why t_p=0 specifically is where this happens, compared to other perturbation times.

One plausible, but untested, explanation: at t_p=0, the perturbation acts on the baseline trajectory before the trajectory has had time to settle toward an attractor. This is consistent with Stage 0's own multistability finding: the system takes time to converge, from a fresh random starting condition. This could plausibly change the local linear routing structure, compared to a perturbation applied once the trajectory has settled.

This note does not test this explanation. The team offers it as a plausible reading, not a confirmed explanation. Per Part 2's result, this explanation does not need to invoke nonlinear attractor-switching, to explain the core pattern.

The question of why node 103 specifically wins, for seed=3000, is addressed further down, in the "Further Follow-Up" section. That section comes from Sections 8-11 of `concentration_regime_notebook.ipynb`.

In short: a static-graph, two-hop relay (source to node 105 to node 103) explains seed=3000's destination. But this does not generalize across Stage 1C's other nine trajectories; their destinations vary between node 103, 130, 35, 55, 152, or no concentration at all.

The phase-dependent Jacobian, J(t) = W cos(theta_j - theta_i), gates which pathway is open, at a given trajectory's t_p=0 state, not the static graph alone. Its individual entries explain the three-trajectory comparison (seed=3000, 3030, 3090) at t=0 specifically.

But Part 3, above, shows that even this account is incomplete. Several of these same Jacobian entries flip sign across the full time interval. A simple time-integral of them also does not track the outcome. The actual time-resolved energy share shows that seed=3000's winner is behind for roughly the first half of the window, before it overtakes.

So the full explanation, at the level of the full time-ordered propagator, remains genuinely open. None of the diagnostics tried so far resolve it: not the snapshot, not the simple time-integral, and not the per-node share curves in Part 3.

This note also does not extend beyond Stage 1B.2's own existing scope limits: one class, one topology (T only, with no graph controls), and the 432 trials already collected. This note gives a finer-grained description of data Stage 1B.2 already gathered. It is not a new experiment.

## How This Relates to Stage 1B.2's Existing Findings

`FINDINGS.md` already documents that the response "does not reduce to the directly stimulated node." It describes redistribution in aggregate, using the source-energy fraction over time.

This note adds a where-and-when description that `FINDINGS.md`'s aggregate treatment did not attempt. Most trials genuinely spread broadly. But one specific, reliably-reproduced (node, t_p) combination instead concentrates the response deterministically onto a single other node. (For one sign and amplitude combination within that cell, it concentrates onto a different single node instead.) This routing is already present in the purely linear part of the response. The nonlinear part changes its strength, but does not create it.

This note refines the frozen finding. It does not contradict it, and it does not need to reopen it.

## Further Follow-Up: The Mechanism Behind the Destination, and Why It Does Not Generalize

The team worked through this interactively, in `concentration_regime_notebook.ipynb`, committed alongside this note. This section summarizes that work, for the permanent record.

**Does direct adjacency in T explain the destination?**

The team checked this directly. Node 152 is a genuine, but weak, direct neighbor of the high-degree source node. It ranks 6th of the source's 7 real edges.

Node 103, the dominant destination in 5 of the 6 concentrated (sign, amplitude) conditions for seed=3000, is not a meaningful direct neighbor at all. It ranks 201st out of about 505 possible nodes, while the source node has only 7 nonzero edges.

So direct adjacency does not explain the dominant case.

**Does a two-hop relay in the static graph explain it?**

Yes, for seed=3000 specifically. Of the source's 7 direct neighbors, exactly one, node 105, has any nonzero edge to node 103. This edge is strong, 0.909, close to the source's own strongest direct edges. All 6 other neighbors have zero weight to node 103.

So seed=3000's routing is a clean two-edge chain: source (node 129) to relay (node 105) to destination (node 103). It is not a diffuse convergence from many nodes.

**Does this two-hop pathway generalize across trajectories?**

No. The team checked the identical (high-degree-node, t_p=0) cell, across all 10 of Stage 1C's baseline trajectories. These are seed=3000, plus the 9 independent trajectories Stage 1C added.

T's edges, including this two-hop bridge, stay identical across every trajectory. But the outcome varies widely:

| Trajectory | Concentrated (of 36) | Destination |
|---|---|---|
| 3000 | 24 | 103 |
| 3010 | 21 | 130 |
| 3020 | 2 | 35 |
| 3030-3070 (5 trajectories) | 0 | -- |
| 3080 | 5 | 55 |
| 3090 | 35 | 152 |

Five of the nine new trajectories show no concentration at all. The four that do concentrate land on four different destinations, and never on node 103 again.

The static graph cannot change between trajectories. So the 105-relay pathway is best read as explaining this one trajectory's (seed=3000's) outcome specifically. It is not a fixed pathway that the topology always routes through.

**What actually explains the difference from trajectory to trajectory: the phase-dependent Jacobian, not the static graph.**

The tangent dynamics follow J_ij(t) = W_ij cos(theta_j(t) - theta_i(t)), defined as `force_jacobian` in `graph_oscillator_field.py`. This depends on W and on the phases, not on W alone.

The team compared J(0) across three trajectories:

- Seed=3000 (routes to node 103, in 24 of 36 trials).
- Seed=3090 (routes to node 152, in 35 of 36 trials).
- Seed=3030 (no concentration, 0 of 36 trials).

They compared J(0) along the two candidate pathways: source to relay to node 103, and source directly to node 152.

| Trajectory | J[129,105] | J[105,103] | J[129,152] | Outcome |
|---|---|---|---|---|
| 3000 | -0.727 (strong) | -0.710 (strong) | +0.643 | Both legs of the 105-relay open -> routes to 103 |
| 3030 | +0.906 (strong) | **+0.074 (weak)** | -0.531 | First leg open, second leg bottlenecked -> no concentration |
| 3090 | **+0.141 (weak)** | -0.865 (strong) | -0.747 (strong) | First leg bottlenecked, but direct edge to 152 strong -> routes to 152 instead |

The same static graph produces three different outcomes. This happens because the phase-dependent cosine term opens or closes different edges of the same candidate pathways, at each trajectory's own t_p=0 phase state.

Seed=3030's relay pathway fails at the second hop, even though it has the strongest first hop of the three trajectories. This is a bottleneck that W alone cannot show.

Seed=3090's relay pathway fails at the first hop instead. But a different, structurally-real pathway, the direct edge to node 152, happens to be strong at that trajectory's phase state, and it wins.

This gives a direct, quantitative account of the initial pathway gating that separates the three observed outcomes. It supports the conclusion that concentration is "an emergent property of the state-dependent network propagator, not static graph geometry." It does not claim that the team has fully broken down the full time-ordered propagator.

**One technical detail applies throughout all of the above. This document has not stated it until now.**

The perturbation is not a pure spike at the source node alone. `stage1b2_core.py` projects the initial impulse through the rotation-removal projector P, before use (`delta0 = P @ e_{node}`, then renormalized). So every node receives some small component of the perturbation, not only the stimulated node.

This does not change any of the findings above. But "the input," throughout this document and `concentration_regime_notebook.ipynb`, should be understood as a rotation-free, projected impulse. It is not a literal single-node spike.

**Still not established**: why seed=3000's and seed=3090's specific phase states happen to open the particular pathways they do.

It remains open whether there is a deeper pattern, that decides which trajectories open which pathways. It also remains open whether this is just an idiosyncratic phase alignment, specific to each trajectory, with no further structure to find. The team did not pursue this question here.
