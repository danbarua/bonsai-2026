# Follow-up note: when does the response relocate to one other node vs. spread broadly?

**Status: a scoped follow-up observation on already-frozen Stage 1B.2
data, not a revision of [Stage 1B.2 findings](FINDINGS.md).** `FINDINGS.md` and Stage 1B.2's
own conclusions are unchanged by this note. This document exists
separately and should be read as an addendum, not an update. Parts 1-2,
Part 5, and the "Further follow-up" section are pure re-analysis of
already-cached data (no new simulation); **Parts 3 and 4 are the
exceptions** and are each flagged as new simulation at the point they're
introduced.

## Where this came from

While building a plain-language report visual 
(`docs/report_visuals/generate_report_visuals.py`), two trial choices --
amplitude=0.8 at t_p=0, and amplitude=0.2 at t_p=0 -- showed the
final-timepoint response energy relocating almost entirely onto a single
node *other than* the stimulated one, rather than spreading broadly
across many nodes (unlike the trial eventually used for the visual,
t_p=2.5 amplitude=0.2, which does spread broadly). This raised an
obvious question: is that a real, characterizable regime, or just two
unrepresentative anecdotes?

## Method

**Pure re-analysis of `results/stage1b2_results.pkl`'s already-computed
432 trials -- no new simulation.** For each trial's `fixed_time_q` (the
tau=T=2.5 energy distribution already saved), computed three
concentration measures:

- `top1`: fraction of total response energy held by the single largest node
- `top2`: fraction held by the two largest nodes combined
- `effective_n = 1 / sum(q_i^2)` (inverse participation ratio, inverted
  so larger = more spread out)
- `argmax_node`: which node holds the largest share, and
  `argmax_is_source`: whether that's the same node that was stimulated

then checked whether these vary systematically with amplitude, t_p
(perturbation time), sign, or stimulated node, versus being roughly
uniform across all 432 trials (below); once a specific (node, t_p) cell
was identified as the locus of the effect, did two focused follow-ups on
that cell specifically: a precise (not aggregate) breakdown of exactly
which trials concentrate (Part 1), and a check of whether the same
concentration-and-destination pattern already exists in the purely
linear (tangent-only) response or is specific to the nonlinear part
(Part 2). Code: `analyze_stage1b2_concentration_regime.py`.

This is descriptive/correlational characterization of an already-
collected dataset, not a new confirmatory hypothesis test -- no
multiplicity correction is applied, and the p-values below describe how
unlikely the observed group differences are under random label
shuffling, not a pre-registered claim.

## Result: real and sharply localized, but not what the original two examples suggested

**Amplitude has no detectable effect.** Kruskal-Wallis across the three
amplitude levels: H=0.154, p=0.926. Spearman rank correlation between
amplitude and `top1`: rho=-0.019, p=0.696. Median `top1` is
0.181-0.182 at every amplitude. My original hypothesis -- that the
amplitude=0.8 trial's concentration was an amplitude effect -- is not
supported by the full data; that trial's concentration had nothing to
do with its amplitude.

**Sign has no detectable effect** (median `top1` 0.182 for both signs).

**t_p (perturbation time) has a real, strong effect on its own**
(Kruskal-Wallis H=43.05, p<0.00001; Spearman rho=-0.236 for `top1`,
+0.284 for `effective_n`, both p<0.00001): t_p=0 trials are much more
concentrated (median `top1`=0.337) than t_p=0.833/1.667/2.5 (median
`top1`~0.17-0.18 at each).

**Stimulated node also has a strong effect on its own**: median `top1`
is 0.338 for the low-degree node, 0.182 for the high-degree node, and
0.082 for the median-degree node -- ordered the opposite of what
"degree" might suggest, and each level's own effect is much larger than
amplitude's null one.

**The real story is an interaction, not two separate main effects.**
Breaking `top1` down by (t_p, stimulated node) together:

| t_p | low-degree node | median-degree node | high-degree node |
|---|---|---|---|
| 0 | 0.350 | 0.090 | **0.622** |
| 0.833 | 0.331 | 0.082 | 0.190 |
| 1.667 | 0.337 | 0.081 | 0.182 |
| 2.5 | 0.345 | 0.082 | 0.174 |

The low- and median-degree nodes show essentially **no** t_p dependence
at all (each column is flat across rows, values within ~0.01-0.02 of
each other). The high-degree node is the exception: at t_p=0 specifically
it jumps to 0.622, roughly 3.3x its own value at any other t_p -- and
this single cell is the only place in the entire design where that
jump happens.

**This cell is cleanly separated from everything else, not a fuzzy
tendency.** Restricting to the (t_p=0, high-degree-node) cell (36 trials,
spanning all combinations of sign, amplitude, and replica): `top1` ranges
from 0.181 to 0.976, with 24 of 36 trials exceeding 0.5 (i.e., a single
node holding more than half the total response energy). **Across the
other 396 trials in the entire design, `top1` never exceeds 0.415** --
there is a hard gap between this one cell and every other combination of
factors.

## Part 1: precise breakdown of the (high-degree-node, t_p=0) cell

The "24 of 36 trials" figure above understates how structured this cell
actually is -- it is not 24 trials scattered probabilistically among 36;
it is **exactly 4 of the 6 (sign, amplitude) combinations, each
concentrated in all 6 of their replicas, and the other 2 combinations,
each concentrated in none of their 6 replicas.** No condition was mixed
(i.e., no (sign, amplitude) pair had some replicas concentrate and
others not) -- replica identity (which nearby-state direction the
perturbation is added to) plays essentially no role in whether a trial
concentrates; only (sign, amplitude) does:

| sign | amplitude | destination node(s) | `top1` range | concentrated in |
|---|---|---|---|---|
| −1 | 0.025 | {103} | 0.616–0.670 | 6/6 replicas |
| −1 | 0.2 | {103} | 0.387–0.446 | 0/6 replicas |
| −1 | 0.8 | **{152}** | 0.181–0.188 | 0/6 replicas |
| +1 | 0.025 | {103} | 0.671–0.720 | 6/6 replicas |
| +1 | 0.2 | {103} | 0.818–0.851 | 6/6 replicas |
| +1 | 0.8 | {103} | 0.976–0.976 | 6/6 replicas |

**Destination-node frequency across all 36 trials: node 103 in 30, node
152 in the remaining 6 (exactly the sign=−1, amplitude=0.8 condition,
consistently across all 6 of its replicas). The dominant node is never
the stimulated node itself, in any of the 36 trials.**

So: five of the six (sign, amplitude) conditions route to the same
destination (node 103); the sixth (sign=−1, amplitude=0.8) routes
consistently to a different node (152) instead, never to the source, and
never inconsistently across its own replicas. "24 of 36 trials"
concentrate, but the correct description is **4 of 6 (sign, amplitude)
conditions, deterministically, not a 67% probability**.

## Part 2: is this linear routing or a nonlinear effect?

Computed the same measures on `q_tangent` (the pure first-order/linear
response, already saved per trial) and `q_residual` (the normalized energy
of the vector residual `z = x_{\text{finite}} - x_{\text{tangent}}`, also
already saved) for the same 36-trial cell. In other words,
`q_{\text{residual}} = \operatorname{normalized\_energy}(z)`, not
`q_finite` minus `q_tangent`.

**`q_tangent` alone already reproduces the destination (node 103) in all
36 of 36 trials**, with `top1` in a narrow band (0.644–0.696) that does
not vary with sign or amplitude at all -- expected, since `q_tangent` is
built from `normalized_energy(epsilon * tangent_direction)`, and
normalizing by total energy removes epsilon's scale and sign entirely.
**`q_residual` also favors node 103 in all 36 of 36 trials**, even more
strongly (`top1` in 0.945–0.961).

**Per the pre-committed decision rule: since `q_tangent` alone already
reproduces the core concentration-and-destination pattern in every trial
of this cell, the core phenomenon is first-order, state-dependent graph
routing -- not evidence of anything nonlinear.** The purely linear
response already "points toward" node 103 with moderate concentration
(~0.65–0.70) before any nonlinear term is added. Per the decision rule
as specified, this result does NOT motivate the attractor-redirection
hypothesis or a comparison against Stage 0's known equilibria, and
neither was run.

**One real, secondary pattern is layered on top, worth naming but not
worth the held-in-reserve further test.** The nonlinear residual
amplifies the linear routing's concentration for sign=+1 as amplitude
increases (0.71 → 0.85 → 0.98) and attenuates it for sign=−1 as
amplitude increases (0.65 → 0.42 → 0.18) -- and at the single most
extreme combination (sign=−1, amplitude=0.8), this attenuation is strong
enough that the actual (finite) response's destination flips away from
node 103 to node 152, even though `q_tangent` and `q_residual`
individually *each still favor node 103* at that same combination. That
implies a partial cancellation between the linear and nonlinear terms
specifically in that one corner of the design, not a case where the
nonlinear part independently "chooses" a different target. This is a
real, secondary effect, but it modulates an already-linear routing
tendency rather than creating a new one -- it does not, on its own, meet
the bar the decision rule set for pursuing attractor redirection further.

## Part 3: does the winning pathway lead the whole way, or overtake late? (time-resolved tangent energy share)

**This part is genuine new simulation, not pure re-analysis of the frozen
cache -- disclosed plainly, unlike Parts 1-2 and everything above.**
`stage1b2_results.pkl` only ever saved `fixed_time_q` at the single
tau=T=2.5 endpoint (plus one event-aligned snapshot); it never retained the
intermediate timepoints of the tangent solution. Answering "does the
eventual winner build up monotonically, or does it overshoot/reverse along
the way" requires the actual time series, so this part re-integrates the
same tangent ODE `run_one_trial` (`stage1b2_core.py`) already solves for
these trials, but keeps all 51 evaluated timepoints in [0, 2.5] instead of
discarding everything but the endpoint. It does not re-solve the separate
nonlinear perturbed-theta system, since `q_tangent(t)` depends only on the
tangent solution `delta_tau(t)` -- see the header docstring of
`analyze_stage1b2_time_resolved_propagator.py` for why this also means the
result is independent of sign and amplitude, depending only on
(baseline_seed, node, replica). It does not touch, modify, or regenerate
`results/stage1b2_results.pkl` or any Stage 1C cache; output lives in a new
`results/stage1b2_time_resolved_propagator.pkl` /
`results/stage1b2_time_resolved_propagator.png`.

This follows directly from the notebook's Sections 10-11 (dynamical
geometry) -- Section 10 found that $J(0)$ alone plausibly explains, for
three trajectories, which of two candidate pathways (a 105-relay into node
103, or a direct edge into node 152) is initially open; Section 11 found
that checking the full interval complicates that story -- several Jacobian
entries flip sign across $[0,2.5]$, and a naive time-integral of individual
entries doesn't cleanly predict the winner either (seed=3000's direct
source->152 edge has the *largest* integrated exposure of its three
candidate edges, yet 3000 still concentrates onto 103). This part goes one
level further: instead of the edge-level Jacobian, it tracks the actual
per-node energy share $q_i(\tau) = \delta_\tau(\tau)_i^2 / \sum_j
\delta_\tau(\tau)_j^2$ throughout the interval, for the three nodes
Sections 8-10 identified as relevant (103, 105, 152), across the same
three trajectories (seed=3000, 3030, 3090), at the same concrete trial
already tabulated in Part 1 (node=high, t_p=0, sign=+1, amplitude=0.025,
replica=0).

**Result: the three trajectories tell three genuinely different stories --
this does not resolve into a single tidy rule.**

- **Seed=3000 (winner: node 103): does NOT lead throughout. It is
  overtaken partway through, in the other direction.** Node 152 rises
  first and faster, reaching q=0.353 at tau=0.95 -- comfortably higher
  than node 103's q at that time (0.180) -- then *declines* for the rest
  of the interval. Node 103 rises more slowly at first but never reverses,
  overtaking node 152 between tau=1.35 (q_103=0.282 vs q_152=0.304) and
  tau=1.40 (q_103=0.305 vs q_152=0.293), then continues to climb to
  q=0.682 by tau=2.5. So the actual final winner spends roughly the first
  half of the interval *behind* the eventual loser, not ahead of it -- a
  genuine overtake, not a monotonic buildup. Node 105 (the relay
  identified in Section 8) shows a separate, small, early transient bump
  (peaking at q=0.095 at tau=0.4) that decays to near-zero by tau=1.0 and
  stays there -- consistent with being a transit point energy passes
  through early, not a node that itself accumulates share.

- **Seed=3090 (winner: node 152): DOES lead essentially the whole way, no
  reversal.** Node 152's share rises from the start and never gives up
  ground to either of the other two nodes (monotonically non-decreasing
  across all 51 timepoints), reaching q=0.751 by tau=2.5. There is a
  visible change in the growth rate around tau=1.5-2.0 (the curve
  accelerates rather than plateauing), but no dip and no competing
  overshoot from node 103 or 105 at any point -- node 103 never exceeds
  q=0.0025 the entire interval, ruling out anything resembling seed=3000's
  crossover for this trajectory. Node 105 shows the same kind of small,
  early, decaying transient bump as in the other two trajectories (peak
  q=0.044 at tau=0.55), again consistent with a transit point rather than
  a competing destination.

- **Seed=3030 (no concentration): a real but insufficient late-arriving
  trend in the direction Section 11's Jacobian result suggested, not
  nothing.** Section 11 found this trajectory's relay edge J(105,103)
  starts weak (+0.074 at t=0, the basis for calling it "bottlenecked") but
  grows to +0.905 by t=2.5. Consistent with that, node 103's share here
  does rise late -- from near zero to q=0.086 by tau=2.4 -- but the
  *node* 105 itself does not show a late buildup (it peaks early, q=0.101
  at tau=0.5, then decays to a low plateau, the same early-transient
  shape seen in the other two trajectories) -- so "the relay strengthens
  late" shows up as a late rise specifically in the *downstream* node
  (103), not in the relay node's own energy share, which is the more
  precise way to state it. Either way, the magnitude never approaches
  concentration: the largest of the three tracked nodes' final shares is
  node 103's 0.085 (using the actual final-timepoint argmax over ALL
  nodes, not just these three: node 153, at 0.176 -- still far below the
  \>0.5 threshold used throughout this note). Node 152 in this trajectory
  is essentially flat and negligible throughout (never exceeds 0.029).

**What this changes about the emerging picture.** Section 10's initial-
gating account and Section 11's "neither a snapshot nor a naive integral
fully explains the winner" finding both stand -- this part adds a third
angle (the actual time-resolved share, not an edge-level proxy) and it
does not converge on a simpler story either. If anything it complicates
the picture further: the two genuinely concentrating trajectories
(3000, 3090) reach the same qualitative endpoint (one node winning
decisively) via visibly different routes -- one with a clean monotonic
buildup, the other with a lead change roughly halfway through the window.
There is no evidence here that "the eventual winner is ahead from early
on" is a general rule; seed=3000 is a direct counterexample to it. This
is reported as a genuine, still-open complication, not smoothed into a
single mechanism -- consistent with Section 11's own conclusion that the
full time-ordered propagator is doing something these simpler diagnostics
(now including this one) can't fully capture.

**Scope of this part specifically**: three trajectories, one fixed trial
selection (node=high, t_p=0, replica=0; sign/amplitude do not affect
`q_tangent(t)`, see above), three tracked nodes (103, 105, 152) chosen
because Sections 8-10 already identified them as the relevant candidates
for this specific (node, t_p) cell -- not a re-run of the full 432-trial
grid, and not evidence about any other cell of the design. Code:
`analyze_stage1b2_time_resolved_propagator.py`; plot:
`results/stage1b2_time_resolved_propagator.png`.

## Part 4: does the early-leader failure generalize beyond seed=3000?

Part 3 found one striking fact about a single trajectory: seed=3000's
early tangent leader (node 152, dominant by tau=0.95) is NOT the eventual
finite winner (node 103) -- a genuine overtake, not a monotonic buildup.
Prompted by the obvious follow-up question (is this one anecdote, or does
it generalize to the other seeds that concentrate?), this part checks
"early leader vs. final winner" across ALL 5 baseline seeds that
concentrate anywhere in the (node=high, t_p=0) cell, not just seed=3000.

**This required additional new simulation**, disclosed same as Part 3:
`generate_frontier_visuals_data.py` (in
`docs/report_visuals/`'s companion data-generation step) computes the
tangent solve at every (seed, replica) combination where at least one
trial actually concentrates -- found by reading the already-cached result
files (`find_concentrating_non_zero_replicas()`), not guessed -- giving
full coverage of all 87 concentrated trials across the 5 concentrating
seeds (3000: 24, 3010: 21, 3020: 2, 3080: 5, 3090: 35), rather than only
the 14 that happen to fall at replica=0 (an earlier pass covered only
that subset and silently dropped seeds 3020/3080 entirely, since neither
ever concentrates at replica=0).

**Result: seed=3000's overtake is not an anecdote -- it is the norm, and
seed=3090's clean buildup is the outlier.** Comparing the early tangent
leader (argmax of `q_tangent` at tau=0.95) against the actual final
winner (argmax of `fixed_time_q`), for every concentrated trial in each
seed:

| Seed | Concentrated trials | Early leader | Final winner(s) | Match? |
|---|---|---|---|---|
| 3000 | 24 | node 152 | node 103 | **0/24 -- never** |
| 3010 | 21 | node 129 | node 130 | **0/21 -- never** |
| 3020 | 2 | node 152 | node 35 | **0/2 -- never** |
| 3080 | 5 | node 154 | node 55 | **0/5 -- never** |
| 3090 | 35 | node 152 | node 152 | **35/35 -- always** |

This is not a mixed 35/87 (~40%) probability spread across trials -- it
is a clean, deterministic **per-seed** split: within every single seed,
either 100% of its concentrated trials match or 0% do (no seed shows a
mix). 4 of the 5 concentrating seeds show total early-leader failure; only
seed=3090 -- the same trajectory Part 3 already found builds up
monotonically with no reversal -- shows total early-leader success. The
early tangent leader is a poor predictor of the eventual winner for the
large majority of trajectories that concentrate at all, not just for the
one trajectory (3000) that happened to be picked as illustrative.

**What this does and doesn't establish.** This generalizes Part 3's
"overtake, not buildup" finding from one trajectory to 4 of 5 -- a real
strengthening of the case that a tau=0.95 snapshot (or any single
early-time read of the tangent solution) cannot be used to predict the
eventual winner, consistent with Section 11's conclusion that only the
full time-ordered propagator determines the outcome. It does NOT explain
*why* seed=3090 is the exception, or what distinguishes a trajectory whose
early tangent leader holds up from one whose leader gets overtaken -- that
remains open, same as Part 3 and Section 11 already left it. It also
does not extend to the 5 non-concentrating seeds (3030, 3040-3070), which
contribute no rows here by construction (no trial in their cell exceeds
the top1>0.5 threshold).

**Scope**: covers the (node=high, t_p=0) cell across all replicas, for the
5 seeds that concentrate anywhere in it -- not a re-run of the full
432-trial grid, and not evidence about any other cell. Code:
`docs/report_visuals/generate_report_visuals.py`'s `plot10_early_leader_vs_final_winner`;
plot: `docs/report_visuals/10_early_leader_vs_final_winner.png`.

**Why the naive time-integral (Section 11) was never guaranteed to
work.** The full propagator is $\Phi(T,0) =
\mathcal{T}\exp(\int_0^T J(t)\,dt)$ -- a *time-ordered* exponential.
Because $J(t_1)J(t_2) \neq J(t_2)J(t_1)$ in general (the Jacobian at
different times doesn't commute with itself), this does not reduce to a
function of each entry's scalar time-integral, even in principle.
Section 11's finding that the naive integral didn't track the outcome
wasn't a surprising empirical failure -- it's the expected consequence
of noncommutativity. The order in which pathways open and close, not
just their accumulated exposure, determines the outcome.

**Closing this investigation.** Static topology determines which
pathways are structurally available (the 105-relay, the direct edge to
152). The evolving phase configuration continuously gates those
pathways through the state-dependent Jacobian. The final routed
response is determined by their time-ordered interaction -- which can
involve transient leaders, pathway bottlenecks, relay activity, and
late overtaking, as Part 3 showed concretely for these three
trajectories. No compact predictor (static adjacency, $J(0)$ snapshot,
integrated exposure, or early leadership) reduces this to a simple
rule, and that is the substantive finding, not a gap in the
investigation. This does not decompose $\Phi(T,0)$ into rigorous
pathwise contributions -- the node-105 transient is consistent with,
but not formal proof of, relay transmission. This thread is closed; it
does not alter Stage 1B.2's frozen `FINDINGS.md`.

## Part 5: decomposing the early-leader failure into its two component transitions

**Motivated by Part 4's finding** that the early tangent leader (argmax of
`q_tangent` at tau=0.95) fails to predict the final winner in 4 of 5
concentrating seeds (35/87 aggregate match) -- itself built from a single
comparison that conflates two genuinely different transitions: (a) time
evolution *within* the linear tangent system itself (early tangent
tau=0.95 -> final tangent tau=T, from `fixed_time_q_tangent`), and (b)
the nonlinear step at the same fixed timepoint (final tangent tau=T ->
final finite tau=T, from `fixed_time_q`). Part 3 already demonstrated
genuine tangent-system overtaking directly for seed=3000 (node 152 leads
until roughly tau=1.35-1.40 before node 103 overtakes it); this part
checks whether that same linear-overtaking account holds for all 4
mismatching seeds, or whether some of Part 4's mismatch is instead
attributable to the nonlinear step.

**Pure re-analysis of already-cached data -- no new simulation.** Reuses
`stage1b2_frontier_visuals_data.pkl`'s `q_tangent_full` (already computed
for Part 4 / `docs/report_visuals/generate_report_visuals.py`'s plot10)
for the early tangent leader, and each trial's own already-cached
`fixed_time_q_tangent` and `fixed_time_q` (both already saved per trial
in `stage1b2_results.pkl` / the Stage 1C result files -- plot9 already
reads `fixed_time_q_tangent` the same way) for the two final states.
Code: `analyze_stage1b2_early_leader_decomposition.py`. Covers the
identical 87 concentrated trials across the same 5 concentrating seeds as
Part 4.

**Transition (a): early tangent (tau=0.95) vs. final tangent (tau=T) --
tests LINEAR overtaking.**

| Seed | Concentrated trials | Match |
|---|---|---|
| 3000 | 24 | 0/24 |
| 3010 | 21 | 0/21 |
| 3020 | 2 | 0/2 |
| 3080 | 5 | 0/5 |
| 3090 | 35 | 35/35 |

Aggregate 35/87 -- an identical, clean per-seed split to Part 4's
headline result. This confirms the mismatch Part 4 reported is entirely a
**linear** phenomenon: whatever overtaking happens, happens within the
tangent solution itself, before tau=T is even reached by the nonlinear
system.

**Transition (b): final tangent (tau=T) vs. final finite (tau=T) --
tests NONLINEAR destination modification.**

| Seed | Concentrated trials | Match |
|---|---|---|
| 3000 | 24 | 24/24 |
| 3010 | 21 | 21/21 |
| 3020 | 2 | 2/2 |
| 3080 | 5 | 5/5 |
| 3090 | 35 | 35/35 |

Aggregate 87/87 -- a perfect match, zero mismatches. The nonlinear step
never flips the destination among these 87 concentrated trials, for any
seed.

**Conclusion: Part 4's early-leader failure is entirely a linear
(tangent-dynamics) phenomenon, not evidence of nonlinear rerouting.**
Whatever mechanism makes the early tangent leader a poor predictor --
time-ordered overtaking of the kind Part 3 showed concretely for
seed=3000 -- operates entirely within the linear tangent system, before
the nonlinear correction is even applied. The nonlinear step, once
applied, never changes which node the trial concentrates onto, for any
of these 87 trials.

**One clarification, not a contradiction: the Part 1 seed=3000/sign=-1/
amplitude=0.8 destination flip (node 103 -> node 152) does not appear
among this section's 87 trials.** That trial's finite `top1` is
0.181-0.188 (see Part 1's table), which never crosses the 0.5
concentration threshold used to define the 87-trial set analyzed
throughout Part 4 and this section -- it is excluded by construction (it
isn't a "concentrated" trial by this note's own definition), not because
the flip stopped happening. Part 2 already established that flip is a
partial linear/nonlinear cancellation specific to that one (sign,
amplitude) corner; nothing here supersedes that finding, since this
section's transition (b) is scoped to the disjoint set of trials that do
cross the concentration threshold.

**Scope**: identical to Part 4 -- the (node=high, t_p=0) cell across all
replicas, the same 5 concentrating seeds, the same 87-trial set. Not a
re-run of the full 432-trial grid. Code:
`analyze_stage1b2_early_leader_decomposition.py`; the corresponding plot
(`docs/report_visuals/generate_report_visuals.py`'s plot10) now shows
both transitions side by side rather than the single conflated
comparison it originally reported.

## What this establishes, and what it doesn't

**Establishes**: for this class-0 learned topology, under Stage 1B.2's
exact design, perturbing the highest-weighted-degree node at the very
start of the baseline trajectory (t_p=0) produces a qualitatively
different final-timepoint response -- energy relocating onto one other
specific node (almost always node 103, deterministically dependent on
sign and amplitude, never the source) rather than spreading broadly.
This does not happen for the low- or median-degree nodes at any t_p, or
for the high-degree node at any t_p other than 0. The routing itself is
present already in the linear (tangent) response; sign and amplitude
modulate its strength and, in one specific combination, its destination,
but do not create it.

**Does not establish**: *why* t_p=0 specifically is where this happens,
versus other perturbation times. One plausible, but untested,
interpretation: t_p=0 perturbations act on the baseline trajectory before
it has had time to settle toward an attractor (consistent with Stage 0's
own multistability finding -- the system takes time to converge from a
fresh random initial condition), which could plausibly change the local
linearization's routing structure compared to a perturbation applied once
the trajectory has settled. This note does not test that mechanism -- it
is offered as a plausible reading, not a confirmed explanation, and per
Part 2's result, does not need to invoke nonlinear attractor-switching to
explain the core pattern.

*Why node 103 specifically, for seed=3000*, is addressed further down
("Further follow-up" section, from `concentration_regime_notebook.ipynb`
Sections 8-11): a static-graph 2-hop relay (source->105->103) explains
seed=3000's destination, but does not generalize across Stage 1C's other
nine trajectories (destinations vary: 103, 130, 35, 55, 152, or nothing).
The phase-dependent Jacobian $J(t)=W\cos(\theta_j-\theta_i)$, not the
static graph, gates which pathway is open at a given trajectory's t_p=0
state, and its individual entries account for the three-trajectory
comparison (seed=3000/3030/3090) at $t=0$ specifically. But Part 3 above
shows even that account is incomplete: several of the same Jacobian
entries flip sign across the full interval, a naive time-integral of
them doesn't track the outcome either, and the actual time-resolved
energy share shows seed=3000's winner is behind for roughly the first
half of the window before overtaking -- so the *why*, at the level of
the full time-ordered propagator, remains genuinely open rather than
resolved by any of the diagnostics tried so far (snapshot, naive
integral, or the per-node share curves in Part 3).

This also does not extend beyond Stage 1B.2's own existing scope
limits: one class, one topology (T only, no graph controls), and the
432 trials already collected. It is a finer-grained characterization of
data Stage 1B.2 already gathered, not a new experiment.

## Relationship to Stage 1B.2's existing findings

`FINDINGS.md` already documents that the response is "not reducible to
the directly stimulated coordinate" and characterizes redistribution in
aggregate (source-energy-fraction over time). This note adds a
*where*-and-*when* characterization that FINDINGS.md's aggregate
treatment did not attempt: most trials genuinely spread broadly, but one
specific, reliably-reproduced (node, t_p) combination instead
concentrates the response deterministically onto a single other node
(or, for one sign/amplitude combination within that cell, a different
single node) -- and that routing is already present in the purely linear
part of the response, with the nonlinear part modulating but not
creating it. This refines, but does not contradict or require reopening,
the frozen finding.

## Further follow-up: mechanism of the destination, and why it doesn't generalize

Worked interactively in `concentration_regime_notebook.ipynb` (committed
alongside this note); summarized here for the permanent record.

**Is the destination explained by direct adjacency in `T`?** Checked
directly: node 152 is a genuine, if weak, direct neighbor of the
high-degree source (rank 6 of its 7 real edges). Node 103 -- the
dominant destination, in 5 of the 6 concentrated (sign, amplitude)
conditions for seed=3000 -- is **not a meaningful direct neighbor at
all**: rank #201 of ~505 possible nodes, when the source has only 7
nonzero edges. Direct adjacency does not explain the dominant case.

**Is it explained by a 2-hop relay in the static graph?** Yes, for
seed=3000 specifically: of the source's 7 direct neighbors, exactly one
(node 105) has any nonzero edge to 103, and it's strong (0.909, close to
the source's own strongest direct edges). All 6 other neighbors have
zero weight to 103. So seed=3000's routing is a clean two-edge chain,
source(129) -> relay(105) -> destination(103), not a diffuse
convergence.

**Does this 2-hop pathway generalize across trajectories?** No.
Checked the identical (high-degree-node, t_p=0) cell across all 10 of
Stage 1C's baseline trajectories (seed=3000 plus the 9 independent
trajectories Stage 1C added). `T`'s edges -- including this 2-hop bridge
-- are identical across every trajectory, yet the outcome varies wildly:

| Trajectory | Concentrated (of 36) | Destination |
|---|---|---|
| 3000 | 24 | 103 |
| 3010 | 21 | 130 |
| 3020 | 2 | 35 |
| 3030-3070 (5 trajectories) | 0 | -- |
| 3080 | 5 | 55 |
| 3090 | 35 | 152 |

Five of the nine new trajectories show no concentration at all; the four
that do land on four different destinations, never 103 again. Since the
static graph can't change between trajectories, the 105-relay pathway is
best read as explaining *this one trajectory's* outcome specifically,
not a fixed pathway the topology always routes through.

**What actually explains the trajectory-to-trajectory difference: the
phase-dependent Jacobian, not the static graph.** The tangent dynamics
are governed by $J_{ij}(t) = W_{ij}\cos(\theta_j(t)-\theta_i(t))$
(`force_jacobian` in `graph_oscillator_field.py`), not by $W$ alone.
Comparing $J(0)$ across three trajectories -- seed=3000 (->103, 24/36),
seed=3090 (->152, 35/36), seed=3030 (no concentration, 0/36) -- along
the two candidate pathways (source->relay->103, and source->152 direct):

| Trajectory | J[129,105] | J[105,103] | J[129,152] | Outcome |
|---|---|---|---|---|
| 3000 | -0.727 (strong) | -0.710 (strong) | +0.643 | Both legs of the 105-relay open -> routes to 103 |
| 3030 | +0.906 (strong) | **+0.074 (weak)** | -0.531 | First leg open, second leg bottlenecked -> no concentration |
| 3090 | **+0.141 (weak)** | -0.865 (strong) | -0.747 (strong) | First leg bottlenecked, but direct edge to 152 strong -> routes to 152 instead |

The same static graph produces three different outcomes because the
phase-dependent cosine term opens or closes different edges of the same
candidate pathways at each trajectory's own $t_p=0$ phase state.
Seed=3030's relay pathway fails at the second hop despite having the
strongest first hop of the three trajectories -- a bottleneck invisible
from $W$ alone. Seed=3090's relay pathway fails at the first hop
instead, but a different, structurally-real pathway (the direct edge to
152) happens to be strong at that trajectory's phase state and wins.
This is a direct quantitative account of the initial pathway gating that
distinguishes the three observed outcomes. It supports the conclusion
that concentration is "an emergent property of the state-dependent
network propagator, not static graph geometry," without claiming that
the full time-ordered propagator has been decomposed.

**One technical detail carried through all of the above, not previously
noted in this document**: the perturbation is not a pure delta at the
source node. `stage1b2_core.py` projects the initial impulse through the
rotation-removal projector $P$ before use (`delta0 = P @ e_{node}`, then
renormalized), so every node receives some small component, not only the
stimulated one. This doesn't change any of the findings above, but "the
input" throughout this document and `concentration_regime_notebook.ipynb`
should be understood as a rotation-free projected impulse, not a literal
single-node spike.

**Still not established**: why seed=3000's and seed=3090's specific
phase states happen to open the pathways they do -- i.e., whether
there's a deeper pattern to which trajectories open which pathways, or
whether this is effectively idiosyncratic per-trajectory phase
alignment with no further structure to find. Not pursued here.
