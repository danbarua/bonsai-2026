# Follow-up note: when does the response relocate to one other node vs. spread broadly?

**Status: a scoped follow-up observation on already-frozen Stage 1B.2
data, not a revision of [Stage 1B.2 findings](FINDINGS.md).** `FINDINGS.md` and Stage 1B.2's
own conclusions are unchanged by this note. This document exists
separately and should be read as an addendum, not an update.

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
of the vector residual $z = x_{\text{finite}} - x_{\text{tangent}}$, also
already saved) for the same 36-trial cell. In other words,
$q_{\text{residual}} = \operatorname{normalized\_energy}(z)$, not
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

**Does not establish**: *why* the high-degree node specifically routes
to node 103 at t_p=0 and not at other t_p values. One plausible, but
untested, interpretation: t_p=0 perturbations act on the baseline
trajectory before it has had time to settle toward an attractor
(consistent with Stage 0's own multistability finding -- the system
takes time to converge from a fresh random initial condition), which
could plausibly change the local linearization's routing structure
compared to a perturbation applied once the trajectory has settled. This
note does not test that mechanism -- it is offered as a plausible
reading, not a confirmed explanation, and per Part 2's result, does not
need to invoke nonlinear attractor-switching to explain the core pattern.

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
