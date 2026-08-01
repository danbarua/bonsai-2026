# Follow-up note: when does the response relocate to one other node vs. spread broadly?

**Status: a scoped follow-up observation on already-frozen Stage 1B.2
data, not a revision of `FINDINGS.md`.** `FINDINGS.md` and Stage 1B.2's
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

then checked whether these vary systematically with amplitude, t_p
(perturbation time), sign, or stimulated node, versus being roughly
uniform across all 432 trials. Code:
`analyze_stage1b2_concentration_regime.py`.

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
spanning all combinations of sign and amplitude): `top1` ranges from
0.181 to 0.976, mean 0.622, with 24 of 36 trials (67%) exceeding 0.5
(i.e., a single node holding more than half the total response energy).
**Across the other 396 trials in the entire design, `top1` never exceeds
0.415** -- there is a hard gap between this one cell and every other
combination of factors. The two dead-end trial examples were both drawn
from exactly this cell (node='high', t_p=0), which is why they looked
the way they did -- but they weren't an unrepresentative pair; they were
two ordinary draws from a genuinely distinct, reliably-reproduced regime
that shows up in 67% of that cell's 36 trials.

## What this establishes, and what it doesn't

**Establishes**: for this class-0 learned topology, under Stage 1B.2's
exact design, perturbing the highest-weighted-degree node at the very
start of the baseline trajectory (t_p=0) produces a qualitatively
different final-timepoint response -- energy relocating onto one other
specific node rather than spreading broadly -- reliably (67% of the
time) and specifically to that one (node, t_p) combination. This does
not happen for the low- or median-degree nodes at any t_p, or for the
high-degree node at any t_p other than 0. Amplitude and sign play no
detectable role in which regime a trial falls into.

**Does not establish**: *why* this happens. One plausible, but untested,
interpretation: t_p=0 perturbations act on the baseline trajectory
before it has had time to settle toward an attractor (consistent with
Stage 0's own multistability finding -- the system takes time to
converge from a fresh random initial condition), so a large-amplitude
disturbance to the most-connected node at that moment may be more able
to redirect the trajectory toward a different attractor's basin than the
same disturbance would be once the trajectory has already settled. This
note does not test that mechanism -- it is offered as a plausible
reading of the pattern, not a confirmed explanation.

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
concentrates the response onto a single other node. This refines, but
does not contradict or require reopening, the frozen finding.
