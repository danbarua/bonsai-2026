# Stage 1D, Part 2: Stochastic-Control Pilot

> **Runtime and variance-allocation pilot. No confirmatory
> topology-specificity inference is drawn from this run.** (DESIGN.md's
> own required framing, restated here verbatim.) Nothing below is a
> claim about whether T outperforms rewired/hist_random/curr_random --
> only about how big the confirmatory run needs to be to test that
> question with adequate power.

## Scope

3 graph realizations (seeds **0, 1, 2** -- chosen for this pilot; not
pinned by DESIGN.md, which fixes trajectory seeds but deliberately
leaves graph-realization seeds open) x the first 3 of Stage 1C's
matched trajectory seeds (3000, 3010, 3020), for each of the three
stochastic controls:

- **rewired**: degree-preserving rewiring (`degree_preserving_rewire`)
- **hist_random**: historical half-edge random, coupling-budget
  normalized (`generate_historical_matched_sparsity_random` +
  `rescale_to_common_budget`)
- **curr_random**: current edge-count-matched random
  (`generate_matched_sparsity_topology`)

Fixed-coordinate intervention only (T's own node indices 17/363/129),
per the task's own explicit narrowing of scope -- role-matched
intervention is secondary/robustness per DESIGN.md and isn't needed to
size the primary comparison. **No disagreement with that scoping call**:
sizing the confirmatory run only needs the fixed-coordinate variance
structure, since that is the primary family DESIGN.md locks Holm
correction over.

T's own topology and ink_mask were reconstructed from the raw KMNIST
class-0 images (first 200 training images, `build_class_topology`'s
defaults) rather than loaded from the cached
`class0_constructions.pkl`, because the cached artifact doesn't retain
`ink_mask` -- required to build the three stochastic controls' fresh
realizations. This reconstruction is verified byte-exact against the
cached T (max abs diff 2.22e-16) before use.

## A real finding surfaced by the pilot itself: fixed-coordinate degeneracy

**hist_random realization seed=2 produced a fully degenerate (NaN)
Delta_map for all 3 of its matched trajectories.** Diagnosis, not a
code bug: T's own fixed 'low' (index 17) and 'high' (index 129) node
indices both happened to land with **weighted degree exactly 0** in
this specific random draw (hist_random places only ~half T's edge
count, independently). An isolated node's response under the
oscillator dynamics is (near-)perfectly linear, so it drives the
tangent-departure diagnostic `E` below `E_min` at every perturbation
time except t_p=0 -- confirmed directly: `event_aligned_valid` was
True for the 'low' and 'high' node labels in only 12/36 trials at t_p=0
and **0/36 at every other t_p**, while 'median' (not isolated) stayed
36/36 valid throughout. With 2 of the 3 node labels invalid, **every**
cross-node-label pairing that `B_node` needs
(`(low,median)`, `(low,high)`, `(median,high)`) loses at least one
side, so `B_node` -- and therefore `Delta_map` -- is undefined for that
realization, at every trajectory.

Two milder, non-fatal instances of the same phenomenon also occurred:
`hist_random` r=0 and `curr_random` r=0 each had **one** isolated node
('low' only). This did not break `Delta_map` (the surviving
`(median, high)` pair still contributes to `B_node`), but is disclosed
here as a caveat on those two realizations' own `B_node` estimate,
which is now supported by fewer valid comparisons than usual.

**Why this doesn't happen to `rewired`**: degree-preserving rewiring
holds each node's exact (unweighted) degree fixed by construction, so
it can never isolate a node T itself didn't already have at that
degree -- T's own 'low' node has degree 1.83 (low, but nonzero), and
every rewired realization preserved that. Both `hist_random` and
`curr_random` place edges by independent resampling, so a node with
few edges in T has a real, nontrivial chance of drawing zero edges
under independent re-placement, especially at (roughly) half T's edge
density (`hist_random`).

**Handling**: hist_random's degenerate realization (seed=2) is
excluded from its family's crossed-variance fit -- not imputed, not
silently averaged over. This mirrors DESIGN.md's own precedent for
degenerate role-matching (disclose the reduced sample rather than pad
it). hist_random's own variance decomposition is therefore fit on only
2 of 3 realizations and flagged **indeterminate** below.

## Raw per-realization Delta_map values

`d_grk = Delta_map(T,k) - Delta_map(g,r,k)`, rows = realizations (r=0,1,2), columns = trajectory seeds (3000, 3010, 3020):

**rewired**
| r \ k | 3000 | 3010 | 3020 |
|---|---:|---:|---:|
| 0 | 0.0241 | -0.0036 | 0.0110 |
| 1 | 0.0205 | 0.0147 | -0.0649 |
| 2 | 0.0092 | -0.0316 | -0.0718 |

d_bar_gr (mean over k) = [0.0105, -0.0099, -0.0314]

**hist_random**
| r \ k | 3000 | 3010 | 3020 |
|---|---:|---:|---:|
| 0 | 0.0421 | 0.0504 | -0.0241 |
| 1 | 0.0328 | 0.0046 | -0.0030 |
| 2 | **NaN** | **NaN** | **NaN** |

d_bar_gr (r=0,1 only) = [0.0228, 0.0115]; r=2 excluded (degenerate, see above)

**curr_random**
| r \ k | 3000 | 3010 | 3020 |
|---|---:|---:|---:|
| 0 | 0.0559 | -0.0085 | -0.0058 |
| 1 | -0.0160 | -0.0380 | -0.0403 |
| 2 | -0.0025 | 0.0413 | -0.0160 |

d_bar_gr (mean over k) = [0.0139, -0.0315, 0.0076]

(All raw per-cell `Delta_map(g,r,k)` values, and every trajectory's own
10,000-permutation validation p-value, are in
`results/stage1d_pilot_analysis.pkl` / the individual
`stage1d_pilot_<family>_r<r>_seed<k>.pkl` checkpoints -- every valid
cell hit the 0.00010 permutation floor, same as T and lattice.)

## Crossed variance decomposition

Fit via the balanced two-way ANOVA method-of-moments estimator for
`d_grk = mu_g + b_gr + tau_k + epsilon_grk` (exact/unbiased for a
balanced crossed design, verified against synthetic data with known
variance components before being applied here -- see commit history).

| family | R used | point sigma^2_b | point sigma^2_tau | point sigma^2_eps | df_r | df_resid |
|---|---:|---:|---:|---:|---:|---:|
| rewired | 3 | 0.000206 | 0.000671 | 0.000698 | 2 | 4 |
| hist_random | **2** (r=2 excluded) | 0.000000 | 0.000451 | 0.000561 | **1** | **2** |
| curr_random | 3 | 0.000341 | 0.000014 | 0.000788 | 2 | 4 |

**Conservative variance estimates** (per DESIGN.md's locked rule: 95%
upper confidence bound via the chi-squared pivot where the relevant
mean-square's denominator df >= 3 -- our own operational threshold for
"reliably estimable" -- else the named 2x-point-estimate fallback):

| family | sigma^2_b conservative | method | sigma^2_eps conservative | method |
|---|---:|---|---:|---|
| rewired | 0.000412 | 2x fallback (df_r=2 < 3) | 0.003931 | chi2 upper CI (df_resid=4) |
| hist_random | 0.000000 | 2x fallback (df_r=1 < 3) | 0.001123 | 2x fallback (df_resid=2 < 3) |
| curr_random | 0.000681 | 2x fallback (df_r=2 < 3) | 0.004433 | chi2 upper CI (df_resid=4) |

hist_random's `sigma^2_b` point estimate is exactly 0 (MS_r < MS_resid
on just 2 realizations) -- reported as-is, not floored to a nonzero
value, since inventing a floor beyond DESIGN.md's own named fallback
would not be a disclosed, prespecified rule.

## Pilot-to-confirmatory allocation

Locked parameters: `delta_min=0.05`, power target 80%, familywise alpha
0.05 across the 4 fixed-coordinate comparisons, Holm approximated as
**option (b)**: alpha=0.05/4=0.0125 per comparison (chosen over jointly
simulating the full 4-comparison Holm procedure, option (a), for
tractability within the pilot's timeframe -- both are sanctioned by
DESIGN.md). Candidate grid R in {10,15,20,25} x K in {3,5,7,10}.

| family | own minimal (R,K) | cost | power at that design | reliable? |
|---|---|---:|---:|---|
| rewired | (15, 3) | 45 | 0.948 | yes |
| hist_random | (15, 3) | 45 | >0.9999 | **no -- indeterminate** |
| curr_random | (15, 3) | 45 | 0.883 | yes |

**hist_random's own requirement is not usable as-is.** Its
near-perfect nominal "power" comes from `sigma^2_b_conservative=0`,
itself an artifact of fitting on only 2 realizations (df_r=1) after
excluding a degenerate draw -- not evidence that hist_random's true
between-realization variance is genuinely near zero. This is reported
honestly as **indeterminate**, not averaged in as if it were a normal
result.

**Locked common (R, K) = (15, 3), cost 45**, selected from the two
*reliable* families (rewired and curr_random), which independently
landed on the identical design -- both requiring R=15 realizations x
K=3 trajectories to hit 80% power for delta_min=0.05 at
alpha=0.0125. hist_random is excluded from this selection, not
because it needs less, but because this pilot cannot say what it
needs.

## Recommendation flagged for follow-up (not acted on here)

Before locking hist_random into the confirmatory run at (15, 3), a
cheap follow-up worth doing: redraw hist_random's degenerate
realization with a different seed (or draw one or two extra
diagnostic realizations) to get a real, non-degenerate 3rd data point
for its own variance estimate. If hist_random's re-estimated variance
turns out to demand a larger (R, K) than (15, 3), that should update
the common design before the confirmatory run starts, per DESIGN.md's
"lock the final (R, K) from this rule's output *before* looking at any
confirmatory-run results." Proceeding with (15, 3) as a provisional
common design in the meantime is reasonable given two of three families
agree exactly, but this gap should not be silently forgotten.
