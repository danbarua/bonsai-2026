Simplified Technical English version of `experiments/stage1d_topology_specificity/PILOT_RESULTS.md`.

# Stage 1D, Part 2: Random-Control Pilot

> **This is a pilot test for runtime and for dividing the test budget.
> It does not give a confirmed result about topology specificity.**
> (This restates DESIGN.md's own required framing, word for word.)
> Nothing below is a claim that T outperforms rewired, hist_random, or
> curr_random. It only tells the team how large the confirmed test run
> needs to be, to test that question with enough power.

## Scope

The team used 3 graph versions (seeds **0, 1, 2**). These were chosen
for this pilot only. DESIGN.md does not fix these seeds. It fixes the
trajectory seeds, but leaves the graph-version seeds open on purpose.
The team crossed these 3 graph versions with the first 3 of Stage 1C's
matched trajectory seeds (3000, 3010, 3020). They did this for each of
the three random controls:

- **rewired**: degree-preserving rewiring (`degree_preserving_rewire`).
  This keeps each node's exact unweighted degree, but swaps which
  nodes connect to which. Note: this project uses the word "rewired"
  for three different graphs in different parts of the project. In
  this document, and everywhere in Stage 1D, "rewired" means only this
  one construction: a double-edge-swap that keeps T's unweighted
  degree sequence. It is not the same graph as the other two "rewired"
  constructions used elsewhere in this project.
- **hist_random**: historical half-edge random, adjusted so total
  coupling weight matches T's
  (`generate_historical_matched_sparsity_random` plus
  `rescale_to_common_budget`). This is a rebuild of an older
  random-control graph from before this project's current tools.
- **curr_random**: current edge-count-matched random
  (`generate_matched_sparsity_topology`). This keeps T's exact edge
  count but places weights differently.

The team used only the fixed-coordinate intervention here (T's own
node indices 17, 363, and 129). This follows the task's own narrower
scope. Role-matched intervention is a secondary, robustness item per
DESIGN.md. It is not needed to size the main comparison. **This is not
a disagreement with that scoping choice**: sizing the confirmed test
run only needs the fixed-coordinate variance pattern. That is the main
family DESIGN.md locks the Holm correction over.

The team rebuilt T's own topology and ink_mask from the raw KMNIST
class-0 images (the first 200 training images, using
`build_class_topology`'s default settings). They did not load these
from the stored `class0_constructions.pkl` file, because that file
does not keep `ink_mask`. The team needed `ink_mask` to build the
three random controls' fresh versions. They checked this rebuilt T
exactly matches the stored T, byte for byte (largest difference:
2.22e-16), before using it.

## A Real Finding From the Pilot: Fixed-Coordinate Degeneracy

**Hist_random version seed=2 produced a fully undefined (NaN)
Delta_map for all 3 of its matched trajectories.** This is a real
finding, not a code bug. T's own fixed 'low' node (index 17) and
'high' node (index 129) both happened to have **weighted degree
exactly 0** in this specific random draw. (Hist_random places only
about half of T's edge count, and does so independently for each
version.)

An isolated node's response under the oscillator dynamics is close to
perfectly linear. This drives the tangent-departure diagnostic, `E`,
below the threshold `E_min` at every perturbation time except t_p=0.
The team confirmed this directly: `event_aligned_valid` was True for
the 'low' and 'high' node labels in only 12 of 36 trials at t_p=0, and
**0 of 36 trials at every other t_p**. The 'median' node label, which
was not isolated, stayed valid in 36 of 36 trials throughout.

With 2 of the 3 node labels invalid, every pairing that `B_node` needs
(`(low,median)`, `(low,high)`, `(median,high)`) loses at least one
side. So `B_node`, and therefore `Delta_map`, is undefined for this
version, at every trajectory.

Two milder, non-fatal cases of the same problem also happened:
`hist_random` version r=0 and `curr_random` version r=0 each had
**one** isolated node ('low' only). This did not break `Delta_map`
(the surviving `(median, high)` pair still contributes to `B_node`).
But the team reports this as a caveat on those two versions' own
`B_node` value, which now rests on fewer valid comparisons than usual.

**Why this does not happen to `rewired`**: degree-preserving rewiring
keeps each node's exact unweighted degree fixed by design. So it can
never isolate a node that T itself did not already have at that same
degree. T's own 'low' node has degree 1.83 (a low value, but not
zero), and every rewired version kept that degree. Both `hist_random`
and `curr_random` place edges by independent resampling. So a node
with few edges in T has a real, non-trivial chance of drawing zero
edges when placed again independently. This is especially true at
roughly half T's edge density, which is the case for `hist_random`.

**How the team handled this**: the team left hist_random's fully
undefined version (seed=2) out of that family's crossed variance fit.
They did not fill in a guessed value, and did not silently average
over it. This follows DESIGN.md's own precedent for handling
degenerate role-matching: report the smaller sample honestly, rather
than pad it. So hist_random's own variance breakdown is fit on only 2
of its 3 versions, and is flagged **not reliable** below.

## Raw Delta_map Values per Version

`d_grk = Delta_map(T,k) - Delta_map(g,r,k)`. Rows are graph versions
(r=0,1,2). Columns are trajectory seeds (3000, 3010, 3020):

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

d_bar_gr (r=0,1 only) = [0.0228, 0.0115]; r=2 left out (undefined, see
above)

**curr_random**
| r \ k | 3000 | 3010 | 3020 |
|---|---:|---:|---:|
| 0 | 0.0559 | -0.0085 | -0.0058 |
| 1 | -0.0160 | -0.0380 | -0.0403 |
| 2 | -0.0025 | 0.0413 | -0.0160 |

d_bar_gr (mean over k) = [0.0139, -0.0315, 0.0076]

(All raw per-cell `Delta_map(g,r,k)` values, and every trajectory's
own 10,000-permutation validity p-value, are in
`results/stage1d_pilot_analysis.pkl` and the individual
`stage1d_pilot_<family>_r<r>_seed<k>.pkl` checkpoint files. Every
valid cell hit the 0.00010 permutation floor, the same as T and
lattice.)

## Crossed Variance Breakdown

The team fit this using the balanced two-way ANOVA method-of-moments
estimator. This estimator fits the model
`d_grk = mu_g + b_gr + tau_k + epsilon_grk`. This method gives an
exact, unbiased fit for a balanced crossed design. The team checked
this method works correctly on test data with known variance values,
before using it here (see the commit history).

| family | R used | point sigma^2_b | point sigma^2_tau | point sigma^2_eps | df_r | df_resid |
|---|---:|---:|---:|---:|---:|---:|
| rewired | 3 | 0.000206 | 0.000671 | 0.000698 | 2 | 4 |
| hist_random | **2** (r=2 left out) | 0.000000 | 0.000451 | 0.000561 | **1** | **2** |
| curr_random | 3 | 0.000341 | 0.000014 | 0.000788 | 2 | 4 |

**Careful (conservative) variance estimates.** (This follows
DESIGN.md's locked rule: use a 95% upper bound from the chi-squared
method, where the relevant value's denominator degrees of freedom is
3 or more. This project's own threshold for "reliably estimable."
Otherwise, use the named fallback of doubling the point estimate.)

| family | sigma^2_b careful value | method | sigma^2_eps careful value | method |
|---|---:|---|---:|---|
| rewired | 0.000412 | 2x fallback (df_r=2 < 3) | 0.003931 | chi2 upper CI (df_resid=4) |
| hist_random | 0.000000 | 2x fallback (df_r=1 < 3) | 0.001123 | 2x fallback (df_resid=2 < 3) |
| curr_random | 0.000681 | 2x fallback (df_r=2 < 3) | 0.004433 | chi2 upper CI (df_resid=4) |

Hist_random's `sigma^2_b` point estimate is exactly 0. (This is
because MS_r is smaller than MS_resid, on just 2 versions.) The team
reports this value as-is. They did not set a floor above zero, because
inventing a floor beyond DESIGN.md's own named fallback would not be a
disclosed, planned-in-advance rule.

## Pilot-to-Confirmed-Run Allocation

Locked settings: `delta_min=0.05` (minimum meaningful difference),
power target 80%, family-wide error rate 0.05 across the 4
fixed-coordinate comparisons. The team approximated Holm correction
using **option (b)**: alpha = 0.05/4 = 0.0125 per comparison. The team
chose this over simulating the full 4-comparison Holm procedure
directly (option (a)), because option (a) was not practical within
the pilot's time limit. DESIGN.md allows both options. The candidate
grid was R in {10, 15, 20, 25} crossed with K in {3, 5, 7, 10}.

| family | own minimal (R,K) | cost | power at that design | reliable? |
|---|---|---:|---:|---|
| rewired | (15, 3) | 45 | 0.948 | yes |
| hist_random | (15, 3) | 45 | >0.9999 | **no -- not reliable** |
| curr_random | (15, 3) | 45 | 0.883 | yes |

**Hist_random's own result here cannot be used as-is.** Its
near-perfect nominal "power" value comes from
`sigma^2_b_conservative=0`. This value itself is an artifact of
fitting on only 2 versions (with 1 degree of freedom for realizations),
after leaving out one fully undefined draw. It is not evidence that
hist_random's true between-version variance is genuinely near zero.
The team reports this honestly as **not reliable**, not averaged in as
if it were a normal result.

**Locked common (R, K) = (15, 3), cost 45.** The team chose this from
the two *reliable* families (rewired and curr_random), which both
independently arrived at the same design. Both families need R=15
versions x K=3 trajectories to reach 80% power for delta_min=0.05 at
alpha=0.0125. The team left hist_random out of this choice. This is
not because hist_random needs less, but because this pilot cannot yet
say what hist_random needs.

## Recommendation Flagged for Follow-Up (Not Done Here Yet)

Before locking hist_random into the confirmed run at (15, 3), one
cheap follow-up step is worth doing: redraw hist_random's undefined
version with a different seed (or draw one or two extra versions) to
get a real, non-degenerate 3rd data point for its own variance
estimate. If hist_random's re-estimated variance needs a larger (R, K)
than (15, 3), the common design should be updated before the confirmed
run starts. This follows DESIGN.md's rule: "lock the final (R, K) from
this rule's output *before* looking at any confirmatory-run results."
Using (15, 3) as a temporary common design in the meantime is
reasonable, since two of the three families agree exactly. But this
gap should not be quietly forgotten.

## Follow-Up: Hist_random Variance Re-Check (Seeds 3, 4)

**This section adds evidence gathered after the fact. The numbers and
the "not reliable" finding above are unchanged. They still reflect the
original 3x3 pilot exactly as it was run.** This section acts on the
recommendation just above. The team drew 2 more hist_random graph
versions (seeds 3, 4, continuing the 0/1/2 sequence). They built these
with the exact same construction method and ran them against the same
3 matched trajectory seeds (3000, 3010, 3020). The team ran the full
432-trial simulation, computed Delta_map, and ran the
10,000-permutation check, exactly as in the original pilot.

**Check for isolated nodes, before running any simulation:**

| seed | degree at 'low' | degree at 'median' | degree at 'high' | isolated? |
|---|---:|---:|---:|---|
| 3 | 1.915 | 3.712 | 1.837 | no |
| 4 | 1.807 | 7.480 | 3.800 | no |

**Neither seed 3 nor seed 4 repeated seed=2's isolated-node problem.**
Both versions' simulations came back fully defined.
(`event_aligned_valid` never dropped to 0 for any node label at any
t_p.) All 6 new trajectories hit the usual 10,000-permutation floor
(p=0.00010).

Taken together, of the 5 hist_random versions drawn so far (seeds
0-4): 1 (seed=2, 20%) was **fully** undefined. 1 (seed=0) was
**mildly** undefined (one isolated node, 'low', that did not break
`Delta_map`; see the original pilot section above). 3 (seeds 1, 3, 4)
showed no isolated fixed-coordinate node at all. The team reports this
as it is: not frequent enough to call the fixed-coordinate method
broken for this construction, but not a one-time fluke either (2 of 5
draws showed *some* isolated node). This is a real, low-probability-
but-not-negligible interaction between T's own sparsest fixed node and
hist_random's roughly half-density independent resampling.

**Refitted crossed variance breakdown**, using the 4 valid versions
(seeds 0, 1, 3, 4; seed 2 is still left out, unchanged):

| | R used | point sigma^2_b | point sigma^2_tau | point sigma^2_eps | df_r | df_resid |
|---|---:|---:|---:|---:|---:|---:|
| hist_random (refit) | **4** | 0.000013 | 0.000784 | 0.000440 | **3** | **6** |

The realization degrees of freedom, df_r, is now 3. This clears this
project's own reliability threshold (3 or more). So **both** variance
parts now get a proper 95% chi-squared upper bound. Neither one needs
the doubling fallback anymore:

| | sigma^2_b careful value | method | sigma^2_eps careful value | method |
|---|---:|---|---:|---|
| hist_random (refit) | 0.004069 | chi2 upper CI (df_r=3) | 0.001613 | chi2 upper CI (df_resid=6) |

**Hist_random's own minimal design, refitted: (R=25, K=3), cost 75,
power=0.827.** The team confirmed this is the true minimum over the
full candidate grid. (Nothing cheaper reaches 80% power; for example,
(20,3) only reaches 0.700.) This is **larger** than the common design
currently locked above, (15, 3), cost 45, and larger than rewired's
and curr_random's own requirements, which this follow-up leaves
unchanged.

**Per this task's own instruction, the common (R, K) is NOT updated
here.** Hist_random alone, now reliably estimated, needs (R=25, K=3).
That is 67% more total trajectory-runs (75 versus 45) than the
currently locked common design. Whether to raise the common design to
(25, 3) for all three families, or to handle hist_random differently,
is an explicit decision left for the next step. This follow-up does
not resolve it alone.

**Resolved in a later step**: the common design has since been raised
to (R=25, K=3) for all three random-control families. This is a
mechanical result of the existing selection rule, now that
hist_random's estimate is reliable. It is not a new judgment call. See
`DESIGN.md`, "Locked Confirmed-Run Allocation: (R=25, K=3)," which
replaces the earlier temporary (15, 3) lock above.

Raw data for this follow-up (all 6 new trajectories' Delta_map values,
the combined 5-version `d_grk` table, and the full candidate-design
grid) are in `results/stage1d_hist_random_followup.pkl`.
