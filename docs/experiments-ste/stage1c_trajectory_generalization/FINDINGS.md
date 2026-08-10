Simplified Technical English version of `experiments/stage1c_trajectory_generalization/FINDINGS.md`

# Stage 1C: Generalization Across Trajectories

## The question

Stage 1B2 found a structured internal transformation. This means the
network turns a small local perturbation into a spatial pattern of
response. The result was significant. It depended on the specific
input. It was locally reproducible.

Stage 1B2 tested only one baseline trajectory. A trajectory is one
full run of the oscillator network from a starting point, through
time. This one trajectory came from KMNIST class 0, the T topology
(the graph learned from that class's images), and seed=3000.

Stage 1B2's own findings stated a clear scope limit. The result was
conditional on that one trajectory. It was not yet shown to
generalize to other trajectories. See the "Scope, essential" section
of `stage1b2_structured_transformation/FINDINGS.md`, and item 1 of
`docs/PROJECT_MEMORY.md`'s "what remains open" list.

Stage 1C asks this question directly. Does the same design, run on
independent baseline trajectories on the same topology, produce the
same result? Or was seed=3000 just a lucky draw?

## Design

Stage 1C uses the exact same design as Stage 1B2. The only difference
is the baseline trajectory itself.

Each trajectory tests:

- 3 nodes, chosen by weighted-degree rank in the topology: low,
  median, and high.
- 2 signs: positive and negative.
- 3 amplitudes: 0.025 (tangent), 0.2 (intermediate), 0.8 (nonlinear).
- Together, these give 18 inputs.
- 4 perturbation times, t_p, in {0, 0.833, 1.667, 2.5}. t_p is the
  point along the baseline trajectory where the team injects a
  perturbation.
- 6 fixed nearby-state replicas per input. A replica is a slightly
  nudged version of the same moment on the same path. It checks that
  a finding does not depend on one exact starting phase. A replica
  is not a repeat of the same trial.
- Together, this gives 432 trials per trajectory.

Every trajectory uses the same KMNIST class-0 T topology. Every
trajectory uses the same permutation test. A permutation test checks
significance by comparing the real result against many randomly
reshuffled versions of the data. Here, the test independently
reshuffles labels per replica, uses 10,000 reshuffles (permutations),
and reports a one-sided Monte Carlo p-value (labeled p_MC).

**10 baseline trajectories.** These are seed=3000 (Stage 1B2's own
reference trajectory) plus 9 new, independent seeds: 3010, 3020,
3030, 3040, 3050, 3060, 3070, 3080, and 3090. Each seed pairs with a
replica-direction seed of baseline+1. So seed 3010 pairs with 3011,
seed 3020 pairs with 3021, and so on up to seed 3090 pairing with
3091. This matches Stage 1B2's own convention: BASELINE_SEED=3000
paired with REPLICA_DIRECTION_SEED=3001.

**Implementation.** `run_stage1c.py` and `analyze_stage1c.py` import
their core logic directly from
`stage1b2_structured_transformation/stage1b2_core.py` and
`analyze_stage1b2.py`. The imported functions are
`get_degree_stratified_nodes`, `generate_reference_baseline`,
`generate_fixed_replica_directions`, `run_one_trial`,
`load_results_as_arrays`, and `run_permutation_test`. Stage 1C does
not copy this code.

This guarantees Stage 1C runs the exact same trial and analysis
logic that Stage 1B2's result rests on. It is not a reimplementation
that could silently drift from the original.

Stage 1C reads seed=3000's 432 trial results directly, read-only,
from Stage 1B2's own committed file, `results/stage1b2_results.pkl`.
Stage 1C does not re-run this trajectory. This keeps Stage 1B2
genuinely frozen as the reference. The 9 new trajectories are Stage
1C's own. Stage 1C caches them independently, in
`results/stage1c_results_seed<N>.pkl`.

**Runtime.** The team measured runtime directly before committing to
the full run. They did not assume it. One trajectory (seed=3010)
took 106.6 seconds end to end, using 432 trials and 9 worker
processes. This was much faster than either earlier estimate. The
original design note expected 15 to 20 minutes. Stage 1B2's own
`run_stage1b2.py` docstring estimates 35 to 45 minutes. All 9 new
trajectories together took under 14 minutes. The permutation
analysis (10,000 permutations across 10 trajectories) added the
rest of the time.

## Results for each trajectory

| Seed | Pooled Delta_map | p_MC | Delta_map, t_p=0 | t_p=0.833 | t_p=1.667 | t_p=2.5 |
|---|---|---|---|---|---|---|
| 3000 (Stage 1B2 reference) | 0.3505 | 0.00010 | 0.3955 | 0.3215 | 0.3395 | 0.3456 |
| 3010 | 0.3318 | 0.00010 | 0.3725 | 0.3122 | 0.3082 | 0.3342 |
| 3020 | 0.2964 | 0.00010 | 0.3174 | 0.2403 | 0.3099 | 0.3180 |
| 3030 | 0.3402 | 0.00010 | 0.3284 | 0.3548 | 0.3276 | 0.3500 |
| 3040 | 0.3389 | 0.00010 | 0.3702 | 0.3409 | 0.3256 | 0.3189 |
| 3050 | 0.3329 | 0.00010 | 0.3478 | 0.3140 | 0.3156 | 0.3542 |
| 3060 | 0.3486 | 0.00010 | 0.3898 | 0.3227 | 0.3312 | 0.3508 |
| 3070 | 0.3077 | 0.00010 | 0.3137 | 0.2947 | 0.3073 | 0.3152 |
| 3080 | 0.3237 | 0.00010 | 0.3308 | 0.3031 | 0.3086 | 0.3524 |
| 3090 | 0.3253 | 0.00010 | 0.3164 | 0.3410 | 0.3378 | 0.3061 |

Every one of the 40 individual t_p-level Delta_map values, across
all 10 trajectories, is positive. Every value is clearly above zero.
The range is 0.24 to 0.40. No trajectory has a t_p where the mapping
becomes non-significant. No trajectory has a t_p where the mapping
reverses direction.

## Results across all 10 trajectories

- Mean pooled Delta_map: 0.3296 (median 0.3324).
- Range: 0.2964 to 0.3505 (span 0.0541).
- Standard deviation: 0.0172 (coefficient of variation about 5.2%).
  This is a tight cluster of values. It is not a wide scatter with a
  few outliers pulling the mean.
- 10 of 10 trajectories hit the Monte Carlo floor. This means p_MC =
  0.00010, or p = 1/10001, the smallest value possible at 10,000
  permutations. Zero trajectories failed to produce a significant
  result.
- Seed=3020 is the weakest trajectory by pooled Delta_map (0.2964).
  It also contains the single lowest individual t_p value in the
  whole dataset (t_p=0.833, Delta_map=0.2403). This value is still
  clearly positive. It is still significant at the Monte Carlo
  floor. It is not a borderline or failing case. It is simply the
  low end of an otherwise tight distribution.

## What this result shows

Structured internal transformation generalizes across independent
baseline trajectories on this topology. This is not a special
property of seed=3000.

9 further, independently drawn baseline trajectories confirm this.
Each uses the identical KMNIST class-0 T topology and the identical
design. Each produces a significant Delta_map of comparable size.
The mean is 0.33. All 10 values fall within a band of about 0.05.

This answers item 1 of `docs/PROJECT_MEMORY.md`'s "what remains
open" list for the dynamics-as-computation programme. That item
asked about generalization across independent baseline
trajectories. The answer is: yes, consistently. The result is not
partial. The result does not depend on the specific trajectory. This
holds at least across this sample of 10 trajectories, on one
topology, for one class.

This strengthens the capability-hierarchy status described in
`docs/PROJECT_MEMORY.md` Part 3. Level 2 (structured internal
transformation) was previously "established, locally," meaning on
one trajectory only. It is now established across 10 independent
trajectories on the same topology. This is a materially stronger
claim. It is still scoped to what was actually tested (see the next
section).

## What this result does not show

- **Topology specificity is still untested for this design.** Every
  trajectory here uses the T topology. No rewired, random, or
  lattice control has been run through Stage 1C's design. Whether T
  produces this generalizing mapping more strongly, more
  efficiently, or more distinctly than the matched controls remains
  exactly as open as `docs/PROJECT_MEMORY.md` already states. Stage
  1C adds trajectory-generalization evidence for T specifically. It
  does not compare T against a control.
- **One class only.** All 10 trajectories are KMNIST class 0.
  Nothing here speaks to the topologies of other classes.
- **The perturbation directions are not fully independent.** Within
  each trajectory, the 6 replica states are nearby-state
  perturbations around that trajectory's own path. This is Stage
  1B2's design, reused here unchanged. The 10 trajectories are
  independent of each other. But within each trajectory, the
  replication is still local. It is not a fresh, independent draw of
  replica geometry for each trajectory, beyond the baseline+1 seed
  offset.
- **External usefulness (Level 3) remains untested**, exactly as
  before. This result is about reproducible structured mapping. It
  does not link that structure to any externally defined task.

## A note on how consistent this result is

The tightness of this result deserves a plain note, not silence. The
coefficient of variation is about 5.2%. Every trajectory hits the
permutation floor.

This means the specific numeric value of Delta_map, for this
topology, this class, and this design, is now a highly stable and
well-characterized quantity: about 0.33, plus or minus 0.02. It is
not merely "significant in each case examined separately."

A future comparison against a graph-control's Delta_map, under this
same design, has a precisely known, low-variance target to compare
against on the T side.

## How to reproduce these results

Run `run_stage1c.py <seeds>`. By default, this uses the 9 new
trajectories listed in `NEW_BASELINE_SEEDS`. It refuses with a
`ValueError` if you pass seed=3000 explicitly. Seed=3000 is Stage
1B2's frozen reference trajectory. This script must never regenerate
it. The script builds and checkpoints each new trajectory's 432
trials to `results/stage1c_results_seed<N>.pkl`.

Run `analyze_stage1c.py <seeds>`. By default, this uses all 10
trajectories listed in `ALL_BASELINE_SEEDS`. It reads seed=3000
read-only from Stage 1B2's own results. It reads the other 9 seeds
from Stage 1C's own cache. It runs the permutation test for each
trajectory. It saves the combined result to
`results/stage1c_final_analysis.pkl`.

Both scripts import their core logic directly from
`stage1b2_structured_transformation/`. Neither script duplicates
that logic. Neither script reads from, or writes to, anything else
under that directory.

Results cached under `results/` are gitignored. This matches this
project's convention for cached artifacts that can be regenerated.
These files are not committed. You can regenerate them in well under
an hour, from this checked-in code.
