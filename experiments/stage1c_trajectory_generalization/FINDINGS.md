# Stage 1C: Trajectory Generalization

## The question

Stage 1B2 established structured internal transformation -- a
significant, input-sensitive, locally reproducible mapping from local
perturbations to spatial response patterns -- on exactly one baseline
trajectory (KMNIST class 0, T topology, seed=3000). That result was
explicitly scoped as conditional on that one trajectory, not yet shown to
generalize (`stage1b2_structured_transformation/FINDINGS.md`'s "Scope,
essential" section, and `docs/PROJECT_MEMORY.md`'s "What remains open,
in priority order", item 1). Stage 1C asks that question directly: does
the same design, applied to independent baseline trajectories on the
same topology, produce the same result, or was seed=3000 a favorable
draw?

## Design

Identical to Stage 1B2 in every respect except the baseline trajectory
itself: 3 nodes (low/median/high weighted-degree in T) x 2 signs x 3
amplitudes (0.025 tangent, 0.2 intermediate, 0.8 nonlinear) = 18 inputs x
4 perturbation times t_p in {0, 0.833, 1.667, 2.5} x 6 fixed nearby-state
replicas = 432 trials per trajectory, same KMNIST class-0 T topology,
same permutation test (independent per-replica label permutation, 10,000
permutations, one-sided Monte Carlo p-value).

**10 baseline trajectories**: seed=3000 (Stage 1B2's own reference) plus
9 new, independent seeds -- 3010, 3020, 3030, 3040, 3050, 3060, 3070,
3080, 3090 -- each paired with a replica-direction seed of baseline+1
(3011, 3021, ..., 3091), matching Stage 1B2's own
BASELINE_SEED=3000/REPLICA_DIRECTION_SEED=3001 offset convention.

**Implementation**: `run_stage1c.py` and `analyze_stage1c.py` import
directly from `stage1b2_structured_transformation/stage1b2_core.py` and
`analyze_stage1b2.py` (`get_degree_stratified_nodes`,
`generate_reference_baseline`, `generate_fixed_replica_directions`,
`run_one_trial`, `load_results_as_arrays`, `run_permutation_test`) rather
than copying them, so Stage 1C is guaranteed to be running the exact
same trial and analysis logic Stage 1B2's result rests on, not a
reimplementation that could silently drift. seed=3000's 432 trial
results are read directly (read-only) from Stage 1B2's own committed
`results/stage1b2_results.pkl` rather than re-run, keeping Stage 1B2
genuinely frozen as the reference. The 9 new trajectories are Stage 1C's
own, independently cached in `results/stage1c_results_seed<N>.pkl`.

**Runtime**: measured directly before committing to the full run, not
assumed -- one trajectory (seed=3010) took 106.6s end-to-end (432 trials,
9 worker processes), dramatically faster than either prior estimate (the
original design note anticipated 15-20 minutes; Stage 1B2's own
`run_stage1b2.py` docstring estimates 35-45 minutes). All 9 new
trajectories together took under 14 minutes; permutation analysis
(10,000 permutations x 10 trajectories) added the remainder.

## Per-trajectory results

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

Every single one of the 40 individual t_p-level Delta_map values across
all 10 trajectories is positive and comfortably above zero (range 0.24
to 0.40) -- there is no t_p, in any trajectory, where the mapping
weakens to non-significance or reverses direction.

## Across all 10 trajectories

- **Mean pooled Delta_map: 0.3296** (median 0.3324).
- **Range: 0.2964 to 0.3505** (span 0.0541).
- **Standard deviation: 0.0172** (coefficient of variation ~5.2%) --
  tight clustering, not a wide scatter with a few outliers pulling the
  mean.
- **10 of 10 trajectories hit the Monte Carlo floor** (p_MC = 0.00010,
  i.e. p = 1/10001, the smallest attainable value at 10,000
  permutations). **Zero trajectories failed to produce a significant
  result.**
- The weakest trajectory by pooled Delta_map is **seed=3020** (0.2964),
  which also contains the single lowest individual t_p value anywhere in
  the dataset (t_p=0.833, Delta_map=0.2403) -- still comfortably
  positive and still significant at the Monte Carlo floor, not a
  borderline or failing case, just the relative low end of an otherwise
  tight distribution.

## What this establishes

**Structured internal transformation generalizes across independent
baseline trajectories on this topology.** This is not a property of
seed=3000 specifically -- 9 further, independently drawn baseline
trajectories on the identical KMNIST class-0 T topology reproduce a
significant, comparably-sized Delta_map (mean 0.33, all 10 within a
~0.05 band) under the identical design. Item 1 of
`docs/PROJECT_MEMORY.md`'s "what remains open" list for the dynamics-
as-computation programme -- generalization across independent baseline
trajectories -- is now answered: **yes, consistently, not partially and
not trajectory-dependently**, at least across this sample of 10
trajectories on one topology and one class.

This upgrades the capability-hierarchy status described in
`docs/PROJECT_MEMORY.md` Part 3: Level 2 (structured internal
transformation) was "established, locally" (one trajectory); it is now
established across 10 independent trajectories on the same topology --
a materially stronger claim, though still scoped to what was actually
tested (see below).

## What this does not establish

- **Topology specificity is still untested for this design.** Every
  trajectory here uses T; no rewired/random/lattice control has been run
  through Stage 1C's design. Whether T produces this generalizing
  mapping more strongly, efficiently, or distinctly than the matched
  controls remains exactly as open as `docs/PROJECT_MEMORY.md` already
  states -- Stage 1C adds trajectory-generalization evidence for T
  specifically, not a T-vs-control comparison.
- **One class only.** All 10 trajectories are KMNIST class 0. Nothing
  here speaks to other classes' topologies.
- **Not fully independent perturbation directions.** Within each
  trajectory, the 6 replica states are nearby-state perturbations around
  that trajectory's own path (Stage 1B2's design, reused unchanged here)
  -- the 10 trajectories are independent of each other, but each one's
  internal replication is still local, not a fresh independent draw of
  replica geometry per trajectory beyond the baseline+1 seed offset.
- **External usefulness (Level 3) remains untested**, exactly as before
  -- this result is about reproducible structured mapping, not about
  linking that structure to any externally defined task.

## Honest note on the consistency itself

The tightness of this result (CV ~5.2%, every trajectory hitting the
permutation floor) is itself worth flagging plainly rather than treating
as unremarkable: it means the specific numeric value of Delta_map for
this topology, this class, and this design is now a highly stable,
well-characterized quantity (~0.33 ± 0.02), not merely "significant
in each case examined separately." A future comparison against a
graph-control's Delta_map under this same design has a precisely known,
low-variance target to compare against on the T side.

## Reproducing these results

`run_stage1c.py <seeds>` (defaults to the 9 new trajectories in
`NEW_BASELINE_SEEDS`; refuses with a `ValueError` if seed=3000 is passed
explicitly, since that trajectory is Stage 1B2's frozen reference and
must never be regenerated by this script) builds and checkpoints each
new trajectory's 432 trials to `results/stage1c_results_seed<N>.pkl`.
`analyze_stage1c.py <seeds>` (defaults to all 10 trajectories in
`ALL_BASELINE_SEEDS`, reading seed=3000 read-only from Stage 1B2's own
results and the other 9 from Stage 1C's own cache) runs
the permutation test per trajectory and saves the aggregate to
`results/stage1c_final_analysis.pkl`. Both import their core logic
directly from `stage1b2_structured_transformation/` rather than
duplicating it; neither reads nor writes anything else under that
directory. Results cached under `results/` are gitignored, matching this
project's convention for regenerable cached artifacts -- not committed,
regenerable in well under an hour from this checked-in code.
