# Stage 1D, Part 1: T vs. Lattice (primary, confirmatory)

**Status: complete.** This is a real, prespecified result -- not gated
on anything else in Stage 1D (the lattice construction is
deterministic, so it needed no piloting; see DESIGN.md's "Lattice is
different" section).

## Question

Does T show any advantage over a lattice control with matched total
edge weight and node count, under the identical Stage 1B2/1C protocol
and fixed-coordinate intervention? This is one of the two remaining
open items from Stage 1B2's own priority list
(`docs/PROJECT_MEMORY.md`).

## Method

- T's per-trajectory Delta_map: read read-only from Stage 1C's own
  already-committed `stage1c_final_analysis.pkl` (never recomputed).
- Lattice: the same 432-trial-per-trajectory design (Option A: 3 nodes
  x 2 signs x 3 amplitudes x 4 t_p x 6 replicas), run fresh for all 10
  of Stage 1C's matched baseline trajectory seeds (3000, 3010, ...,
  3090), on the class-0 lattice construction already cached in
  `class0_constructions.pkl` (verified byte-exact against the
  historical artifact elsewhere in this repo).
- **Fixed-coordinate intervention**: the perturbed nodes are T's own
  low/median/high weighted-degree indices (17, 363, 129), used
  identically on the lattice graph regardless of what role those
  indices play in the lattice's own (near-uniform) degree
  distribution -- per DESIGN.md's "fixed graph coordinates" protocol.
- Every one of the 20 trajectories involved (T's 10, lattice's 10) gets
  the full 10,000-permutation validation test, exactly as Stage
  1B2/1C's own convention -- reported below as a sanity check, not used
  as a filter.
- **Primary test, locked**: two-sided paired t-test on
  `d_k = Delta_map(T,k) - Delta_map(lattice,k)` for the 10 matched
  seeds. Robustness: exact sign-flip test (all 2^10 sign patterns) and
  Wilcoxon signed-rank, both on the same 10 `d_k` values.

## Results

| seed | Delta_map(T) | Delta_map(lattice) | d_k = T - lattice | lattice's own permutation p |
|------|-------------:|--------------------:|------------------:|----------------------------:|
| 3000 | 0.3505 | 0.3270 |  0.0236 | 0.00010 |
| 3010 | 0.3318 | 0.3446 | -0.0129 | 0.00010 |
| 3020 | 0.2964 | 0.3465 | -0.0501 | 0.00010 |
| 3030 | 0.3402 | 0.3424 | -0.0022 | 0.00010 |
| 3040 | 0.3389 | 0.3245 |  0.0144 | 0.00010 |
| 3050 | 0.3329 | 0.3593 | -0.0264 | 0.00010 |
| 3060 | 0.3486 | 0.3457 |  0.0030 | 0.00010 |
| 3070 | 0.3077 | 0.3456 | -0.0379 | 0.00010 |
| 3080 | 0.3237 | 0.3298 | -0.0060 | 0.00010 |
| 3090 | 0.3253 | 0.3160 |  0.0093 | 0.00010 |

(0.00010 is the minimum attainable Monte Carlo p-value at 10,000
permutations -- every trajectory of both constructions hits the floor,
confirming each individually shows a structured mapping. This validity
check is separate from, and does not feed into, the topology-family
comparison below -- per DESIGN.md's "What the permutation test does and
does not answer.")

Mean Delta_map: T = 0.3296 (matching Stage 1C's own reported value),
lattice = 0.3381. Lattice is nominally slightly *higher* on average.

**Mean d_k = -0.0085, SD = 0.0235 (n=10).**

- **Primary: two-sided paired t-test**: t = -1.146, df = 9,
  **p = 0.2815** -- not significant.
- **Robustness, exact sign-flip test** (all 1024 sign patterns of the
  10 `d_k` values): **p = 0.2871**.
- **Robustness, Wilcoxon signed-rank**: statistic = 19.0,
  **p = 0.4316**.

All three tests agree: **no detectable difference between T and the
matched lattice control** on the primary Delta_map endpoint, across
these 10 sampled initial trajectories.

## What this establishes, precisely

- T does not show a detectable advantage over a lattice control with
  matched total coupling budget and node count, on class 0, under this
  protocol, across these 10 trajectories. If anything, lattice's point
  estimate is nominally slightly higher, but the difference is well
  within what 10 trajectories' own sampling noise can produce (T's own
  between-trajectory SD was ~0.017 per Stage 1C; the observed |mean
  d_k| of 0.0085 is smaller than that).
- This is a **bounded, class-0-only, this-lattice-only claim** across
  **10 sampled trajectories** -- not a population-level statement about
  learned-vs-lattice topology families in general (DESIGN.md's own
  scoping).
- This does not mean lattice "has no structure" -- both T and lattice
  hit the permutation floor on every single trajectory tested. The
  finding is that the *degree* of structured transformation (Delta_map)
  is statistically indistinguishable between the two constructions at
  this sample size, not that either construction lacks the phenomenon.
- Consistent with, and extending, Stage 1A's own re-verification, which
  found no surviving advantage for historical-random or current-random
  controls and left rewiring genuinely inconclusive: lattice now joins
  the "no detected advantage" side of that ledger, using Stage
  1B.2/1C's matched-trajectory-seed, paired-test design rather than
  Stage 1A's original one.

## What this does not establish

- Whether T outperforms the three *stochastic* controls (rewired,
  historical-random, current-random) -- that is the confirmatory run
  this Stage 1D pilot (see `PILOT_RESULTS.md`) is sizing, not yet run.
- Anything about role-matched intervention (T's node roles vs. each
  construction's own degree-stratified nodes) -- out of scope for this
  pass, per the task that produced this result; fixed-coordinate only.
- A general claim across KMNIST classes 1-9 -- explicitly deferred
  (DESIGN.md's own scoping).
