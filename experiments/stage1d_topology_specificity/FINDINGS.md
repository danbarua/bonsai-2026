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

# Stage 1D, Part 2: T vs. the three stochastic controls (confirmatory)

**Status: complete.** Uses the locked (R=25, K=3) common allocation
(`DESIGN.md`, "Locked confirmatory-run allocation") and, for
hist_random, the locked pre-screening/conditional-estimand protocol
(`DESIGN.md`, "Historical-random: pre-screening and a conditional
estimand"). Ran on GPU via the bug-fixed JAX pipeline
(`experiments/stage1d_topology_specificity_gpu/`, `FINDINGS.md` --
correct `generate_fixed_replica_directions()`, proper
`event_aligned_valid` gating).

## Seeds: an explicit choice, not something already pinned

Neither `DESIGN.md` nor `PILOT_RESULTS.md` pins confirmatory-run
graph-realization seeds -- both leave this open (the pilot itself notes
its own realization seeds "are not pinned by DESIGN.md"). Checked
before running anything. **Decision made here**: fresh seeds for all
three families, starting at 5 and continuing upward, strictly
continuing past the pilot's own range (rewired/curr_random used seeds
0-2; hist_random used 0-4 after its variance follow-up) rather than
reusing any pilot draw. This keeps pilot and confirmatory data cleanly
disjoint, avoiding any non-independence concern between the two runs.
Matched trajectory seeds are unchanged from the pilot and Stage
1C -- **3000, 3010, 3020** -- per the locked K=3 design (DESIGN.md:
"drawn as the first K of Stage 1C's 10 seeds... in order").

## Construction

- `rewired`, `curr_random`: 25 realizations each, seeds 5-29 directly --
  no pre-screening applied (rewired cannot show fixed-coordinate
  isolation by construction; curr_random showed only one mild,
  non-fatal instance in the pilot, not full degeneracy).
- `hist_random`: DESIGN.md's locked pre-screening protocol. Candidate
  seeds drawn sequentially from 5 upward; before any simulation, the
  weighted degree of `nodes_T`'s three fixed coordinates (low/median/
  high) was checked in each candidate graph, rejecting on any zero
  (isolated) reading. **7 of 32 candidates drawn were rejected (seeds
  8, 17, 20, 25, 29, 30, 33) before 25 evaluable realizations were
  obtained** (accepted seeds: 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
  18, 19, 21, 22, 23, 24, 26, 27, 28, 31, 32, 34, 35, 36).
  **Unconditional rejection rate: 7/32 = 21.9%, 95% Clopper-Pearson CI
  [9.3%, 40.0%]** -- consistent with, and a tighter estimate than, the
  pilot's own disclosed rate (2 of 5 draws showing some isolated node).
  This is reported as a disclosed secondary characteristic of the
  hist_random family, not folded into the primary comparison below.
- All three families' graph realizations were rebuilt directly on the
  GPU session from the same deterministic seeded construction functions
  (`degree_preserving_rewire`, `generate_historical_matched_sparsity_random`
  + `rescale_to_common_budget`, `generate_matched_sparsity_topology`,
  copied unmodified from `src/bonsai/dynamics/`, not reimplemented) --
  rather than uploading a 155MB local pickle, which failed once on
  upload. **Verified, not assumed, to be byte-identical to the locally
  built realizations**: a per-family checksum (sum of weights + a
  degree-weighted positional sum) computed on both sides matched
  exactly (e.g. rewired: 13026068.53542050 both locally and on GPU).

## GPU run

225 trajectories (25 realizations x 3 matched trajectory seeds x 3
families) simulated via the same fixed `build_432_batch` /
`run_one_trial_jax_faithful` pipeline already verified in this folder's
own `FINDINGS.md`. T's own Delta_map for seeds 3000/3010/3020 was read
directly from Stage 1C's cached `stage1c_final_analysis.pkl` -- **not
resimulated**. Per-trajectory Delta_map (per-t_p and pooled) was
computed on the GPU session itself via the real `analyze_stage1b2.py`
functions (`load_results_as_arrays`, `compute_W_B_deltamap`), with
`event_aligned_q` gated to `None` on invalid trials before being handed
to those functions -- matching the numpy contract, not the bug fixed
earlier tonight. Total GPU simulation time: **204.09s** for all 225
trajectories (835.76s wall time including per-trajectory CPU-side
baseline-trajectory solves and per-realization graph swaps) --
consistent with this pipeline's previously-measured ~0.9s/trajectory
GPU cost.

**One real, disclosed degeneracy surfaced in the confirmatory run
itself**: `curr_random` realization seed=21 has its 'median'
fixed-coordinate node isolated (weighted degree 0.0) -- the same
mechanism diagnosed in the pilot, occurring here in curr_random (which
was not pre-screened, per DESIGN.md's scoping) rather than
hist_random. This did not NaN out that realization's Delta_map (the
surviving (low, high) node-label pair still supports `B_node`) --
288/432 trials valid per trajectory (144 missing = exactly the
'median'-node trial block: 4 t_p x 6 replicas x 2 signs x 3
amplitudes), same mild, non-fatal pattern the pilot already disclosed
for a different curr_random realization. No other family or
realization showed any invalid node label; no trajectory anywhere in
the 225 had a fully-NaN pooled Delta_map.

## Results

For each family, `d_grk = Delta_map(T,k) - Delta_map(g,r,k)` aggregated
within realization to `d_bar_gr` (mean over the 3 matched trajectory
seeds), across all 25 realizations (none excluded in any family --
hist_random's pre-screening already removed the realizations that
would have produced a fully-undefined `Delta_map`):

| family | R used | mean d_bar_gr | SD | realizations where control beats T | primary t-test (df=24) | sign-flip | Wilcoxon | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rewired | 25 | -0.0020 | 0.0125 | 14/25 | t=-0.812, p=0.4246 | p=0.4215 | p=0.4418 | [-0.0103, 0.0061] |
| hist_random | 25 | -0.0025 | 0.0154 | 16/25 | t=-0.824, p=0.4179 | p=0.4217 | p=0.2521 | [-0.0109, 0.0060] |
| curr_random | 25 | -0.0004 | 0.0158 | 15/25 | t=-0.132, p=0.8958 | p=0.8950 | p=0.8119 | [-0.0096, 0.0084] |

(Sign-flip: all 2^25 = 33,554,432 sign patterns, computed exactly via
vectorized iterative distribution-doubling, not a per-pattern loop or
Monte Carlo approximation. Bootstrap: 20,000 hierarchical resamples,
realizations then matched trajectories within realization.)

**hist_random's primary result above is the conditional estimand**
`E[Delta_T - Delta_hist_random | evaluable]` -- it says nothing about
the 21.9% of candidate draws that were unevaluable by construction; that
question is answered only by the rejection-rate disclosure above, per
DESIGN.md's locked framing.

All three families' point estimates are small and near zero, with signs
that are *nominally negative* (control very slightly outperforming T on
average) but every single one of the primary tests, and every
robustness test run alongside it, is far from significant --
mirroring lattice's own result almost exactly (lattice: mean d_k =
-0.0085; all three stochastic families here: -0.0004 to -0.0025).

## Holm correction across the 4-way fixed-coordinate family

| rank | comparison | raw p | Holm-adjusted p |
|---|---|---:|---:|
| 1 | lattice | 0.2815 | 1.0000 |
| 2 | hist_random | 0.4179 | 1.0000 |
| 3 | rewired | 0.4246 | 1.0000 |
| 4 | curr_random | 0.8958 | 1.0000 |

All four Holm-adjusted p-values saturate at 1.0 -- unsurprising given
even the smallest raw p (lattice's 0.2815) already exceeds 0.25, so no
correction was ever going to rescue significance here. **None of the
four fixed-coordinate comparisons (lattice, rewired, hist_random,
curr_random) shows a detectable Delta_map advantage for T**, at R=25
realizations (10 trajectories for lattice/T, matched-pair design) with
80% power targeted for a minimum meaningful effect of 0.05 -- an effect
size roughly 3x T's own between-trajectory SD.

## What this establishes, precisely

- **T shows no detectable topology-specificity advantage over any of
  the four tested controls** (lattice, degree-preserving rewiring,
  historical-random, current-random) on class 0, under the identical
  Stage 1B.2/1C fixed-coordinate protocol, at a design powered for an
  effect size (`delta_min=0.05`) roughly 3x T's own between-trajectory
  noise. This closes Stage 1D's primary question in the negative:
  Bonsai-style graph-oscillator dynamics appear to produce this
  structured Delta_map transformation on graphs matching T's broad
  statistics (size, total coupling budget, and for rewired/lattice,
  degree structure too), not something specific to what T learned from
  the KMNIST class-0 image population.
- This does **not** mean the phenomenon itself is spurious or
  uninteresting -- every trajectory across all four constructions
  (T, lattice, rewired, hist_random, curr_random) hit the 10,000-
  permutation floor (p=0.00010), confirming a real, structured,
  reproducible internal transformation is present in every case. What
  is not established is that *T's specific learned structure* is what
  produces it, rather than the broad statistics any of these
  constructions share with T.
- **Scope, precisely as before**: class-0 only, this specific T, these
  25 (or 10, for lattice) sampled realizations/trajectories,
  fixed-coordinate intervention only. Not a general claim across
  KMNIST classes, not a role-matched-intervention claim, not a claim
  that no learned topology anywhere could show specificity.
- hist_random's result is additionally scoped to the conditional
  estimand (evaluable realizations only); its own ~22% unconditional
  rejection rate is itself a disclosed, real property of that
  construction at T's edge density, not swept into the headline number.

## What this does not establish

- Whether T outperforms these controls under role-matched intervention
  (each construction's own degree-stratified nodes) -- out of scope,
  per DESIGN.md, secondary/robustness only and not run here.
- A general claim across KMNIST classes 1-9 -- explicitly deferred.
- Anything about Level 3 (external usefulness) -- a separate, larger
  question untouched by this design.
