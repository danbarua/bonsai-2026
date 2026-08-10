Simplified Technical English version of `experiments/stage1d_topology_specificity/FINDINGS.md`.

# Stage 1D, Part 1: T vs. Lattice (Main Test, Confirmed)

**Status: complete.** This is a real result. The team planned it in
advance. It does not depend on any other part of Stage 1D. The lattice
construction is fixed. It uses no random seed. So the team did not
need to run a pilot test for it first. See DESIGN.md's "Lattice Is
Different" section.

## Question

Does T show any advantage over a lattice control? T is the learned
topology: the graph learned from a class's image population. A lattice
control is a simple grid graph. It connects each pixel to its
neighbors on the grid. It has no learning in it. This is the plainest
control graph in this project.

The lattice control used here has the same total edge weight and the
same number of nodes as T. The team ran this test under the same
Stage 1B2/1C method as before. They used the same fixed-coordinate
intervention (see below). This is one of two open items left from
Stage 1B2's own priority list (see `docs/PROJECT_MEMORY.md`).

## Method

- The team read T's per-trajectory Delta_map value straight from
  Stage 1C's own file, `stage1c_final_analysis.pkl`. Delta_map is the
  main score for this project. It measures whether the network turns
  different starting kicks into reliably different patterns. A higher
  score means a clearer, more reliable pattern. The team did not
  recompute T's value.
- The team ran the lattice construction fresh. They used the same
  design as before: 432 trials per trajectory (3 nodes x 2 signs x 3
  sizes x 4 start times x 6 replicas). A replica is a slightly
  different but nearby starting point for a trial. It checks that the
  result does not depend on one exact starting position. The team ran
  this design on all 10 of Stage 1C's matched baseline trajectory
  seeds (3000, 3010, ..., 3090). They used the class-0 lattice graph
  already stored in `class0_constructions.pkl`. The team checked this
  stored graph exactly matches a historical version, byte for byte.
- **Fixed-coordinate intervention**: the team perturbed the same three
  nodes on every graph. These are T's own low, median, and high
  weighted-degree node indices: 17, 363, and 129. The team used these
  same three index numbers on the lattice graph too. This is true even
  though the lattice graph's own degree pattern is close to uniform,
  so these three indices may not mean "low, median, high" on the
  lattice itself. This method follows DESIGN.md's "fixed graph
  coordinates" protocol.
- Every one of the 20 trajectories (T's 10, lattice's 10) got the full
  10,000-permutation test. This matches Stage 1B2/1C's standard
  practice. The team reports this test below as a check, not as a
  filter.
- **Main test, locked in advance**: a two-sided paired t-test. This
  test compares `d_k = Delta_map(T,k) - Delta_map(lattice,k)` for the
  10 matched seeds. As a check on this result, the team also ran an
  exact sign-flip test (checking all 2^10 possible sign patterns) and
  a Wilcoxon signed-rank test. Both checks used the same 10 `d_k`
  values.

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

The value 0.00010 is the smallest p-value the test can produce at
10,000 permutations. Every trajectory of both graphs hits this floor
value. This confirms each graph, on its own, shows a structured
pattern. This check is separate from the main topology comparison
below. It does not feed into that comparison. See DESIGN.md's "What
the Permutation Test Does and Does Not Answer" section.

Mean Delta_map: T = 0.3296 (this matches Stage 1C's own reported
value). Lattice = 0.3381. The lattice value is nominally a little
higher on average.

**Mean d_k = -0.0085, standard deviation (SD) = 0.0235, across 10
seeds.**

- **Main test: two-sided paired t-test**: t = -1.146, df = 9,
  **p = 0.2815**. This is not a significant result.
- **Check, exact sign-flip test** (all 1024 sign patterns of the 10
  `d_k` values): **p = 0.2871**.
- **Check, Wilcoxon signed-rank test**: statistic = 19.0,
  **p = 0.4316**.

All three tests agree. **The team found no detectable difference
between T and the matched lattice control** on the main Delta_map
score. This holds across the 10 sampled starting trajectories.

## What This Result Shows, Exactly

- T does not show a detectable advantage over a lattice control with
  the same total coupling budget and the same number of nodes. This is
  true for class 0, under this method, across these 10 trajectories.
  If anything, the lattice's point value is nominally a little higher.
  But the difference is well within the noise 10 trajectories can
  produce on their own. (T's own trajectory-to-trajectory SD was about
  0.017 in Stage 1C. The observed size of mean d_k, 0.0085, is smaller
  than that.)
- This is a **bounded claim**. It applies only to class 0 and only to
  this lattice construction, across **10 sampled trajectories**. It is
  not a general statement about learned graphs versus lattice graphs
  as a whole family. See DESIGN.md's own scope statement.
- This does not mean the lattice "has no structure." Both T and the
  lattice hit the permutation floor on every single trajectory tested.
  The finding is that the *amount* of structured transformation
  (Delta_map) is statistically the same between the two graphs at this
  sample size. It does not mean either graph lacks the pattern itself.
- This result is consistent with, and builds on, Stage 1A's own
  re-check. That earlier check found no lasting advantage for the
  historical-random or current-random controls. It also left the
  rewiring question genuinely unsettled. Lattice now joins the "no
  detected advantage" side of that record. This new result uses Stage
  1B2/1C's matched-trajectory-seed, paired-test design, not Stage 1A's
  original design.

## What This Result Does Not Show

- Whether T outperforms the three *random* controls (rewired,
  historical-random, current-random). Note: this project uses the word
  "rewired" for three different graphs in different parts of the
  project. Here, and everywhere in this document, it means only Stage
  1D's own construction: a graph built by swapping pairs of edges,
  which keeps each node's exact unweighted degree the same as in T.
  Whether T outperforms these three controls is the confirmed test
  this Stage 1D pilot (see `PILOT_RESULTS.md`) was sizing. It had not
  yet been run at the time of this test.
- Anything about role-matched intervention. This means testing T's
  node roles against each graph's own degree-stratified nodes. This is
  out of scope for this pass. Only the fixed-coordinate method was
  used here.
- Any general claim across KMNIST classes 1-9. This is explicitly left
  for later work. See DESIGN.md's own scope statement.

# Stage 1D, Part 2: T vs. the Three Random Controls (Confirmed Test)

**Status: complete.** This test used the locked allocation of R=25
versions and K=3 trajectory seeds (see DESIGN.md, "Locked Confirmed-
Run Allocation"). For hist_random, it also used the locked
pre-screening method and a conditional estimand (see DESIGN.md,
"Historical-Random: Pre-Screening and a Conditional Estimand"). The
team ran this test on GPU hardware. They used the bug-fixed JAX
pipeline from `experiments/stage1d_topology_specificity_gpu/` (see
that folder's own `FINDINGS.md`). That fix corrected the
`generate_fixed_replica_directions()` function and the
`event_aligned_valid` gate.

## Seeds: A Clear Choice, Not Fixed Before

Neither `DESIGN.md` nor `PILOT_RESULTS.md` fixes the graph-version
seeds to use for this confirmed test. Both documents leave this choice
open. (The pilot's own notes say its version seeds "are not pinned by
DESIGN.md.") The team checked this before running anything.

**Decision made here**: use fresh seeds for all three graph families.
Start at seed 5 and count upward. This continues strictly past the
pilot's own seed range. (The pilot used seeds 0-2 for rewired and
curr_random. It used seeds 0-4 for hist_random, after a variance
follow-up test.) The team did not reuse any pilot seed. This keeps the
pilot data and the confirmed-test data cleanly separate. It avoids any
concern about the two runs not being independent.

The matched trajectory seeds stay the same as in the pilot and in
Stage 1C: **3000, 3010, 3020**. This follows the locked K=3 design
(DESIGN.md: "drawn as the first K of Stage 1C's 10 seeds... in
order").

## How the Team Built the Graphs

- For `rewired` and `curr_random`: the team built 25 versions of each,
  using seeds 5 through 29 directly. They applied no pre-screening.
  (Rewired cannot show the fixed-coordinate problem described below,
  by how it is built. Curr_random showed only one mild, non-serious
  case of the problem in the pilot, not a full failure.)
- For `hist_random`: the team used DESIGN.md's locked pre-screening
  method. They drew candidate seeds one after another, starting at 5.
  Before running any simulation, they checked the weighted degree of
  the three fixed nodes (`nodes_T`'s low, median, and high nodes) in
  each candidate graph. They rejected any candidate where one of these
  three nodes had zero degree (was isolated). **7 of the 32 candidates
  drawn were rejected** (seeds 8, 17, 20, 25, 29, 30, 33) before the
  team reached 25 usable versions. (Accepted seeds: 5, 6, 7, 9, 10,
  11, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 24, 26, 27, 28, 31, 32,
  34, 35, 36.) **The unconditional rejection rate was 7/32 = 21.9%**,
  with a 95% Clopper-Pearson confidence interval of [9.3%, 40.0%].
  This matches, and gives a tighter estimate than, the pilot's own
  rate (2 of 5 draws showed some isolated node). The team reports this
  rate as a known property of the hist_random family. It is not folded
  into the main comparison below.
- The team rebuilt all three families' graphs directly on the GPU
  session. They used the same fixed, seeded construction functions
  (`degree_preserving_rewire`,
  `generate_historical_matched_sparsity_random` plus
  `rescale_to_common_budget`, and `generate_matched_sparsity_topology`).
  These functions were copied unchanged from `src/bonsai/dynamics/`,
  not rewritten. The team did this instead of uploading a large,
  155MB local file, which had failed once before during upload. **The
  team checked, and did not just assume, that the GPU-built graphs
  exactly match the locally-built ones.** They computed a checksum for
  each family (the sum of weights plus a degree-weighted position sum)
  on both sides. The checksums matched exactly (for example, rewired:
  13026068.53542050 on both the local machine and the GPU).

## GPU Run

The team simulated 225 trajectories in total: 25 versions x 3 matched
trajectory seeds x 3 graph families. They used the same fixed pipeline
as before (`build_432_batch` and `run_one_trial_jax_faithful`),
already checked in this folder's own `FINDINGS.md`. T's own Delta_map
values, for seeds 3000, 3010, and 3020, came directly from Stage 1C's
stored file, `stage1c_final_analysis.pkl`. **The team did not
resimulate T.**

The team computed Delta_map for each trajectory, both per start time
and pooled, on the GPU session. They used the real
`analyze_stage1b2.py` functions (`load_results_as_arrays`,
`compute_W_B_deltamap`). For invalid trials, they set
`event_aligned_q` to `None` before passing data to these functions.
This matches the standard numpy method, not the earlier bug.

Total GPU simulation time: **204.09 seconds** for all 225
trajectories. Total wall time was 835.76 seconds, including CPU-side
baseline solves and per-version graph swaps. This matches the
pipeline's known cost of about 0.9 seconds per trajectory on the GPU.

**One real problem was found and is reported here.** In the confirmed
run itself, `curr_random` version seed=21 had its 'median'
fixed-coordinate node isolated (weighted degree 0.0). This is the same
problem seen in the pilot, but this time it happened in curr_random,
not hist_random. (Curr_random was not pre-screened, per DESIGN.md's
plan.) This problem did not make that version's Delta_map undefined.
The surviving (low, high) node-label pair still supports `B_node`. In
this version, 288 of 432 trials were valid per trajectory. (The 144
missing trials are exactly the 'median'-node block: 4 start times x 6
replicas x 2 signs x 3 sizes.) This is the same mild problem the pilot
already reported for a different curr_random version. No other family
or version showed an invalid node label. No trajectory in all 225 had
a fully-undefined (NaN) pooled Delta_map.

## Results

For each graph family, the team computed
`d_grk = Delta_map(T,k) - Delta_map(g,r,k)`. They combined this within
each version (mean over the 3 matched trajectory seeds) to get
`d_bar_gr`, across all 25 versions. (No version was left out in any
family. Hist_random's pre-screening had already removed the versions
that would give a fully-undefined `Delta_map`.)

| family | R used | mean d_bar_gr | SD | versions where control beats T | main t-test (df=24) | sign-flip test | Wilcoxon test | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rewired | 25 | -0.0020 | 0.0125 | 14/25 | t=-0.812, p=0.4246 | p=0.4215 | p=0.4418 | [-0.0103, 0.0061] |
| hist_random | 25 | -0.0025 | 0.0154 | 16/25 | t=-0.824, p=0.4179 | p=0.4217 | p=0.2521 | [-0.0109, 0.0060] |
| curr_random | 25 | -0.0004 | 0.0158 | 15/25 | t=-0.132, p=0.8958 | p=0.8950 | p=0.8119 | [-0.0096, 0.0084] |

(Sign-flip test: the team checked all 2^25 = 33,554,432 possible sign
patterns exactly. They used a fast, vectorized method, not a
per-pattern loop and not an approximate Monte Carlo method. Bootstrap:
20,000 combined resamples, drawing first from graph versions, then
from matched trajectories within each version.)

**Hist_random's main result above is the conditional estimand**
`E[Delta_T - Delta_hist_random | evaluable]`. This means it only
covers the graphs that passed pre-screening. It says nothing about the
21.9% of candidate draws that were unusable by construction. That
separate question is answered only by the rejection-rate result
reported above, per DESIGN.md's locked plan.

All three families' point values are small and close to zero. All
three signs are *nominally negative* (the control very slightly
outperforms T, on average). But every one of the main tests, and every
check run alongside it, is far from significant. This closely matches
the lattice result reported above. (Lattice: mean d_k = -0.0085. The
three random families here: -0.0004 to -0.0025.)

## Holm Correction Across the Four Fixed-Coordinate Tests

| rank | comparison | raw p | Holm-adjusted p |
|---|---|---:|---:|
| 1 | lattice | 0.2815 | 1.0000 |
| 2 | hist_random | 0.4179 | 1.0000 |
| 3 | rewired | 0.4246 | 1.0000 |
| 4 | curr_random | 0.8958 | 1.0000 |

All four Holm-adjusted p-values reach the top value of 1.0. This is
not surprising, since even the smallest raw p-value (lattice's 0.2815)
already exceeds 0.25. No correction method could rescue a significant
result here. **None of the four fixed-coordinate comparisons
(lattice, rewired, hist_random, curr_random) shows a detectable
Delta_map advantage for T.** This test used R=25 versions (10
trajectories for lattice and T, in a matched-pair design). The design
targeted 80% power to detect a minimum meaningful effect of 0.05. This
effect size is roughly 3 times T's own trajectory-to-trajectory SD.

## What This Result Shows, Exactly

- **T shows no detectable topology-specificity advantage over any of
  the four tested controls** (lattice, degree-preserving rewiring,
  historical-random, current-random). This holds for class 0, under
  the same Stage 1B2/1C fixed-coordinate method. The test was designed
  to detect an effect size (`delta_min=0.05`) roughly 3 times T's own
  trajectory-to-trajectory noise. This closes Stage 1D's main
  question, with a negative answer: Bonsai-style graph-oscillator
  dynamics appear to produce this structured Delta_map pattern on any
  graph matching T's broad statistics (size, total coupling budget,
  and, for rewired and lattice, degree structure too). It does not
  appear to depend on the specific structure T learned from the
  KMNIST class-0 image population.
- This does **not** mean the pattern itself is a false result or is
  uninteresting. Every trajectory across all four constructions (T,
  lattice, rewired, hist_random, curr_random) hit the 10,000-
  permutation floor (p=0.00010). This confirms a real, structured,
  repeatable internal pattern is present in every case. What is not
  shown is that *T's specific learned structure* is what produces this
  pattern, rather than the broad statistics that any of these
  constructions share with T.
- **Scope, exactly as before**: this result applies only to class 0,
  only to this specific T, and only to these 25 (or 10, for lattice)
  sampled versions and trajectories. Only the fixed-coordinate
  intervention was used. This is not a general claim across KMNIST
  classes. It is not a claim about role-matched intervention. It is
  not a claim that no learned graph, anywhere, could ever show this
  kind of specificity.
- The hist_random result is scoped further, to the conditional
  estimand (evaluable versions only). Its own roughly 22% unconditional
  rejection rate is a real, disclosed property of that construction at
  T's edge density. It is not folded into the headline number.

## What This Result Does Not Show

- Whether T outperforms these controls under role-matched intervention
  (each construction's own degree-stratified nodes). This is out of
  scope. Per DESIGN.md, this is a secondary, robustness-only item, and
  it was not run here.
- A general claim across KMNIST classes 1-9. This is explicitly left
  for later work.
- Anything about Level 3 (external usefulness). This is a separate,
  larger question that this design does not touch.
