Simplified Technical English version of `experiments/stage1d_topology_specificity/DESIGN.md`.

# Stage 1D: Is the Learned Graph's Structure Special?

## The Question

Stage 1B2 found a significant, repeatable structured internal pattern
(Delta_map about 0.35, at the permutation floor) on one baseline
trajectory. It used KMNIST class 0's learned topology, T. T is the
graph learned from a class's image population. Stage 1C showed this
pattern holds across 10 independent trajectories on the same T (mean
Delta_map 0.3296, CV about 5.2%, all 10 trajectories hitting the Monte
Carlo floor).

Neither result shows that T itself is doing anything special. Both
results are equally consistent with a different explanation: that this
pattern is a general property of graph-oscillator dynamics on *any*
graph with T's broad statistics (size, density, degree pattern). Under
that explanation, the pattern would not depend on anything specific
that T learned from the image population.

**Stage 1D asks directly**: does T produce this structured pattern
more strongly, more steadily, or in a different way than matched
control graphs? The team tested this under the identical Stage
1B2/1C method.

This is the second of three open items on Stage 1B2's own priority
list (see `docs/PROJECT_MEMORY.md`). Item 1, generalization across
trajectories, is now closed by Stage 1C. This design closes item 2.
Item 3, external usefulness ("Level 3"), is not touched by this
design. It needs its own, separately-scoped effort. It is a different
kind of question: it asks whether the pattern can be linked to
anything outside the system, not whether one graph beats another at
producing the pattern.

## What Is Ready, and What a "Stage 2" Would Actually Be

Every construction this design needs already exists and has been
checked independently: `degree_preserving_rewiring.py`,
`historical_matched_sparsity_random.py`, `matched_sparsity_ablation.py`,
and `lattice_construction.py`. Each one reproduces its matching
historical stored file exactly, byte for byte. (For the historical
random construction, the match is structural, and the byte-for-byte
gap is clearly documented.) This design could not be run before
tonight's construction-recovery work. It can be run now.

If T does not outperform, or does not meaningfully differ from, the
matched controls, that does not close the dynamics-as-computation
programme. It changes the scope of the claim. Instead of "the learned
Bonsai topology specifically supplies the computational advantage,"
the claim becomes "Bonsai-style graph-oscillator dynamics produce this
structured pattern on graphs with these general statistics." Both are
real, reportable findings. This design is built to tell the two apart.
It does not assume which one is true in advance.

## Main Score, and What Stays Secondary

**Main score**: trajectory-level Delta_map, computed the same way as
Stage 1B2/1C's own definition. This uses the same pooling across the 4
start times, the same 432-trial-per-trajectory design, and the same
permutation test. This is the planned, overall statistic this whole
programme has used since Stage 1B2. It stays the main score here, for
direct comparison.

**Secondary scores, named precisely rather than described loosely.**
Stage 1B2 produced two separate common-support quantities. This design
compares both directly, rather than picking one without saying so:
1. common-support overall score, Delta_map^(-S)
   (`analyze_stage1b2_common_support_exclusion.py`)
2. node-specific restricted score, Delta_node^(-S)
   (`analyze_stage1b2_common_support_node_test.py`)

The research question asks whether T is "stronger, more steady, or
different in kind." Each part of that question gets its own named
score, rather than being folded into the main Delta_map comparison
alone:
- **Strength**: mean trajectory-level Delta_map (the main score
  above).
- **Steadiness**: the within-graph standard deviation (SD) of
  Delta_map across the matched trajectories. This is the formal
  steadiness measure, and the team always reports it. The team reports
  CV (SD divided by mean) only when a graph's mean Delta_map is above
  **0.05**. This threshold is locked. It is roughly T's own
  trajectory-to-trajectory SD from Stage 1C. The team chose this
  threshold so that CV is only computed where the mean is clearly away
  from zero. CV becomes unstable or misleading as the mean gets close
  to zero, and a construction with a weak or absent pattern plausibly
  could have a mean close to zero. Between-version variance (for the
  random controls) and within-trajectory SD are measured at different
  levels. The team reports each one with its level clearly labeled.
  They are not treated as directly comparable numbers.
- **Straight-line versus non-straight-line makeup**: tangent-only and
  residual Delta_map. Tangent response is the straight-line (linear)
  approximation of the network's response. Residual is what is left
  after subtracting that straight-line part from the real response:
  the purely non-linear part. This checks whether any topology-specific
  effect lives in the linear part of the response or the non-linear
  part.
- **Source-independent structure**: both common-support quantities
  above.
- **"Different in kind"** stays a descriptive idea, not a formal
  score, unless a specific pattern is named in advance, before running
  anything. It is not a license to go looking for *some* difference
  after seeing the results.

**Explicitly not a main score**: the concentration/routing pattern
(`CONCENTRATION_REGIME_NOTE.md`). This mechanistic idea is valuable.
It may be worth checking across constructions later. But it is a
single-case finding (high-degree node, t_p=0) found after the fact,
not a score planned in advance. A construction's concentration
behavior must not substitute for its Delta_map result.

## Two Kinds of Intervention, Run and Reported Separately

These test different things. They must not be mixed into one number.

1. **Fixed graph coordinates.** Perturb the same three pixel-linked
   node indices in every construction. These are T's own low, median,
   and high-degree nodes, already used throughout Stage 1B2/1C. Use
   these same three indices regardless of what role they play in each
   construction's own degree pattern. This isolates the effect of
   changing *which graph* connects a fixed set of physical points,
   while holding the intervention itself constant.
2. **Role-matched nodes.** Choose each construction's *own* low,
   median, and high-degree nodes independently. Do this using
   `get_degree_stratified_nodes`, already used throughout this
   project. This makes the comparison "does T's structure matter under
   the same structural roles," rather than "does T's structure matter
   at these specific pixels."

**Undefined role-matching, locked directly, rather than papered over
with a tie-break rule.** A fixed tie-break rule is right for choosing
among several nodes *within* a genuine degree group. It is not a
substitute for that group actually existing. If a construction's
degree pattern does not support three separate groups, role-matched
intervention is declared **undefined** for that construction and
reported as such. (This is a real risk for the lattice construction
specifically. Its edge weights are uniform, so weighted degree there
is really just a count of connections. This can easily produce many
exact ties.) Picking three tied nodes at random and calling them
"low/median/high" would invent a role difference that does not really
exist. It would not measure a real one.

**If lattice's role-matched condition is undefined, the role-matched
Holm family has 3 tests, not 4 padded slots.** Holm correction works
on the p-values that were actually produced. An undefined comparison
produces no p-value to correct. So it is left out of the family, not
kept as an empty placeholder. The team reports this directly (for
example: "role-matched family: 3 of 4 planned comparisons, lattice
undefined"). This way, the smaller family size is never mistaken for
4 tests having actually been run.

Report both kinds of intervention. If they agree, that is a solid
specificity finding either way. If they disagree, that disagreement is
itself the finding. Report which nodes differ and why. Do not average
over the disagreement.

## Scope and Order of Claims

**Main scope: class 0 only.** The team used the construction pipeline
and 10-trajectory design already built and checked. This shows, or
fails to show, topology specificity for this one learned graph. This
is a real, bounded claim. It is not yet a general claim about "Bonsai
topologies" as a whole.

**A general claim would need repeating this test across multiple
KMNIST classes.** Class would be the unit of analysis, the same logic
used in the Stage 1A re-check's class-level design. This is explicitly
out of scope for this design's first pass. It is a natural next step,
to be decided separately, if the class-0 result is positive and worth
generalizing.

**The nesting that matters**: trajectories within one graph version
are repeated looks at *that* graph. They are not independent copies of
the topology family. For T and lattice (fixed constructions, no seed),
this comes down to a single version with 10 trajectories each. (T's
version already exists via Stage 1C. Lattice needs its own
10-trajectory run.) For the three random constructions -- rewiring,
historical-random, and current-random -- the correct structure is:

```
trajectory  (10 per version, matching Stage 1C's seed convention)
  |_ graph version  (built from one specific seed)
       |_ topology family  (rewiring / hist-random / curr-random)
```

Note on the word "rewired": this project uses that word for three
different graphs in different parts of the project. In this document,
and everywhere in Stage 1D, "rewired" means only Stage 1D's own
construction, described below: a double-edge-swap that keeps T's
unweighted degree sequence. Do not assume "rewired" elsewhere in the
project means this same graph.

**Number of graph versions for the random controls**: this is settled
by a dedicated pilot test, not fixed here in advance. See "Pilot
Versus Confirmed Test" below, which sets out the pilot's scope and
purpose. It also states the clear rule that the pilot cannot be used
for the confirmed test's conclusions.

## Main Test Structure

**The core fix, made after review**: the original plan compared T's 10
Stage 1C trajectory-level values against each random control's
version-level means. It used an unpaired Mann-Whitney test. This
treats T's 10 repeated trajectories as if they were 10 independent
topology versions. At the same time, it correctly reduces each control
to one version-level value per graph. This mixes two different levels
of comparison. It is not simply a weaker test choice; it is an
inconsistent one. The design below replaces it with a matched design
that fixes this problem.

**Matched trajectory seeds.** Use the same baseline trajectory seeds
for T and for every random control's graph versions. Do not
re-randomize them independently for each construction. This turns
trajectory-to-trajectory variation into a controllable, paired factor,
instead of an uncontrolled noise source that would contaminate the
comparison between constructions. The confirmed test run uses **K
matched trajectory seeds**. The pilot-driven allocation rule below
sets K; it is not fixed at 10 in advance. The team draws these as the
first K of Stage 1C's 10 seeds (3000, 3010, 3020, and so on) in order,
for direct comparison with T's already-computed values. (T and
lattice, both fixed constructions, still use the full 10 -- see
below.)

**Definitions.** For random construction g, graph version r, and
trajectory seed k (k = 1 to K, shared across all constructions):

```
d_grk = Delta_map(T, k) - Delta_map(g, r, k)
```

Combine within each version first (mean over the K matched
trajectories):

```
d_bar_gr = (1/K) * sum_k(d_grk)
```

The **graph version**, not the trajectory, is the unit of analysis for
random controls. The score being estimated is:

```
theta_g = E_r[d_bar_gr]
```

This is the expected advantage of this specific learned T over a graph
drawn from control family g. It is conditional on class 0 and the
intervention method tested.

**Main test: two-sided one-sample t-test, locked as the single main
decision rule.** theta_g is explicitly an expected value (a mean), not
a median or other location value. A signed-rank test or sign-flip test
targets a median-like value under a symmetry assumption. That does not
test theta_g itself. **Locked main test**: a two-sided one-sample
t-test on the version-level mean differences, mean_r(d_bar_gr). This
is Holm-corrected across the four fixed-coordinate comparisons. The
test is two-sided, not one-sided: the Stage 1D question explicitly
allows for T being stronger, weaker, or simply different from a
control, not only "T is stronger." A one-sided test would quietly
narrow the question being asked. The studentized bootstrap interval,
the Wilcoxon signed-rank test, and the exact sign-flip test on
{d_bar_gr} are kept as checks. They are not alternative main tests to
pick between after seeing the data.

**Main reporting**, for each random construction g:
- every version-level d_bar_gr value, not just the test statistic
- the mean version-level difference (the main estimate) and the median
  (a check)
- the two-sided t-test result (main), plus the studentized bootstrap
  interval, signed-rank test, and sign-flip test results (checks)
- the share of control versions that outperform T (where
  d_bar_gr < 0)
- within-version trajectory variability (the SD of d_grk across the K
  matched seeds, per version). The team reports this using the crossed
  variance breakdown described in "Pilot Versus Confirmed Test" below,
  not a plain per-version SD.

**Checks**: the studentized bootstrap interval, Wilcoxon signed-rank
test, and exact sign-flip test on {d_bar_gr} (see above), plus a
combined bootstrap that resamples both graph versions and matched
trajectory seeds. This is similar to the Stage 1A re-check's own
bootstrap method.

**Lattice is different: it is a fixed construction, so pair directly.**
T and lattice both need no seed. There is no version dimension to
combine over. Use the same 10 matched trajectory seeds directly:

```
d_k = Delta_map(T, k) - Delta_map(lattice, k),  k = 1 to 10
```

**Main test, locked to a single choice, matched to the same
mean-effect logic as the random-control comparisons**: a two-sided
paired t-test on the 10 d_k values. (T's values already exist via
Stage 1C. Lattice needs its own 10 trajectories on the identical
seeds.) The exact sign-flip test and exact signed-rank test on the
same 10 values are kept as checks, not as alternative main tests to
pick between after seeing the results. This fixes the same "or"
ambiguity already fixed for the random-control main test. An unpaired
test here would throw away a naturally available, more powerful paired
structure, for no reason. The resulting claim stays bounded: this
class-0 learned graph differs, or does not differ, from this fixed
lattice, across the 10 sampled starting trajectories. It is not a
population-level claim about learned-versus-lattice topology families
in general.

**Fixed-coordinate is the main test; role-matched is a secondary
check. This choice was locked before running anything, not decided
after seeing results.** Both kinds of intervention could otherwise be
used to claim "topology specificity," giving two chances at a positive
result. Fixed-coordinate holds the physical intervention constant
while changing only the graph. This is the more direct test of the
question. Role-matched changes node identity too, and answers a
related but different question: does structure matter under
equivalent structural roles? In practice:
- Holm-correct the 4 fixed-coordinate construction comparisons
  (rewired, hist_random, curr_random, lattice) as the main family.
- Holm-correct the 4 role-matched comparisons as a separate family.
- **A significant role-matched result cannot rescue a failed
  fixed-coordinate result.** If the two disagree, report the
  disagreement itself as useful, mechanistic information (which nodes
  differ, and why). Do not resolve it by picking whichever intervention
  happened to be significant.

## Pilot Versus Confirmed Test: The 3x3 Run Is Not a Confirmed Result

With only 3 graph versions per random construction, even an
all-positive one-sided exact sign test on {d_bar_gr} has
p_min = 1/2^3 = 0.125. No multiplicity correction can rescue that. So
the 3-versions-by-3-trajectories run must be labeled and treated
exactly as:

> **A runtime and variance-allocation pilot. No confirmed
> topology-specificity conclusion will be drawn from this run.**

**Variance breakdown, corrected: the pilot's data are crossed, not
simply nested.** Because the same 3 trajectory seeds are shared across
every graph version (per the matched-seed design above), a plain
variance of the 3 version-level means would mix true between-graph
variance with leftover trajectory-sampling noise. The correct model
is:

```
d_grk = mu_g + b_gr + tau_k + epsilon_grk
```

Here `b_gr` is the graph-version effect. `tau_k` is the shared
trajectory-seed block effect: the same k represents the same baseline
trajectory identity across every version. `epsilon_grk` is the
graph-by-trajectory remainder. The team fit this with a crossed
variance breakdown, or mixed model (treating version and
trajectory-seed as crossed random effects), for pilot allocation
purposes. They did not use the raw variance of `d_bar_gr` across the 3
versions, since that is not a pure estimate of `sigma^2_between-graphs`.
With only 3x3 data, all resulting variance estimates are rough. They
are fit for budget allocation, not for substantive claims.

**The pilot-to-confirmed-test allocation rule, locked before the pilot
ran, not decided after seeing its results:**

1. Minimum scientifically meaningful difference `delta_min` in
   Delta_map: **0.05**, locked. This is informed by Stage 1C's own
   trajectory-to-trajectory SD for T, about 0.017. (delta_min is set
   at roughly 3 times that SD, so the design targets a difference
   clearly outside T's own observed trajectory noise, not merely
   detectable in principle.)
2. Desired power: **80%**, locked.
3. Family-wide error rate for the main (fixed-coordinate) family of
   comparisons, under Holm correction: **overall 0.05**, locked.
4. Candidate grid of (versions R, trajectories-per-version K) pairs:
   **R in {10, 15, 20, 25}, K in {3, 5, 7, 10}**, locked as the
   starting grid.
5. **Selection rule**: using the crossed variance parts estimated from
   the pilot (using a careful, conservative choice: prefer the upper
   confidence bound over the point estimate, where possible), simulate
   the planned version-level analysis (the mean-effect test above)
   over every candidate (R, K) pair. Select the lowest-*cost* design
   (cost = R x K, the total trajectory-runs needed) that reaches at
   least the planned power for `delta_min`, under Holm-adjusted alpha.

**Fixed tie-breaks for the selection rule, to close the remaining open
operational questions:**
- **One common (R, K) across all three random-control families**, not
  a separately-optimized allocation per family. The team selects this
  using whichever of the three families' pilot variance estimates is
  most demanding (needs the largest (R, K) to reach the power target).
- **If multiple candidate designs tie on lowest cost**, choose the one
  with the larger R. The graph version is the unit of analysis, so
  more versions is the more informative direction to break the tie,
  over more trajectories per version.
- **Conservative variance estimate, defined precisely**: use a
  set-in-advance 95% upper confidence bound on the relevant variance
  part, where it can be reliably estimated from the 3x3 pilot. Where
  it cannot be reliably estimated at that sample size, use the larger
  of the point estimate and a named conservative fallback (proposed:
  double the point estimate), rather than leaving "where feasible"
  undefined.
- **Power under Holm, made runnable**: "Holm-adjusted alpha" alone is
  not a single, runnable target, since Holm's per-comparison threshold
  depends on the order of the actual observed p-values, not a fixed
  alpha per test. So use either (a) simulate the complete
  four-comparison Holm procedure jointly across candidate (R, K)
  pairs, or (b), as a simpler and more conservative approximation,
  give each random-control comparison power individually at
  alpha=0.0125 (0.05/4, the Bonferroni bound that Holm can only
  improve on). Prefer option (a) if the simulation is practical within
  the pilot's time limit. Document which option was used, either way.

Stage 1C's own finding was low trajectory-to-trajectory variance for T
(CV about 5.2%). Given that, more graph versions are plausibly more
valuable than more trajectories per version. But this is an
expectation that the pilot's crossed variance breakdown should confirm
with real data, not simply assume. Lock the final (R, K) from this
rule's output *before* looking at any confirmed-run results. "10 to 20
versions, 3 to 5 trajectories" is a plausible range this rule might
land on. It is not itself the rule.

## Locked Confirmed-Run Allocation: (R=25, K=3)

The 3x3 pilot (`PILOT_RESULTS.md`) first produced a temporary common
design of (R=15, K=3). This was selected from rewired's and
curr_random's own minimal needs, while explicitly flagging
hist_random's own estimate as **not reliable**. Hist_random's crossed
variance breakdown was fit on only 2 of 3 versions. (One, seed=2, was
left out for a fixed-coordinate problem: an isolated intervention node
with zero weighted degree in that version. See `PILOT_RESULTS.md`, "A
Real Finding From the Pilot.") This gave df_r=1, below this project's
own reliability threshold (df_r of 3 or more) for a proper confidence
bound on the between-version variance part.

A follow-up (`PILOT_RESULTS.md`, "Follow-Up: Hist_random Variance
Re-Check (Seeds 3, 4)") drew two more hist_random versions. This
reached df_r=3 and a reliable 95% chi-squared upper confidence bound
on both variance parts. Hist_random's own minimal design, under this
reliable estimate, is **(R=25, K=3), cost 75**. The team confirmed
this is the true minimum over the full candidate grid. It is larger
than rewired's and curr_random's own (15, 3) needs, which this refit
leaves unchanged.

**This section's lock is a mechanical use of the pre-existing
selection rule above, not a new judgment call**: "one common (R, K)
across all three random-control families... selected using whichever
of the three families' pilot variance estimates is most demanding."
With hist_random's estimate now reliable, it is the most demanding of
the three. **The locked common allocation for the confirmed run is
(R=25, K=3), cost 75 per random-control family (225 total across
rewired, hist_random, curr_random).** This replaces the earlier
temporary (R=15, K=3). Rewired and curr_random run at this same (25,
3) allocation despite their own lower individual needs, per the
one-common-design rule. This is not because their own pilot estimates
changed.

## Historical-Random: Pre-Screening and a Conditional Estimand

The pilot found a real, disclosed failure pattern specific to
historical-random (`PILOT_RESULTS.md`, "A Real Finding From the
Pilot"). Because hist_random places edges by independent resampling,
at roughly half T's edge density, a real share of draws isolate one of
T's three fixed intervention coordinates (zero weighted degree at that
node in that version). An isolated node's response is close to
perfectly linear, so it fails the tangent-departure validity check at
every start time except 0. This makes the fixed-coordinate pattern
undefined for that version. Of 5 pilot draws (seeds 0-4): 1 was fully
undefined, 1 was mildly undefined (and survived), and 3 were clean.
This is not frequent enough to call the method broken, but it is not a
one-time event either.

**Rewired does not have this problem.** Degree-preserving rewiring
keeps each node's exact degree fixed by design, so it can never
isolate a node that T itself did not already have at that same degree.
This protocol is therefore specific to hist_random (and, in principle,
curr_random, which showed one mild case in the pilot but no full
undefined case). The team states it here for hist_random specifically,
since that is where the confirmed run's sizing depends on it.

**Protocol, locked before any confirmed-run trajectory simulation
starts:**

1. **Pre-screen every candidate hist_random version before simulating
   anything.** Compute the weighted degree of all three of `nodes_T`'s
   fixed coordinates (low, median, high) in the candidate graph. This
   is a cheap, static graph check that needs no simulation. If any of
   the three is zero (isolated), reject the version without running
   any trajectory on it. This directly matches the already-confirmed
   mechanism: an isolated node reliably fails `event_aligned_valid` at
   every start time except 0.
2. **Draw replacement candidates until 25 usable versions are obtained
   for hist_random specifically.** Rewired and curr_random do not need
   this replacement-draw step, since they do not show this failure
   pattern at a rate that calls for it.
3. **Record every rejected candidate.** Report the rejection rate,
   with a binomial confidence interval, as a disclosed property of the
   hist_random family under this protocol. Do not silently discard it,
   and do not fold it into the main estimate's sample size.
4. **The main hist_random comparison is conditional on
   fixed-coordinate usability.** State this explicitly, both in this
   document and in whatever findings document reports the confirmed
   result: the score being tested for hist_random is
   `E[Delta_T - Delta_hist_random | evaluable]`, not an unconditional
   claim about the full hist_random family. The unconditional
   question, how often hist_random even produces a usable
   fixed-coordinate comparison at all, is answered separately, by the
   rejection rate from step 3. It must not be folded into the main
   theta_g estimate above.

## What the Permutation Test Does and Does Not Answer

The topology-specificity comparison above works directly on Delta_map
*values* (the d_grk differences and their combined results). This is
a separate question from Stage 1B2/1C's own 10,000-permutation test,
which asks, per trajectory: does *this* graph and trajectory contain a
structured pattern at all? Do not combine per-trajectory permutation
p-values into the topology-family test. The family-level test uses
Delta_map point values, not permutation significance.

**A trajectory stays in the topology comparison regardless of its own
permutation test's result.** The 10,000-permutation test is reported
alongside every trajectory as a validity check, not used as a filter.
A trajectory whose own pattern test is not significant still
contributes its Delta_map value to the topology-family comparison,
exactly like any other trajectory. Silently dropping non-significant
trajectories would turn a disclosed validity check into a hidden
exclusion step. It would bias the comparison toward whichever
construction happens to produce more individually-significant
trajectories, rather than measuring the real quantity of interest (the
Delta_map advantage).

**Every trajectory in the confirmed run (T, lattice, and all
random-control versions) gets the full 10,000-permutation test.** This
matches Stage 1B2/1C's own convention exactly, so significance is
checked the same way across every construction. The team reports this
as a validity check. It confirms each construction/trajectory pair
that enters the comparison actually shows a structured pattern in the
first place, the same precondition Stage 1B2/1C established for T. It
is not an input to the topology-family test itself, which is answered
by the mean-effect analysis above, regardless of each trajectory's own
permutation result.

## Rules That Must Hold in the Implementation

**Identical experimental randomness across constructions, wherever
mathematically possible.** Use the same baseline initial-phase seeds,
the same nearby-state replica offset seeds, the same perturbation
times, the same sign and size grid, the same event-alignment rule, and
the same numerical solver settings (RTOL, ATOL, MAX_STEP). Reuse these
exactly from `stage1b2_core.py` and Stage 1C's constants, across every
construction. Otherwise, topology differences would become mixed up
with different replica neighborhoods, instead of staying isolated.

**For role-matched node selection specifically**, record for every
graph version:
- the selected node indices (low, median, high-degree)
- their weighted degrees and degree percentiles
- the fixed tie-breaking rule applied
- an explicit confirmation that all three selected nodes are distinct

This matters especially for constructions with compressed or heavily
tied degree patterns, in particular the lattice construction. It has
uniform edge weights and is likely to produce many degree ties. This
makes low, median, and high selection ambiguous without a disclosed,
fixed tie-break rule. (See the undefined-role-matching rule above,
which takes priority if a construction's ties are severe enough that
no genuine grouping exists.)

**Common-support mask, defined separately for each intervention
method. This matters because this project has already been caught
once by an exclusion mask that leaked input identity through its own
position** (Stage 1B2's three-round q_excl_node fix; see `CLAUDE.md`'s
principle 11). So the team states this rule explicitly, rather than
assuming it carries over correctly.
- **Fixed-coordinate protocol**: straightforward. The mask removes the
  same three T-defined candidate coordinates from every trial, in
  every construction, identically.
- **Role-matched protocol**: each graph version has its *own* three
  candidate source coordinates (that version's own low, median,
  high-degree nodes). The mask must be (a) identical across every
  trial *within* that version. All three of that version's own
  candidates are zeroed, regardless of which one was actually
  stimulated in a given trial. This exactly matches the common-support
  principle already established for Stage 1B2. And the mask must be
  (b) explicitly documented as version-specific when comparing across
  versions, since different versions' masks cover different coordinate
  positions. Do not reuse one version's mask for another version's
  trials.

## Files to Create

```
experiments/stage1d_topology_specificity/
  DESIGN.md                          <- this document, moved here once
                                         the folder is created (see below)
  run_stage1d.py                     <- driver: builds each construction,
                                         runs the trajectory grid under
                                         both intervention definitions
  analyze_stage1d.py                 <- main test and reporting
  FINDINGS.md                        <- filled in after analysis
  results/                           <- gitignored (not saved in git)
```

**Note on folder naming**: at the time of writing, this document lives
in `experiments/stage1d/`, matching where `IDEAS.md` was. Once work on
the design starts, consider renaming the folder to
`experiments/stage1d_topology_specificity/`. This matches this
project's set naming pattern for stage folders (`stage1a_...`,
`stage1b_...`, `stage1c_...` all carry a descriptive name after the
stage number). This is a small, low-risk housekeeping step, not a
design decision. It is safe to leave until the first implementation
commit.

## What This Design Does Not Do

- Does not address Level 3 (external usefulness). This is a separate,
  larger question, scoped differently.
- Does not extend to classes 1-9 in this first pass. This is
  explicitly left for later, pending the class-0 result.
- Does not treat the concentration/routing pattern as a main score.
  Checking whether it appears in other constructions may be a
  reasonable follow-up once the main comparison is complete.
- Does not draw confirmed topology-specificity conclusions from the
  3x3 pilot run. The pilot's role is limited to estimating the two
  variance parts that set the final, separately-locked version and
  trajectory counts.
- Does not let a significant role-matched result rescue a failed
  fixed-coordinate result, or the other way around. Disagreement
  between the two intervention kinds is reported as mechanistic
  information, not resolved by picking whichever one was significant.

## Review Status

The design went through three rounds of outside statistical review.
The team took in every point raised.

**Round 1** flagged the main test's core reasoning error: comparing
T's 10 trajectory-level values against version-level control means,
using an unpaired Mann-Whitney test, which mixes up two different
inferential levels. Verdict: "scientifically approved; the statistical
test structure needs one revision before locking." The team addressed
this with the matched-trajectory-seed design, version-level
combination, and the paired lattice comparison.

**Round 2** resolved seven remaining points: matching the main test to
the mean estimand, a pilot-to-confirmed-run allocation rule, a crossed
variance model, an undefined-role-matching rule, per-protocol
common-support masking, a CV instability safeguard, and clarifying the
permutation test's role. Verdict: "scientifically approved. Lock after
specifying the main mean-effect test, the pilot-to-confirmed-run
allocation rule, and the treatment of undefined role matching."

**Round 3** closed the remaining operational gaps that Round 2's fixes
had introduced: the main test was locked to a single choice (two-sided
one-sample t-test, not "t-test or bootstrap"); all "proposed, confirm
before locking" values (delta_min=0.05, power=80%, family-wide error
rate=0.05, the R/K candidate grid, the CV floor=0.05) were adopted as
final, locked values; the design made explicit that a trajectory stays
in the topology comparison regardless of its own permutation test's
significance; and it specified that an undefined lattice
role-matching condition reduces the role-matched Holm family to 3
tests, not 4 padded slots.

**Round 4** (this revision) closed three remaining operational gaps.
The design was scientifically locked, but not yet fully runnable, on
these points: (1) the lattice main test was pinned to a single choice
(two-sided paired t-test, matching the same mean-effect logic as the
random-control comparisons, not "signed-rank or sign-flip"); (2) the
pilot-to-confirmed-run allocation rule was given fixed tie-breaks: one
common (R, K) across all three random families, selected from the
most demanding one; larger-R as the cost-tie preference; a
precisely-defined conservative variance estimate; and a runnable
Holm-power rule; (3) the heading "Additional details to lock before
implementation" was renamed to "Rules That Must Hold in the
Implementation," since its contents are now locked prescriptions, not
open decisions.

The design is considered fully locked and operationally runnable. No
further review points remain open.
