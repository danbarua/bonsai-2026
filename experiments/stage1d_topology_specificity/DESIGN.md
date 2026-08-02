# Stage 1D: Topology Specificity of the Stage 1B.2/1C Mapping

## The question

Stage 1B.2 established a significant, reproducible structured internal
transformation (Delta_map ~0.35, permutation floor) on one baseline
trajectory using KMNIST class 0's learned topology T. Stage 1C showed
this holds across 10 independent trajectories on the same T (mean
Delta_map 0.3296, CV ~5.2%, all 10 hitting the Monte Carlo floor).
Neither result establishes that T itself is doing anything special --
both are entirely consistent with the mapping being a generic property
of graph-oscillator dynamics on *any* graph with T's broad statistics
(size, density, degree distribution), not something specific to what T
learned from the image population.

**Stage 1D asks directly**: does T produce this structured
transformation more strongly, more stably, or qualitatively differently
than matched controls, under the identical Stage 1B.2/1C protocol?

This is the second of the three open items in Stage 1B.2's own priority
list (`docs/PROJECT_MEMORY.md`) -- generalization across trajectories
(item 1) is now closed by Stage 1C; this closes item 2. External
usefulness (item 3, "Level 3") remains untouched by this design and
would need its own, separately-scoped effort -- it is a qualitatively
different question (can the mapping be linked to anything outside the
system) from a specificity comparison, and is not addressed here.

## What's ready, and what a "Stage 2" would actually be

Every construction this design needs already exists and is
independently verified: `degree_preserving_rewiring.py`,
`historical_matched_sparsity_random.py`, `matched_sparsity_ablation.py`,
and `lattice_construction.py`, all reproducing their respective
historical cached artifacts byte-exact (or, for the historical random
construction, structurally verified with the byte-exact gap explicitly
documented). This design was not executable before tonight's
construction-recovery work; it is now.

If T does not outperform or meaningfully differ from the matched
controls, that does not close the dynamics-as-computation programme --
it changes the scope of the claim from "the learned Bonsai topology
specifically supplies the computational advantage" to "Bonsai-style
graph-oscillator dynamics produce this structured transformation on
graphs with these general statistics." Both are genuine, reportable
findings; this design is built to distinguish between them, not to
presuppose which one is true.

## Primary endpoint, and what stays secondary

**Primary**: trajectory-level Delta_map, computed identically to
Stage 1B.2/1C's own definition (pooled across the 4 t_p values, same
432-trial-per-trajectory design, same permutation test). This is the
prespecified omnibus statistic this whole programme has used since
Stage 1B.2; it stays primary here for direct comparability.

**Secondary, named precisely rather than referred to broadly.** Stage
1B.2 produced two distinct common-support quantities, and this design
compares both explicitly rather than picking one implicitly:
1. common-support omnibus, Delta_map^(-S)
   (`analyze_stage1b2_common_support_exclusion.py`)
2. node-specific restricted, Delta_node^(-S)
   (`analyze_stage1b2_common_support_node_test.py`)

The research question asks whether T is "stronger, more stable, or
qualitatively different" -- each gets its own named estimand rather than
being folded into the primary Delta_map comparison alone:
- **Strength**: mean trajectory-level Delta_map (the primary comparison
  above).
- **Stability**: within-graph SD of Delta_map across the matched
  trajectories is the formal stability measure, always reported. CV
  (SD/mean) is reported only when a graph's mean Delta_map exceeds
  **0.05** (locked -- roughly T's own between-trajectory SD from
  Stage 1C, chosen so CV is only computed where the mean is comfortably
  away from zero) -- CV becomes unstable or misleading as the mean
  approaches zero, which a construction showing weak or no structured
  mapping plausibly could. Between-realization variance (stochastic
  controls) and within-trajectory SD are measured at different levels
  and are reported with that level labeled explicitly, not treated as
  directly comparable numbers.
- **Linear vs. nonlinear composition**: tangent-only and residual
  Delta_map, checking whether any topology-specific effect lives in the
  linear or nonlinear part of the response.
- **Source-independent structure**: both common-support quantities
  above.
- **"Qualitatively different"** stays descriptive rather than a formal
  estimand, unless a specific phenotype is prespecified before running
  anything -- it is not a license to go looking for *some* difference
  post hoc.

**Explicitly not primary**: the concentration/routing phenotype
(`CONCENTRATION_REGIME_NOTE.md`). That mechanistic thread is valuable
and may be worth checking across constructions later, but it is a
single-cell (high-degree-node, t_p=0) diagnostic discovered post hoc,
not a prespecified estimand. Do not let a construction's concentration
behavior substitute for its Delta_map result.

## Two intervention definitions, run and reported separately

These test different things and must not be conflated into one number.

1. **Fixed graph coordinates.** Perturb the same three pixel-linked
   node indices (T's own low/median/high-degree nodes, as already used
   throughout Stage 1B.2/1C) in every construction, regardless of what
   role those indices play in that construction's own degree
   distribution. This isolates the effect of changing *which graph*
   connects a fixed set of physical points, holding the intervention
   itself constant.
2. **Role-matched nodes.** Select each construction's *own*
   low/median/high-degree nodes independently (via
   `get_degree_stratified_nodes`, already used throughout this project),
   so the comparison is "does T's structure matter under equivalent
   structural roles" rather than "does T's structure matter at these
   specific pixels."

**Degenerate role-matching, locked explicitly rather than papered over
with tie-breaking.** A deterministic tie-break is appropriate for
choosing among several nodes *within* a genuine degree stratum -- it is
not a substitute for that stratum existing. If a construction's degree
distribution doesn't support three distinguishable strata (a real risk
for the lattice construction specifically, whose uniform edge weights
mean weighted degree reduces to simple connectivity count, plausibly
producing many exact ties), role-matched intervention is declared
**degenerate** for that construction and reported as such -- arbitrarily
selecting three tied nodes and calling them "low/median/high" would
manufacture a role distinction that doesn't actually exist, not measure
one.

**If lattice's role-matched condition is degenerate, the role-matched
Holm family contains 3 tests, not 4 padded slots.** Holm correction
operates over the p-values actually produced; a degenerate comparison
produces no p-value to correct, so it is excluded from the family
rather than retained as an empty placeholder. Report this explicitly
(e.g. "role-matched family: 3 of 4 planned comparisons, lattice
degenerate") so the reduced family size is never mistaken for 4 tests
having been run.

Report both. If they agree, that's a robust specificity finding either
way. If they disagree, that disagreement is itself the finding --
report which nodes differ and why, don't average over it.

## Scope and inferential hierarchy

**Primary scope: class 0 only**, using the construction pipeline and
10-trajectory design already built and verified. This establishes (or
fails to establish) topology specificity for this one learned graph --
a real, bounded claim, not yet a general one about "Bonsai topologies."

**A general claim would require repeating this across multiple KMNIST
classes**, with class as the inferential unit (same logic as the
Stage 1A re-verification's class-level design) -- explicitly out of
scope for this design's first pass; a natural, separately-decided
extension if the class-0 result is positive and worth generalizing.

**The nesting that matters**: trajectories within one graph realization
are repeated observations of *that* graph, not independent replicates
of the topology family. For T and lattice (deterministic constructions,
no seed), this collapses to a single realization with 10 trajectories
each (T's already exists via Stage 1C; lattice needs its own 10-trajectory
run). For the three stochastic constructions (rewiring,
historical-random, current-random), the correct hierarchy is:

```
trajectory  (10 per realization, matching Stage 1C's seed convention)
  |_ graph realization  (a specific seed's construction)
       |_ topology family  (rewiring / hist-random / curr-random)
```

**Number of graph realizations for stochastic controls**: settled by a
dedicated pilot rather than fixed here in advance -- see "Pilot vs.
confirmatory" below, which specifies the pilot's scope, purpose, and
the explicit rule that it cannot be used for confirmatory inference.

## Primary test structure

**The core fix, per review**: the original proposal compared T's 10
Stage 1C trajectory-level values against each stochastic control's
realization-level means via unpaired Mann-Whitney -- this treats T's 10
repeated trajectories as if they were 10 independent topology
realizations, while correctly collapsing each control to one
realization-level value per graph. That's an inconsistent comparison
across inferential levels, not just a suboptimal test choice. Replaced
below with a matched design that fixes this.

**Matched trajectory seeds.** Use the same baseline trajectory seeds for
T and for every stochastic control's graph realizations -- not
independently re-randomized per construction. This makes trajectory-to-
trajectory variation a controllable, paired nuisance factor rather than
an uncontrolled source of noise contaminating the between-construction
comparison. The confirmatory run uses **K matched trajectory seeds**,
where K is determined by the pilot-driven allocation rule below, not
fixed at 10 in advance -- drawn as the first K of Stage 1C's 10 seeds
(3000, 3010, 3020, ...) in order, for direct comparability with T's
already-computed values. (T and lattice, both deterministic, still use
the full 10 -- see below.)

**Definitions.** For stochastic construction g, graph realization r,
and trajectory seed k (k = 1..K, shared across all constructions):

```
d_grk = Delta_map(T, k) - Delta_map(g, r, k)
```

Aggregate within realization first (mean over the K matched
trajectories):

```
d_bar_gr = (1/K) * sum_k(d_grk)
```

The **graph realization**, not the trajectory, is the inferential unit
for stochastic controls. The estimand is:

```
theta_g = E_r[d_bar_gr]
```

-- the expected advantage of this specific learned T over a graph drawn
from control family g, conditional on class 0 and the tested
intervention protocol.

**Primary test: two-sided one-sample t-test, locked as the single
primary decision rule.** theta_g is explicitly an expectation, not a
median or location parameter -- a signed-rank or sign-flip test targets
the latter under a symmetry assumption, which doesn't test theta_g
itself. **Locked primary test**: two-sided one-sample t-test on the
realization-level mean differences mean_r(d_bar_gr), Holm-corrected
across the four fixed-coordinate comparisons. Two-sided, not one-sided:
the Stage 1D question explicitly allows T to be stronger, weaker, or
simply different from a control -- not only "T is stronger" -- so a
one-sided test would silently narrow the question being asked. The
studentized bootstrap interval, Wilcoxon signed-rank test, and exact
sign-flip test on {d_bar_gr} are retained as robustness analyses, not
as alternative primary tests to choose between after seeing the data.

**Primary reporting**, for each stochastic construction g:
- every realization-level d_bar_gr value, not just the test statistic
- mean realization-level difference (primary estimate) and median
  (robustness)
- the two-sided t-test result (primary), plus the studentized bootstrap
  interval, signed-rank, and sign-flip results (robustness)
- proportion of control realizations outperforming T (d_bar_gr < 0)
- within-realization trajectory variability (SD of d_grk across the K
  matched k's, per realization), reported using the crossed variance
  decomposition described in "Pilot vs. confirmatory" below, not a
  naive per-realization SD

**Robustness**: studentized bootstrap interval, Wilcoxon signed-rank,
and exact sign-flip tests on {d_bar_gr} (see above), plus a
hierarchical bootstrap resampling both graph realizations and matched
trajectory seeds, analogous to the Stage 1A re-verification's
bootstrap.

**Lattice is different: deterministic, so pair directly.** T and
lattice both require no seed -- there is no realization dimension to
aggregate over. Use the same 10 matched trajectory seeds directly:

```
d_k = Delta_map(T, k) - Delta_map(lattice, k),  k = 1..10
```

**Primary test, locked to a single choice, aligned with the same
mean-effect logic as the stochastic-control comparisons**: two-sided
paired t-test on the 10 d_k values (T's values already exist via
Stage 1C; lattice needs its own 10 trajectories on the identical
seeds). Exact sign-flip and exact signed-rank tests on the same 10
values are retained as robustness, not as alternative primary tests to
choose between after seeing the data -- the same "or" ambiguity already
fixed for the stochastic-control primary test. An unpaired test here
would discard a naturally available, more powerful paired block for no
reason. The resulting claim stays bounded: this class-0 learned graph
differs, or does not differ, from this deterministic lattice across the
10 sampled initial trajectories -- not a population-level claim about
learned-vs-lattice topology families in general.

**Fixed-coordinate is primary; role-matched is secondary robustness --
locked before running anything, not decided after seeing results.**
Both intervention definitions could otherwise be used to declare
"topology specificity," creating two chances at a positive result.
Fixed-coordinate holds the physical intervention constant while
changing only the graph, which is the more direct test of the
question; role-matched changes node identity too and answers a
related-but-different question (does structure matter under equivalent
structural roles). Concretely:
- Holm-correct the 4 fixed-coordinate construction comparisons
  (rewired, hist_random, curr_random, lattice) as the primary family.
- Holm-correct the 4 role-matched comparisons as a separate family.
- **Role-matched significance cannot rescue a failed fixed-coordinate
  result.** If the two disagree, report the disagreement itself as
  mechanistic information (which nodes differ and why), not something
  to resolve by picking whichever intervention definition happened to
  be significant.

## Pilot vs. confirmatory: the 3x3 run is not confirmatory

With only 3 graph realizations per stochastic construction, even an
all-positive one-sided exact sign test on {d_bar_gr} has
p_min = 1/2^3 = 0.125 -- no multiplicity correction rescues that. The
3-realizations-by-3-trajectories run must therefore be labeled and
treated explicitly as:

> **Runtime and variance-allocation pilot. No confirmatory
> topology-specificity inference will be drawn from this run.**

**Variance decomposition, corrected: the pilot's data are crossed, not
simply nested.** Because the same 3 trajectory seeds are shared across
every graph realization (per the matched-seed design above), a naive
variance of the 3 realization-level means conflates true between-graph
variance with residual trajectory-sampling noise. The correct model is:

```
d_grk = mu_g + b_gr + tau_k + epsilon_grk
```

where `b_gr` is the graph-realization effect, `tau_k` is the shared
trajectory-seed block effect (the same k represents the same baseline
trajectory identity across every realization), and `epsilon_grk` is the
graph-by-trajectory remainder. Fit this via a crossed variance
decomposition or mixed model (realization and trajectory-seed as
crossed random effects) for pilot allocation -- not the raw variance of
`d_bar_gr` across the 3 realizations, which is not a pure estimate of
`sigma^2_between-graphs`. With only 3x3 data, all resulting variance
estimates will be rough -- suitable for budget allocation, not
substantive claims.

**The pilot-to-confirmatory allocation rule, locked before the pilot is
run, not decided after seeing its results:**

1. Minimum scientifically meaningful difference `delta_min` in
   Delta_map: **0.05**, locked -- informed by Stage 1C's own
   between-trajectory SD for T of ~0.017 (delta_min set at roughly 3x
   that SD, so the design targets a difference clearly outside T's own
   observed trajectory noise, not merely detectable in principle).
2. Desired power: **80%**, locked.
3. Familywise alpha for the primary (fixed-coordinate) family of
   comparisons under Holm correction: **overall FWER 0.05**, locked.
4. Candidate grid of (realizations R, trajectories-per-realization K)
   pairs: **R in {10, 15, 20, 25}, K in {3, 5, 7, 10}**, locked as the
   starting grid.
5. **Selection rule**: using the crossed variance components estimated
   from the pilot (conservatively -- prefer the upper confidence bound
   over the point estimate where feasible), simulate the planned
   realization-level analysis (the mean-effect test above) over every
   candidate (R, K) pair. Select the lowest-*cost* design (cost = R x K,
   the total trajectory-runs needed) that achieves at least the
   prespecified power for `delta_min` under Holm-adjusted alpha.

**Deterministic tie-breaks for the selection rule, closing the
remaining operational ambiguity:**
- **One common (R, K) across all three stochastic-control families**,
  not a separately-optimized allocation per family -- selected using
  whichever of the three families' pilot variance estimates is most
  demanding (i.e. requires the largest (R, K) to hit the power target).
- **If multiple candidate designs tie on minimum cost**, choose the one
  with the larger R -- graph realization is the inferential unit, so
  more realizations is the more informative tie-break direction over
  more trajectories per realization.
- **Conservative variance estimate, defined precisely**: use a
  prespecified 95% upper confidence bound on the relevant variance
  component where it's estimable from the 3x3 pilot; where it isn't
  reliably estimable at that sample size, use the larger of the point
  estimate and a named conservative fallback (proposed: 2x the point
  estimate) rather than leaving "where feasible" undefined.
- **Power under Holm, made executable**: "Holm-adjusted alpha" alone
  isn't a uniquely executable target, since Holm's per-comparison
  threshold depends on the ordering of the realized p-values, not a
  fixed alpha per test. Either (a) simulate the complete four-comparison
  Holm procedure jointly across candidate (R, K) pairs, or (b) as a
  simpler, more conservative approximation, power each stochastic
  comparison individually at alpha=0.0125 (0.05/4, the Bonferroni bound
  Holm can only improve on). Prefer (a) if the simulation is tractable
  within the pilot's timeframe; document which was used either way.

Given Stage 1C's own finding of low trajectory-to-trajectory variance
for T (CV ~5.2%), more graph realizations are plausibly more valuable
than more trajectories per realization -- but this is an expectation the
pilot's crossed variance decomposition should confirm empirically, not
assume. Lock the final (R, K) from this rule's output *before* looking
at any confirmatory-run results -- "10 to 20 realizations, 3 to 5
trajectories" is a plausible range this rule might land on, not itself
the rule.

## Locked confirmatory-run allocation: (R=25, K=3)

The 3x3 pilot (`PILOT_RESULTS.md`) initially produced a provisional
common design of (R=15, K=3), selected from rewired's and curr_random's
own minimal requirements while explicitly flagging hist_random's own
estimate as **indeterminate** -- its crossed variance decomposition was
fit on only 2 of 3 realizations (one, seed=2, excluded for a fixed-
coordinate degeneracy -- an isolated intervention node with zero
weighted degree in that realization; see `PILOT_RESULTS.md`, "A real
finding surfaced by the pilot itself"), giving df_r=1, below this
project's own reliability threshold (df_r>=3)
for a proper confidence bound on the between-realization variance
component.

A follow-up (`PILOT_RESULTS.md`, "Follow-up: hist_random variance
re-estimation (seeds 3, 4)") drew two further hist_random realizations,
reaching df_r=3 and a reliable 95% chi-squared upper confidence bound on
both variance components. hist_random's own minimal design under this
reliable estimate is **(R=25, K=3), cost 75** -- confirmed as the true
minimum over the full candidate grid, and larger than rewired's and
curr_random's own (15, 3) requirements, which this refit leaves
unchanged.

**This section's lock is a mechanical application of the pre-existing
selection rule above, not a new judgment call**: "one common (R, K)
across all three stochastic-control families... selected using
whichever of the three families' pilot variance estimates is most
demanding." With hist_random's estimate now reliable, it is the most
demanding of the three. **The locked common allocation for the
confirmatory run is (R=25, K=3), cost 75 per stochastic-control family
(225 total across rewired, hist_random, curr_random)**, superseding the
earlier provisional (R=15, K=3). Rewired and curr_random run at this
same (25, 3) allocation despite their own lower individual requirement,
per the one-common-design rule -- not because their own pilot estimates
changed.

## Historical-random: pre-screening and a conditional estimand

The pilot surfaced a real, disclosed failure mode specific to
historical-random (`PILOT_RESULTS.md`, "A real finding surfaced by the
pilot itself"): because hist_random places edges by independent
resampling at roughly half T's edge density, a nontrivial fraction of
draws isolate one of T's three fixed intervention coordinates (zero
weighted degree at that node in that realization) -- an isolated node's
response is (near-)perfectly linear, so it fails the tangent-departure
validity gate at every t_p except 0, making the fixed-coordinate mapping
undefined for that realization. Of 5 pilot draws (seeds 0-4): 1 fully
degenerate, 1 mildly degenerate (survived), 3 clean -- not frequent
enough to call the protocol broken, but not a one-off either.

**Rewired does not have this problem** -- degree-preserving rewiring
holds each node's exact degree fixed by construction, so it can never
isolate a node T itself didn't already have at that degree. This
protocol is therefore specific to hist_random (and, in principle,
curr_random, which showed one mild instance in the pilot but no full
degeneracy); it is stated here for hist_random specifically since that
is where the confirmatory run's sizing depends on it.

**Protocol, locked before any confirmatory trajectory simulation runs:**

1. **Pre-screen every candidate hist_random realization before
   simulating anything.** Compute the weighted degree of all three of
   `nodes_T`'s fixed coordinates (low/median/high) in the candidate
   graph -- a cheap, static graph check, no simulation needed. If any of
   the three is zero (isolated), reject the realization without running
   any trajectory on it. This directly matches the already-confirmed
   mechanism: an isolated node trivially fails `event_aligned_valid` at
   every t_p except 0.
2. **Draw replacement candidates until 25 evaluable realizations are
   obtained for hist_random specifically.** Rewired and curr_random do
   not need this replacement-draw step, since they do not show this
   failure mode at a rate that warrants it.
3. **Record every rejected candidate.** Report the rejection rate (with
   a binomial confidence interval) as a disclosed secondary
   characteristic of the hist_random family under this protocol -- not
   silently discarded, and not folded into the primary estimate's
   sample size.
4. **The primary hist_random comparison is conditional on fixed-
   coordinate evaluability.** State explicitly, in this document and in
   whatever findings document reports the confirmatory result: the
   estimand being tested for hist_random is `E[Delta_T - Delta_hist_random
   | evaluable]`, not an unconditional claim about the full hist_random
   family. The unconditional question -- how often hist_random even
   produces a usable fixed-coordinate comparison at all -- is answered
   separately, by the rejection rate from step 3, and must not be
   folded into the primary theta_g estimate above.

## What the permutation test does and does not answer

The topology-specificity comparison above operates directly on
Delta_map *values* (the d_grk differences and their aggregates) -- it
is a separate question from Stage 1B.2/1C's own 10,000-permutation test,
which asks, per trajectory: does *this* graph and trajectory contain a
structured mapping at all? Do not combine per-trajectory permutation
p-values into the topology-family test; the family-level test consumes
Delta_map point estimates, not permutation significance.

**A trajectory remains in the topology comparison regardless of its own
permutation test's outcome.** The 10,000-permutation test is reported
alongside every trajectory as a validation check, not used as a filter
-- a trajectory whose own mapping test is non-significant still
contributes its Delta_map value to the topology-family comparison
exactly like any other. Silently dropping non-significant trajectories
would turn a disclosed validation check into an undocumented exclusion
step, biasing the comparison toward whichever construction happens to
produce more individually-significant trajectories rather than
measuring the actual quantity of interest (the Delta_map advantage).

**Every trajectory in the confirmatory run (T, lattice, and all
stochastic-control realizations) gets the full 10,000-permutation test**,
matching Stage 1B.2/1C's own convention exactly, so significance is
assessed identically across every construction. This is reported as a
validation check (confirming each construction/trajectory combination
that enters the comparison actually shows a structured mapping in the
first place, the same precondition Stage 1B.2/1C established for T) --
not as an input to the topology-family test itself, which is answered
by the mean-effect analysis above regardless of each trajectory's own
permutation result.

## Implementation invariants

**Identical experimental randomness across constructions, wherever
mathematically possible.** Same baseline initial-phase seeds, same
nearby-state replica offset seeds, same perturbation times, same
sign/amplitude grid, same event-alignment rule, same numerical solver
settings (RTOL, ATOL, MAX_STEP), reused exactly from
`stage1b2_core.py`/Stage 1C's constants across every construction.
Otherwise topology differences become entangled with different replica
neighborhoods rather than isolated.

**For role-matched node selection specifically**, record for every
graph realization:
- selected node indices (low/median/high-degree)
- their weighted degrees and degree percentiles
- the deterministic tie-breaking rule applied
- explicit confirmation that all three selected nodes are distinct

This matters especially for constructions with compressed or heavily
tied degree distributions -- the lattice construction in particular,
which has uniform edge weights and is likely to produce many degree
ties, making low/median/high selection ambiguous without a disclosed,
deterministic tie-break (see the degenerate-role-matching rule above,
which takes precedence if a construction's ties are severe enough that
no genuine stratification exists).

**Common-support mask, defined separately for each intervention
protocol -- this project has already been burned once by an exclusion
mask that leaked input identity through its own position (Stage 1B.2's
three-round q_excl_node fix; see `CLAUDE.md`'s principle 11),
so this is worth stating explicitly rather than assuming it carries over
correctly.**
- **Fixed-coordinate protocol**: straightforward -- the mask removes the
  same three T-defined candidate coordinates from every trial, in every
  construction, identically.
- **Role-matched protocol**: each graph realization has its *own* three
  candidate source coordinates (that realization's own low/median/high-
  degree nodes). The mask must be (a) identical across every trial
  *within* that realization -- all three of that realization's own
  candidates zeroed regardless of which was actually stimulated in a
  given trial, exactly matching the common-support principle already
  established for Stage 1B.2 -- and (b) explicitly documented as
  graph-realization-specific when comparing across realizations, since
  different realizations' masks cover different coordinate positions.
  Do not reuse one realization's mask for another's trials.

## Files to create

```
experiments/stage1d_topology_specificity/
  DESIGN.md                          <- this document, moved here once
                                         the folder is created (see below)
  run_stage1d.py                     <- driver: builds each construction,
                                         runs the trajectory grid under
                                         both intervention definitions
  analyze_stage1d.py                 <- primary test + reporting
  FINDINGS.md                        <- populated after analysis
  results/                           <- gitignored
```

**Note on folder naming**: this document currently lives in
`experiments/stage1d/`, matching where `IDEAS.md` was. Once
implementation starts, consider renaming to
`experiments/stage1d_topology_specificity/` to match this project's
established naming convention for stage folders (`stage1a_...`,
`stage1b_...`, `stage1c_...` all carry a descriptive suffix) -- a small,
low-risk housekeeping step, not a design decision, and safe to defer
until the first implementation commit.

## What this does not do

- Does not address Level 3 (external usefulness) -- a separate,
  larger, differently-scoped question.
- Does not extend to classes 1-9 in this first pass -- explicitly
  deferred pending the class-0 result.
- Does not treat the concentration/routing phenotype as a primary
  endpoint, though checking whether it appears in other constructions
  may be a reasonable follow-up once the primary comparison is in hand.
- Does not draw confirmatory topology-specificity conclusions from the
  3x3 pilot run -- its role is limited to estimating the two variance
  components that determine the final, separately-locked realization
  and trajectory counts.
- Does not let role-matched significance rescue a failed
  fixed-coordinate result, or vice versa -- disagreement between the
  two intervention definitions is reported as mechanistic information,
  not resolved by picking whichever was significant.

## Review status

Three rounds of external statistical review, all incorporated.

**Round 1** flagged the primary test's core inferential error (comparing
T's 10 trajectory-level values against realization-level control means
via unpaired Mann-Whitney, conflating inferential levels). Verdict:
"scientifically approved; statistical test structure requires one
revision before lock." Addressed via the matched-trajectory-seed design,
realization-level aggregation, and paired lattice comparison.

**Round 2** resolved seven remaining points: aligning the primary test
with the mean estimand, a pilot-to-confirmatory allocation rule, a
crossed variance model, a degenerate-role-matching rule, per-protocol
common-support masking, a CV instability safeguard, and clarifying the
permutation test's role. Verdict: "scientifically approved. Lock after
specifying the primary mean-effect test, the pilot-to-confirmatory
allocation rule, and the treatment of degenerate role matching."

**Round 3** closed the remaining operational gaps Round 2's fixes had
introduced: the primary test was locked to a single choice (two-sided
one-sample t-test, not "t-test or bootstrap"); all "proposed/confirm
before locking" values (delta_min=0.05, power=80%, FWER=0.05, the R/K
candidate grid, the CV floor=0.05) were adopted as final, locked values;
made explicit that a trajectory stays in the topology comparison
regardless of its own permutation test's significance; specified that a
degenerate lattice role-matching condition reduces the role-matched
Holm family to 3 tests, not 4 padded slots.

**Round 4** (this revision) closed three remaining operational
ambiguities the design was scientifically locked but not yet fully
executable on: (1) the lattice primary test was pinned to a single
choice (two-sided paired t-test, matching the same mean-effect logic as
the stochastic-control comparisons, not "signed-rank or sign-flip"); (2)
the pilot-to-confirmatory allocation rule was given deterministic
tie-breaks -- one common (R, K) across all three stochastic families
selected from the most demanding one, larger-R as the cost-tie
preference, a precisely-defined conservative variance estimate, and an
executable Holm-power rule; (3) the "Additional details to lock before
implementation" heading was renamed "Implementation invariants," since
its contents are now locked prescriptions, not open decisions.

Considered fully locked, operationally executable. No further review
points outstanding.
