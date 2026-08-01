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
- **Stability**: within-graph SD/CV of Delta_map across the 10 matched
  trajectories, plus (for stochastic controls) the between-realization
  variance sigma^2_between-graphs estimated by the pilot.
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

**Matched trajectory seeds.** Use the identical 10 baseline trajectory
seeds (3000, 3010, ..., 3090, Stage 1C's own convention) for T and for
every stochastic control's graph realizations -- not independently
re-randomized per construction. This makes trajectory-to-trajectory
variation a controllable, paired nuisance factor rather than an
uncontrolled source of noise contaminating the between-construction
comparison.

**Definitions.** For stochastic construction g, graph realization r,
and trajectory seed k (k = 1..10, shared across all constructions):

```
d_grk = Delta_map(T, k) - Delta_map(g, r, k)
```

Aggregate within realization first (mean over the 10 matched
trajectories):

```
d_bar_gr = (1/10) * sum_k(d_grk)
```

The **graph realization**, not the trajectory, is the inferential unit
for stochastic controls. The estimand is:

```
theta_g = E_r[d_bar_gr]
```

-- the expected advantage of this specific learned T over a graph drawn
from control family g, conditional on class 0 and the tested
intervention protocol.

**Primary test**: one-sample exact signed-rank test (or exact sign-flip
test, whichever is more appropriate given the realization count decided
by the pilot -- see below) on the set of {d_bar_gr} values across
realizations r. Not Mann-Whitney against T's raw trajectory values --
that comparison is exactly what the review correctly identified as
invalid.

**Primary reporting**, for each stochastic construction g:
- every realization-level d_bar_gr value, not just the test statistic
- mean and median realization-level difference
- an interval estimate across graph realizations
- proportion of control realizations outperforming T (d_bar_gr < 0)
- within-realization trajectory variability (SD of d_grk across the 10
  matched k's, per realization)

**Robustness**: a hierarchical bootstrap resampling both graph
realizations and matched trajectory seeds, analogous to the Stage 1A
re-verification's bootstrap.

**Lattice is different: deterministic, so pair directly.** T and
lattice both require no seed -- there is no realization dimension to
aggregate over. Use the same 10 matched trajectory seeds directly:

```
d_k = Delta_map(T, k) - Delta_map(lattice, k),  k = 1..10
```

tested via exact paired signed-rank or exact sign-flip across these 10
matched seeds (T's values already exist via Stage 1C; lattice needs its
own 10 trajectories on the identical seeds). An unpaired test here would
discard a naturally available, more powerful paired block for no
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

Its actual purpose is estimating two variance components:
- sigma^2_between-graphs (variance of d_bar_gr across realizations r)
- sigma^2_within-graph-across-trajectories (variance of d_grk across
  the 10 matched k's, within one realization)

These estimates determine where the real computational budget belongs.
Given Stage 1C's own finding of low trajectory-to-trajectory variance
for T (CV ~5.2%), more graph realizations are plausibly more valuable
than more trajectories per realization -- but this is an expectation to
be checked empirically by the pilot, not assumed in advance. A
plausible final shape, subject to revision once the pilot's variance
estimates are in hand, is roughly **10-20 graph realizations with 3-5
matched trajectories each**, rather than the inverse (3 realizations
with 10 trajectories each). Lock the final realization/trajectory
counts from the pilot's variance estimates *before* looking at any
confirmatory-run results.

## Additional details to lock before implementation

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
deterministic tie-break.

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

Reviewed and revised per external statistical review (this revision).
Verdict on the reviewed design: "scientifically approved; statistical
test structure requires one revision before lock" -- the experimental
question, controls, intervention definitions, endpoints, and scope were
assessed as correct; the primary test structure (matched trajectory
seeds, realization-level aggregation, paired lattice comparison,
explicit pilot/confirmatory separation) has been revised accordingly
in this version. Considered locked pending any further review of this
revision.
