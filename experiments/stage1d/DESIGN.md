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

**Secondary**: common-support node discrimination
(`analyze_stage1b2_common_support_exclusion.py`'s statistic) and the
tangent/residual decomposition (whether topology specificity, if found,
lives in the linear or nonlinear part of the response).

**Explicitly not primary**: the concentration/routing phenotype
(`CONCENTRATION_REGIME_NOTE.md`). That mechanistic thread is valuable
and may be worth checking across constructions later, but it is a
single-cell (high-degree-node, t_p=0) diagnostic discovered post hoc,
not the prespecified comparison this design is testing. Do not let a
construction's concentration behavior substitute for its Delta_map
result.

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

**Number of graph realizations for stochastic controls**: given the
cost of a full 10-trajectory Stage 1C-style run per realization (not a
cheap per-instance AUC like Stage 1A's re-verification), a large sweep
(e.g., Stage 1A's S=25) is very likely computationally impractical
without first measuring actual per-trajectory runtime for these
constructions. Propose starting with **3 graph realizations per
stochastic construction**, each with a smaller trajectory count per
realization (**3 trajectories**, not all 10) as the initial, tractable
design -- explicitly a pilot scope, not a final statistical target.
Estimate runtime from one (construction, realization, trajectory)
triple before committing to even this reduced scope; revise the
realization/trajectory counts based on measured cost, the same way
Stage 1B, Stage 1C, and the Stage 1A re-verification all did before
committing to their full runs.

## Primary test structure

For each construction g in {rewired, hist_random, curr_random, lattice}
and each intervention definition (fixed-coordinate, role-matched):

- T's comparison values: the 10 Delta_map values already computed by
  Stage 1C (no new computation for T itself).
- g's comparison values: for T vs. lattice (deterministic), 10 newly-run
  trajectories on the lattice construction. For T vs. each stochastic
  control, the nested realization/trajectory structure above --
  aggregate within each graph realization first (mean Delta_map across
  that realization's trajectories), then compare T's 10 trajectory-level
  values against the stochastic control's realization-level means,
  analogous to how Stage 1A's re-verification aggregated within class
  before testing across classes.
- Paired comparison: given T's values and a construction's
  (aggregated) values are not naturally paired 1:1 the way Stage 1A's
  per-class differences were, use an unpaired, two-sample comparison
  (Mann-Whitney U, exact where feasible) rather than forcing an
  artificial pairing. Report the actual values, not just the test
  statistic -- effect size and overlap matter more than a p-value alone
  here, given the small sample sizes at the realization level.
- Holm correction across the 4 planned construction comparisons, within
  each intervention definition (8 tests total: 4 constructions x 2
  intervention definitions), treated as two separate families (one per
  intervention definition) rather than one family of 8, since the two
  intervention definitions are answering different questions and
  correcting across both would conflate them.

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
- Does not commit to a specific realization/trajectory count for the
  stochastic controls beyond the proposed pilot scope (3 realizations x
  3 trajectories) -- revise based on measured runtime before running the
  full design, the same discipline this project has applied to every
  prior stage.
