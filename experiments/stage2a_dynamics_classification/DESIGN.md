# Stage 2A: Does Runtime Oscillator Evolution Improve Classification?

*Fourth draft, revised per a third external review round (the first
review's ten corrections and the second review's five corrections
already incorporated into the second and third drafts respectively;
this round's own one substantive correction and three operational
closures incorporated below, plus a wording fix). The reviewer's own
verdict on the third draft: "scientifically approved. Lock after
correcting the active-support prior statement and specifying
standardization, the encoder-seed robustness subset, and mechanical
go/no-go thresholds." Not yet locked -- follow this project's
established convention (Stage 1D's own path through four review rounds
before any code ran).*

## The question, precisely -- corrected framing

**Correction from review, load-bearing**: this design does not test
"dynamics vs. no dynamics." `_local_converged_phases` already performs
150 iterations of a locally-coupled, image-anchored oscillator system
to produce `theta(0)` -- the "unevolved" condition is not a static
algebraic encoding, it already contains real local dynamics. The actual
comparison this design tests is:

> Does subsequent runtime evolution on the tested graph improve
> classification over the already dynamically-encoded phase state?

This is still a valid, well-motivated Level 3 test -- it isolates the
specific causal contribution of *graph-level* evolution, holding the
*local* encoding dynamics fixed and identical across conditions. But
calling condition 2 "no evolution" invites exactly the wrong reading
(a purely static image encoding), and every reference to it below is
corrected to "encoded, pre-graph-evolution" throughout.

**A genuinely static control, deferred but named explicitly**:
`theta_static = pi * x` (no `_local_converged_phases` step at all) would
separate the value of the local-convergence encoder itself from the
value of subsequent graph evolution. Not primary for this design --
worth adding as a follow-up once the primary comparison (condition 3 vs.
2) is settled, to know whether *any* of the value found (if any) is
already present before graph evolution even starts.

Part 1 (`docs/PROJECT_MEMORY.md`) closed classification via *static
exported features* in the negative. Stage 2A's actual callback, stated
precisely: does runtime graph evolution, on top of an already-dynamical
local encoding, add classification value a linear readout can use --
not a test of dynamics against a non-dynamical baseline.

## A design fork resolved explicitly: topology selection

Class-conditioned topology (each class evolved on its own class-specific
topology) is circular for a genuine predictive classifier -- selecting
which topology to use requires already knowing the class. **Locked:
single fixed topology for all classes, regardless of true label** --
class 0's T for the primary run, chosen for continuity with the rest of
this project, arbitrarily but consistently. The confirmatory expansion
tests whether that pinned choice matters.

**Disclosure, worth stating plainly rather than leaving implicit --
corrected by a fourth review round, load-bearing**: T was learned from
class-0 images drawn from the official training population. This is
**not test leakage** -- the official test set remains untouched
regardless -- but the previous statement that "the pre-evolution
condition does not depend on T's structure" was only partly true.
`active_indices` -- the 505-node support the pre-evolution feature
vector is restricted to (see "Encoding," below) -- is itself a product
of the class-0 topology-construction pipeline. The pre-evolution
condition does not use T's *edge weights* or graph evolution, but it
does inherit T's class-0-derived *active-pixel support*. The correct,
three-way distinction:

- **Raw pixels**: no class-0-derived topology prior at all.
- **Encoded pre-evolution**: class-0-derived active-node *support*, but
  no T edges and no graph evolution.
- **Evolved T**: that same support, plus class-0-derived *edge
  structure*, plus graph evolution.

**The encoded-pre-evolution condition already inherits class 0's
active-node support, because its 505 features are selected using T's
`active_indices`. The primary evolved-vs.-pre-evolution comparison
therefore isolates the incremental contribution of graph evolution and
T's edge structure specifically, while holding that class-0-derived
support fixed across both conditions. Raw pixels alone contain neither
prior.** This is, if anything, a cleaner causal design than the second
draft's disclosure implied: the primary contrast was already isolating
edge structure and evolution, not accidentally comparing "some
class-0 prior" against "no class-0 prior."

This still may benefit class 0 differently from the other nine classes,
particularly under evolution on T (which carries both priors, not just
support). **Per-class recall and the full confusion matrix remain
particularly important outputs**, not optional detail -- an aggregate
accuracy or log-loss number alone could mask a class-0-specific effect.

## Encoding: the full pipeline, dimensions made explicit

**Correction from review, load-bearing**: `_local_converged_phases`
returns all 784 pixel phases; T (and the matched controls) operate on
only the 505 *active* nodes recorded in `active_indices` at construction
time. The design must state the restriction step explicitly, not move
silently from a 784-dim field to a 505-dim representation:

```
theta_0^784 = encode(x)                          # _local_converged_phases, full 28x28 field
theta_0^505 = theta_0^784[active_indices]         # restrict to T's active support
theta_T^505  = F_W^2.5(theta_0^505)               # graph evolution, W = T (or a control)
```

The 279 discarded coordinates participate in local convergence (each
pixel's phase is influenced by its 4-connectivity neighbors regardless
of whether that pixel ends up active) but not in graph evolution --
defensible, but must be visible, which it now is.

**Every one of T's matched controls (lattice, rewired, random) must use
this identical 505-node support and ordering** -- they already do, by
construction (all built from the same `active_indices` as T throughout
this project), but this is stated here as a locked requirement, not an
assumption: any topology-comparison difference must come from edge
structure, not from a different or differently-ordered pixel support.

**Encoder RNG, previously unspecified**: `_local_converged_phases`
initializes `pi*x + eta`, `eta ~ N(0, 0.01^2)`, default seed 0. Using
seed 0 for every image means every image receives the *identical*
spatial noise pattern -- deterministic and reproducible, but a shared
positional template across the whole dataset, not independent per-image
noise. **Locked: seed 0 for every image**, as the primary condition --
most faithfully reuses the encoder exactly as already established
elsewhere in this project.

**Robustness check, fully specified -- correction from a fourth review
round, load-bearing: "a subset," selection, and evaluation criterion
were previously unfixed.** Locked:
- **Subset**: the same fixed, training-derived validation subset already
  used for feasibility stage 2 (up to 5,000 official-training images) --
  reused, not redrawn, so this check adds no new data-selection
  decision.
- **Seeding**: recompute the encoded-pre-evolution and evolved-T features
  for this subset with each image's encoder seed set to that image's
  immutable dataset index, replacing the shared seed-0 noise template.
- **Refit**: both readouts (pre-evolution, evolved T) refit on this
  subset using the identical fold assignment and `C`-selection procedure
  already locked above.
- **Report**: the change in the validation-set log-loss difference
  (independent-seed vs. seed-0) for this subset.
- **Status of this check, stated explicitly**: this is a **descriptive
  robustness result on training-derived validation data, computed before
  the official test set is ever touched** -- it cannot replace, override,
  or be substituted for the seed-0 primary analysis against the official
  test set, and no formal inference test is attached to it.

## Feature representation: reference-node gauge, primary; circular-mean, secondary

**Correction from a third review round, load-bearing: the previous
conditional fallback (circular mean when `R >= 0.05`, reference-node
otherwise) is replaced.** Switching gauge rule based on the state's own
`R(theta)` created two different feature coordinate systems selected by
the state itself -- images near the threshold could switch
representation discontinuously, and the floor (0.05) was provisional
and untested. That design is dropped as the primary representation.

**Locked, single gauge rule, used for every image and every condition,
with no state-dependent switching**:

```
h(theta) = [cos(theta_i - theta_ref), sin(theta_i - theta_ref) for all i]
```

**Locked reference node**: `theta_ref = theta_{363}` -- T's own
`nodes_T['median']` active-node index, already an established landmark
in this project's own node taxonomy (one of the three fixed
low/median/high degree-stratified nodes reused throughout Stage
1B.2/1C/1D), chosen for being a typical rather than extremal node and
for being fixed **before** any Stage 2A classification outcome is
examined -- not selected by inspecting results. The same index is used
identically before and after evolution, and identically for every
topology (T, lattice, rewired, random) -- no condition or topology gets
its own reference node.

**The circular order parameter `R(theta) = |mean(exp(1j*theta))|` is
still recorded for every state, pre- and post-evolution, for every
image** -- a required dynamical diagnostic, not a gauge-selection input.
Low `R` states (phases close to uniformly scattered) are not given a
different feature representation; they are simply reported, since a
low order parameter is itself potentially informative about that
image/topology's dynamics.

**Circular-mean-centered features remain available as a secondary,
robustness representation** (`mu = angle(mean(exp(1j*theta)))`, as in
the second draft) -- reusing `stage1b2_core.py`'s rotation-removal
convention (there applied to a phase *difference*; here to a phase
*configuration* directly -- analogous, not identical, since a raw phase
configuration lives on a torus and the project's linear rotation-removal
*projector* doesn't transfer to it the same way a phase-difference shift
does). If the primary (reference-node) and secondary (circular-mean)
representations disagree materially on the confirmatory result, that
disagreement is reported as a finding about gauge sensitivity, not
resolved by picking whichever was significant.

## Data: official KMNIST split, not a fresh custom one

**Correction from review**: the prior draft specified "fixed train/test
split (SEED=42)," implying a freshly-drawn split. KMNIST already
provides an official train/test division
(`train-images-idx3-ubyte`/`t10k-images-idx3-ubyte`, already used
elsewhere in this project for the GPU work's ink-mask computation).
**Locked**: use the official KMNIST training set for fitting and
cross-validation, the official test set exactly once for final
evaluation. `SEED=42` governs fold assignment, solver-perturbation, and
feasibility-stage subsampling -- it does not redefine what "the test
set" means.

**Locked, to prevent silent test-set leakage during development**: a
fixed validation subset drawn from the *official training data* is used
for every feasibility-stage and encoder/representation/regularization
decision. The official test set's labels remain untouched until the
design and implementation are fully locked and only one final evaluation
pass is run against them.

## Minimal feasibility pass: a staged ladder, not the full dataset at once

**Correction from review**: "full 10-class KMNIST" for a feasibility
pass is up to 70,000 images, each requiring local convergence plus a
505-node ODE solve -- not a minimal mechanical test even with the
validated GPU infrastructure.

**Correction from a third review round, load-bearing: the boundary
between "feasibility" and "confirmatory" was not sharp enough** -- the
second draft's own stage 3 ("full dataset") and the confirmatory
section's "official test set touched exactly once" needed clearer
separation between the training-side ladder and the one singular
test-side evaluation. **Locked, four stages, strictly separated**:

1. **1,000 training images** (100/class) -- end-to-end correctness only.
2. **Up to 5,000 official-training images**, plus the fixed
   training-derived validation subset (never the official test set) --
   throughput measurement and development.
3. **The full 60,000-image official KMNIST training set** -- final
   feature generation and model selection (regularization search,
   representation checks), still entirely on training-derived data.
4. **One locked evaluation on the untouched 10,000-image official test
   set.** This is the only stage that touches test-set labels, and it
   happens exactly once, after stages 1-3 and the rest of this design
   are fully settled.

Stages 1-3 may guide engineering and representation choices using
training-derived validation data; **they must not generate repeated
official-test evaluations** -- a disappointing validation-stage result
may justify stopping the programme before stage 4, but must not trigger
a redesign-then-re-evaluate-on-the-real-test-set cycle.

**Go/no-go criteria for advancing between stages 1-3, locked as
mechanics, not as a promised effect direction** (a "no-go" on any of
these is a real, reportable outcome, not a failure of the experiment).
**Correction from a fourth review round, load-bearing: these criteria
must be executable thresholds, not placeholders** -- fixed here, not
deferred to implementation time:
- **Zero non-finite feature vectors** (NaN/inf) anywhere in the
  pipeline, at any stage -- any occurrence is a stop, not a rate to
  tolerate.
- **Zero silent solver failures**: every local-convergence or
  graph-evolution ODE solve either completes and reports its own
  convergence status, or is logged as a failure -- none may fail
  silently and be treated as a success.
- **Recoverable solver failure rate**: a solver call that fails but is
  recoverable (e.g. a retry with tighter tolerance succeeds) is
  tolerated up to **0.1%** of calls at any stage; **above 0.1%, scaling
  to the next stage stops pending investigation** -- this is not itself
  a Stage 2A finding, it is a pipeline-health gate.
- **Runtime and storage**: stage 2's measured per-image cost is
  extrapolated to the full 60,000-image stage-3 set; this projection is
  documented and must be explicitly approved before stage 3 is launched
  (not silently assumed acceptable because stage 2 ran without error).
- **`R(theta)` diagnostic**: the complete distribution of `R(theta)`
  (pre- and post-evolution, every condition) is reported at every stage;
  any noticeable mass at the numerical limits (`R` near 0 or near 1) is
  flagged explicitly. This is a reporting requirement, not a gauge
  trigger -- per the locked single reference-node rule above, `R(theta)`
  is never used to change which feature representation is computed.
- **Classifier convergence**: the linear readout converges (per the
  classifier-implementation lock above) in every condition at every
  stage; any non-convergence is reported explicitly, per that section's
  own requirement, not silently absorbed as a routine stage-advancement
  blocker.

Three conditions per the original design (raw pixels -> linear; encoded
pre-evolution -> linear; evolved on T -> linear) at every ladder stage,
same classifier architecture and feature dimension for conditions 2 and
3 throughout. Stage 3 additionally includes the secondary evolved graphs
(lattice, canonical rewired, canonical random -- see "Confirmatory
expansion," below) for model-selection purposes, ahead of the single
stage-4 evaluation.

## Confirmatory endpoint and test: locked before any full-test-set result is examined

**Correction from review, load-bearing**: the prior draft said "any
signal... that evolution changes held-out accuracy" -- not a locked
primary metric or test.

**Correction from a third review round, load-bearing: the sole primary
comparison must be named, not left implicit across several graphs.**
The confirmatory expansion (below) evolves T, lattice, a canonical
rewired graph, and a canonical random graph -- if every evolved-vs-
pre-evolution comparison were treated as equally primary, several graphs
would create several independent chances to declare "graph evolution
helps," despite only one endpoint being locked. **Locked: T-evolved vs.
encoded-pre-evolution is the sole primary Level 3 comparison** --
consistent with the continuity rationale already used to justify
choosing class-0 T throughout this project. Every other evolved-graph
comparison (lattice, canonical rewired, canonical random, each vs.
encoded-pre-evolution; evolved T vs. the other evolved graphs) is
**secondary and explicitly graph-specific**, reported descriptively and
not used to rescue a null primary result.

**Locked direction convention**, stated explicitly so "improvement"
means one specific thing throughout: for held-out image `i`,

```
d_i = ell_i(evolved T) - ell_i(encoded pre-evolution)
```

where `ell_i` is that image's multiclass log-loss under the respective
classifier. **Improvement is `E[d_i] < 0`** (evolved T's log-loss lower,
i.e. better, than the pre-evolution baseline's). The same sign
convention applies to every secondary evolved-vs-pre-evolution
comparison, substituting the relevant graph for T.

**Locked, primary test -- fully executable, not left as a category**:
- 20,000 paired, class-stratified bootstrap resamples of the 10,000
  official test images (stratified: each resample preserves each
  class's original count, drawn with replacement within class).
- Compute the mean per-image `d_i` on each resample.
- Report a two-sided 95% **percentile** interval over the 20,000
  resampled means.
- **Success criterion**: the primary result supports an improvement only
  if the entire interval is below zero (analogously, only if the entire
  interval is above zero would it support the pre-evolution baseline
  winning). An interval straddling zero is a null primary result --
  it is not rescued by any secondary comparison's outcome.
- **Explicit scope of what this interval does and does not capture**:
  it measures sampling uncertainty from which official test images were
  drawn, conditional on the already-fitted classifiers and the already-
  fixed feature pipeline and topology. It does **not** include
  uncertainty from retraining the feature pipeline, re-selecting
  regularization, or redrawing the topology -- those are each fixed,
  singular choices in this design, not resampled.

**Secondary**: McNemar's exact test for accuracy disagreement (more
interpretable, if less statistically complete, than log-loss alone);
macro-F1; per-class recall; raw accuracy; the same paired-bootstrap
procedure applied to each secondary graph-specific comparison, reported
descriptively (not Holm-corrected as a family, since none of them is a
second chance at the primary claim).

The feasibility ladder (below) may report raw differences descriptively,
without formal inference -- the metric and test above are locked
specifically for the one, final, confirmatory evaluation against the
official test set.

## Confirmatory expansion: graph-specific, not family-general, unless scaled up

**Correction from review, load-bearing, directly reapplying Stage 1D's
own lesson**: a single rewired graph or single random-construction
realization does not support a family-level claim ("rewiring" or
"random construction" in general) -- Stage 1D's entire R=25-realization
design existed specifically because one realization isn't a reliable
estimate of a stochastic family's behavior.

Two legitimate scopes, and this design must pick one explicitly rather
than drift into the wrong claim by default:

- **Fixed, prespecified graph instances** (cheaper): one prespecified
  realization each of lattice, rewired, and one random construction,
  compared alongside T. **The claim must stay explicitly graph-specific**
  -- "this compares these four specific graphs," never "T vs. rewiring
  in general" or any family-level language.
- **Topology-family comparison** (more expensive, Stage-1D-style):
  multiple realizations per stochastic control, aggregated at the
  realization level, supporting an actual family-level claim.

**Locked for this first Level 3 result: fixed, prespecified graph
instances.** Cheaper, and a first result doesn't need family-level
generality to be informative -- but every reported claim about
lattice/rewired/random in this design's confirmatory results must be
phrased as being about that one specific realized graph, not the family
it was drawn from. Escalating to a family-level design is a legitimate,
separate future extension if this first result motivates it.

**Wording correction from a fourth review round**: "representative
graphs" is replaced with "prespecified graph instances" throughout this
design -- a single draw was prospectively selected (named below, before
any Stage 2A result exists), but one draw cannot establish that it is
statistically *representative* of its family, only that it is a fixed,
prespecified instance of it. This is a wording fix, not a scope change --
the graph-specific-only claim above already stated this correctly.

**Correction from a third review round, load-bearing: these instances
must not become outcome-selected.** The exact rewired and random graphs
are named here, before any Stage 2A result exists, reusing already-
existing project artifacts rather than drawing fresh ones for this
design specifically:

- **Lattice**: T's already-canonical matched lattice construction
  (`class0_constructions.pkl`'s cached `lattice`) -- deterministic, no
  seed, already used throughout Stage 1D.
- **Canonical rewired**: `degree_preserving_rewire(W_T, ink_mask_active,
  seed=0)` -- Stage 1D's own first pilot realization seed
  (`experiments/stage1d_topology_specificity/results/stage1d_constructions.pkl`),
  reused here rather than drawing a new seed for this design.
- **Canonical random**: `generate_matched_sparsity_topology(W_T,
  ink_mask_active, seed=0)` -- **current-random (curr_random)
  specifically, not historical-random**, chosen because it needs no
  coupling-budget rescaling step and matches T's edge count exactly by
  construction, and because hist_random's own isolated-fixed-coordinate
  failure mode (DESIGN.md, Stage 1D) is a real complication this design
  does not need to inherit for a single prespecified-instance choice.
  Same seed=0, same rationale as rewired above.

**Graph statistics, recorded for all four graphs (T, lattice, canonical
rewired, canonical random) rather than left implicit**:

| graph | n edges | total weight | mean weighted degree | min degree | max degree |
|---|---:|---:|---:|---:|---:|
| T | 1051 | 1959.98 | 3.881 | 0.902 | 12.153 |
| lattice | 935 | 1959.98 | 3.881 | 1.048 | 4.193 |
| rewired (seed=0) | 1051 | 1959.98 | 3.881 | 0.903 | 12.122 |
| curr_random (seed=0) | 1051 | 1959.98 | 3.881 | 0.000 | 12.083 |

**Disclosed, not silently absorbed**: `curr_random` seed=0 has one
isolated node (weighted degree 0.0) under this specific draw. Unlike
Stage 1D's perturbation-response measurement (where an isolated fixed
intervention node breaks the tangent/event-alignment machinery
entirely), an isolated node under Stage 2A's plain unperturbed evolution
simply never changes its phase during evolution (`dtheta_i/dt = 0` when
every `W_ij` for that row is 0) -- not a pipeline failure, but a
real, disclosed property of this specific prespecified graph instance,
worth reporting alongside its results rather than silently absorbed.

Per-topology conditions: encoded-pre-evolution and evolved, both using
the shared 505-node encoding pipeline above (the encoding step doesn't
depend on topology; only the evolution step does).

## Baselines: parameter-matched AND separately, a competent ordinary network

The oscillator readout has `2*505*10 + 10 = 10,110` trainable parameters
(circular-embedding input dimension x 10 classes + bias). A parameter-
matched MLP (`R^784 -> Linear(784,H) -> ReLU -> Linear(H,10)`) needs
`795*H + 10` parameters; matching to 10,110 gives `H ≈ 12.7`, so
**H=13** (10,345 parameters, verified by direct calculation, not just
estimated).

**Correction from review**: H=13 is a very narrow MLP -- a valid
parameter-matched control, but not a fair test of "does this do as well
as a competent ordinary classifier." **Locked: report both**,
clearly separated:
- **H=13 MLP**: strict parameter-matched control (fairness-by-parameter-
  count).
- **A separate, competent-context MLP, `H=128` locked** (correction from
  a third review round: "128 or similar" was not yet a fixed value) --
  not parameter-matched, reported explicitly as practical context, not
  as a fairness control.

For both baselines and the oscillator-readout conditions, report:
trainable parameter count, total forward-pass compute (state which:
FLOPs or wall-clock), feature dimension, training epochs/optimization
steps, and regularization search budget. The oscillator's own evolution
is frozen (no gradient through it) but its runtime compute is real and
must be reported alongside the MLPs', not treated as free.

## Linear readout: regularization grid, locked

**Correction from a third review round**: the regularization grid,
selection procedure, and tie-break were left unspecified. **Locked**:
- Classifier: multinomial logistic regression (softmax), L2-penalized.
- Regularization grid: `C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3,
  1e4}` (9 values, log-spaced).
- Selection: fixed 5-fold stratified cross-validation within the
  official training data (same fold assignment, governed by `SEED=42`,
  reused across every condition for comparability).
- Select the `C` minimizing mean validation log-loss.
- **Deterministic tie-break**: if multiple `C` values tie on mean
  validation log-loss, select the smaller `C` (stronger regularization).
- **Each feature condition (raw pixels, encoded-pre-evolution, evolved
  T, and every secondary evolved graph) selects its own `C` independently**,
  using the identical grid and fold assignments -- not a single `C`
  shared across conditions.

**Correction from a fourth review round, load-bearing: "standardized...
only if" left whether standardization happens at all unresolved, and
said nothing about fold-safe scaling.** Locked, precisely:
- **Every logistic-regression condition uses per-feature standardization**
  (zero mean, unit variance per dimension) -- not optional, not
  per-condition discretion.
- **Fold-safe fitting**: for each condition, during cross-validation, the
  scaler is fit separately on only that fold's *training* partition and
  applied, unchanged, to that fold's validation partition -- never fit on
  validation data.
- **After `C` selection**: a new scaler is fit on the complete official
  training set for that condition, and applied unchanged to that
  condition's official test features. This final scaler is fit once,
  after the design and `C` are both already locked.
- **Each condition (raw pixels, encoded-pre-evolution, evolved T, every
  secondary evolved graph) gets its own independently-fit scaler** at
  every stage above -- raw, pre-evolution, and evolved features have
  different distributions, so "the same standardization" means the same
  *procedure* applied consistently, never the same numerical means and
  variances shared across conditions.

**Classifier implementation, locked to prevent platform-dependent
defaults from silently becoming part of the experiment**:
- Solver: `lbfgs` (scikit-learn's default multinomial solver; stated
  explicitly rather than left to whatever the installed version
  defaults to).
- Convergence tolerance: `tol=1e-4`.
- Maximum iterations: `max_iter=1000` (higher than scikit-learn's own
  default of 100, since 10-class multinomial fits with 505-1010
  features can need more iterations to converge cleanly, especially at
  weak regularization).
- Class weighting: uniform (`class_weight=None`) -- KMNIST's 10 classes
  are balanced by construction, so no reweighting is applied.
- Random seed: `random_state=42` wherever the solver accepts one (governs
  any internal stochastic tie-breaking; does not affect the CV fold
  assignment, which is separately governed by `SEED=42` above).
- **Non-convergence**: if a fit does not converge within `max_iter` for
  any `(condition, fold, C)` combination, this is logged and reported
  explicitly (which combination, at what iteration count) -- not
  silently accepted or silently re-run with a larger `max_iter`. A
  pattern of non-convergence concentrated in one condition or one
  `C` region is itself a reportable diagnostic.

## Named watched-for outcomes (unchanged from the first draft)

1. Dynamics useful, topologies equivalent (across the specific graphs
   tested -- see the graph-specificity scoping above).
2. Dynamics useful, one specific graph wins -- would show `Delta_map`
   equivalence does not imply task-utility equivalence.
3. Dynamics not useful -- Level 2 structure without demonstrated Level 3
   value under this task/readout, a real negative result.
4. Encoded-pre-evolution state already helps, graph evolution adds no
   further value -- the useful contribution would be the *local*
   encoding dynamics, not graph-level evolution specifically.

## What this does not do

- Does not attempt denoising (Stage 2B) or generation -- deferred, per
  the first draft's reasoning, unchanged.
- Does not test role-matched intervention or classes' own topologies for
  their own images (circular, rejected above).
- Does not test a genuinely static (`pi*x`, no local convergence)
  encoding as primary -- named above as a well-motivated future
  extension, not required for this design's primary claim.
- Does not support a topology-*family*-level claim under the locked
  fixed-prespecified-graph-instances scope -- escalating to that is a
  separate, larger future design if motivated by this one's results.
- Does not treat any secondary graph-specific comparison (lattice,
  canonical rewired, canonical random, or evolved-T-vs-other-evolved-
  graphs) as capable of rescuing a null primary (T-evolved vs.
  encoded-pre-evolution) result -- per the locked primary/secondary
  hierarchy above.

## Status

Fourth draft. The first review's ten corrections (second draft) and the
second review round's five load-bearing corrections (third draft,
summarized below) are unchanged from the third draft; this draft
incorporates a **third review round's one substantive correction and
three operational closures**, on a third draft the reviewer judged
"substantially improved... close to lock":

1. **Sole primary comparison designated**: T-evolved vs. encoded-
   pre-evolution, with an explicit direction convention
   (`d_i = ell_i(evolved T) - ell_i(pre-evolution)`, improvement =
   `E[d_i] < 0`). Every other evolved-graph comparison is secondary and
   explicitly graph-specific, unable to rescue a null primary result.
2. **Bootstrap fully specified and executable**: 20,000 paired,
   class-stratified resamples of the 10,000 official test images,
   two-sided 95% percentile interval on the mean per-image log-loss
   difference, success criterion = entire interval below zero, with an
   explicit statement of what uncertainty the interval does and does not
   capture.
3. **Gauge fixed to a single, non-switching rule**: reference-node
   centering (`theta_ref` = T's own median-degree node, active index
   363) as the primary representation for every image and condition;
   the circular-mean representation demoted to a secondary robustness
   check rather than a state-triggered fallback.
4. **Feasibility ladder and confirmatory run sharply separated** into
   four stages (1,000 images / up to 5,000 training images / full
   60,000-image training set / one locked 10,000-image test evaluation),
   with mechanics-based (not effect-direction-based) go/no-go criteria
   between stages.
5. **Remaining implementation choices locked**: the regularization grid
   (`C` in 9 log-spaced values, 5-fold stratified CV, deterministic
   tie-break toward stronger regularization, each condition selecting
   its own `C`), the competent-MLP width (`H=128`, fixed), and the exact
   prespecified-graph-instance seeds (rewired and curr_random, both
   seed=0, reused from Stage 1D's own pilot artifacts, with graph
   statistics recorded and one disclosed isolated node in curr_random's
   draw).

**This draft's own corrections**:

1. **Corrected the class-0 prior disclosure**: the pre-evolution
   condition is not prior-free -- it inherits T's class-0-derived
   `active_indices` *support*, just not T's edge structure. The primary
   comparison isolates evolution-and-edge-structure specifically, since
   both conditions already share the same support -- a cleaner causal
   design than the previous disclosure implied, not a weaker one.
2. **Standardization fully locked**: per-feature, fold-safe (fit on
   training-fold-only, applied to that fold's validation data; refit
   once on the complete official training set after `C` selection,
   applied unchanged to test), and independently fit per condition.
   Classifier implementation (solver, tolerance, max iterations, class
   weighting, random seed, non-convergence handling) locked alongside it
   to prevent platform-dependent defaults from entering the experiment.
3. **Encoder-RNG robustness check fully specified**: reuses the fixed
   feasibility-stage-2 validation subset, seeds by each image's dataset
   index, refits both readouts on identical folds/`C`-selection, reports
   the validation log-loss-difference change -- explicitly descriptive,
   never able to replace the seed-0 primary analysis.
4. **Go/no-go thresholds made executable**: zero non-finite features,
   zero silent solver failures, a named 0.1% recoverable-failure-rate
   stop threshold, an explicitly-approved runtime/storage projection
   before stage 3, and full `R(theta)` distribution reporting (never a
   gauge trigger).
5. **Wording**: "representative graphs" replaced with "prespecified
   graph instances" throughout -- a single draw is prospectively fixed,
   not a demonstrated statistical representative of its family.

The reviewer's own verdict on this draft: "scientifically approved.
Lock after correcting the active-support prior statement and specifying
standardization, the encoder-seed robustness subset, and mechanical
go/no-go thresholds... no further conceptual redesign is needed."
Whether this fourth draft is now ready to lock is a decision for a
separate step -- not made here.
