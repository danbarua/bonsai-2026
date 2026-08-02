# Stage 2A: Does Runtime Oscillator Evolution Improve Classification?

*Third draft, revised per a second external review round (the first
review's ten corrections already incorporated into the second draft;
this round's own five load-bearing corrections incorporated below).
The reviewer's own verdict on the second draft: "scientifically
approved, not yet locked." Not yet locked -- follow this project's
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

**Disclosure, worth stating plainly rather than leaving implicit**: T
was learned from class-0 images drawn from the official training
population. This is **not test leakage** -- the official test set
remains untouched regardless -- but it does mean the fixed feature
extractor (T's own edge structure) carries a class-0-derived prior
before any classifier ever sees a single image. This may benefit class 0
differently from the other nine classes under evolution on T
specifically (though not under the pre-evolution or raw-pixel
conditions, which do not depend on T's structure). **Per-class recall
and the full confusion matrix are therefore particularly important
outputs**, not optional detail, for the confirmatory result -- an
aggregate accuracy or log-loss number alone could mask a class-0-specific
effect that has nothing to do with graph evolution being generically
useful.

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
elsewhere in this project. **Robustness check, required before the
confirmatory result is reported**: repeat with independent, deterministic
per-image seeds (e.g. seed = image index) for a subset, confirming any
classification difference found does not depend on that one fixed noise
realization.

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
these is a real, reportable outcome, not a failure of the experiment):
- no non-finite (NaN/inf) features at any pipeline stage;
- solver failure rate (local-convergence or graph-evolution ODE) within
  a prespecified acceptable bound;
- runtime and storage extrapolation to the full dataset within stated
  bounds;
- `R(theta)` diagnostic distribution inspected for gross pathology (not
  used to change the gauge, per the locked single reference-node rule
  above -- purely a sanity check that dynamics look reasonable);
- the linear readout converges successfully in every condition.

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

- **Fixed representative graphs** (cheaper): one prespecified realization
  each of lattice, rewired, and one random construction, compared
  alongside T. **The claim must stay explicitly graph-specific** -- "this
  compares these four specific graphs," never "T vs. rewiring in
  general" or any family-level language.
- **Topology-family comparison** (more expensive, Stage-1D-style):
  multiple realizations per stochastic control, aggregated at the
  realization level, supporting an actual family-level claim.

**Locked for this first Level 3 result: fixed representative graphs.**
Cheaper, and a first result doesn't need family-level generality to be
informative -- but every reported claim about lattice/rewired/random
in this design's confirmatory results must be phrased as being about
that one specific realized graph, not the family it was drawn from.
Escalating to a family-level design is a legitimate, separate future
extension if this first result motivates it.

**Correction from a third review round, load-bearing: "representative"
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
  does not need to inherit for a single representative-graph choice.
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
real, disclosed property of this specific representative graph, worth
reporting alongside its results rather than silently absorbed.

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
- Features standardized (zero mean, unit variance per dimension) only if
  the identical, prespecified standardization is applied consistently
  across every condition being compared -- no per-condition tuning of
  whether or how to standardize.
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
  fixed-representative-graphs scope -- escalating to that is a separate,
  larger future design if motivated by this one's results.
- Does not treat any secondary graph-specific comparison (lattice,
  canonical rewired, canonical random, or evolved-T-vs-other-evolved-
  graphs) as capable of rescuing a null primary (T-evolved vs.
  encoded-pre-evolution) result -- per the locked primary/secondary
  hierarchy above.

## Status

Third draft. The first review's ten corrections (incorporated into the
second draft) plus a second review round's five further load-bearing
corrections are now incorporated:

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
   representative-graph seeds (rewired and curr_random, both seed=0,
   reused from Stage 1D's own pilot artifacts, with graph statistics
   recorded and one disclosed isolated node in curr_random's draw).

Also added: an explicit disclosure that T's class-0-derived learning
history is not test leakage but may create a class-0-specific prior,
making per-class recall and the confusion matrix required outputs, not
optional detail.

The reviewer's own verdict on the prior draft: "scientifically approved,
not yet locked... no broader redesign needed." Whether this third draft
is now ready to lock, or needs a further review pass, is a decision for
a separate step -- not made here.
