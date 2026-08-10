Simplified Technical English version of `experiments/stage2a_dynamics_classification/DESIGN.md`

# Stage 2A: Does Runtime Oscillator Evolution Improve Classification?

*LOCKED (fourth draft, final). "Locked" means the design is fixed. No
one may change it after this point, except through a disclosed
amendment. Three earlier review rounds made corrections: the first made
ten corrections, the second made five corrections, the third made one
important correction plus three smaller closures. All of these are
already folded into drafts two through four. This final draft adds a
fourth review round's two engineering fixes (a working non-convergence
stop rule, and a corrected recovery policy for a failed ODE solve) and
one note about the feature representation (the reference node's own two
constant feature columns are dropped; the true feature count is 1008,
not 1010). The reviewer's verdict: "design approved and ready to lock…
no further conceptual redesign needed." This follows the project's usual
practice (Stage 1D went through four review rounds before any code ran).
Work on feasibility stage 1 begins from this document.*

**Amendment made after locking, during feasibility stage 2**: `max_iter`
(the maximum number of solver steps) was raised from `1000` to `10000`
in the "Linear classifier implementation" section below. The team found
a real case where the classifier fit did not converge. They
investigated and found the cause: severe feature multicollinearity
(features that are almost duplicates of each other) specific to
`evolved_T`. This was not a problem that more training images would fix
on their own. See `FINDINGS.md` for the full investigation. This is the
only change made to this document after locking. It is disclosed here,
not silently changed.

## The question, precisely — corrected framing

**Correction from review, load-bearing**: this design does not test
"dynamics versus no dynamics." A function called
`_local_converged_phases` already runs 150 steps of local, image-based
oscillator dynamics to build the starting phase state, `theta(0)`. So
the "not yet evolved" condition already contains real local dynamics.
It is not a plain algebraic encoding. The real comparison this design
tests is:

> Does further runtime evolution on the graph being tested improve
> classification, compared to the already dynamically-encoded phase
> state?

This is still a valid, well-motivated test. It isolates the specific
effect of graph-level evolution, while holding the local encoding
dynamics fixed and identical across every condition. But calling
condition 2 "no evolution" would wrongly suggest a plain, static image
encoding. Every mention of it below is corrected to "encoded,
pre-graph-evolution."

**A genuinely static control, named but not run yet**: a proposed
alternative, `theta_static = pi * x` (turning each pixel straight into a
phase value, with no `_local_converged_phases` step at all), would
separate the value of the local-convergence encoder from the value of
later graph evolution. This is not part of the primary design. It is a
worthwhile follow-up, once the primary comparison (condition 3 versus
condition 2) is settled, to check whether any of the value found (if
any) is already present before graph evolution even starts.

An earlier part of this project (see `docs/PROJECT_MEMORY.md`, Part 1)
already closed the question of classification from static exported
features, with a negative answer. Stage 2A asks a different, precise
question: does runtime graph evolution, on top of an already-dynamical
local encoding, add classification value that a linear readout can use?
It does not compare dynamics against a non-dynamical baseline.

## A design choice made explicitly: which topology to use

Using a class-specific topology (evolving each class's images on that
class's own topology) is circular for a real predictive classifier. You
would need to already know the class to choose the topology. **Locked
choice: use one single, fixed topology for every class, regardless of
the true label.** The team chose class 0's topology, `T`, for the
primary run. They chose it for continuity with the rest of this
project, arbitrarily but consistently. The confirmatory expansion (see
below) tests whether this fixed choice matters.

**A disclosure worth stating plainly, corrected by a fourth review
round, load-bearing**: `T` was learned from class-0 images in the
official training set. This is **not test leakage** — the official test
set stays untouched regardless. But an earlier statement, "the
pre-evolution condition does not depend on T's structure," was only
partly true. `active_indices` is the set of 505 nodes the pre-evolution
feature vector is restricted to (see "Encoding," below). This set of
nodes itself comes from the class-0 topology-building process. So the
pre-evolution condition does not use T's edge weights or graph
evolution, but it does inherit T's class-0-derived choice of active
pixels. The correct, three-way distinction:

- **Raw pixels**: uses no class-0-derived choice of pixels at all.
- **Encoded pre-evolution**: uses the class-0-derived set of active
  nodes, but not T's edges, and no graph evolution.
- **Evolved T**: uses that same set of active nodes, plus T's
  class-0-derived edge structure, plus graph evolution.

**The encoded-pre-evolution condition already inherits class 0's set of
active nodes, because its 505 features are chosen using T's
`active_indices`. So the primary comparison, evolved versus
pre-evolution, isolates the added effect of graph evolution and T's
edge structure specifically, while holding that class-0-derived choice
of nodes fixed across both conditions. Raw pixels alone contain neither
of these class-0-derived choices.** This is, if anything, a cleaner
comparison than the second draft's disclosure suggested: the primary
comparison was already isolating edge structure and evolution, not
accidentally comparing "some class-0 influence" against "no class-0
influence."

This design may still help class 0 more than other classes, especially
under evolution on T, since T carries both the node choice and the edge
structure. **Per-class recall (how well the model finds each class) and
the full confusion matrix remain required outputs, not optional
detail.** A single overall accuracy or log-loss number could hide a
class-0-specific effect.

## Encoding: the full pipeline, with dimensions stated clearly

**Correction from review, load-bearing**: `_local_converged_phases`
returns phase values for all 784 pixels. T, and its matched control
graphs, use only the 505 *active* nodes recorded in `active_indices`
when they were built. The design must state this restriction step
directly, not silently move from a 784-value field to a 505-value one:

```
theta_0^784 = encode(x)                          # _local_converged_phases, full 28x28 field
theta_0^505 = theta_0^784[active_indices]         # restrict to T's active support
theta_T^505  = F_W^2.5(theta_0^505)               # graph evolution, W = T (or a control)
```

The 279 dropped coordinates still take part in local convergence — each
pixel's phase is affected by its 4 neighboring pixels, whether or not
that pixel ends up active — but they take no part in graph evolution.
This choice is reasonable, but it must be visible, and now it is.

**Every one of T's matched control graphs (lattice, rewired, random)
must use this identical 505-node set of active pixels, in the identical
order.** They already do, since they were all built from the same
`active_indices` as T throughout this project. This rule is now stated
here as a locked requirement, not left as an assumption. Any difference
found when comparing topologies must come from edge structure, never
from a different or differently-ordered set of pixels.

**The encoder's random-number seed, not previously stated**:
`_local_converged_phases` starts from `pi*x + eta`, where `eta` is drawn
from a normal distribution with mean 0 and standard deviation 0.01, using
random seed 0 by default. Using seed 0 for every image means every image
gets the *identical* random noise pattern. This is deterministic and
reproducible, but it means every image shares the same positional noise
template, rather than each image getting independent noise. **Locked:
use seed 0 for every image**, as the primary condition. This most
closely reuses the encoder exactly as it is already used elsewhere in
this project.

**A robustness check, now fully defined — correction from a fourth
review round, load-bearing: earlier drafts left the subset, the
selection method, and the pass/fail rule undefined.** Locked:
- **Subset**: reuse the same fixed, training-derived validation subset
  already used in feasibility stage 2 (up to 5,000 official-training
  images). Do not draw a new subset. This adds no new data-selection
  decision.
- **Seeding**: recompute the encoded-pre-evolution and evolved-T
  features for this subset, but set each image's encoder seed to that
  image's own fixed dataset index, instead of the shared seed 0.
- **Refit**: refit both readouts (pre-evolution, evolved T) on this
  subset, using the identical fold assignment and `C`-selection
  procedure already locked below.
- **Report**: report the change in the validation-set log-loss
  difference, comparing independent-seed results against seed-0
  results, for this subset.
- **What this check does and does not show, stated explicitly**: this
  is a **descriptive robustness check on training-derived validation
  data, run before the official test set is ever touched.** It cannot
  replace, override, or stand in for the seed-0 primary analysis against
  the official test set. No formal statistical test is attached to it.

## Feature representation: reference-node centering as primary, circular-mean as secondary

**Correction from a third review round, load-bearing: an earlier
conditional rule is now replaced.** The earlier rule switched between
two different feature methods (circular mean when the order parameter R
was at least 0.05, reference-node centering otherwise). Switching the
method based on the state's own value created two different feature
coordinate systems, chosen by the state itself. Images near the
threshold could switch representation with no warning, and the 0.05
cutoff was never tested. This rule is dropped as the primary method.

**Locked, single rule, used for every image and every condition, with no
switching**:

```
h(theta) = [cos(theta_i - theta_ref), sin(theta_i - theta_ref) for all i]
```

This rule turns each node's phase into two numbers, based on its
difference from one fixed reference node.

**Locked reference node**: `theta_ref = theta_{363}` — this is T's own
`nodes_T['median']` active-node index. This node is already a fixed
landmark used throughout this project (one of three fixed
low/median/high degree nodes reused across several stages). The team
chose it because it is a typical node, not an extreme one, and because
it was fixed **before** any Stage 2A classification result was seen, not
chosen by looking at results. The same node index is used identically
before and after evolution, and identically for every topology (T,
lattice, rewired, random). No condition or topology gets its own
reference node.

**The circular order parameter, `R(theta) = |mean(exp(1j*theta))|`, is
still recorded for every state, before and after evolution, for every
image.** R measures how synchronized (in step) the phases are. This is a
required measurement to report, not an input used to choose the feature
method. Low-R states (phases spread out, not in step) do not get a
different feature representation. They are simply reported, since a low
order parameter may itself be informative about that image and
topology's dynamics.

**Circular-mean-centered features remain available as a secondary,
robustness check** (`mu = angle(mean(exp(1j*theta)))`, as in the second
draft). This reuses `stage1b2_core.py`'s method for removing rotation
(there it was applied to a phase *difference*; here it is applied to a
phase *configuration* directly — similar, but not identical, since a raw
phase configuration lives on a torus, and the project's linear
rotation-removal method does not carry over to it the same way it does
for a phase-difference shift). If the primary (reference-node) and
secondary (circular-mean) representations disagree in an important way
on the confirmatory result, that disagreement is reported as a finding
about sensitivity to the choice of representation. It is not resolved by
simply picking whichever representation gave a significant result.

**Correction from a fourth review round, load-bearing: the reference
node's own two features are trivially constant.** By construction,
`cos(theta_363 - theta_363) = 1` and `sin(theta_363 - theta_363) = 0`
for every image, every condition, always. These two columns carry zero
variance and zero information. **Locked**: drop these two constant
columns on purpose, before the features reach the classifier. Do not
leave this to whatever a standardization library happens to do with a
zero-variance column, since that behavior can differ silently across
platforms or software versions. **The effective feature count is
therefore 1008, not 1010** (`2*505 - 2`), for both the reference-node
representation and the parameter counts described below that depend on
it. This is expected, not a flaw in the design. A unit test confirms the
two dropped columns are always exactly `(1, 0)` before removal, for
every image.

## Data: the official KMNIST split, not a freshly-made one

**Correction from review**: an earlier draft said "fixed train/test
split (SEED=42)," which implied a freshly-drawn split. KMNIST already
provides an official split into training and test data
(`train-images-idx3-ubyte` and `t10k-images-idx3-ubyte`, already used
elsewhere in this project for the GPU work's ink-mask computation).
**Locked**: use the official KMNIST training set for fitting and
cross-validation, and use the official test set exactly once, for the
final evaluation. `SEED=42` controls fold assignment, solver
tie-breaking, and feasibility-stage subsampling. It does not change what
counts as "the test set."

**Locked, to prevent the test set from leaking into development by
accident**: use a fixed validation subset drawn from the *official
training data* for every feasibility-stage and encoder, representation,
or regularization decision. The official test set's labels stay
untouched until the design and code are both fully locked, and only one
final evaluation pass runs against them.

## The feasibility ladder: staged steps, not the full dataset at once

**Correction from review**: running a feasibility check on the "full
10-class KMNIST" set is up to 70,000 images. Each image needs local
convergence plus a 505-node ODE solve. This is not a small mechanical
check, even with validated GPU infrastructure.

**Correction from a third review round, load-bearing: the line between
"feasibility" and "confirmatory" was not sharp enough.** The second
draft's own stage 3 ("full dataset") and the confirmatory section's
"official test set touched exactly once" needed a clearer separation
between the training-side ladder and the one, single test-side
evaluation. **Locked, four stages, kept strictly separate**:

1. **1,000 training images** (100 per class) — checks the pipeline
   works correctly, end to end. Nothing more.
2. **Up to 5,000 official-training images**, plus the fixed
   training-derived validation subset (never the official test set) —
   measures throughput, and supports development work.
3. **The full 60,000-image official KMNIST training set** — the final
   feature generation and model selection step (regularization search,
   representation checks). This still uses only training-derived data.
4. **One locked evaluation on the untouched 10,000-image official test
   set.** This is the only stage that touches test-set labels, and it
   happens exactly once, after stages 1 through 3 and the rest of this
   design are fully settled.

Stages 1 through 3 may guide engineering and representation choices
using training-derived validation data. **They must not run repeated
evaluations on the official test set.** A disappointing validation-stage
result may justify stopping the project before stage 4, but it must not
trigger a cycle of redesigning and then re-evaluating on the real test
set.

**Go/no-go rules for moving between stages 1 through 3, locked as
mechanical checks, not as a promised direction of effect.** A "no-go" on
any of these is a real, reportable outcome, not a failure of the
experiment. **Correction from a fourth review round, load-bearing: these
rules must be exact, working thresholds, fixed now, not left for
implementation time to decide:**
- **Zero non-finite feature values** (NaN or infinity) anywhere in the
  pipeline, at any stage. Even one occurrence stops the process. This is
  not a rate to tolerate.
- **Zero silent solver failures**: every local-convergence or
  graph-evolution ODE solve either completes and reports its own success
  status, or is logged as a failure. None may fail silently and be
  treated as a success.
- **Recoverable solver failure rate**: a solver call that fails but can
  be recovered is tolerated up to **0.1%** of calls at any stage.
  **Above 0.1%, the team stops moving to the next stage and
  investigates.** This is not itself a Stage 2A scientific finding. It
  is a pipeline-health check. **Correction from a fourth review round**:
  the earlier example recovery policy ("retry with a tighter tolerance")
  was backwards. A tighter tolerance demands *more* from the solver,
  which makes recovery less likely, not more. **Locked recovery policy,
  tried in this order**: (1) retry with a smaller `MAX_STEP`; (2) retry
  with a larger `max_steps` limit; (3) fall back to a different,
  pre-chosen solver (`Radau`, which handles stiff problems better). The
  exact retry order is coded once and logged the same way every time a
  failure is recovered. This is engineering work, not a further design
  choice.
- **Runtime and storage**: the per-image cost measured at stage 2 is
  used to project the cost of the full 60,000-image stage 3 run. This
  projection must be documented and explicitly approved before stage 3
  starts. It is not assumed acceptable just because stage 2 ran without
  error.
- **The `R(theta)` check**: the complete distribution of `R(theta)`
  (before and after evolution, for every condition) is reported at every
  stage. Any noticeable amount of R sitting near the numerical limits (0
  or 1) is flagged directly. This is a reporting requirement, not a
  trigger for changing the feature method — per the locked single
  reference-node rule above, `R(theta)` never changes which feature
  representation gets computed.
- **Classifier convergence**: the linear readout must converge (as
  defined by the classifier-implementation rule in "Linear readout,"
  below) in every condition, at every stage. **Stated exactly the same
  way here as there, not a second, different phrasing**: any fit that
  fails to converge, in any required fold and `C` combination, stops
  moving to the next stage until the team investigates. This is a hard
  stop, not just something to log.

Per the original design, three conditions run at every ladder stage: raw
pixels into a linear model; encoded pre-evolution into a linear model;
evolved on T into a linear model. Conditions 2 and 3 use the same
classifier design and feature count throughout. Stage 3 also adds the
secondary evolved graphs (lattice, canonical rewired, canonical random —
see "Confirmatory expansion," below) for model-selection purposes,
before the single stage-4 evaluation.

## The confirmatory endpoint and test: locked before any full-test-set result is seen

**Correction from review, load-bearing**: an earlier draft said "any
signal… that evolution changes held-out accuracy." That is not a locked
primary metric or test.

**Correction from a third review round, load-bearing: the one primary
comparison must be named directly, not left spread across several
graphs.** The confirmatory expansion (below) evolves T, lattice, a
canonical rewired graph, and a canonical random graph. If every
evolved-versus-pre-evolution comparison were treated as equally primary,
several graphs would give several separate chances to claim "graph
evolution helps," even though only one endpoint is locked. **Locked: the
sole primary comparison is T-evolved versus encoded-pre-evolution.**
This matches the continuity reasoning already used to justify choosing
class-0's T throughout this project. Every other evolved-graph
comparison (lattice, canonical rewired, canonical random, each against
encoded-pre-evolution; and evolved T against the other evolved graphs)
is **secondary and specific to that one graph**. These are reported
descriptively. They cannot rescue a null (no-effect) primary result.

**Locked direction rule**, stated plainly, so "improvement" means one
specific thing everywhere below. For held-out image `i`:

```
d_i = ell_i(evolved T) - ell_i(encoded pre-evolution)
```

`ell_i` is that image's multiclass log-loss (a measure of prediction
error; lower is better) under the matching classifier. **Improvement
means `E[d_i] < 0`** (evolved T's log-loss is lower — better — than the
pre-evolution baseline's log-loss). The same direction rule applies to
every secondary evolved-versus-pre-evolution comparison, using the
matching graph in place of T.

**Locked, primary test — fully defined and ready to run, not left as a
category**:
- Take 20,000 paired, class-stratified bootstrap resamples of the
  10,000 official test images. "Stratified" means each resample keeps
  each class's original image count, drawing with replacement within
  each class.
- Compute the mean per-image `d_i` on each resample.
- Report a two-sided 95% **percentile** interval over the 20,000
  resampled means.
- **Success rule**: the primary result supports improvement only if the
  entire interval is below zero. In the same way, it would support the
  pre-evolution baseline winning only if the entire interval is above
  zero. An interval that includes zero is a null primary result. No
  secondary comparison can rescue it.
- **What this interval does and does not capture, stated explicitly**:
  it measures uncertainty coming from which official test images were
  drawn, given the classifiers, the feature pipeline, and the topology
  are all already fixed. It does **not** include uncertainty from
  retraining the feature pipeline, re-selecting regularization, or
  redrawing the topology. Each of those is a fixed, one-time choice in
  this design, not something resampled here.

**Secondary tests**: McNemar's exact test for accuracy disagreement
(easier to interpret, though less statistically complete, than log-loss
alone); macro-F1; per-class recall; raw accuracy; and the same paired
bootstrap method applied to each secondary graph-specific comparison,
reported descriptively. These are not Holm-corrected as a family, since
none of them is a second chance at the primary claim.

The feasibility ladder (above) may report raw differences descriptively,
with no formal statistical test. The metric and test described here are
locked specifically for the one, final, confirmatory evaluation against
the official test set.

## Confirmatory expansion: about specific graphs, not a claim about a whole family, unless scaled up

**Correction from review, load-bearing, directly applying Stage 1D's own
lesson**: a single rewired graph, or a single random-construction
result, does not support a claim about a whole family ("rewiring" or
"random construction" in general). Stage 1D used 25 realizations for
exactly this reason — one realization is not a reliable estimate of a
random family's behavior.

There are two valid scopes here, and this design must pick one on
purpose, rather than drifting into the wrong claim by default:

- **Fixed, pre-chosen graph instances** (cheaper): one pre-chosen
  version each of lattice, rewired, and one random construction,
  compared against T. **The claim must stay specific to these graphs**
  — never "T versus rewiring in general" or any claim about a family.
- **Topology-family comparison** (more expensive, in the style of Stage
  1D): many versions of each stochastic control graph, combined at the
  level of each realization, supporting a real family-level claim.

**Locked for this first Level 3 result: fixed, pre-chosen graph
instances.** This is cheaper, and a first result does not need
family-level generality to be informative. But every reported claim
about lattice, rewired, or random in this design's confirmatory results
must be stated as being about that one specific, realized graph, not the
family it came from. Moving to a family-level design is a legitimate,
separate, future extension if this first result motivates it.

**Wording correction from a fourth review round**: "representative
graphs" is replaced with "prespecified graph instances" throughout this
design. A single draw was chosen ahead of time (named below, before any
Stage 2A result exists), but one draw cannot prove it represents its
whole family statistically. It only proves it is a fixed, pre-chosen
example of that family. This is a wording fix, not a scope change — the
graph-specific-only claim above already stated this correctly.

**Correction from a third review round, load-bearing: these graph
instances must not be chosen after seeing an outcome.** The exact
rewired and random graphs are named here, before any Stage 2A result
exists, reusing files this project already had, rather than drawing new
ones specifically for this design:

- **Lattice**: T's already-standard matched lattice construction
  (`class0_constructions.pkl`'s cached `lattice`). Deterministic, no
  random seed. Already used throughout Stage 1D.
- **Canonical rewired**: `degree_preserving_rewire(W_T, ink_mask_active,
  seed=0)` — this is Stage 1D's own first pilot version
  (`experiments/stage1d_topology_specificity/results/stage1d_constructions.pkl`),
  reused here rather than drawing a new random seed for this design.
- **Canonical random**: `generate_matched_sparsity_topology(W_T,
  ink_mask_active, seed=0)` — this is **curr_random specifically, not
  hist_random**. The team chose curr_random because it needs no
  coupling-budget rescaling step, and it matches T's edge count exactly
  by construction. hist_random has its own isolated-fixed-coordinate
  failure mode (see `DESIGN.md`, Stage 1D) — a real complication this
  design does not need to inherit for a single pre-chosen instance. Same
  seed=0, same reasoning as rewired above.

**Graph statistics, recorded for all four graphs (T, lattice, canonical
rewired, canonical random) rather than left unstated**:

| graph | n edges | total weight | mean weighted degree | min degree | max degree |
|---|---:|---:|---:|---:|---:|
| T | 1051 | 1959.98 | 3.881 | 0.902 | 12.153 |
| lattice | 935 | 1959.98 | 3.881 | 1.048 | 4.193 |
| rewired (seed=0) | 1051 | 1959.98 | 3.881 | 0.903 | 12.122 |
| curr_random (seed=0) | 1051 | 1959.98 | 3.881 | 0.000 | 12.083 |

**Disclosed directly, not quietly absorbed — corrected from an earlier
undercount**: under this specific draw, `curr_random` seed=0 has **13
isolated nodes** (weighted degree 0.0), not the "one isolated node"
originally stated in this table. The original check only tested
`nodes_T`'s three fixed nodes for isolation (Stage 1D's own limited
screening), not all 505 nodes. A full scan (prompted by the Laplacian
spectrum check in `FINDINGS.md`'s stage-3 pre-check) found 13. This
splits the graph into 14 separate connected pieces (13 single-node
pieces, plus one 492-node main piece). This was confirmed directly, by
looking at the graph Laplacian's eigenvalues (14 values at or near zero,
compared to exactly 1 for T, lattice, and rewired, which are each a
single connected piece). Unlike Stage 1D's perturbation-response
measurement (where an isolated fixed intervention node breaks that
project's tangent and event-alignment machinery entirely), an isolated
node under Stage 2A's plain, unperturbed evolution simply never changes
its phase during evolution (`dtheta_i/dt = 0`, since every `W_ij` value
for that row is 0). This is not a pipeline failure. It is a real,
disclosed property of this specific pre-chosen graph, worth reporting
alongside its results, not hidden.

Each topology gets two conditions: encoded-pre-evolution and evolved,
both using the shared 505-node encoding pipeline above. The encoding
step does not depend on the topology; only the evolution step does.

## Baselines: a parameter-matched model, and separately, a competent ordinary network

The oscillator readout has `1008*10 + 10 = 10,090` trainable parameters.
(**Corrected from `2*505*10 + 10 = 10,110`** — the reference node's two
constant columns are dropped, per the effective-feature-count correction
above, so the true input size is 1008, not 1010.) A parameter-matched MLP
(a standard multi-layer network: `R^784 -> Linear(784,H) -> ReLU ->
Linear(H,10)`) needs `795*H + 10` parameters. Matching this to 10,090
gives `H ≈ 12.68`, so **H=13** (10,345 parameters, verified by direct
calculation, not just estimated). This is the same locked value as
before this correction, since 12.68 and the earlier estimate of 12.7
both round to 13.

**Correction from review**: H=13 is a very narrow MLP. It is a valid
parameter-matched control, but it is not a fair test of "does this do as
well as a competent ordinary classifier." **Locked: report both,
clearly kept separate**:
- **H=13 MLP**: a strict, parameter-matched control (fair by parameter
  count).
- **A separate, competent-context MLP, `H=128`, locked** (correction
  from a third review round: "128 or similar" was not yet a fixed
  value). This one is not parameter-matched. It is reported explicitly
  as practical context, not as a fairness control.

For both baselines and the oscillator-readout conditions, report:
trainable parameter count, total compute for one forward pass (state
whether this is measured in FLOPs or wall-clock time), feature count,
training epochs or optimization steps, and the regularization search
budget. The oscillator's own evolution step does not receive gradient
updates (it is frozen), but its runtime compute is real and must be
reported alongside the MLPs', not treated as free.

## Linear readout: the regularization grid, locked

**Correction from a third review round**: the regularization grid, the
selection method, and the tie-break rule were left undefined. **Locked**:
- Classifier: multinomial logistic regression (softmax), with L2
  regularization.
- Regularization grid: `C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3,
  1e4}` (9 values, evenly spaced on a log scale). `C` controls how
  strongly the classifier is regularized: a smaller `C` means stronger
  regularization.
- Selection: fixed 5-fold cross-validation on the official training
  data, with each class kept balanced across folds. The same fold
  assignment (controlled by `SEED=42`) is reused across every condition,
  so results are comparable.
- Select the `C` with the lowest mean validation log-loss.
- **Deterministic tie-break rule**: if multiple `C` values tie on mean
  validation log-loss, choose the smaller `C` (stronger regularization).
- **Each feature condition (raw pixels, encoded-pre-evolution, evolved
  T, and every secondary evolved graph) selects its own `C`
  independently**, using the identical grid and fold assignments. No `C`
  value is shared across conditions.

**Correction from a fourth review round, load-bearing: an earlier draft
said "standardized… only if," leaving unclear whether standardization
happens at all, and said nothing about keeping the folds' scaling
separate.** Locked, stated precisely:
- **Every logistic-regression condition uses per-feature
  standardization** (each feature is rescaled to zero mean and unit
  variance). This is not optional, and it is not left to per-condition
  choice.
- **Fold-safe fitting**: for each condition, during cross-validation, the
  standardizer (called a "scaler") is fit only on that fold's *training*
  portion of the data, and then applied unchanged to that fold's
  validation portion. The scaler is never fit on validation data.
- **After `C` is selected**: a new scaler is fit on the complete official
  training set for that condition, and applied unchanged to that
  condition's official test features. This final scaler is fit exactly
  once, after the design and `C` are both already locked.
- **Each condition (raw pixels, encoded-pre-evolution, evolved T, every
  secondary evolved graph) gets its own independently-fit scaler** at
  every stage above. Raw, pre-evolution, and evolved features have
  different value distributions, so "the same standardization" means the
  same *procedure*, applied consistently, never the same numerical means
  and variances shared across conditions.

**Classifier implementation, locked to prevent platform-dependent
default settings from silently becoming part of the experiment**:
- Solver: `lbfgs` (scikit-learn's default multinomial solver, stated
  here directly rather than left to whatever the installed software
  version happens to default to).
- Convergence tolerance: `tol=1e-4`.
- Maximum iterations: **`max_iter=10000`** (raised from the original
  `max_iter=1000` — feasibility stage 2 hit a real non-convergence at
  1000 iterations. The team investigated and found it was caused by
  severe multicollinearity specific to `evolved_T`'s standardized
  feature matrix, not by a separability problem that more sample size
  could fix. See `FINDINGS.md`'s "Why evolved_T specifically fails to
  converge" section for the full investigation. The change was applied
  uniformly to all three conditions, not only the affected one, for
  consistency. **If `10000` iterations still does not converge for some
  condition, that counts as a new, separate finding** — it is not
  grounds to raise `max_iter` again, but a signal that a more careful
  fix (a documented minimum ridge floor, or decorrelating or whitening
  the features before classification) is the right next step, rather
  than pushing the optimizer harder.)
- Class weighting: uniform (`class_weight=None`) — KMNIST's 10 classes
  are balanced by construction, so no reweighting is applied.
- Random seed: `random_state=42` wherever the solver accepts one
  (controls any internal random tie-breaking; it does not affect the
  cross-validation fold assignment, which is separately controlled by
  `SEED=42` above).
- **Non-convergence**: if a fit does not converge within `max_iter`, for
  any combination of condition, fold, and `C`, this is logged and
  reported explicitly (which combination, and at what iteration count).
  It is never silently accepted or silently re-run with a larger
  `max_iter`. A pattern of non-convergence concentrated in one condition
  or one part of the `C` grid is itself a reportable finding. **Correction
  from a fourth review round: logging alone does not satisfy the
  "converges successfully in every condition" go/no-go rule above.**
  Locked, the clear consequence: **any non-converged fit, in any
  required fold and `C` combination, stops the team from moving to the
  next stage until they investigate.** Both logging and stopping are
  required — logging is not a substitute for stopping.

## Named possible outcomes (unchanged from the first draft)

1. Dynamics are useful, and the topologies are equivalent (across the
   specific graphs tested — see the graph-specificity scope above).
2. Dynamics are useful, but one specific graph wins — this would show
   that equivalence on `Delta_map` (a different metric) does not mean
   equivalence in task usefulness.
3. Dynamics are not useful — the network has Level 2 structure, but no
   demonstrated Level 3 value under this task and this readout. This
   would be a real negative result.
4. The encoded-pre-evolution state already helps, and graph evolution
   adds no further value — in that case, the useful part would be the
   *local* encoding dynamics, not graph-level evolution specifically.

## What this design does not do

- It does not attempt denoising (that is Stage 2B) or generation —
  deferred, for the same reasons as in the first draft.
- It does not test role-matched intervention, or classes using their own
  topologies for their own images — this would be circular, as rejected
  above.
- It does not test a genuinely static (`pi*x`, no local convergence)
  encoding as a primary condition — this is named above as a
  well-motivated future extension, not required for this design's
  primary claim.
- It does not support a claim about a topology *family*, under the
  locked fixed-prespecified-graph-instances scope — moving to that would
  be a separate, larger, future design, if motivated by this one's
  results.
- It does not treat any secondary, graph-specific comparison (lattice,
  canonical rewired, canonical random, or evolved-T-versus-other-evolved-
  graphs) as able to rescue a null primary (T-evolved versus
  encoded-pre-evolution) result — per the locked primary and secondary
  ranking above.

## Status

Fourth draft. The first review's ten corrections (kept in the second
draft) and the second review round's five load-bearing corrections
(kept in the third draft, summarized below) are unchanged from the third
draft. This draft adds a **third review round's one important
correction and three smaller closures**, on a third draft the reviewer
judged "substantially improved… close to lock":

1. **The one primary comparison is named**: T-evolved versus
   encoded-pre-evolution, with an explicit direction rule (`d_i =
   ell_i(evolved T) - ell_i(pre-evolution)`, improvement means `E[d_i] <
   0`). Every other evolved-graph comparison is secondary and specific
   to that graph. None can rescue a null primary result.
2. **The bootstrap is fully defined and ready to run**: 20,000 paired,
   class-stratified resamples of the 10,000 official test images, a
   two-sided 95% percentile interval on the mean per-image log-loss
   difference, with success meaning the entire interval is below zero,
   and an explicit statement of what uncertainty the interval does and
   does not capture.
3. **The feature method is fixed to a single, non-switching rule**:
   reference-node centering (`theta_ref` = T's own median-degree node,
   active index 363) as the primary method for every image and
   condition. The circular-mean method is demoted to a secondary
   robustness check, instead of a rule that switches based on the
   state itself.
4. **The feasibility ladder and the confirmatory run are sharply
   separated** into four stages (1,000 images; up to 5,000 training
   images; the full 60,000-image training set; one locked 10,000-image
   test evaluation), with go/no-go rules based on mechanics, not on the
   direction of effect, between stages.
5. **Remaining implementation choices are locked**: the regularization
   grid (`C` at 9 evenly log-spaced values, 5-fold cross-validation with
   balanced classes, a deterministic tie-break toward stronger
   regularization, each condition selecting its own `C`), the competent
   MLP's width (`H=128`, fixed), and the exact pre-chosen graph-instance
   seeds (rewired and curr_random, both seed=0, reused from Stage 1D's
   own pilot files, with graph statistics recorded — later corrected to
   13 disclosed isolated nodes in curr_random's draw, not the originally
   stated one; see "Confirmatory expansion," above).

**This draft's own corrections**:

1. **The class-0 prior disclosure is corrected**: the pre-evolution
   condition is not free of class-0 influence — it inherits T's
   class-0-derived `active_indices` (choice of nodes), just not T's edge
   structure. The primary comparison isolates evolution and edge
   structure specifically, since both conditions already share the same
   choice of nodes. This is a cleaner design than the previous
   disclosure implied, not a weaker one.
2. **Standardization is now fully locked**: per-feature, fold-safe
   (fit on the training fold only, applied to that fold's validation
   data; refit once on the complete official training set after `C` is
   selected, applied unchanged to the test set), and independently fit
   per condition. The classifier implementation (solver, tolerance,
   maximum iterations, class weighting, random seed, non-convergence
   handling) is locked alongside it, to prevent platform-dependent
   default settings from entering the experiment.
3. **The encoder-random-seed robustness check is now fully defined**: it
   reuses the fixed feasibility-stage-2 validation subset, sets seeds by
   each image's dataset index, refits both readouts on identical folds
   and `C`-selection, and reports the change in validation log-loss.
   This check is explicitly descriptive. It can never replace the
   seed-0 primary analysis.
4. **The go/no-go thresholds are now executable**: zero non-finite
   features, zero silent solver failures, a named 0.1% recoverable
   failure-rate stop threshold, an explicitly-approved runtime and
   storage projection before stage 3, and full `R(theta)` distribution
   reporting (never used to trigger a change in feature method).
5. **Wording**: "representative graphs" is replaced with "prespecified
   graph instances" throughout — a single draw is fixed ahead of time,
   not proven to be a statistical representative of its family.

**A fourth review round confirmed this draft closes every substantive
issue**, and asked only for two engineering clarifications and one
representational note, none of which needed further scientific review:

1. **The non-convergence stop rule is made unambiguous**: logging a
   non-converged fit no longer stands alone — it now explicitly stops
   the team from moving to the next stage until they investigate (see
   the classifier-implementation section above).
2. **The ODE-recovery policy is corrected**: the earlier "retry with a
   tighter tolerance" example was backwards (a tighter tolerance demands
   more from the solver, not less). The locked recovery policy now tries
   a smaller `MAX_STEP`, then a larger `max_steps` limit, then a
   different, pre-chosen solver (`Radau`), in that order (see the
   feasibility-ladder go/no-go section above).
3. **A note about the reference-node representation**: `theta_ref`'s own
   two circular features are trivially constant (`cos(0)=1`, `sin(0)=0`)
   for every image. They are dropped on purpose, so the effective
   feature count is **1008**, not 1010 (see the feature-representation
   section above; the oscillator-readout parameter count in "Baselines"
   is corrected to match: 10,090, not 10,110. `H=13` remains the locked,
   parameter-matched MLP width regardless).

**Reviewer's final verdict: "design approved and ready to lock. Begin
Stage 2A feasibility stage 1."** This document is now **LOCKED**.
Feasibility stage 1 (1,000 training images, checking the pipeline works
correctly, end to end — explicitly not an early scientific result, per
this document's own feasibility-ladder section) is the next step,
implemented directly against this locked design.
