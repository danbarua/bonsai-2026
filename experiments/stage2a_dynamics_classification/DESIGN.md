# Stage 2A: Does Runtime Oscillator Evolution Improve Classification?

*Second draft, revised per external review of the first. Not yet locked
-- follow this project's established convention (Stage 1D's own path
through four review rounds before any code ran).*

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

## Feature representation: circular embedding, with a locked singularity fallback

```
mu = angle(mean(exp(1j * theta)))
h(theta) = [cos(theta_i - mu), sin(theta_i - mu) for all i]
```

Reuses `stage1b2_core.py`'s rotation-removal convention (there applied
to a phase *difference*; here to a phase *configuration* directly --
analogous, not identical, since a raw phase configuration lives on a
torus and the project's linear rotation-removal *projector* doesn't
transfer to it the same way a phase-difference shift does).

**Correction from review, load-bearing**: the circular mean `mu` is
numerically unstable when the order parameter magnitude
`R(theta) = |mean(exp(1j*theta))|` is near zero (phases close to
uniformly scattered) -- a small phase perturbation near this point can
swing the computed mean angle wildly, injecting spurious rotation into
the feature vector unrelated to any real structure.

**Locked protocol**:
- Record `R(theta)` for every state, pre- and post-evolution, for every
  image -- a required diagnostic output, not optional.
- Prespecify a floor (default: `R(theta) < 0.05`, revisit once real
  `R` values from the feasibility pass are in hand) below which
  mean-angle centering is considered unreliable.
- **Fallback when below floor**: align to a fixed reference node's phase
  (`theta_ref`, a single prespecified active-node index) instead of the
  circular mean, rather than allowing an ill-defined angle to rotate the
  whole feature vector. Record how often the fallback triggers, per
  condition -- if it triggers often, that is itself worth reporting, not
  silently absorbed.

The shift (mean-angle or fallback) is computed fresh, independently, for
whichever state is being featurized -- pre-evolution and post-evolution
each get their own, matching the prior draft.

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
validated GPU infrastructure. **Locked staged ladder**, each stage
gated on the previous one actually working before scaling up:

1. **100 images per class** (1,000 total) -- end-to-end correctness
   only: does the full pipeline run without error, are all outputs
   finite, does `R(theta)` look reasonable, does the linear readout fit
   at all.
2. **1,000-5,000 training images**, plus the fixed held-out-from-training
   validation subset -- throughput measurement and signal *direction*
   (does condition 3 move relative to condition 2 at all, in either
   direction) -- still not confirmatory.
3. **Full dataset** -- only after stage 2's runtime/storage extrapolation
   and numerical checks (finite features, reasonable `R(theta)`
   distribution, fallback-trigger rate) pass.

At every stage: report honestly, do not treat any stage's result as the
Stage 2A finding. Only the final, fully-locked confirmatory run (below)
produces the actual claim.

Three conditions per the original design (raw pixels -> linear; encoded
pre-evolution -> linear; evolved on T -> linear), same classifier
architecture and feature dimension for conditions 2 and 3 throughout
every ladder stage.

## Confirmatory endpoint and test: locked before any full-test-set result is examined

**Correction from review, load-bearing**: the prior draft said "any
signal... that evolution changes held-out accuracy" -- not a locked
primary metric or test. **Locked**:

- **Primary endpoint**: held-out multiclass log-loss difference between
  condition 3 and condition 2 (retains confidence information; more
  statistically informative than accuracy alone for this comparison).
- **Primary test**: paired confidence interval via test-image bootstrap
  (conditions 2 and 3 classify the *same* held-out images -- this is a
  paired comparison, not two independent samples, and must be treated as
  such).
- **Secondary**: McNemar's exact test for accuracy disagreement (more
  interpretable, if less statistically complete, than log-loss alone);
  macro-F1; per-class recall; raw accuracy.
- The feasibility ladder (above) may report raw differences descriptively,
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
- **A separate, modestly-sized ordinary MLP** (H chosen for reasonable
  practical competence, not parameter equality -- a specific value to be
  set when this is implemented, e.g. H=128 or similar, stated plainly as
  "not parameter-matched, included for practical context").

For both baselines and the oscillator-readout conditions, report:
trainable parameter count, total forward-pass compute (state which:
FLOPs or wall-clock), feature dimension, training epochs/optimization
steps, and regularization search budget. The oscillator's own evolution
is frozen (no gradient through it) but its runtime compute is real and
must be reported alongside the MLPs', not treated as free.

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
- Does not yet fix the exact cross-validated regularization grid, the
  `R(theta)` reliability floor's final value (default 0.05, pending real
  data), or the competent-MLP's exact hidden width -- implementation-
  level specifics to confirm at lock time, not fixed here.

## Status

Second draft. Ten load-bearing corrections from the first review
incorporated: precise framing of what "evolution" adds on top of an
already-dynamical encoder; explicit active-node restriction; official
KMNIST split with test-set-leakage prevention; a staged feasibility
ladder instead of a full-dataset first pass; explicit encoder-RNG
handling with a robustness check; a circular-mean singularity floor and
fallback; a locked primary endpoint (log-loss) and paired test structure;
explicit graph-specific (not family-general) confirmatory scoping; and
a dual parameter-matched-plus-competent MLP baseline, arithmetic
verified independently (H=13, 10,345 parameters). Not yet locked --
ready for a further review round if anything here still needs
tightening, otherwise ready to move toward implementation.
