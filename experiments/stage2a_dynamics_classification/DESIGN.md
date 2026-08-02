# Stage 2A: Does Runtime Oscillator Evolution Improve Classification?

*First draft, synthesized from two rounds of discussion (not yet
implemented, not yet locked by external review -- follow this project's
established convention: design first, reviewed, then built, the same
path Stage 1D took through four review rounds before any code ran).*

## The question, precisely

Part 1 (`docs/PROJECT_MEMORY.md`) closed classification via *static
exported features* in the negative: E and R, read off a frozen snapshot,
added nothing over generic spatial/DCT controls. Stage 2A asks the
directly analogous question one level up: **does the oscillator
system's runtime evolution -- not a frozen snapshot, the actual
trajectory -- provide information a linear readout can use for
classification that the unevolved, merely-encoded state does not
already provide?**

This is deliberately the first Level 3 test, not denoising (Stage 2B,
deferred -- see "What this does not do"). Classification is the tighter
scientific callback to Part 1's own closed result, has substantially
fewer undetermined design choices than a diffusion-style task (no noise
schedule, no prediction-target choice, no reverse-sampler), and reuses
labels already present in the data this whole project has used
throughout.

## A design fork that must be resolved explicitly, not left implicit

Classification requires images from multiple classes. T is specifically
KMNIST class 0's learned topology. This creates a real architectural
choice neither an earlier draft of this proposal nor its review resolved
explicitly:

- **Class-conditioned topology** (each class's images evolved on *its
  own* class-specific topology, matching how Part 1's original
  benchmark-programme scoring worked) is **circular for a genuine
  predictive classifier** -- selecting which topology to evolve an image
  on requires already knowing its class, which is exactly what the
  classifier is supposed to predict.
- **Single fixed topology for all classes** (every image, regardless of
  true class, evolved on the *same* one topology) avoids this circularity
  entirely and is the only version of this design that is a genuine
  held-out classifier, not a scoring/verification setup.

**Locked choice: single fixed topology for all classes.** For the
feasibility pass and initial confirmatory run, this is **class 0's T**
(`class0_constructions.pkl`), chosen for continuity with every other
stage in this project, arbitrarily but consistently -- not because
there's a principled reason class-0's T should be privileged for images
of other classes. The confirmatory expansion (below) is exactly what
tests whether that pinned choice matters, by comparing against lattice/
rewired/random evolved under the identical protocol.

## Encoding: reuse the established convention, specified precisely

**Locked choice: `_local_converged_phases` (`learned_topology_construction.py`)
as the canonical encoder**, used unaltered -- this is already the
project's defined interface between image structure and phase state,
and inventing a parallel encoder for this task risks a subtle,
unnecessary discrepancy from everything else in this project. Concretely,
per image `x` (normalized to `[0,1]`, `28x28`):

1. `target_phase = pi * x` -- the target phase field this function
   already uses internally.
2. Run `_local_converged_phases` to convergence (same steps/dt/k_coupling/
   k_bias defaults already used throughout this project) -- this is
   `theta(0)`, the initial phase configuration handed to evolution.
3. Evolve `theta(0)` under the chosen topology's dynamics
   (`dtheta/dt = sum_j W_ij sin(theta_j - theta_i)`, the same RHS used
   everywhere else in this project, **no perturbation, no tangent
   system** -- this is unperturbed evolution of an encoded image, not a
   perturbation-response measurement) for `T_HORIZON=2.5` (reusing the
   existing constant), reading off `theta(T)` at the fixed endpoint. No
   event-alignment logic (`tau_star`, `E`/`C`) applies here -- that
   machinery is specific to measuring *perturbation response departure
   from a tangent prediction*, not applicable to plain evolution of an
   encoded image with nothing perturbing it.

**What determines what, stated explicitly** (per the review's own
request that this be unambiguous): the image determines a *target phase
field* used for local convergence, which produces `theta(0)` -- not the
initial phases directly, and not intrinsic frequencies. This is a single
primary mapping, not a combination.

## Feature representation: circular embedding, rotation-corrected

A raw phase vector is a poor readout input because of angular wrapping
(a phase near 0 and near `2*pi` are physically identical but numerically
distant). Use, for any phase configuration `theta` being featurized
(whether pre- or post-evolution):

```
shift = angle(mean(exp(1j * theta)))          # circular mean of this configuration
h(theta) = [cos(theta_i - shift), sin(theta_i - shift) for all i]   # 2N-dim
```

**This reuses an established project convention, applied one level
earlier than its original use.** `stage1b2_core.py`'s `get_outputs_at`
already computes `shift = angle(mean(exp(1j * diff_phase)))` and
subtracts it before measuring displacement, specifically because the
dynamics are invariant to a global rotation of all phases together. The
same reasoning applies here, just to a phase *configuration* rather than
a phase *difference* -- **this is analogous, not identical**, to the
project's rotation-removal *projector* (`P = I - ones(n,n)/n`) used
elsewhere for perturbation directions: `theta` lives on a torus, not in
a vector space, so the linear projector doesn't transfer directly to a
raw phase configuration the way it does to a tangent-space direction
vector. Circular-mean subtraction is the correct torus-appropriate
analog, not the same operation reused verbatim.

**The shift is computed fresh, independently, for whichever state is
being featurized** -- the pre-evolution encoded state gets its own
shift removed; the post-evolution state gets its own, separately
computed shift removed. Both the no-evolution and evolved conditions
use this identical procedure (same formula, independently applied),
so neither condition is advantaged or disadvantaged by how the rotation
ambiguity is resolved.

## Minimal feasibility pass (not a confirmatory claim)

Smaller than the eventual confirmatory design, deliberately. One
topology (T), three conditions, same train/test split (fixed seed,
reusing this project's `SEED=42` convention), same classifier type,
optimizer, and regularization search across conditions 2 and 3 (which
also share the same feature dimension by construction, `2 x 505`, since
both are the circular embedding of the same topology's node states):

1. **Raw pixels -> linear classifier.** `x` (784-dim) directly into a
   multinomial logistic regression. Context only, does not isolate
   dynamics -- not the decisive comparison.
2. **Encoded, no evolution -> linear classifier.** `h(theta(0))` (the
   circular embedding of the *pre-evolution* encoded state) into the
   same classifier architecture as condition 3.
3. **Evolved -> linear classifier.** `h(theta(T))` (the circular
   embedding of the state *after* evolution on T) into the same
   classifier architecture as condition 2.

**The decisive comparison is condition 3 vs. condition 2** -- this
isolates the causal contribution of runtime evolution, holding the
encoding, feature representation, classifier architecture, and training
procedure fixed.

Classifier specification: multinomial logistic regression (softmax),
L2-regularized, regularization strength chosen via a small cross-
validated grid on the training split, identical grid and selection
procedure for conditions 2 and 3.

**Dataset scope for this pass**: full 10-class KMNIST (classification
requires multiple classes), fixed train/test split (`SEED=42`), every
image evolved on class-0's T regardless of its own true class, per the
locked single-fixed-topology choice above.

**What the feasibility pass should answer** (mechanical/practical, not
the scientific claim itself):
- Does the full image -> phase -> evolution pipeline run at useful
  throughput across the full 10-class dataset?
- Are evolved features finite and numerically stable across the whole
  dataset (no NaN/inf from the local-convergence or evolution steps)?
- Does the linear readout train successfully in both conditions 2 and 3?
- Is there *any* signal, in either direction, that evolution changes
  held-out accuracy relative to condition 2?
- What are the actual runtime and storage requirements, measured, for
  scaling this to the full confirmatory design below?

**This pass is not used for a confirmatory claim** -- report what it
shows honestly, but do not treat a positive or negative signal here as
the Stage 2A result. Its job is feasibility, not inference.

## Confirmatory expansion (only after the feasibility pass runs mechanically)

Expand to:
- **Topologies**: T, lattice, rewired, and **one** random construction
  (not both hist_random and curr_random -- Stage 1D already established
  these behave similarly on the internal-mapping endpoint; no need to
  duplicate both here unless the feasibility result makes topology-
  family distinctions central to the story, in which case revisit this).
- **Conditions per topology**: no-evolution counterpart and evolved
  counterpart, both using that topology's own encoding pipeline (the
  encoding step itself doesn't depend on topology, only the evolution
  step does -- so "no-evolution" is the same `h(theta(0))` regardless of
  which topology's confirmatory row it's reported alongside; it's
  included per-topology in the results table for clarity, not because
  it's a genuinely different computation each time).
- **Raw-pixel linear baseline**: unchanged from the feasibility pass,
  reported once, not per-topology.
- **Parameter-matched ordinary shallow network**, specified exactly (not
  left as a category): `x in R^784 -> Linear(784, H) -> ReLU ->
  Linear(H, 10)`, with `H` chosen so trainable parameter count is as
  close as possible to the oscillator-readout condition's own parameter
  count. Report, alongside this baseline: trainable parameter count,
  total forward-pass compute (FLOPs or wall-clock, stated which),
  feature dimension, training epochs/optimization steps, and
  regularization search budget -- parameter count alone is not a
  sufficient specification of a matched control. The oscillator's own
  evolution is frozen (no gradient through it), but its runtime compute
  is real and should be reported alongside the MLP's, not treated as free.

## Named watched-for outcomes (stated before running anything)

Stage 1D found T/lattice/rewired/hist_random/curr_random statistically
indistinguishable on internal mapping strength (`Delta_map`). Stage 2A
tests whether downstream task utility is likewise topology-generic, or
dissociates from that internal-mapping result. Four possible outcomes,
named now, not reconstructed after seeing scores:

1. **Dynamics useful, topologies equivalent.** Runtime evolution
   improves classification (condition 3 > condition 2, consistently
   across topologies), but no topology beats the others. Coherent story:
   runtime dynamics provide useful features; the learned topology isn't
   uniquely responsible, consistent with Stage 1D's own finding
   extending to this new endpoint.
2. **Dynamics useful, one topology wins.** Runtime evolution helps, and
   T (or some other specific construction) shows a real advantage over
   the others despite Stage 1D finding no `Delta_map` difference between
   them. This would show `Delta_map` equivalence does NOT imply task-
   utility equivalence -- internal reproducibility and externally useful
   representation quality would be genuinely distinct properties, not
   two views of the same thing.
3. **Dynamics not useful.** Condition 3 does not beat condition 2 (or
   the raw-pixel/MLP baselines) on any topology. Level 2 structure
   (established, Stage 1B.2/1C/1D) exists without demonstrated Level 3
   value under this specific task and readout -- a real, reportable
   negative result, not a failure of the experiment.
4. **Unevolved encoding already helps, evolution does not add further
   value.** Condition 2 already beats raw pixels/the MLP baseline, but
   condition 3 does not meaningfully improve on condition 2. The useful
   contribution here would be the phase *representation* itself (the
   encoding step), not runtime dynamics specifically -- a real,
   mechanistically distinct finding from either "dynamics help" or
   "nothing helps."

## What this does not do

- **Does not attempt denoising** (Stage 2B) -- deferred until Stage 2A
  establishes whether the pipeline carries externally useful information
  at all. If Stage 2A is negative, that does not prove denoising must
  fail, but it would be a real reason for caution before building a
  larger diffusion-style stack.
- **Does not attempt image generation** -- compounds too many additional
  variables (noise schedule, reverse sampler, accumulated approximation
  error, perceptual evaluation) to be a sensible first Level 3 test.
- **Does not test role-matched intervention, other stochastic-control
  seeds beyond the one chosen, or classes' own topologies for their own
  images** (the class-conditioned scheme explicitly rejected above as
  circular for prediction).
- **Does not yet specify the exact cross-validated regularization grid,
  local-convergence hyperparameter defaults to inherit, or exact runtime/
  storage budget** -- these are implementation-level specifics to settle
  when this design is actually reviewed and locked, not fixed here.

## Status

First draft. Not yet reviewed, not yet locked, not yet implemented.
Following this project's established convention (Stage 1D's own path),
this should go through external review before any code is written
against it.
