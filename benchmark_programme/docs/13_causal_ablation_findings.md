# Causal Ablation Study: What Is the Learned Topology Actually Doing?

*(Revised following review feedback that narrowed several overstated
claims to exactly what the interventions establish. Changes from the
previous version are substantive, not cosmetic -- several conclusions
below are meaningfully more bounded than in the prior draft.)*

## Context and motivation

Following full-test-set verification (90.89%, 95% CI 90.33%-91.45%,
correcting an earlier 94.0% figure measured against a repeatedly-tuned
200-image validation sample), the next question was identified as causal,
not another accuracy number: *what part of the topology construction is
causally responsible for the class information the classifier uses?* Six
controls were run to separate the contribution of oscillator dynamics,
specific pairwise edge arrangement, node-level degree profile, class-
conditioned aggregation, and generic dimensionality reduction.

## The strongest supported finding

Within Bonsai's fixed topology-scoring architecture, high classification
performance depends jointly on oscillator-evolved states, class-
conditioned population aggregation, and the specific learned assignment of
connections between pixel pairs. None of the tested alternative
constructions reproduced the learned topology's performance -- graph density alone,
marginal edge-weight distribution alone, node degree sequence alone, a
generic 20-dimensional projection, and class-agnostic population topology
were all tested directly and all fell far short.

## Methodology, precisely stated

**The no-dynamics ablation used full downstream retraining, not a reused
head.** This matters and needs to be explicit: calibration statistics and
the logistic-regression head were refitted from scratch using the no-
dynamics features (new population statistics, new topologies, new
calibration baselines from the same 200 calibration images rescored
against the no-dynamics topologies, new hybrid-head training data
rescored the same way, and a freshly-fit `LogisticRegression`). This
measures the best performance obtainable under the same downstream
protocol, not compatibility with a head trained on converged-phase
features -- the 20.87-point decline is not confounded by feeding a shifted
feature distribution into an incompatible, stale head.

**Data accounting, by ablation:**
- *Matched-sparsity random topology* and *degree-preserving rewiring*:
  both are built by transforming the frozen reference topologies (which
  themselves derive from 200 images/class = 2,000 images), reusing their
  edge values and ink-mask candidate pools. The downstream classifier head
  was fit fresh on a separate 500-image set (50/class, indices [500:550],
  never used elsewhere) and evaluated on a fresh, stratified 200-image
  test subset (seed=31415, also never used elsewhere).
- *Mixed-class topology*: built entirely independently of the frozen
  topologies. Ten topology "slots" were each constructed from 200 images
  (20/class), but the ten slots were independently sampled (not identical,
  not disjoint) from a shared pool of 200 images/class [indices 700:900].
  Verified directly: **1,297 distinct images were touched in total across
  all ten slots and all ten classes** (out of a possible 2,000 with no
  overlap, or 200 with complete overlap) -- roughly 62-70% of each
  class's 200-image pool was used at least once. Each individual slot's
  topology used exactly 200 images, matching the frozen configuration's
  budget per class.
- *Calibration*: the same 200-image calibration pool (seed=555) was reused
  as raw images across every ablation; the baseline statistics computed
  from those images were recomputed fresh for each ablation's specific
  topology, since baselines depend on which topology is being scored
  against.
- *Random-feature control*: no topology machinery at all -- 20 random
  linear projections of the raw pixel vector, same fresh 500-train/200-
  test images as the topology ablations.

**Statistical treatment of the cross-seed standard deviations.** These
measure control-construction variation only, on one fixed 200-image
sample -- not population-level uncertainty. At n=200, ordinary sampling
uncertainty (several percentage points; see the earlier full-test-set
correction, where a 3.1-point gap proved statistically indistinguishable
from noise) is far larger than any of the reported cross-seed spreads. The
correct reading of, e.g., "75.8% +/- 0.85%" is: *matched-sparsity controls
scored 75.0%, 75.5%, and 77.0% on the same fixed 200-image evaluation
subset; the 0.85-point spread quantifies variation from which random
control was drawn, not uncertainty about the wider MNIST population.* A
paired analysis (McNemar's test or paired bootstrap over the shared 200
test images) would be more informative than comparing aggregate
accuracies alone, and has not yet been done.

## Results

| Configuration | Accuracy on its evaluation sample | Scale |
|---|---|---|
| **Real, learned topology** | **89.0%** | fresh 500-train/200-test sample |
| Matched-sparsity random topology | 75.0%, 75.5%, 77.0% (3 seeds, same fixed sample) | fresh 500-train/200-test sample |
| Random-feature control (20 random projections) | 58.5%, 60.0%, 64.0% (3 seeds, same fixed sample) | fresh 500-train/200-test sample |
| Degree-preserving rewired topology | 49.5%, 50.5%, 51.0% (3 seeds, same fixed sample) | fresh 500-train/200-test sample |
| Mixed-class topology | 47.0% (single run) | fresh 500-train/200-test sample |
| | | |
| With oscillator dynamics, full retraining | 90.89% (95% CI 90.33%-91.45%) | full 10,000-image test set |
| Without dynamics, full retraining | 70.02% | full 10,000-image test set |

Chance level for all rows: 10%. An evaluation-time class-label permutation
check (8 random derangements) collapsed to 9.96% +/- 2.39%, as expected.

## What each intervention establishes -- narrowed to exactly what was tested

**No-dynamics (initial encoded phase vs. converged phase): the largest,
most decisive effect, -20.87 points on the full test set.** A useful side
observation, not yet mechanistically explained: without dynamics, per-
class connection counts became wildly inconsistent (10-64 for most
digits, 3,408 for digit 1) versus a much more even 836-1,450 range with
dynamics. **What remains unestablished is *why*** -- candidate mechanisms
include within-class variance compression, nonlinear denoising, spatial
consensus formation, attractor-like canonicalization, improved threshold
conditioning, or genuine higher-order relational organization. The study
establishes the total contribution of the dynamics, not its internal
decomposition.

**Matched-sparsity random topology: -13.2 points, tightly reproducible.**
Rules out edge count, candidate-pool restriction, and marginal edge-value
distribution as sufficient on their own.

**Degree-preserving rewiring: -38.7 points.** The precise, bounded claim:
*preserving the exact degree sequence while destroying learned pairings
does not preserve useful performance; degree structure is not
independently sufficient, and whatever information it contributes appears
tightly coupled to relational edge placement.* This does **not** establish that node participation patterns
contribute nothing in the intact topology -- degree may still interact
with edge identity, spatial arrangement, and weights in ways this
ablation cannot isolate.

**Mixed-class topology: -42.0 points, but the least secure result in this
study.** A single run, following correction of an implementation bug
(the first attempt used identical image indices for every slot, producing
ten literally identical topologies -- corrected to genuinely independent
per-slot sampling, verified above at 1,297 distinct images used). The
defensible statement: *a correctly implemented mixed-class topology
control performed substantially worse in one run, providing strong
preliminary evidence that class-conditioned population aggregation
matters.* It should not carry the same evidential weight as the three-seed
randomization and rewiring controls until replicated.

**Random-feature control: -28.2 points.** Establishes that a generic
20-dimensional projection does not match the learned representation.

**Class-label permutation: collapsed to chance as expected.** The bounded
claim: *correct classification depends on the intended alignment between
class-conditioned feature channels and output labels; no class-
independent shortcut survived evaluation-time derangement.* This is a
successful pipeline sanity check, not a universal leakage audit -- it does
not rule out every conceivable form of leakage, only alignment-dependent
shortcuts.

## Interpreting the control ordering -- offered as interpretation, not established fact

The ordering (learned 89.0% > matched-sparsity 75.8% > random-projection
60.8% > degree-preserving 50.3% > mixed-class 47.0%) is not monotonic in
how much superficial graph structure was preserved -- degree-preserving
rewiring and mixed-class topology, which retain more structure from the
real construction than a generic random projection does, both
underperformed that generic projection. This suggests the scoring system
is sensitive not merely to how much structure survives, but to whether
that structure remains internally coherent: matched-sparsity graphs may
act as relatively neutral, broadly-distributed probes, while degree-
preserving rewiring retains a highly nonuniform node-participation profile
but assigns those nodes inappropriate partners, potentially producing
systematically distorted (not merely uninformative) features; mixed-class
topologies may actively suppress the between-class differences the
scoring mechanism depends on. **This remains interpretation. The current
experiments do not identify the exact cause of the negative-control
ordering.** Additional targeted controls would be required to distinguish
between these candidate explanations.

## Scientific claim now supported

Bonsai demonstrates that a single-pass, local oscillator process can
transform image populations into class-conditioned relational topologies
containing substantial information useful for discrimination within this
pipeline. In a fixed topology-
matching pipeline, removing oscillator evolution reduced full-test MNIST
accuracy from 90.89% to 70.02%. On a smaller fresh evaluation,
randomizing the learned edge assignments while preserving graph size and
edge-value statistics reduced accuracy from 89.0% to 75.8%, while exact
degree-preserving rewiring reduced it to 50.3%. These controls show
performance depends on more than generic sparsity, weight distributions,
or node-degree profiles -- the particular dynamics-derived pairwise
structure matters. This is meaningful even though the complete system
remains below raw-pixel KNN and PCA+KNN on MNIST under an equal-image-
budget comparison.

## What has not been established

- That Bonsai is computationally cheaper end-to-end than conventional
  lightweight baselines.
- That the dynamics discover semantically meaningful structure rather than
  highly effective task-specific regularization.
- That the effect transfers beyond MNIST.
- That the precise oscillator equations are necessary, rather than a
  simpler nonlinear spatial process (untested alternatives include
  deterministic spatial smoothing, learned or fixed convolutional filters,
  raw correlation with adaptive thresholds, class-conditioned covariance
  features, graph features built without oscillator simulation, and random
  features followed by a nonlinear or wider head).
- That the learned topology beats strong, matched, class-conditioned
  statistical features built without any simulation at all.
- That the method remains effective when the downstream supervised head is
  removed.
- That the observed topology effects scale favorably with image size or
  class count.
- That the observed topology representations are stable across different
  random training-set draws -- the current work freezes one topology
  construction (one specific 200-images-per-class sample); whether
  rebuilding from a different 200 images per class produces comparable
  representation quality is untested, and is a distinct question from
  transfer to another dataset.

These are the frontier questions. They do not undermine the ablation
result.

## Overall assessment

Before this study, the project had shown that a topology-derived
representation could classify MNIST, but had not identified whether the
result came from oscillator dynamics, graph density, class partitioning,
node participation, or a generic low-dimensional head. After this study:
useful performance arises from an interaction between oscillator
evolution, class-conditioned aggregation, and specific learned pairwise
connectivity. None of the tested marginal properties reproduces the intact
representation. The unresolved question is no longer whether the dynamics
or the learned topology matter -- they clearly do, within this pipeline --
but what computation the dynamics perform that a cheaper conventional
transformation cannot.

## Reproducing these results

Code: `nodynamics_ablation.py`, `matched_sparsity_ablation.py`,
`degree_preserving_rewiring.py`, and the mixed-class/random-projection
scripts. All reuse `class_topologies_200.pkl` (the frozen reference
topologies), the same threshold (0.9), background-exclusion rule, and
20-feature (simple + cosine) hybrid-head architecture as the frozen
configuration, with full retraining of calibration and the classifier
head for each ablation condition.
