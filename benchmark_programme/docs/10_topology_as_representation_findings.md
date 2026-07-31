# Topology-as-Representation: Findings

## The reframing that made this work

Every earlier attempt at using learned structure for classification tonight
treated topology as a **prior for dynamics** — build a coupling structure,
run a simulation through it, read out the resulting phase state. That
failed decisively when tested: applying digit-1's learned long-range
connections to a digit-0 image's dynamics roughly *halved* accuracy,
because forcing an inappropriate structural prior onto data it doesn't
belong to actively corrupts the representation.

The fix: **treat topology as the representation itself**, compared
directly, never wired into a running simulation with borrowed structure.
For a test image, compute its own per-image topology (the same population-
style pairwise correlation statistic, but from just one image), and
classify by how well it matches each class's reference topology — a direct
structural comparison, not a reconstruction-fit or dynamics-with-borrowed-
coupling.

## Result: this became the best classifier of the entire session

*(Note: the "200-image test set" throughout this section is the
repeatedly-tuned validation sample discussed in "Corrected headline
result" below, not a held-out test set. Figures here are accurate as
internal comparisons made during development; see the corrected section
for the defensible full-test-set number.)*

![Accuracy vs training sample size](accuracy_vs_training_size.png)

| Training images/class (reference topology) | Overall accuracy (10-class) | Digit 0 specifically |
|---|---|---|
| 20 | 68.0% | 35% |
| 100 | 72.5% | 55% |
| **200** | **82.5%** | **85%** |
| 500 | 81.0% | 85% |

**82.5% at 10 classes (chance=10%) is 8.25x chance — clearly ahead of
tonight's previous best result** (edge-residual, ~5.5-6.5x chance),
achieved with training sample sizes (20-500 images/class) that are tiny by
conventional ML standards.

**The trend plateaus, not indefinitely improves** — 500/class (81.0%) sits
very slightly *below* 200/class (82.5%), on a 200-image test set where a
~1.5pp difference is plausibly within noise. This is the same character of
result as the earlier calibration-sample-size plateau (which also
saturated around n=200, for a different part of the pipeline) — a real,
useful finding in its own right: more data helps substantially up to a
point, then stops mattering for this specific mechanism.

## Two distinct problems were found and fixed along the way

### 1. Raw match scores are not comparable across classes without calibration

Different classes' reference topologies have different intrinsic "generic
matchability" — e.g. digit 2's baseline match score (against ANY random
image) was unusually high and tight (mean 0.88, std 0.03), while digit 1's
was notably lower (mean 0.70). Comparing raw scores directly caused a
severe, confirmed bias (digits 2 and 4 over-predicted 2-3x their fair
share; digits 1, 6, 9 almost never predicted). Fixed by z-scoring each
class's raw score against its own baseline (mean/std measured from a
broad, mixed-class calibration population) before comparing across
classes -- overall accuracy went from 45.5% (raw scores) to 62.5% (n=50
calibration) to 68.0% (n=200 calibration, where it plateaued).

### 2. Reference topology coverage, confirmed as a real, distinct issue from class overlap

Digit 0 and digit 8 were both weak at small training sizes, but for
different, diagnosable reasons (confusion-matrix analysis): digit 0's
errors were concentrated on specific competitors (2 and 6), while digit 8's
were diffuse across many classes. Looking at individual misclassified
digit-0 images directly (not just the confusion matrix) revealed two
different failure modes hiding inside the same label: genuine close
competition (both classes score well, the wrong one edges ahead -- likely
real handwriting ambiguity) versus the reference topology failing to
recognize the test image as belonging to its own class at all (negative
self-match z-score). Scaling training data from 20 to 200 images/class
fixed the majority of the coverage-failure cases directly (confirmed on
specific tracked examples, not just the aggregate number) -- e.g. one
specific stubborn image's self-match z-score went from -1.615 (100
images) to -1.253 (200 images), still short of flipping positive, while
two others flipped cleanly to positive and were fixed.

![Stubborn zeros](stubborn_zeros.png)

Visual inspection of the remaining hardest cases suggests they represent
*different* deviations from the "typical" loop shape (notably narrow/
compressed, irregular/gapped, and tilted), not one shared failure pattern
-- consistent with needing specifically diverse training coverage, not
just more of the same random sample, to fully resolve.

## Visualizing the learned topologies directly

![Class topology visualization](class_topologies_visualization.png)

Every surviving connection across every class tested was positive
(same-phase), zero negative -- among ink pixels that survive a strict
correlation threshold, none are strongly anti-phase. Digit 0's and 6's
connection webs visibly concentrate around and across their loop
structures; digit 8's looked comparably sparser and less internally
coherent, consistent with its diffuse (not structured) confusion pattern.
Precise fine-grained interpretation (e.g. "this connects the top of the
loop to the bottom") wasn't reliably readable at this density -- a higher
threshold or an animated view of the pruning process would likely be
needed for that level of detail.

## Pruning threshold: a real effect at the extremes, a broad plateau in the middle

Swept from 0.7 to 0.99 (connection count drops steeply: ~35k at 0.7, ~1300-
2800 at 0.9, single digits at 0.99). Both loose (0.8: 58.5%) and tight
(0.95: 66.0%) thresholds clearly hurt accuracy relative to the middle.
Refined sweep (0.87-0.93) showed a broad, relatively flat plateau (68-70%)
rather than one sharp peak -- our original, unexamined choice of 0.9 landed
inside that plateau, reasonably close to optimal, though not provably the
single best value.

## Overlap between class topologies doesn't cleanly explain confusion patterns

Raw connection-count overlap between digit 0's topology and each other
class's did NOT match the actual confusion pattern found in classification
(0's highest raw overlaps were with 4 and 5, not the 2 and 6 it actually
got confused with) -- meaning classification errors depend on *which
specific connections* overlap and how strongly they score for a given
test image, not just how many positions are shared. This was confirmed
directly by examining individual misclassified images' full z-score
profiles rather than relying on the aggregate overlap statistic.

## Reproducing these findings

Core files: `developmental_pruning.py` (population statistic
computation), `topology_matching_classifier.py` (per-image topology,
match scoring, calibrated classification), `learned_topology_encoder.py`
(the earlier, now-superseded dynamics-with-borrowed-topology approach --
kept as a negative-result reference, not for reuse).

```python
from developmental_pruning import population_developmental_stat
from topology_matching_classifier import (
    per_image_topology, compute_class_baselines, classify_by_normalized_topology_match
)

# Build reference topologies (population-level, unsupervised -- labels only
# organize which images inform which class, not used to supervise dynamics)
class_topologies = {}
for c in range(10):
    images = <200 images of class c, normalized to [0,1]>
    stat = population_developmental_stat(images)
    ink_mask = images.mean(axis=0).flatten() > 0.15
    pruned = np.where(np.abs(stat) > 0.9, stat, 0.0)
    pruned[np.outer(~ink_mask, ~ink_mask)] = 0.0  # exclude background-background pairs
    class_topologies[c] = pruned

# Calibrate against a broad, mixed population
baselines = compute_class_baselines(<~200 mixed unlabeled images>, class_topologies)

# Classify
predicted, z_scores = classify_by_normalized_topology_match(test_image, class_topologies, baselines)
```

## Corrected headline result (full 10,000-image test set, run once, frozen configuration)

**A code review correctly identified that the 200-image sample used
throughout this investigation had been reused for every design decision
(scoring function, threshold, top-K, hybrid head, feature count) and
therefore functioned as a validation set, not a held-out test set. The
94.0% figure below is real for that specific, repeatedly-tuned sample, but
is not the defensible MNIST accuracy claim.**

**Full, untouched 10,000-image MNIST test set, frozen configuration, run
once: 90.89% (95% CI: 90.33%-91.45%), 911 errors.** The 3.1pp gap between
this and the 200-image figure is a direct, measured demonstration of the
validation-set-reuse effect the review predicted -- not a separate bug,
the exact mechanism the review named.

### Per-class precision/recall (full test set)

| Digit | Precision | Recall |
|---|---|---|
| 0 | 0.937 | 0.924 |
| 1 | 0.926 | 0.971 |
| 2 | 0.891 | 0.880 |
| 3 | 0.862 | 0.871 |
| 4 | 0.922 | 0.939 |
| 5 | 0.908 | 0.922 |
| 6 | 0.946 | 0.951 |
| 7 | 0.931 | 0.874 |
| 8 | 0.884 | 0.865 |
| 9 | 0.883 | 0.891 |

Digit 8's errors remain diffuse (spread across every other digit, no
dominant competitor) at full scale -- the same qualitative pattern found
on the 200-image sample, holding up even though the headline number needed
correcting.

### Trivial baselines, identical 5,000-image training budget, identical full test set

The review asked directly: how much of the result comes from the learned
topology, versus the supervised head repairing a moderately informative
score vector? Answer, tested honestly rather than assumed:

| Method | Accuracy |
|---|---|
| PCA(50) + KNN(k=3) | 94.40% |
| Cosine KNN(k=3) | 94.21% |
| Raw-pixel 1-NN | 93.51% |
| Raw-pixel KNN(k=3) | 93.40% |
| **Topology matching, 20 features + hybrid head (this project)** | **90.89%** |
| Linear classifier on raw pixels | 89.59% |
| Topology, 10 simple-only features | 89.46% |
| Topology, 10 cosine-only features | 88.72% |
| Random projection(50) + KNN | 88.50% |
| Nearest centroid | 81.04% |

**Several trivial baselines beat this project's result outright, using the
identical training budget** -- raw-pixel KNN and PCA+KNN both exceed
90.89% by 2.5-3.5pp. This is an honest, important recalibration of the
project's significance: MNIST is a benchmark where the simplest possible
methods are already excellent, so "does this beat KNN" was probably never
the right test of whether something scientifically interesting happened
here. What remains true: a completely different computational paradigm
(local, unsupervised, no-backprop Hebbian-style dynamics feeding a shallow
supervised readout) reached a real, mechanistically-understood,
well-above-chance result. What is no longer claimed: that this is an
efficient or advantageous way to solve MNIST specifically.

## Data accounting ledger

| Role | Images | Labels used? | Used to choose hyperparameters? |
|---|---|---|---|
| Topology learning (reference topologies) | 2,000 (200/class, train indices [0:200]) | Yes, to partition images by class | Indirectly -- threshold=0.9 was tuned against the 200-image dev-validation sample |
| Calibration (z-score baselines) | 200 (unstratified random draw from the full 60,000-image train pool, seed=555) | No | Yes -- baselines directly determine every z-score feature |
| Hybrid-head fitting | 3,000 (300/class, train indices [200:500], disjoint from topology range) | Yes | Yes -- used to fit the LogisticRegression head |
| Development validation ("200-image test") | 200 (stratified sample, MNIST TEST split, seed=999) | Yes | Yes -- reused for every design decision throughout this investigation; functions as a validation set, not a test set |
| Final test | 10,000 (full MNIST TEST split) | Yes, evaluation only | No -- frozen configuration, evaluated exactly once |

**The "5,000 total training images" claim is accurate for the two labeled,
directly-fit components (topology + hybrid head) and should be stated with
that scope explicit.** It does not include the 200 calibration images,
which are a separate pool drawn from the same 60,000-image train split
without deliberate deduplication against the topology-construction range
-- expected overlap by chance is ~6-7 images (200 x 2000/60000), not
controlled for. A fully precise claim: **5,000 labeled training images
(2,000 for topology, 3,000 for the classifier head) plus 200 additional
images for unsupervised calibration, drawn from the same pool with minor,
uncharacterized overlap.**

## How the classifier head is actually trained (precision requested by review)

`sklearn.linear_model.LogisticRegression`, default solver = **lbfgs, a
gradient-based quasi-Newton method.** This means:
- **"No backpropagation through the representation/dynamics"** -- true.
  Backpropagation specifically refers to chain-rule gradient computation
  through a multi-layer differentiable graph; nothing here does that to
  the oscillator dynamics themselves.
- **"No gradient descent anywhere in the pipeline"** -- **false**, and this
  framing should not be used going forward. The classifier head is fit via
  gradient-based optimization. The narrower, correct claim: unsupervised
  local Hebbian dynamics feed a supervised, gradient-trained, but shallow
  and non-end-to-end linear readout.



### Sequence of improvements found on the 200-image validation sample

The table below reflects the sequence of real, properly-tested
improvements found *during development*, all measured against the
200-image validation sample -- useful for understanding what each change
contributed, but see the corrected full-test-set section above for the
defensible accuracy number. Each step was tested honestly against
alternatives (not assumed), which is what makes the sequence itself
worth keeping on record even though the absolute numbers below are
validation-sample figures, not test-set figures:

| Change | Accuracy on 200-image validation sample |
|---|---|
| Topology matching, simple scoring, argmax (200/class reference) | 82.5% |
| + cosine similarity scoring (untested variant, suggested in review) | 87.5% |
| + top-K adaptive pruning instead of fixed threshold | 86.0% (does not beat threshold-based pruning) |
| + hybrid classifier head, 10 features (cosine only, 300/class training) | 93.0% |
| + hybrid classifier head, 20 features (simple + cosine, 300/class training) | 94.0% |

**Top-K pruning was a legitimate, well-motivated hypothesis** (different
classes have different intrinsic score distributions, confirmed via the
calibration baselines -- a fixed absolute threshold seemed like it should
be unfair across classes) **that turned out not to matter** -- tested
properly across a full parameter sweep (k=100 to 2000) and both scoring
functions, its best result (86.0%) never beat simple threshold-based
pruning (87.5%) paired with cosine scoring on the validation sample. A
real, useful negative result, not a wasted effort -- it rules out
cross-class scoring asymmetry as the dominant remaining limitation.

**The hybrid classifier head very nearly gave a false negative on the
validation sample.** First tested with only 50 held-out training
images/class: 87.0%, statistically indistinguishable from (very slightly
below) argmax's 87.5% -- on its own, that result would have suggested the
z-score vectors don't carry additional structure beyond simple argmax.
Scaled to 300 held-out images/class (genuinely separate from the 200/class
used to build the reference topologies -- no data leakage): 93.0% on the
validation sample, a decisive jump. The lesson generalizes past this one
experiment: a negative result from an under-powered test isn't evidence
the underlying idea is wrong, it's evidence the test needs to be re-run at
adequate scale before either conclusion is trusted -- the exact same
lesson the spike-train and Oja-trained-coupling comparisons taught earlier
in this project, now confirmed a third time. (This lesson about
under-powered tests is independent of, and unaffected by, the separate
validation-sample-reuse issue that required correcting the headline number
-- both are real, both matter, and neither invalidates the other.)

**On richer features (simple+cosine concatenated)**: a further gain on the
validation sample (93.0% to 94.0%), confirming the two scoring modes carry
genuinely complementary information. This also holds on the full test
set -- the corrected section above shows 20-feature (90.89%) beating both
10-feature variants (89.46% simple-only, 88.72% cosine-only) -- so this
particular finding survived the transition from validation sample to full
test set, even though the absolute numbers moved.

## Open items

- Whether a hybrid head trained on features from multiple pruning
  thresholds simultaneously (e.g. concatenating scores from topologies
  built at 0.85/0.90/0.95) could push past 90.89% on the full test set --
  the simple+cosine concatenation half of this idea has now been tested
  (confirmed real, +1.4-2.2pp on the full test set: 89.46%/88.72%
  single-mode vs 90.89% combined); the multi-threshold half remains
  untested.
- Whether 300/class is a genuine ceiling for the hybrid head on the full
  test set, or an artifact specific to the 200-image validation sample --
  the plateau was only characterized against the validation sample and
  should be re-checked now that the validation-sample-reuse issue is
  understood.
- The remaining hardest digit-0 case (identified via validation-sample
  inspection, not touched afterward) -- to be investigated only now that
  the frozen full-test-set run is complete, per the review's explicit
  caution against inspection-driven fixes before freezing.
- The full ablation suite the review proposed (random topologies with
  matched sparsity, shuffled class assignment, topology from mixed
  classes, topology without background exclusion, degree-preserving
  rewiring) -- not yet run. Would isolate how much of the 90.89% comes
  from the specific learned edge structure versus generic properties
  (sparsity, spatial locality) that a random topology of the same shape
  might reproduce.
- Transfer to Fashion-MNIST or KMNIST with a frozen architecture, to test
  whether this generalizes as a representation principle or is exploiting
  digit-specific regularities.

