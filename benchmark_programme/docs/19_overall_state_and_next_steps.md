# Bonsai — Overall State of Exploration & Next Steps

## Headline result (corrected after full-test-set verification)

A code review correctly identified that the 200-image sample used
throughout the topology-as-representation investigation had been reused
for every design decision and therefore functioned as a validation set,
not a held-out test set. Running the frozen configuration once against
the full, untouched 10,000-image MNIST test set gives the defensible
number: **90.89% accuracy (95% CI: 90.33%-91.45%), from 5,000 labeled
training images (2,000 for reference topologies, 3,000 for a shallow
classifier head) plus 200 additional images for unsupervised calibration.**
The 3.1pp gap from the earlier 94.0% figure is a direct, measured
demonstration of the validation-set-reuse effect, not a separate error.

**Equally important, and requested by the same review**: trivial
baselines (raw-pixel KNN, PCA+KNN) using the identical 5,000-image budget
and identical full test set **beat this result outright** (93.4-94.4% vs
90.89%). This means the project's significance can no longer be framed as
"an efficient way to solve MNIST" -- it wasn't, once checked honestly.
What remains true and worth keeping: a completely different computational
paradigm (local, unsupervised, no-backprop Hebbian-style dynamics feeding
a shallow, gradient-trained but non-end-to-end linear readout) reached a
real, mechanistically-understood, well-above-chance result from a
genuinely small amount of data. See `topology_as_representation_findings.md`
for the full baseline comparison table and data-accounting ledger.

**A further, separate precision**: the classifier head is fit via
`sklearn.LogisticRegression`'s default solver (lbfgs), which is
gradient-based. "No backpropagation through the representation" remains
true; "no gradient descent anywhere in the pipeline" does not, and
shouldn't be claimed going forward.


## The project's actual thesis, restated

Can genuinely useful representations emerge from local, unsupervised,
Hebbian-style physics — no backprop, no labels touching the dynamics
themselves — cheaply enough to avoid the expense of simulating a real
brain? Tonight's answer, across many independent threads, leans real yes,
with specific, characterized limits.

## Major threads and their current status

### 1. Foundation: Hebbian and Predictive Hebbian models (verified against theory)
A genuine coupling-sign bug was found and fixed in three places (silently
turned attractive coupling repulsive, invisible at exact synchrony). Fixed
and verified directly against Bronski et al. (2017)'s actual stability
theorem, not informal thresholds. Test suite reorganized (150 passed, 17
skipped, 0 failed, stable). Real classification numbers established:
Predictive Hebbian gets 83-98% on a small 7-character alphabet from 3
examples/class; the mechanism (closed-loop sensory correction vs.
Hebbian's open-loop drift) is understood, not just measured.
**Status: solid, validated foundation. Not revisited further tonight.**

### 2. MNIST baselines and methodology
Established real, mechanistic findings: full-2π phase encoding aliases
pixel intensities and costs ~6.6pp accuracy (understood why: background and
ink can map to the same point); sample-size regime changes which encoding
wins; NearestCentroid isn't universally better than KNN, it depends on
whether the encoding is redundant. Built the reusable few-shot evaluation
harness used throughout the rest of the night.
**Status: complete, reusable infrastructure.**

### 3. Local oscillator field (scalar Kuramoto-style)
Built local (not all-to-all) coupling, closed-loop anchoring, partial-arc
phase mapping — deliberately assembled from validated pieces, not any one
source ported wholesale. Confirmed ~240x speedup over all-to-all. Found and
understood a real ceiling: fixed coupling initialized near the input
barely moves the state from a trivial transform (mostly denoising, not
novel computation) — confirmed directly on real MNIST, both scalar and a
D=4 vector-valued (AKOrN-inspired) extension.
**Status: ceiling found and understood. Superseded by the topology-based
approach below for classification purposes, but the model itself remains
a validated building block.**

### 4. Readout experiments on the local field (five variants tested)
First-spike-time: real signal, underperforms raw pixels, caught and fixed
a genuine initialization bug along the way. Bare eigenvalue spectrum of a
temporal-coincidence graph: real, modest, above-chance signal. Self-
referential and cross-signal GFT: informative negative results, traced to
a specific circularity mechanism. Per-class/idealized-template spectral
resonance: decisively confounded across three independent framings
(closed-loop graph topology is unconditionally "nicer" to reconstruct
against, regardless of content) — a real, interesting fact about graph
spectral theory, not a usable classification feature. **Edge-residual**
(subtract the low-frequency graph reconstruction, keep both the residual
and the aligned component) was the standout: nearly doubled every other
spectral variant, and the *reason* the remaining gap to raw pixels
persisted was diagnosed, not just observed (ruled out the oscillator
dynamics as the culprit; confirmed the aggressive low-frequency discard
was partly to blame).
**Status: complete investigation, edge-residual was the best single-image
readout found, now superseded in absolute accuracy by the topology-
matching approach (thread 6) but methodologically important — it's the
origin of "subtract what's common, keep what's specific," which resurfaces
as the core idea behind topology-as-representation.**

### 5. Complex-valued Hopf oscillators with power-coupling
Followed a real paper (Bandyopadhyay et al. 2023, verified against the
primary source, not a summary) into complex-valued (not unit-norm)
oscillators, power-coupling, and Oja-normalized Hebbian learning for local
coupling weights. Built a full, checked-in test suite (10 tests) verifying
the dynamics against known Hopf theory before testing anything downstream.
Found and fully characterized a real coupling-strength vs. amplitude-
fidelity tradeoff. The Oja-trained coupling violated a "safe range"
heuristic and was tested anyway rather than assumed bad: real, scale-
dependent result (KNN improved and the gap widened with more data;
NearestCentroid's finding reversed at confident scale) — an important
lesson in itself about not trusting small-sample reads.
**Status: a real, working, but smaller-scale investigation than thread 6.
Not revisited after the audio pivot.**

### 6. Topology-as-representation (tonight's headline result)
Started from a real failure: applying one class's learned long-range
connections as a dynamics prior for a different class's image roughly
halved accuracy. The fix was a genuine reframing — compare topology
directly as a feature, never wire it into borrowed dynamics. Built
incrementally with real controls at every step (a mixed-class population
control that held up, unlike the earlier resonance-classifier confound),
found and fixed two distinct calibration problems (raw-score class bias;
reference-topology coverage gaps, confirmed down to individual tracked
test images), characterized three independent plateaus (calibration size,
reference training size, hybrid-head training size — all around a few
hundred images), and took a code review seriously enough to test its
suggestions properly rather than nod along, which is what produced the
final result: cosine scoring (real win), top-K pruning (real, honest
negative result), hybrid classifier head (near-false-negative at small
scale, real +5.5pp win at adequate scale), richer features (+1pp further).
**Status: this is where the frontier of the investigation currently sits.**

### 7. Audio / Hopf cochlea pivot
A verified, real scientific connection was found (not assumed): the Hopf
oscillator dynamics built for MNIST classification structurally match a
real ~15-year research line on modeling the mammalian cochlea. A scoped,
disciplined fresh-start prompt was built for a separate agent (PyCharm/
JetBrains AI) to pursue this independently, with an amended plan (no
coupling in v1, exact governing equation with correct ω-scaling, specific
falsifiable verification targets — the 1/3-power compression law, cubic
distortion products — rather than vague "looks nonlinear" checks).
**Status: handed off, running independently in a separate environment.
Not part of tonight's MNIST work from here on.**

## What generalizes across all seven threads — the actual transferable lessons

1. **A hypothesis that fails a confounded test hasn't been tested.** True
   for the resonance-classifier work (only became trustworthy once tested
   against a deliberately mixed control) and for the calibration-bias
   diagnosis (needed the confusion matrix and individual image inspection,
   not just aggregate accuracy).
2. **A negative result from an under-powered test isn't evidence the idea
   is wrong.** True at least three times tonight: the Oja-trained complex
   coupling (small-scale read gave the wrong sign for one classifier), the
   hybrid classifier head (50/class training looked like a null result,
   300/class gave +5.5pp), and implicitly in every plateau-vs-genuine-
   ceiling distinction made along the way.
3. **"Does this mechanism do anything beyond a trivial baseline" is the
   right standing question for any new piece**, and it's cheap to ask
   before building further on top of something unverified.
4. **Subtracting out shared/generic structure, not just measuring
   resonance with it, is where the exploitable signal actually lives** —
   found independently in edge-residual (thread 4) and echoed conceptually
   in the background-pixel exclusion that made topology-as-representation
   work at all (thread 6).
5. **Real prior science, checked at the primary source, is worth more than
   an appealing analogy** — the power-coupling paper's own follow-up work
   revealed a nuance (Hebbian-only learning insufficient for amplitude) a
   surface read would have missed; the Hopf-cochlea connection was
   confirmed via the actual governing equation, not just family resemblance.

## Next steps to explore, roughly prioritized

### Immediate, cheap, direct continuations of thread 6
- **Full-test-set verification: done.** 90.89% (95% CI: 90.33%-91.45%),
  confirmed against trivial baselines using the identical 5,000-image
  budget — several of which (raw-pixel KNN, PCA+KNN) beat this result
  outright. This changes the priority ordering below.
- **New top priority, given the baseline comparison**: the review's
  proposed ablation suite (random topologies with matched sparsity,
  shuffled class assignment, topology from mixed classes, degree-
  preserving rewiring) — needed to establish whether the *specific
  learned edge structure* is doing anything a random topology of the same
  shape and sparsity wouldn't, now that "beats simple baselines" is off
  the table as the source of significance.
- Concatenating scores from multiple pruning thresholds (0.85/0.90/0.95
  simultaneously) as additional hybrid-head features — the other half of
  the "richer features" idea; the simple+cosine half is done and confirmed
  on the full test set (89.46%/88.72% single-mode vs 90.89% combined).
- Resolving the one remaining stubborn digit-0 test case with deliberately
  diverse (not just more random) training coverage — now appropriate to
  investigate, since the frozen full-test-set run is complete.

### Medium-effort extensions
- Whether topology-as-representation transfers to the small 7-character
  alphabet (thread 1's original success case), to a genuinely different
  dataset (Fashion-MNIST or KMNIST, per the review's suggestion, with a
  frozen architecture), or to translated/rotated MNIST — all now higher
  priority than before, since the ablation suite above and these transfer
  tests together are what would establish whether something general was
  learned, independent of whether it happens to beat MNIST-specific
  baselines.
- Revisiting per-class Hebbian *coupling* (not just topology-as-feature)
  with the calibration and cosine-scoring lessons applied — the original
  version of this idea (thread 6's precursor) was abandoned after a
  dynamics-with-borrowed-topology failure, but that failure mode is now
  understood and might not apply to a properly reframed version.
- Extending the complex Hopf field (thread 5) with the topology-matching
  approach instead of dynamics-based classification, now that thread 6 has
  shown topology-as-feature outperforms topology-as-prior.

### Larger, separate efforts
- The audio/Hopf-cochlea thread (thread 7), progressing independently.
- Whether the "amino-acid-chain-folding" higher-order structure idea that
  originally motivated topology-as-representation has more to give beyond
  what's been extracted so far — tonight only tested one layer of
  emergent long-range structure, not genuinely hierarchical/multi-scale
  folding.
- A proper, larger-N field topology study for the vectorized/AVX
  oscillator field experiments referenced but not detailed tonight — a
  natural place where the topology-as-representation lessons could
  compound with genuine performance engineering.
