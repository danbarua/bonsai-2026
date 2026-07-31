# Bonsai — Full Session Wrap-Up

## Part 1: Foundation work (Hebbian/Predictive Hebbian, verified against theory)

- Found and fixed a genuine coupling-sign bug present in three places
  (`models/hebbian/minimalist.py`, `HebbianKuramotoOperator`,
  `PredictiveHebbianOperator`'s within-layer term) — silently turned
  attractive Kuramoto coupling into repulsive, invisible at exact synchrony
  (`sin(0)=0` either way), only surfacing once real phase spread existed.
  Verified against Bronski et al. (2017)'s actual equations and stability
  theorem, not just informal thresholds (`tests/test_hebbian_kuramoto_bronski.py`).
- Fixed a `GraphLaplacian` normalization bug and built out its Bronski-
  stability-matrix machinery (`from_bronski_stability_matrix`,
  `is_bronski_stable`).
- Full test suite reorganization (Phase A: Hebbian, Phase B: Predictive) —
  retired duplicate test files (porting unique content first), fixed
  order-dependent flakiness, replaced arbitrary thresholds with evidence-
  based ones. Result: 150 passed, 17 skipped (AKOrN, correctly
  deprioritized), 0 failed, stable across repeated runs.
- Real classification numbers established for the existing models: Hebbian
  Kuramoto sits at chance under a windowed-average readout; Predictive
  Hebbian gets 83-98% clean / 60-83% under noise-occlusion from 3 examples
  per class — traced to a mechanistic reason (Hebbian's perturbation is an
  open-loop bias to phase *velocity*, drifting and seed-dependent;
  Predictive's sensory error is a closed-loop correction toward a fixed
  target phase, stable and reproducible).

## Part 2: MNIST baselines — the methodology lessons

- **Full 2x2 grid** (raw pixels / cos-sin encoding x untrained centroid /
  trained classifier): raw+trained (0.9261) beats cos-sin+trained (0.8605)
  by 6.6pp, consistently across 9 of 10 digit classes (digit '1' the one
  exact tie) — traced to a real mechanism: mapping intensity to a *full*
  `[0,2*pi]` rotation aliases the two most common pixel values (background,
  ink) to the identical point, destroying real information.
- **Sample-size sensitivity is real and large**: a 5k-sample comparison and
  a 60k-sample comparison can disagree about which encoding is better,
  because a classifier's own parameter-to-sample ratio changes what's
  measurable — Bonsai's actual regime (3-5 examples/class) is much closer
  to the data-starved end.
- Built a reusable, verified few-shot evaluation harness
  (`few_shot_harness.py`) and a classifier sweep — confirmed
  `NearestCentroid` isn't universally the right default (loses to KNN on
  raw pixels) but is the right one specifically for redundant/aliased
  representations, where KNN collapses to near-chance.

## Part 3: The new oscillator-field model — built, characterized, iterated

Local (not all-to-all) coupling, closed-loop anchoring, partial-arc phase
mapping — deliberately assembled from validated pieces (Predictive Hebbian's
sensory anchor, the field-dynamics notebook's local topology, our own
MNIST-baseline lesson about arc aliasing), not any one source ported
wholesale.

- Confirmed ~240x faster per step than all-to-all at 28x28 (0.2ms vs 48ms) —
  the scaling problem genuinely solved.
- Caught and corrected a real self-inflicted mistake: an early claim that
  deterministic initialization "resolved" multistability was actually
  tautological (the test accidentally overwrote the thing it was supposed
  to vary). Redone properly: real multistability exists for hard-edged
  synthetic patterns (a genuine unstable equilibrium at exactly `sin(pi)=0`
  phase differences), and resolves cleanly for realistic, anti-aliased
  images — verified, not assumed, on real MNIST digits.
- **Direct test of "does the dynamics do anything"**: found the fixed-
  coupling model, initialized near the answer, moved the state by only
  ~0.07-0.1 rad from a trivial no-simulation transform — mostly denoising,
  not novel computation. Confirmed via the exact real-MNIST numbers: fixed
  coupling tracked raw-pixel performance almost exactly, both for the
  scalar model and a vector-valued (D=4, AKOrN-inspired cross-channel
  mixing) extension.
- Read AKOrN and a Microsoft KoPE paper in full for grounding: both
  confirmed "learnable" coupling in this literature means backprop on a
  supervised/self-supervised loss, never a Hebbian rule — a real fork
  Bonsai deliberately didn't take, staying in the unsupervised/no-backprop
  regime throughout.
- Population-level Hebbian-adaptive local coupling (2 shared, direction-
  specific weights): found a real, small, directionally-sensible signal
  (digit '1' showed the most vertical-vs-horizontal split, '7' almost none)
  but too weak to move classification numbers, and confirmed a genuine
  cross-class-cancellation effect when classes are mixed in one shared
  statistic.

## Part 4: Readout experiments — five real variants tested, one clear winner

| Encoding | NC @ n=10 | KNN @ n=10 | Note |
|---|---|---|---|
| Raw pixels (baseline) | 0.716 | 0.732 | |
| Final-phase (fixed coupling) | ~raw pixels | ~raw pixels | mostly denoising |
| First-spike-time | 0.49 | 0.56 | real but underperforms |
| Bare eigenvalue spectrum | 0.36 | 0.36 | above chance, modest |
| Self-referential GFT amplitude | 0.22 | 0.16 | circularity hurt it |
| **Edge-residual (aligned+residual)** | **0.61** | **0.61** | **best oscillator result** |
| Spike-train cross-correlation | 0.57 | 0.55 | comparable-to-worse |

- **First-spike-time**: real signal, caught and fixed a genuine bug along
  the way (degenerate all-identical initialization gave zero variance;
  restoring a deterministic spatial phase gradient — dropped when porting
  from the source notebook — fixed it).
- **Spectral resonance (per-class or idealized-template reference graphs)**:
  decisively confounded, run to ground across three independent framings
  (raw scores, column-normalized, row-centered) — closed-loop/ring graph
  topology is *unconditionally* a better generic smooth-signal reconstructor
  than open/line topology, regardless of what's being reconstructed. A
  real, interesting structural fact about graph spectral theory; not a
  classification feature.
- **Edge-residual** (subtract the low-frequency reconstruction, keep the
  residual, then add the low-frequency part back in alongside it): the
  actual breakthrough of the session. Nearly doubled the bare-eigenvalue
  baseline. Diagnosed *why* the remaining gap to raw pixels persists: not
  the oscillator dynamics discarding information (ruled out directly —
  bypassing them to build the graph straight from pixels did much worse),
  partly the low-frequency discard being too aggressive (confirmed —
  keeping it recovered several points).
- **Spike-train cross-correlation**: tested properly (not just assumed) at
  matched, low-noise sample size — genuinely comparable at n=5, but
  measurably *worse* than simple first-spike-time at n=10. An honest
  negative result on this specific implementation, not a general verdict
  on richer temporal codes.

## What actually generalizes from tonight, beyond any one number

1. **A hypothesis that fails a confounded test hasn't been tested** — the
   resonance/spectral work only became trustworthy once tested against a
   deliberately mixed control and 6x more data, ruling out the two most
   obvious alternative explanations before accepting the real one.
2. **"Does the dynamics do anything beyond a trivial transform" is the
   right standing question for any new mechanism** — it caught the fixed-
   coupling ceiling, and an independent paper (KoPE) validated the same
   standard from a completely different direction.
3. **Closed-loop beats open-loop for input encoding** — confirmed
   independently at least three times (Predictive Hebbian's sensory error,
   the Colab notebook's `bias = c - z`, and implicitly in why fixed local
   coupling alone plateaus).
4. **Full-scale and few-shot regimes can give opposite answers** — always
   worth checking both when the two might disagree, not assuming small-
   scale results extrapolate.
5. **Subtracting out shared/generic structure, not just measuring resonance
   with it, was the move that actually worked** — the single biggest
   result of the night came from explicitly discarding what's common
   across images and keeping what's specific to one.

## Open, well-defined threads for next time

- Tune `edge_encoder`'s parameters further (sigma, k_coupling) now that
  cutoff_idx is settled near 22-25.
- Test aligned+residual at n=50+ to see if its gap-narrowing holds at scale.
- Revisit per-class Hebbian coupling with a much larger per-class
  population (hundreds of images, not tens).
- The spike-train result leaves open whether a *different* summary signal
  (inter-spike interval, rather than raw count) or different tolerance
  window would do better — not yet tested.
- `models/akorn/` was never checked for the same coupling-sign bug pattern
  found in the other two Bonsai models.
