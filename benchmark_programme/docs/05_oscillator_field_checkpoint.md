# Bonsai Oscillator Field — Session Checkpoint

## What was built

- `local_oscillator_field.py` -- local (4-neighbor) coupled Kuramoto field,
  closed-loop anchoring to input, partial-arc phase mapping, optional shared
  natural frequency (omega). ~240x faster per step than all-to-all at 28x28.
- `vector_oscillator_field.py` -- D=4 channel extension with cross-channel
  (omega-matrix) mixing, borrowed from oscillator_field_dynamics.ipynb / AKOrN.
- `hebbian_local_field.py` -- population-level Hebbian-adaptive local
  coupling (shared vertical/horizontal weights, no backprop).
- `spike_time_encoder.py` -- first-spike-time readout (Nadasdy-style),
  requires a deterministic spatial phase-gradient initialization (a real bug
  was found and fixed here -- see below).
- `spectral_coincidence_encoder.py` / `resonance_classifier.py` /
  `shape_resonance_encoder.py` -- temporal-coincidence graph (CTM-inspired)
  + GraphLaplacian spectral decomposition, several variants.

## Findings, in order

1. **Fixed coupling (scalar or vector-valued) does little beyond raw
   pixels.** Measured directly: oscillator-field encoding tracked raw-pixel
   nearest-centroid/KNN performance almost exactly on real MNIST. Root
   cause, confirmed: the model was initialized at (or very near) the input
   target, so local coupling had almost no room to do anything beyond a
   small amount of blur-like smoothing.
2. **Fixed cross-channel (omega) mixing between identical channels adds
   nothing** -- confirmed on real MNIST, vector model tracked scalar model
   within noise across all classifiers and sample sizes tested.
3. **First-spike-time readout: real, working signal, but underperforms raw
   pixels.** Local spatial coupling gives a consistent, growing-with-N
   benefit over no coupling (+3.5 to +6.5pp across two classifiers) -- a
   cleaner result than the phase-value readout's coupling benefit, which
   was small and classifier-specific. But absolute accuracy (0.45-0.70)
   stays well below raw pixels (0.65-0.85) at matched sample sizes.
   - A real bug was caught and fixed here: initializing every pixel at the
     literally identical phase (deterministic, but degenerate) gave the
     uncoupled control ZERO variance across pixels -- restoring a
     deterministic spatial phase gradient (dropped when porting from the
     original notebook) fixed it.
4. **Population-level Hebbian-adaptive local coupling finds a real, small,
   directionally-sensible signal** (digit '1', the most vertical shape,
   showed the largest vertical-vs-horizontal weight split; digit '7', more
   diagonal, showed almost none) but the effect is small (~2-3% of the
   coupling scale) and washes out almost entirely when classes are mixed
   together in one shared population statistic (confirms a real
   cross-class-cancellation effect, not just noise).
5. **Temporal-coincidence graph (CTM-inspired) + spectral eigenvalue
   features: real, above-chance signal from a genuinely different
   mechanism** (0.32-0.36 on a 3-class subset, chance=0.33 -- wait, this was
   ABOVE the 3-class chance rate reported at the time; full context in the
   test-by-test record). Untuned, and below raw-pixel baselines, but a
   working, non-trivial signal from an approach nothing else tonight used.
6. **Eigenvector-based resonance classification (per-class or idealized-
   template reference bases): decisively confounded, run to ground across
   three independent framings.** Comparing a test signal's reconstruction
   fit against different reference graphs is dominated by which reference
   graph has "nicer" (closed-loop/ring) topology, completely independent of
   what's being reconstructed -- confirmed via (a) class-specific vs mixed-
   class reference comparison, (b) robustness to 6x more reference data
   (ruling out small-sample noise), and (c) idealized geometric templates
   (ring/vertical/horizontal/diagonal) with two rounds of normalization
   (column z-score, row-centering) -- "ring" won for 12/12 test images in
   the final, most-normalized version, regardless of true digit. This is a
   genuine, interesting structural fact about graph spectral theory (closed-
   loop graphs are unconditionally better generic smooth-signal
   reconstructors than open/line graphs) -- not a classification feature,
   and not fixable by more data or the normalizations tried so far.

## Open, well-defined next steps

- Self-referential GFT (project a signal onto ITS OWN graph's eigenvectors,
  not someone else's) sidesteps the cross-graph confound entirely --
  natural next thing to try, in progress as of this checkpoint.
- A properly topology-blind fit metric (if one can be constructed) would be
  needed to salvage the reference-basis-comparison approach.
- Per-class Hebbian coupling could still be worth revisiting with much
  larger per-class populations (hundreds, not tens, of images) given the
  directionally-correct but weak signal found.
