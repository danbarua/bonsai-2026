# Bonsai — Oscillator Networks for Audio/Amp Modeling — Fresh Session Prompt

## Context (Bonsai, briefly)

Bonsai is a research playground exploring Kuramoto/Hopf oscillator networks
as an alternative to conventional neural computation — local, largely
unsupervised (Hebbian, no backprop) dynamics as the representation, with a
separate trained or untrained readout evaluated on top. A long prior session
built and rigorously tested oscillator-based encoders for MNIST digit
classification; see the attached `bonsai_complex_hopf_findings.md` and
`bonsai_full_session_wrapup.md` for the full account. **You don't need to
read either in full to start** — the relevant parts are pulled out below.

## What just happened, and why this new session exists

While exploring Hopf-oscillator networks for MNIST, we independently arrived
at a dynamical structure (`dz/dt = z(λ + iω − |z|²) + coupling + input`)
that turns out to closely match a real, ~15-year-old line of research:
**Hopf cochlea models** — the mammalian ear modeled as a bank of coupled
Hopf oscillators poised near a bifurcation, governed by (from the
literature, confirmed via direct search, not assumed):

```
dz/dt = (μ + i)·ω_ch·z − ω_ch·|z|²·z − ω_ch·F(t)
```

This literature attributes specific, well-documented phenomena to
Hopf-bifurcation dynamics that map directly onto **guitar amp/effect
modeling** (the actual target application, inspired by
github.com/cmajor-lang/GuitarLSTM, an existing LSTM-based tool that learns
an amp's transfer function from paired dry/wet recordings):

- **"Generation of combination tones"** (new frequency components from an
  input tone) ↔ harmonic distortion, the core phenomenon overdrive/tube
  amps produce.
- **"Compression of the dynamic range"** ↔ compressor/overdrive behavior —
  exactly what a Hopf oscillator's self-regulating amplitude (stable limit
  cycle) does by construction.
- **A driven oscillator responding to F(t)** ↔ the amp/effect processing an
  input signal.

## The goal for THIS session

**Not** to beat or replicate GuitarLSTM yet — that's premature and would
need real recorded amp data we don't have here. The scoped, honest first
step, in the same spirit as the MNIST work's "verify each mechanism in
isolation before composing" discipline:

**Does a driven Hopf-oscillator bank (with power-coupling, from
Bandyopadhyay et al. 2023 — see findings doc) reproduce basic, well-known
nonlinear audio phenomena — harmonic/combination-tone generation and
amplitude compression — on a SYNTHETIC test signal (e.g. a pure sine wave,
or two sine waves for combination-tone testing), before ever touching a
real amp recording?**

If that checks out, natural next steps (don't jump ahead of this one):
tuning per-oscillator characteristic frequencies (a 1D "cochlea" of
oscillators each resonant at a different frequency, not the 2D image grid
used for MNIST), then eventually fitting to real paired audio.

## Methodological discipline to carry over — please actually follow these, not just note them

This is the single most important thing to bring from the prior session.
Applying it caught real bugs and real wrong assumptions repeatedly:

1. **Verify each mechanism in isolation before composing it with anything
   else.** The MNIST work's core dynamics were verified against Hopf theory
   (amplitude → √λ, phase rotation rate → ω) *before* any spatial coupling
   was added. Do the same here: verify a single oscillator's response to a
   pure tone before building any kind of bank/array.
2. **"Does this mechanism actually do anything beyond a trivial baseline"
   is the standing question for any new piece.** Caught a real issue in
   MNIST (fixed coupling was ~equivalent to a trivial no-simulation
   transform) — the audio equivalent might be "does power-coupling produce
   genuinely different harmonic content than just clipping/saturating the
   input directly," and that's worth testing explicitly, not assuming.
3. **A hypothesis that fails a confounded test hasn't been tested.** The
   MNIST resonance work looked like a real signal on the first pass and
   turned out to be a topology confound, caught only by deliberately testing
   a mixed/control condition. Build the equivalent controls here before
   trusting a result.
4. **Small-sample or short-test results can reverse at scale — check
   before trusting.** Happened twice in the MNIST work (spike-train, and
   the Oja-trained NearestCentroid finding). If a synthetic-signal test
   looks promising, try a longer signal / different frequency / different
   amplitude before generalizing.
5. **Verify primary sources directly rather than trusting a summary.** The
   Bandyopadhyay paper's own follow-up work was checked directly (found via
   search, not assumed) and revealed an important nuance (Hebbian learning
   alone isn't sufficient for amplitude learning in their architecture) that
   a surface-level read would have missed. Do the same for any new audio/DSP
   claims that come up.
6. **Write real, checked-in tests alongside exploratory code**, not just
   scratch scripts — see `test_complex_hopf_field.py` for the pattern
   (verify against known theory with actual assertions, not narrated
   results).

## Files provided (tarball)

- `complex_hopf_field.py` — the core `ComplexLocalOscillatorField` class:
  intrinsic Hopf dynamics + local power-coupling + closed-loop input
  anchoring. **The Hopf math is directly reusable; the 2D image-grid
  topology (H×W neighbors) is NOT** — audio needs a different topology
  (most naturally: a 1D array of oscillators, each with its own
  characteristic frequency, all driven by the same scalar input signal
  F(t) — closer to the literal Hopf cochlea structure than to the MNIST
  spatial grid). Expect to restructure this, not use it as-is.
- `complex_hebbian_training.py` — Oja-normalized Hebbian training for the
  coupling weights, unsupervised, population-level. Same caveat: built
  around the 2D grid's shared vertical/horizontal weight structure, will
  need rethinking for a 1D or frequency-organized oscillator bank.
- `test_complex_hopf_field.py` — the verification suite; use as a template
  for how to structure tests here (isolate one mechanism per test class,
  assert against known theory, document *why* each tolerance/threshold is
  what it is).
- `bonsai_complex_hopf_findings.md`, `bonsai_full_session_wrapup.md` — full
  prior context, for reference, not required reading to start.

## What to bring from your end (Dan)

- Nothing else is strictly required to start the isolated-oscillator
  verification step (synthetic sine-wave inputs, no real audio needed yet).
- If/when this moves toward fitting a real amp: paired dry/wet `.wav`
  recordings (same format GuitarLSTM expects) would be the natural next
  input — not needed for this session's scoped goal.

## Suggested first concrete action

Build a single driven Hopf oscillator (or a small array with a few
different characteristic frequencies), drive it with a pure sine tone, and
check directly (plot + actual assertions, not just eyeballing) whether the
output spectrum shows the expected harmonic/combination-tone content and
amplitude-compression behavior — before building anything resembling a
full "cochlea" bank or touching real audio.
