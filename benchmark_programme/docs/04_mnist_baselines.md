# Bonsai MNIST — Baseline Investigation (Stages 0, 0.25, 0.5, 0.75)

## Why this exists

Before evaluating any oscillator-based encoding against MNIST, we needed to
know what bar it has to clear, and — more importantly — **what question we
were actually asking**. "Does an oscillator encoding help" turns out to
depend entirely on what it's paired with downstream (an untrained
nearest-centroid readout, matching Bonsai's own methodology, vs. a trained
classifier), and conflating those two questions was the single biggest
source of confusion in this investigation. This doc lays out the full 2x2
grid, what each cell means, and the one clean, mechanistic finding that came
out of it.

## The full grid (real MNIST, 60k train / 10k test) — COMPLETE

| | Untrained (nearest-centroid) | Trained (LogisticRegression) |
|---|---|---|
| **Raw pixels** | 0.8203 | 0.9261 |
| **cos/sin phase encoding** | 0.7758 | 0.8605 |

Where "cos/sin phase encoding" is: pixel intensity `p in [0,1]` -> phase
`p * 2*pi` -> feature `[cos(phase), sin(phase)]`, concatenated across all
784 pixels to a 1568-dim vector. This reproduces the `direct_encode`
ablation from two uploaded Colab notebooks exploring Nadasdy-style
phase-field encoding for MNIST (`augmented_phase_encoder.ipynb` and a
second, differently-configured notebook informally called
"...TensorFlow.ipynb" despite actually being PyTorch code).

**Raw pixels beat the cos/sin encoding in both regimes** -- 0.8203 vs 0.7758
untrained (a 4.45pp gap), 0.9261 vs 0.8605 trained (a 6.56pp gap). The
encoding is a genuine, consistent cost, not just an artifact of one
methodology.

## What each cell means, and why we needed all four

- **Raw + untrained (0.8203)**: the essential control. Any oscillator-based
  encoding paired with an untrained, few-shot-style readout (Bonsai's actual
  methodology) needs to beat this to show the *dynamics themselves* are
  adding separability over doing nothing.
- **cos/sin + trained (0.8605)**: reproduces the notebooks' own ablation
  methodology (pixel encoding + a classifier that gets to learn from
  labels). Taken alone, this number is genuinely ambiguous -- it beats
  0.8203, which could look like "the encoding helps," but that comparison
  changes two things at once (encoding *and* classifier).
- **Raw + trained (0.9261)**: the missing control that isolates the
  encoding's effect from the classifier's. Same classifier as the cell
  above, only the input representation differs.
- **cos/sin + untrained (0.7758)**: confirms the prediction directionally --
  worse than raw+untrained (0.8203), a 4.45pp gap -- but the gap is actually
  *smaller* than the trained comparison's 6.56pp, which is the opposite of
  what was predicted going in (reasoning at the time: an untrained method
  should have *less* ability to compensate for the aliasing than a trained
  one, so the gap should be larger, not smaller). Best available explanation,
  held loosely rather than verified: the untrained centroid method has a
  lower performance ceiling overall regardless of encoding, so it has
  proportionally less to lose from any given information deficit than a
  trained classifier that's actually good enough to exploit fine intensity
  gradations when they're present.

### Digit 1 replicates across both comparisons -- this is the strongest single piece of evidence for the mechanism

| digit | raw+untrained | cos/sin+untrained | raw wins by |
|---|---|---|---|
| 0 | 0.8959 | 0.8490 | +0.0469 |
| 1 | 0.9621 | 0.9630 | **-0.0009** |
| 2 | 0.7568 | 0.7490 | +0.0078 |
| 3 | 0.8059 | 0.7396 | +0.0663 |
| 4 | 0.8259 | 0.7322 | +0.0937 |
| 5 | 0.6861 | 0.5740 | +0.1121 |
| 6 | 0.8633 | 0.8319 | +0.0314 |
| 7 | 0.8327 | 0.8210 | +0.0117 |
| 8 | 0.7372 | 0.6961 | +0.0411 |
| 9 | 0.8067 | 0.7562 | +0.0505 |

Digit 1 was an exact tie (+0.0000) in the trained comparison and is an
essentially exact tie again here (-0.0009, technically reversed, well
within noise) -- the same digit, the same near-zero effect, across two
completely different classifiers. That's real cross-validation of "aliasing
destroys fine intensity information, not spatial location information, and
a near-binary thin-stroke shape has little of the former to lose" -- not a
fluke of one method.

## The actual finding: cos/sin encoding costs ~6.6 points, and we know why

Comparing raw+trained (0.9261) against cos/sin+trained (0.8605) -- the one
clean, single-variable isolation in this grid -- **raw pixels beat the
cos/sin encoding in every single digit class except one exact tie**:

| digit | raw + trained | cos/sin + trained | raw wins by |
|---|---|---|---|
| 0 | 0.9776 | 0.9184 | +0.0592 |
| 1 | 0.9780 | 0.9780 | +0.0000 |
| 2 | 0.9012 | 0.8178 | +0.0834 |
| 3 | 0.9149 | 0.8505 | +0.0644 |
| 4 | 0.9379 | 0.8381 | +0.0998 |
| 5 | 0.8711 | 0.7691 | +0.1020 |
| 6 | 0.9530 | 0.9154 | +0.0376 |
| 7 | 0.9241 | 0.8872 | +0.0369 |
| 8 | 0.8768 | 0.7864 | +0.0904 |
| 9 | 0.9158 | 0.8206 | +0.0952 |

**Mechanism**: mapping pixel intensity to a *full* `[0, 2*pi]` rotation means
`p=0` and `p=1` map to the identical point on the unit circle
(`cos(0)=cos(2*pi)=1`, `sin(0)=sin(2*pi)=0`). MNIST pixel intensities are
strongly bimodal -- mostly exactly 0 (background) or near 1 (ink) -- so this
wraparound collapses the two most common, most informative pixel values
together, destroying exactly the simple linear "how much ink is here"
signal a classifier reading raw intensity gets for free.

**Digit 1 being the one exact tie is a nice confirming detail, not a
coincidence**: it's essentially a thin vertical stroke -- almost pure
presence/absence of ink at each position, with very little continuous
intensity variation to lose. The aliasing specifically destroys *fine
intensity* information, not *spatial location* information, so a
near-binary shape has little to lose from it. The digits with the biggest
deltas (4, 5, 8, 9) are exactly the ones with more curves and antialiased
edges -- more continuous intensity structure at stake.

### A note on how we got here (worth being honest about)

The first pass at this comparison (Stage 0's untrained+raw vs. Stage 0.5's
trained+cos/sin) showed a per-class improvement pattern that *didn't* match
this mechanism -- the classes that improved most were simply the ones
Stage 0 already found hardest (5, 2, 8), which looked more like "trained
classifier rescues confusable classes" than an encoding-specific effect. We
initially retracted the aliasing hypothesis on that basis. That retraction
was itself premature: that comparison changed two variables at once
(classifier *and* encoding), and the confound was masking the real signal.
Once Stage 0.75 gave us a same-classifier comparison, the mechanism shows up
cleanly. Lesson: a hypothesis that fails a confounded test hasn't actually
been tested -- isolate one variable at a time before concluding anything.

## Sample-size sensitivity (why Stage 0.5 alone was also misleading)

A LogisticRegression on 1568 features has ~15,690 parameters. At 5,000
training samples, that's 0.32 samples per parameter -- a badly
underdetermined fit. At 60,000, it's 3.8 -- thin by ML standards but
workable, especially with sklearn's default L2 regularization. This is why
an earlier matched 5k-vs-5k comparison showed cos/sin+trained *losing* to
raw+untrained (0.7708 vs 0.8149), while the full-60k comparison shows a
completely different picture. Both numbers are real; they're measuring
different regimes. **This matters a lot for Bonsai specifically**, since its
own methodology (3-5 examples per template) sits much closer to the
data-starved 5k-regime than the well-powered 60k one.

## What this means going forward

1. **Any future oscillator-based pixel encoding should map intensity to a
   partial arc (e.g. `[0, pi]`), not a full `[0, 2*pi]` rotation.** A full
   rotation aliases the two most common, most informative pixel values
   together for free, at no benefit -- a straightforwardly avoidable design
   mistake now that we know to look for it.
2. **Full-scale (60k) comparisons and few-shot comparisons can disagree, and
   Bonsai's real operating regime is the few-shot one.** Any oscillator
   encoding needs to be evaluated at small training sizes (5-50 examples per
   class), not just at scale, to say anything meaningful about Bonsai's
   actual use case.
3. **Isolate one variable at a time.** Two of the four grid cells were
   individually ambiguous or actively misleading until the full grid existed
   to compare against. The lesson generalizes to whatever oscillator model
   comes next: always have the matched, single-variable-changed control
   before drawing a conclusion from a single number.

## Files

- `mnist_loader.py` -- pure-NumPy IDX file parser.
- `stage0_raw_pixel_baseline.py` -- raw pixels + untrained nearest-centroid (0.8203).
- `stage0_25_cossin_untrained_baseline.py` -- cos/sin encoding + untrained
  nearest-centroid (0.7758).
- `stage0_5_direct_encoding_baseline.py` -- cos/sin encoding + trained
  LogisticRegression (0.8605).
- `stage0_75_raw_pixel_trained_baseline.py` -- raw pixels + trained
  LogisticRegression (0.9261).

## Open items

- Build a few-shot (5/10/50 examples/class) evaluation harness, encoding-
  and classifier-agnostic, since that's the regime that actually matters for
  Bonsai's methodology -- in progress.
- Scaffold the actual new oscillator-field model (local coupling, NumPy,
  closed-loop anchoring, partial-arc phase mapping) once the harness exists
  to evaluate it properly.
