# Stage 2B: Feasibility Ladder Stage 1

**Status: mechanical validation only, per `DESIGN.md`'s own explicit
framing. This is NOT a scientific result and must not be read as one.**
Its job is to confirm the pipeline runs correctly end-to-end at n=1,000
before scaling up. The ridge/stats numbers in the final section are an
in-sample machinery smoke test, not the locked confirmatory design, and
are labeled as such throughout -- same discipline as Stage 2A's own
feasibility-stage FINDINGS.

## Scope

1,000 official KMNIST training images, drawn by the locked nested
stratified partition (`SEED=42`, `stage1_indices` a prefix of the
5,000-image stage-2 draw). Corrupted per the locked forward process
(`SHA256(split:index:42)` seeding, `alpha_bar=0.5`), against original
dataset indices -- verified directly, not assumed: recomputing
`epsilon_for` from each drawn image's ORIGINAL index and re-deriving
`forward_corrupt` reproduces the corpus bit-exact at three spot-checked
rows. Run entirely on a Colab A100 via `run_ladder_stage1.py`, which
fetches one pinned commit of this repository rather than being uploaded
with its dependencies, and every artifact through `ensure_artifact`, so
a dead session would resume having lost at most one step.

## Part 1 -- first run: an honest FAIL, as designed

The encoder-on-noisy-inputs gate's first execution on real, majority-
censored KMNIST (commit `7723b96`, `ENCODER_STEPS=150`) failed the
pre-registered threshold:

| quantity | clean | noisy |
|---|---|---|
| median final-Delta | 2.177485e-07 | 3.698480e-05 |
| p95 final-Delta | 9.971726e-07 | 1.784018e-04 |
| non-finite phases / deltas | 0 / 0 | 0 / 0 |

**rho = 169.851** against threshold 10 -- roughly 17x over. Zero
non-finite values anywhere: a clean ratio failure, not a numerical
blow-up. Per `DESIGN.md`'s locked stop-gate this halted the stage
immediately (`STAGE1_FAIL`, session torn down, nothing billing);
steps 5-10 never ran. Confirmed against the live bucket listing before
any further work: 11 objects existed under `stage2b/train/stage1/`,
none for `theta_T`, `features`, `ridge_cv`, `ridge_final`, or
`stats_smoke` -- the claim that nothing downstream had run was checked,
not assumed.

## Part 2 -- diagnosis (not part of the locked pipeline)

Investigated in `diagnose_encoder_gate_failure.py`, run entirely on CPU
(`_local_converged_phases` has no JAX/GPU dependency; this bills
nothing). It regenerates the exact stage-1 corpus and corruption locally
and verifies that reconstruction bit-for-bit against the failed run's
own reported identity-baseline MSE before trusting anything computed
from it -- confirmed exact, relative diff `0.000e+00`, and the
independently-recomputed steps=150 row reproduced the cloud run's median
deltas and rho to full reported precision.

**Measurement 1, convergence curve** (five step counts, full 784-grid):

| steps | median clean | p95 clean | median noisy | p95 noisy | rho |
|---|---|---|---|---|---|
| 75 | 8.663e-05 | 2.267e-04 | 1.298e-03 | 2.977e-03 | 14.98 |
| 150 | 2.177e-07 | 9.972e-07 | 3.698e-05 | 1.784e-04 | 169.9 |
| 300 | 1.538e-12 | 2.030e-11 | 2.945e-08 | 6.812e-07 | 1.915e4 |
| 600 | **0.0** | **0.0** | 1.776e-14 | 1.043e-11 | 17.76 |
| 1200 | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |

Noisy final-Delta decays geometrically all the way to exact float64
zero -- median AND p95, every one of 1,000 images -- by 1,200 steps, the
same fixed point clean reaches. No floor above a meaningful scale
exists; the encoder converges on censored inputs, it is simply slower
to.

**A second, independent defect is visible in the same table.** The rho
column is non-monotone (14.98, 169.9, 1.915e4, 17.76, 0.0) because clean
and noisy cross their own float64 floors at different step counts. At
steps=600, clean's median had already hit exact 0.0 while noisy's sat at
1.776e-14 -- nine orders below the smallest meaningful measured value
(2.177e-07) -- yet the gate reported **FAIL at rho=17.76**, because
`max(0.0, 1e-15)` silently turned a ratio gate into an absolute test
against the 1e-15 floor. A ratio between two quantities that have each
decayed to numerical dust measures which one underflowed first, not
whether the mechanism converged.

**Measurement 2, state drift vs. between-image scale** (noisy, full
784-grid, same reduction on both sides -- max absolute wrapped
difference):

- Drift, 150->600 steps, n=1000: median **7.573e-04**, p95 4.702e-03
- Between-image scale at 150 steps, n=5000 pairs: median **1.806**, p95
  2.146
- **Ratio: 0.0004**

The phase field has, for all practical purposes, stopped moving relative
to the scale that distinguishes one image from another -- by a factor of
roughly 2,400 -- long before the Delta metric says so. Independent
corroboration of Measurement 1's reading: genuine, if slow, convergence,
not a qualitatively different regime on noisy inputs.

## Part 3 -- disclosed post-lock amendment

Full text: `DESIGN.md`'s "Encoder-on-noisy-inputs gate" section and
Review History. Two changes, both required by the diagnosis above:

1. **`ENCODER_STEPS` raised 150 -> 1200**, uniformly (every encoding
   site, clean and noisy identically). Mirrors Stage 2A's own
   `max_iter` 1,000 -> 10,000 precedent: halt honestly, diagnose
   mechanism, amend with disclosure, re-verify.
2. **Gate formula gains an absolute-convergence escape**: PASS if
   `rho <= 10` OR both medians are already below `ABS_CONV_EPS=1e-12`
   (5+ orders below the smallest meaningful measured Delta, well above
   observed float64 dust). Non-finite auto-fail stays unconditional.

**The decision rule that selected S\*=1200 is stated as verdict-
invariant, not merely correct.** An earlier, looser reading of the rule
("some S\* brings noisy within 10x of clean-at-150") was caught before
being applied -- it selects S\*=300, which immediately fails its own
same-step re-run at rho=1.915e4. The corrected rule (same-step, both
series required to have genuinely converged) passes at S\*=1200 under
EITHER reading, so the correction did not select this outcome. No finer
scan between 600 and 1200 was run: 1200 is the only step count in the
five-point scan that passes robustly (exact zero, both medians and both
p95s), rather than sitting near the fragile crossover band a smaller,
untested value might land in.

A second, independent bug was found and fixed in the same investigation,
by tracing "every encoding site" through the actual call graph rather
than assuming the constant bump alone was sufficient: `_encode_one`'s
returned theta went through `stage2a_core.encode_and_restrict`, which
has no `steps` parameter and is hardwired to `_local_converged_phases`'s
bare default (150, Stage 2A's own unrelated convention, load-bearing for
~14 of its own already-verified pipeline files). Final-Delta reflected
the requested step count; the returned theta silently did not --
invisible while `ENCODER_STEPS` also happened to be 150, a real defect
the moment it stopped being. The same bug, independently, was in the
driver's own step-5 sanity check, which compared against the same
hardwired function. Both fixed at the source: `_local_converged_phases`
is now called directly, at the caller's own requested step count,
everywhere in Stage 2B's pipeline.

New `CLAUDE.md` principle (23): a ratio gate between two quantities that
each decay to a numerical floor measures which one floored first, not
the mechanism.

## Part 4 -- re-run: full pipeline completion

Commit `32b6688`. The gate now passes robustly -- not marginally:

```
encoder-on-noisy-inputs gate: PASS
  median final-Delta clean : 0.000000e+00
  median final-Delta noisy : 0.000000e+00
  rho                      : 0 (threshold 10)
  absolute convergence     : True (both medians < 1.0e-12)
  non-finite phases/deltas : 0/0, 0/0
```

Exact float64 zero on both sides, matching the diagnostic's own
steps=1200 measurement precisely. Because the gate passed, the driver
continued automatically through steps 5-10 in the same run, per the
amendment's own instruction -- the halt rule is satisfied by a passing
verdict, no separate authorization needed. **This is the first time any
Stage 2B code has run graph evolution, ridge fitting, or the statistics
machinery against real data of any kind.**

**Step 5 (restrict)**: the new, corrected sanity check passed on the
real run -- confirms the fix (comparing against a fresh encode at the
gate's own step count, not the stale hardwired-150 oracle) is correct in
production, not only in the isolated unit test that caught the bug.

**Step 6 (evolution)**: all four canonical graphs (`T`, `lattice`,
`rewired`, `curr_random`), batched JAX evolution -- **0 failed of 1,000
for every graph**, and the CPU reference cross-check on image 0
succeeded for all four (`RK45`, primary attempt, no recovery step
needed). 10.0s total.

**Step 7 (features)**: all seven ridge conditions built --
`pre_evolution`, `T`, `lattice`, `rewired`, `curr_random` (1008-dim
each), `raw_505` (505-dim), `raw_784` (784-dim). 4.4s.

**Step 8 (ridge)**: cross-validation and the real-data ridge equivalence
gate, for the first time ever on non-synthetic features (every prior
equivalence number in this project was synthetic). All seven conditions
passed **with enormous margin**:

| condition | selected alpha | max abs pred diff | tol | alpha agrees |
|---|---:|---:|---:|:---:|
| raw_505 | 1000 | 6.928e-14 | 1e-8 | yes |
| raw_784 | 1000 | 7.511e-13 | 1e-8 | yes |
| pre_evolution | 1000 | 1.811e-13 | 1e-8 | yes |
| T | 100 | 8.159e-13 | 1e-8 | yes |
| lattice | 10 | 6.465e-13 | 1e-8 | yes |
| rewired | 1000 | 1.151e-12 | 1e-8 | yes |
| curr_random | 1000 | 6.568e-13 | 1e-8 | yes |

Every difference is 4+ orders below the 1e-8 gate. The n-dependent
scaler-centering tolerance (`1e-9 * (n/1000)**0.5`, this project's most
recent post-lock amendment before this one) held with wide margin at
every condition -- `curr_random`, the condition it was specifically
raised for, sits at `margin_ratio=0.076` (its `||mean(X)||` at ~7.6% of
tolerance, close to the 12.7x margin the amendment's own derivation
predicted). Worth recording plainly, not smoothed over: the evolved and
pre-evolution conditions' fold condition numbers are extreme
(pre_evolution ~6e14, T/lattice ~1.2-1.6e14, rewired/curr_random
~2-7e13, versus raw_505/raw_784's 14-490) -- exactly the regime the
JAX-SVD ridge implementation was designed and stress-tested for, and the
equivalence numbers above show it holding under real, not merely
synthetic, ill-conditioning.

**Step 9 (stats smoke)**: the full statistics machinery (primary paired
bootstrap, both Holm families, the branched winner rule) exercised
end-to-end against real, in-sample ridge output for the first time --
the ridge-output-to-stats-input glue no test had crossed before this
run. Artifact's first line, verbatim:
`SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE,
NON-INFERENTIAL, NOT A RESULT`. Mean per-image clipped MSE, in-sample,
all conditions -- recorded because the design permits recording it, not
because n=1,000 in-sample numbers support any claim:

| condition | mean clipped MSE (in-sample) |
|---|---:|
| identity baseline | 0.1995 |
| raw_784 | 0.0397 |
| raw_505 | 0.0508 |
| pre_evolution | 0.0530 |
| lattice | 0.0562 |
| T | 0.0616 |
| curr_random | 0.0621 |
| rewired | 0.0709 |

Every learned/raw condition beats the identity baseline by a wide
margin, as expected of any fitted readout against "return the input
unchanged." No condition-vs-condition ordering here should be read as
evidence of anything -- in-sample MSE at n=1,000 with no held-out split
is not the locked confirmatory design (20,000-resample paired bootstrap
against the official 10,000-image test set), and DESIGN.md explicitly
scopes this step to exercising the machinery, not producing a result.

## Runtime

**596.7s (9.9 minutes) total**, measured end-to-end on the A100, not
projected:

| step | seconds |
|---|---:|
| bootstrap (clone + pip install) | 18.1 |
| stage_kmnist (download) | 0.7 |
| preflight | 0.0 |
| corpus | 0.1 |
| topologies | 0.1 |
| corruption | 0.3 |
| corruption diagnostics | 0.1 |
| **encoder gate** | **423.2** |
| restrict | 0.1 |
| evolution (4 graphs) | 10.0 |
| features | 4.4 |
| ridge (7 conditions, CV + equivalence) | 128.8 |
| stats smoke | 10.9 |

The encoder gate dominates at 71% of total wall clock -- expected, at
1200 iterations of a local update per image, two full encoder passes
per image (the `_encode_one` fix's own byproduct: down from three).
Every other step completed in seconds.

## Code and artifacts

`stage2b_encoder_gate.py` (the gate, amended), `run_ladder_stage1.py`
(the driver), `diagnose_encoder_gate_failure.py` (the investigation,
diagnostic-only), `stage_kmnist_inputs.py` (one-time input staging).
Tests: `tests/test_stage2b_encoder_gate.py`,
`tests/test_stage2b_ladder_stage1.py`. Every stage-1 artifact lives in
the public-read bucket `bonsai-2026-stage2b-cache` under
`stage2b/train/stage1/`; the pre-amendment `encoder_gate.npz` (steps=150
FAIL) remains alongside the post-amendment `encoder_gate_s1200.npz`
(PASS) as the historical record of the first real run -- not deleted,
not silently superseded.

## Next step

Feasibility ladder stage 2 (5,000-image development subset) -- see
below.

# Stage 2B: Feasibility Ladder Stage 2

**Status: mechanical validation only, same framing as stage 1.** Its job
is runtime and feature-validity measurement at 5x scale, the production
SVD's own condition-number diagnostic, ridge-grid behaviour, the
ladder's second real-data ridge equivalence gate, and the first CNN
training against real data. The CNN-vs-identity and in-sample stats
numbers below are mechanical/development reporting, not a result --
`DESIGN.md` scopes this stage that way explicitly.

## Scope

5,000 official KMNIST training images, the same nested stratified draw
stage 1's 1,000 is a prefix of (`SEED=42`) -- checked explicitly (not
merely trusted from construction) and, further, checked bit-exact
against stage 1's own cached corruption artifact at the three shared
prefix rows. Corrupted per the same locked forward process. Topologies
and staged KMNIST inputs reused directly from stage 1's cached objects,
not rebuilt or re-staged. Run on a Colab A100 via `run_ladder_stage2.py`,
same architecture as the stage-1 driver.

## A false start, fixed before any real cost was incurred

The first attempt crashed immediately, at module scope, before `main()`
ever started: `NameError: name '__file__' is not defined`. A refactor
that imported `KMNIST_FILES` from `run_ladder_stage1` at module scope
(to avoid duplicating the dict) relied on `os.path.dirname(os.path.
abspath(__file__))` to locate it -- which fails under `mighty-colab exec
-f script.py`: the file's TEXT is transmitted directly into an existing
IPython kernel cell, not run as a script or imported as a module, so
`__file__` is never defined there. Past that, `run_ladder_stage1.py`
does not exist anywhere on the exec'd kernel's filesystem until
`bootstrap_repo()` clones the repo -- which happens INSIDE `main()`,
after every module-scope statement has already run. Confirmed against
the live session list and the bucket before any fix was written: zero
objects under `stage2b/train/stage2/`, no leaked session -- the crash
cost provisioning and package-install overhead only.

Fixed by reverting to plain duplication (stage 1's own pattern), with
two new durable guards added directly from this failure: a static check
that neither driver references `__file__` anywhere, and a dynamic check
that execs each driver's actual source into a namespace with no
`__file__` key, reproducing the real execution model exactly -- which
the ordinary import-based local verification could not, since Python's
own import machinery sets `__file__` correctly for a real imported
module. That gap is exactly how the bug reached a billing A100 uncaught
by every other local check run beforehand.

## Result: complete, all ten steps, first attempt after the fix

**`STAGE2_OK`.** Commit `a84fac9`. Total wall clock: **1,722.0s (28.7
minutes)** -- well under the 90-minute budget reserved for it. No
non-finite features anywhere; the corruption cross-stage spot-check
matched stage 1's cache bit-exact at all three shared rows.

| step | seconds |
|---|---:|
| bootstrap (clone + pip install) | 17.7 |
| stage_kmnist (download, reused from stage 1) | 8.4 |
| preflight | 0.0 |
| corpus | 5.6 |
| topologies (reused from stage 1) | 1.7 |
| corruption | 13.1 |
| corruption diagnostics | 2.5 |
| **encode (noisy only, diagnostic)** | **1,099.2** |
| restrict | 0.1 |
| evolution (4 graphs) | 41.0 |
| features | 46.8 |
| ridge (7 conditions, CV + equivalence pass 2) | 305.5 |
| CNN (3 seeds) | 113.1 |
| stats smoke | 67.1 |

**Step 6 (evolution)**: all four graphs, **0 failed of 5,000 for every
graph**, CPU reference cross-check succeeded on all four. **Step 8
(ridge)**: all seven conditions passed the ladder's SECOND real-data
equivalence gate, every difference 4+ orders below the 1e-8 tolerance
(1.2e-14 to 1.1e-12), every alpha selection agreeing between JAX and
sklearn, no condition at a grid edge. The n-dependent centering
tolerance held at every condition (see named item 2, below).

## Named items, on their own terms

**(1) Measured encode cost and the stage-3 projection.** **218.28
ms/image** at 1,200 steps, single-worker, n=5,000 (1,091.4s measured).
Linearly projected to stage 3's 54,000-image fit side: **11,787s (3.27
hours)** for encoding alone -- by far the dominant cost in the whole
pipeline at that scale, and the number that should drive any stage-3
planning decision about parallelizing or otherwise restructuring this
step. Not validated at that scale; a linear projection from one
measurement, stated as such.

**(2) `curr_random` centering margin vs. the amendment's ~0.075
prediction.** Measured: **`margin_ratio = 0.0807`** (`||mean(X)|| =
1.804e-10` against tolerance `2.236e-09` at this fold's n=4,000 training
rows) -- close agreement with the amendment's own derivation, confirmed
now at real n=5,000 scale rather than only the n=1,000 rung it was
first measured to hold at. Every other condition's margin sits inside
0.0009 to 0.081 (`T` and `lattice` tightest, `curr_random` still the
widest of the seven, consistent with it being the condition the
amendment's tolerance was specifically set to protect).

**(3) The condition-number table, all seven conditions**, from the
production SVD's own singular values (`fold_cond`), no second `cond()`
call:

| condition | mean fold cond(X) | alpha |
|---|---:|---:|
| raw_505 | 4.25 | 1000 |
| raw_784 | 5.34 | 1000 |
| pre_evolution | 171.0 | 1000 |
| curr_random | 1.49e6 | 100 |
| rewired | 1.87e6 | 10 |
| T | 4.78e6 | 1 |
| **lattice** | **4.82e8** | 1 |

Recorded plainly, not interpreted: `lattice`'s condition number is
roughly two orders of magnitude worse than `T`'s, and `T` in turn is
worse than `curr_random`/`rewired` by a factor of ~3. All seven passed
the equivalence gate regardless -- the diagnostic is descriptive, not a
gate, exactly as designed.

**(4) CNN: best seed, best_epoch, val MSE vs. identity.** Best seed
**0** (`best_epoch=94`, ran the full 100 epochs without early stopping),
**clipped validation MSE = 0.063678** against the identity baseline's
**0.199770** on the same locked 6,000-image validation partition -- a
substantial, consistent margin; all three seeds landed within
0.0637-0.0642 of each other (seed 1: 0.064229, stopped early at epoch
63/74; seed 2: 0.064156, stopped early at epoch 81/92). Total CNN
wall-clock across all three seeds: 99.5s. Labeled explicitly,
matching the design's own framing: mechanical sanity ("does the CNN
beat doing nothing on its own validation data"), NON-INFERENTIAL, not a
locked comparison -- the first CNN training against real data this
project has ever run.

## Stats smoke (in-sample, non-inferential)

Ran (not skipped): projected at 54.3s from stage 1's own measurement,
actually took 67.1s -- the linear projection understated real cost by
about 24%, worth noting as a fact about the projection's own accuracy
rather than a concern (the decision threshold has 60s of margin below
stage 1's own measured value, and this rung's real cost still landed
close to it). Artifact's first line, verbatim, matching stage 1's:
`SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE,
NON-INFERENTIAL, NOT A RESULT`.

## Stage-3 projections, per pipeline stage (linear, unvalidated)

| stage | measured at n=5,000 | projected at n=54,000 (10.8x) |
|---|---:|---:|
| encode | 1,091.4s | 11,787s (3.27h) |
| evolution | 41.0s | 443.2s (7.4min) |
| ridge | 305.5s | 3,299.7s (55.0min) |
| CNN | 99.5s | 1,074.9s (17.9min), basis differs -- see caveat below |

Never one blended rate. The CNN projection is explicitly weaker than the
others: CNN cost scales with epochs x batches, not simply n, and early
stopping means the epoch count itself is not fixed by corpus size --
this row scales wall-clock linearly as a first approximation only, not
a validated model of CNN training cost at scale.

## Code and artifacts

`run_ladder_stage2.py` (the driver). Tests:
`tests/test_stage2b_ladder_stage2.py`. Every stage-2 artifact lives in
the public-read bucket under `stage2b/train/stage2/`; topologies and
KMNIST inputs are read from `stage2b/train/stage1/` directly (reused,
not duplicated).

## Next step

Feasibility ladder stage 3 -- Phase A complete, see below.

# Stage 2B: Feasibility Ladder Stage 3, Phase A (encoding)

**Status: Phase A only. Phase B (evolution, ridge, CNN) has NOT run, and
stage 3 has produced no denoising number of any kind.** This section
records the encoding phase because it is a complete, measured unit with
a durable artifact -- and because this project's own Part 4 lesson is
that an unwritten result does not survive the session that produced it.

## The two-phase split, and why encoding moved off the GPU

Stage 2 measured encoding at 1,099s of a 1,722s total -- 64% of the run,
and the only genuinely CPU-bound step in the pipeline. Evolution, ridge
and the CNN are what actually use the A100. Running the encode inside a
provisioned GPU session would leave a metered A100 idle for most of
stage 3's wall-clock, at ten times stage 2's corpus size.

Stage 3 therefore splits: **Phase A** (corpus, corruption, encode,
restrict) runs on local CPU cores and writes only the encoded array to
GCS; **Phase B** reads it and regenerates corruption and clean targets
in-session, both being deterministic and cheap. That puts 218 MB across
the boundary instead of 775 MB, for about 2.5 minutes of cloud CPU.

This is a disclosed post-lock amendment, not a quiet deviation --
`DESIGN.md`'s "Computational strategy" and "Review history" both carry
it. The convention it amends says generation happens "entirely in the
cloud environment... never round-tripped through local upload", and
names its own reason in the same sentence: Stage 2A's Colab **session
upload** limit. A direct local→GCS write never touches that mechanism;
it uses the same client and the same chunked, crc32c-verified transport
in `stage2b_gcs.py` that a cloud-side write uses, and
`stage_kmnist_inputs.py` has moved KMNIST that way for both prior rungs.

## Measured, before the decision was acted on: cross-architecture reproducibility

Encoding on this Mac put the encoder on Apple Silicon for the first
time; stages 1 and 2 both encoded on Colab's x86. Rather than assume
equivalence, three comparisons were run on the same images first:

| comparison | result |
|---|---|
| same machine, encode run twice | **bit-exact** |
| two different Colab sessions, same 1,000 images | **bit-exact**, 784,000/784,000 coordinates |
| Mac (ARM) vs Colab (x86), same images | 93.4% coordinates identical; **max 3 ULP** (4.441e-16), mean 0.07 ULP, max relative 3.98e-16 |

The middle row matters and nearly went unchecked: an initial assumption
that "Colab hardware varies too, so local loses nothing" is simply
false -- two separate Colab sessions produced byte-identical encodings.
The difference is cross-architecture, not cross-session.

It cannot amplify. The encoder is a contraction toward a fixed point:
measured residual against the 1,200-step result falls 8.370e-07 (300
steps) → 8.062e-13 (600) → exactly 0.0 (1,200). Both platforms resolve
the SAME fixed point; a minority of coordinates land on an adjacent
representable float64. Downstream, these phases feed an ODE solver at
`rtol=1e-6`, eight or more orders above the difference.

Accepted deliberately, with disclosure, on the same standard this
project already applies to cuML's non-bit-reproducible GPU logistic
regression: what is guaranteed is within-architecture determinism, and
the encoded array in GCS is the artifact of record.

## Result

**54,000 fit-side images, 1,200 steps, 9 workers (Darwin arm64, 10
cores).** The 6,000-image locked validation partition is deliberately
not encoded -- the CNN consumes validation images as raw corrupted
grids, and ridge selects alpha by internal cross-validation on the fit
side, so an encoded validation array would be an artifact nothing reads.

| quantity | measured |
|---|---:|
| encode wall-clock | **475.3s (7.9 min)** |
| per image | **8.80 ms** |
| upload (206 MB compressed, chunked) | 102.9s |
| non-finite theta / delta | 0 / 0 |

Faster than the 9.5-minute projection made from a 180-image sample --
larger chunks amortise process-pool startup better. Against stage 2's
Colab-CPU measurement of 218.28 ms/image single-worker, this is roughly
a 25x wall-clock reduction, of which ~3.4x is per-core speed and the
rest parallelism.

## A scale-dependent finding: the convergence tail is not exactly zero

Stage 1 concluded final-Delta reaches "exact float64 zero -- median AND
p95, every one of 1,000 images", and stage 2 measured a max of 2.22e-14
across 5,000. At 54,000 the maximum is **2.468e-10** -- four orders
larger, and only visible at this scale:

| final-Delta | count of 54,000 |
|---|---:|
| exactly 0.0 | 53,921 (99.854%) |
| > 0 | 79 (0.146%) |
| > 1e-13 | 9 |
| > 1e-12 | 4 |
| > 1e-10 | 2 |

median 0.0, p95 0.0, max 2.468e-10.

Recorded rather than smoothed over, with its consequences stated
precisely. It does not affect the encoder gate, which keys on the
MEDIAN (0.0 here, so the absolute-convergence escape fires regardless of
the tail). It does not affect the pipeline: the worst image sits 4,053x
below the ODE solver's `rtol=1e-6`. What it does do is narrow stage 1's
claim -- "every one of 1,000 images" was true of 1,000 images, and at 54
times that corpus size a 0.146% tail of not-quite-settled images
appears. Two images in 54,000 remain above 1e-10 after 1,200 steps.

## Code and artifacts

`encode_stage3_local.py`, run by `make stage2b-encode-stage3-local`
(local, CPU-only, provisions nothing, bills nothing). Composes
`corrupt_corpus` and `encode_with_final_delta_batch` unchanged -- same
numerics as both prior rungs, different machine. Output:
`stage2b/train/stage3/common/encoded_fit_s1200.npz` (206.1 MB,
crc32c-verified), carrying the encoded array, per-image final-Deltas,
the fit indices and active support for self-description, and the run
summary. The object name carries the step count, so a future
`ENCODER_STEPS` change mints a new object rather than silently resuming
a stale one -- the same self-invalidation discipline the stage-1
encoder-gate artifact uses.

218 MB crossed `ensure_artifact`'s 64 MB auto-chunk threshold, so the
resumable chunked upload path engaged without the call site asking for
it. That safeguard was added speculatively earlier in the same week;
this is its first use on an artifact large enough to need it.

## Next step

Phase B: a GPU-session driver that reads
`stage2b/train/stage3/common/encoded_fit_s1200.npz`, regenerates
corruption and clean targets in-session, and runs evolution, ridge (with
the ladder's third real-data equivalence gate) and CNN training at full
scale. Not yet written. It should carry a spot-check that one image's
encoding re-derived in-session matches the stored array, with the
tolerance stated as ULP-level rather than exact, for the
cross-architecture reason recorded above.
