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
pre-registered `rho <= 10` threshold:

| quantity | clean | noisy |
|---|---|---|
| median final-Delta | 2.177485e-07 | 3.698480e-05 |
| p95 final-Delta | 9.971726e-07 | 1.784018e-04 |
| non-finite phases / deltas | 0 / 0 | 0 / 0 |

**rho = 169.851** against threshold 10 -- roughly 17x over. Zero
non-finite values anywhere: a clean ratio failure, not a numerical
blow-up.

**What this failure was, and what it was not.** The original Stage 1
failure was not itself a numerical-floor artefact. It showed that 150
iterations were insufficient for noisy inputs under the original
relative gate. The noisy median, 3.698e-05, and its p95, 1.784e-04, sit
far above any numerical floor -- seven orders above the `ABS_CONV_EPS`
of 1e-12 adopted later, and nine or more above the float64 dust Part 2
observes at 1e-14 to 1e-16. The
floor artefact identified in Part 2 is a distinct defect with a
distinct cause: it is a property of the DIAGNOSTIC trajectory at 600
steps and above, where both series have decayed into numerical dust.
The two are not the same failure mode and neither implies the other.

What the gate's own summary reports is the median, the p95 and the
non-finite counts -- not per-image counts of exactly-zero and nonzero
final-Delta, and not the max. The per-image `delta_clean` and
`delta_noisy` arrays are stored inside `encoder_gate.npz`, so those
counts are recoverable from the artifact rather than lost.

Per `DESIGN.md`'s locked stop-gate this halted the stage
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

Noisy final-Delta decays geometrically: by 1,200 steps both its median
and its p95 are exact float64 zero, the same fixed point clean reaches.
No floor above a meaningful scale exists; on censored inputs the encoder
reaches numerical convergence below any practically relevant tolerance,
it is simply slower to get there.

**Two quantiles are not a distribution.** The median and the p95 are the
only order statistics this table carries, and median and p95 ratios do
not by themselves prove a whole-distribution shift -- they show a shift
at two quantiles. Concretely, the steps=1,200 row supports "median and
p95 are exactly zero across these 1,000 images", not "every one of
1,000 images is exactly zero": no per-image exact-zero count and no max
was reported at this rung. The 54,000-image measurement recorded further
down shows the stronger reading is false at larger corpus sizes. The
per-image arrays are stored in the stage-1 `encoder_gate*.npz`
artifacts, so the unreported counts remain recoverable from the record.

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
Review History. **The 1,200-step budget and absolute-convergence clause
were prospective amendments made after Stage 1 failure and before
downstream confirmatory evaluation; they were not preregistered
components of the original design.** Two changes, answering two
different failures:

1. **`ENCODER_STEPS` raised 150 -> 1200**, uniformly (every encoding
   site, clean and noisy identically). This answers the Part 1 failure:
   150 iterations were insufficient budget for noisy inputs. Mirrors
   Stage 2A's own `max_iter` 1,000 -> 10,000 precedent: halt honestly,
   diagnose mechanism, amend with disclosure, re-verify.
2. **Gate formula gains an absolute-convergence escape**: PASS if
   `rho <= 10` OR both medians are already below `ABS_CONV_EPS=1e-12`
   (5+ orders below the smallest meaningful measured Delta, well above
   observed float64 dust). Non-finite auto-fail stays unconditional.
   This answers the separate Part 2 defect -- the ratio gate flooring
   out at 600 steps and above -- and would not have rescued the Part 1
   failure, where both medians sit far above 1e-12.

### Why S\* = 1,200: three separate things, kept separate

**(a) The contemporaneous decision-time reason, as it was actually
given.** 1,200 was the only step count in the five-point scan at which
both reported statistics -- median and p95, on the clean series and the
noisy series alike -- sat at exact float64 zero. That was read at the
time as maximal distance from the fragile crossover band where clean and
noisy pass their own numerical floors at different step counts, the band
that produced the spurious rho=17.76 FAIL at 600. No finer scan between
600 and 1,200 was run, on the reasoning that a smaller untested value
might land inside that band.

The decision rule that selected S\*=1200 was stated as verdict-invariant,
not merely correct. An earlier, looser reading of the rule ("some S\*
brings noisy within 10x of clean-at-150") was caught before being
applied -- it selects S\*=300, which immediately fails its own same-step
re-run at rho=1.915e4. The corrected rule (same-step, both series
required to have genuinely converged) passes at S\*=1200 under EITHER
reading, so the correction did not select this outcome.

**(b) The durable rationale, identified retrospectively.** 600 is the
first measured budget satisfying the amended absolute gate: noisy median
1.776e-14 and clean median exactly 0, both below `ABS_CONV_EPS=1e-12`,
where 300's clean median of 1.538e-12 is not. 1,200 is a conservative
twofold safety margin over that first passing budget, frozen before any
downstream fitting. This reading is derivable from the scan table above
and does not rest on (a)'s exact-zero premise.

**(c) Subsequent evidence narrowing (a)'s premise.** (a) rests on exact
zero being a property of the converged encoder, as the n=1,000 scan's
two reported order statistics showed at the time it was reasoned. That
premise did not survive scale, and degraded monotonically with corpus
size: stage 2's n=5,000 encode already recorded a nonzero max
(2.22e-14), and the 54,000-image encoding recorded further down found 79
images with nonzero final-Delta, max 2.468e-10. (a) is preserved here as
the decision record -- what was actually reasoned, on the evidence then
available -- not as a claim that still stands unmodified. (b) is
untouched by this: it turns on medians, which are exactly zero at every
corpus size where a median is on the record, n=1,000 and n=54,000.

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

Both medians at exact float64 zero, matching the diagnostic's own
steps=1200 measurement precisely. What the gate's summary reports is
medians, p95s and non-finite counts (0/0 on both sides); it reports no
per-image count of exactly-zero or nonzero final-Delta and no max, and a
median of zero is not a per-image claim -- see the 54,000-image tail
below. What this run establishes is numerical convergence below any
practically relevant tolerance, not exact convergence for every image.
Because the gate passed, the driver
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
measurement, stated as such. It was projected against a 54,000-image
stage-3 scale; the stage-3 populations are fixed differently -- see
"Population roles" in the Phase A section below.

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

Every row was projected against a 54,000-image stage-3 scale. The
ridge CV and final-fit corpus is the full 60,000-image training side
(see "Population roles" below), so the ridge row's population is not the
one that will actually be fitted; no revised projection is stated here,
because rescaling one row by a population ratio is exactly the
cross-stage extrapolation this project has already been bitten by.

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

Feasibility ladder stage 3 -- Phase A complete at 54,000 images and
pending regeneration at 60,000, see below.

# Stage 2B: Feasibility Ladder Stage 3, Phase A (encoding)

**Status: Phase A only, and at the wrong population -- this run encoded
54,000 images. It has since been SUPERSEDED by the 60,000-image
regeneration recorded in the next section, which is the authoritative
Phase A artifact. Phase B (evolution, ridge, CNN) has NOT run, and stage
3 has produced no denoising number of any kind.** This section
records the encoding phase because it is a complete, measured unit with
a durable artifact -- and because this project's own Part 4 lesson is
that an unwritten result does not survive the session that produced it.
Its numbers stand: the regeneration reproduced every one of them
bit-exactly.

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
cores).** This run encoded the CNN fit side only, on the reading that
the CNN consumes validation images as raw corrupted grids and ridge
selects alpha by internal cross-validation on the fit side, so an
encoded validation array would be an artifact nothing reads. That
reading is wrong for the ridge path: ridge CV and the final refits run
on the full 60,000-image training side, so the 6,000 locked-validation
images must be encoded too. Phase A will be regenerated at 60,000; the
measurements below are the 54,000-image run's own.

**Population roles.** `DESIGN.md:479` defines "full training" at stage 3
as the 54,000 fit plus 6,000 locked validation composite, and `:492`'s
compute cell reads `~48-60k x 1008`, where 48,000 = 0.8 x 60,000 is a
five-fold CV training portion of 60,000. The 54,000 / 6,000 split
governs CNN weight-fitting and model selection only: the 6,000 are held
out from CNN gradient updates, not from training-side analysis.

| role | n |
|---|---:|
| official training corpus | 60,000 |
| CNN weight-fit subset | 54,000 |
| CNN validation / model-selection subset | 6,000 |
| **ridge CV and final-fit corpus** | **60,000** |

Frozen as Freeze 2 in `AUDIT_PROTOCOL.md`, which is authoritative for
these roles.

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

Stage 1 reported final-Delta at exact float64 zero at both the median
and the p95 across 1,000 images, and stage 2 measured a max of 2.22e-14
across 5,000. At 54,000 the maximum is **2.468e-10** -- four orders
larger, and only visible at this scale. Neither earlier rung reported a
count of exactly-zero or nonzero final-Delta; both stored their
per-image arrays, so both counts are recoverable from the artifacts
without re-encoding.

Population: **the 54,000 fit-side images of the official KMNIST training
split**, encoded by this Phase A run. The 6,000-image locked validation
subset has never been encoded, so no tail measurement exists for it and
none of the counts below speak to it. Nothing here touches the test
split.

| final-Delta, over the 54,000 encoded fit-side images | count |
|---|---:|
| exactly 0.0 | 53,921 (99.854%) |
| > 0 | **79 (0.146%)** |
| > 1e-13 | 9 |
| > 1e-12 | 4 |
| > 1e-10 | 2 |
| non-finite | 0 |

median 0.0, p95 0.0, max 2.468e-10; non-finite theta 0, non-finite delta
0. Derived from the counts above rather than measured directly: p99 is
0.0 (only 79 of 54,000 values are nonzero), and p99.9 lies in
(0, 1e-13] -- it falls inside the nonzero tail, but only 9 values exceed
1e-13.

Recorded rather than smoothed over, with its consequences stated
precisely. It does not affect the encoder gate, which keys on the
MEDIAN (0.0 here, so the absolute-convergence escape fires regardless of
the tail). It does not affect the pipeline: the worst image sits 4,053x
below the ODE solver's `rtol=1e-6`. What it does do is narrow stage 1's
claim -- "every one of 1,000 images" was never established by the two
order statistics stage 1 actually reported, and at 54 times that corpus
size a 0.146% tail of not-quite-settled images appears. Two images in
54,000 remain above 1e-10 after 1,200 steps.

The standing wording for what the encoder achieves is therefore
**numerical convergence below any practically relevant tolerance**,
never exact convergence as a universal claim. Aggregate quantiles
sitting at zero do not mean every image sits at zero, and this is the
in-project proof of it.

## Code and artifacts

`encode_stage3_local.py`, run by `make stage2b-encode-stage3-local`
(local, CPU-only, provisions nothing, bills nothing). Composes
`corrupt_corpus` and `encode_with_final_delta_batch` unchanged -- same
numerics as both prior rungs, different machine. Output:
`stage2b/train/stage3/common/encoded_fit_s1200.npz` (206.1 MB,
crc32c-verified), carrying the encoded array for the 54,000 fit-side
images, their per-image final-Deltas, the fit indices and active support
for self-description, and the run summary. The object name carries the
step count, so a future
`ENCODER_STEPS` change mints a new object rather than silently resuming
a stale one -- the same self-invalidation discipline the stage-1
encoder-gate artifact uses.

218 MB crossed `ensure_artifact`'s 64 MB auto-chunk threshold, so the
resumable chunked upload path engaged without the call site asking for
it. That safeguard was added speculatively earlier in the same week;
this is its first use on an artifact large enough to need it.

## Next step

Regeneration at 60,000 -- done, and recorded in the next section.

# Stage 2B: Feasibility Ladder Stage 3, Phase A regenerated at 60,000

**Status: the authoritative Phase A artifact.
`stage2b/train/stage3/common/encoded_train_s1200.npz` supersedes
`encoded_fit_s1200.npz`, which stays in the bucket as the baseline this
run was checked against. Phase B has still NOT run and stage 3 has still
produced no denoising number.**

This is also the first Stage 2B artifact of any kind published with
provenance attached. Every earlier one carries none.

## Why it was rerun, and what that cost

The original Phase A encoded the 54,000-image CNN fit side on the
reading that the 6,000 locked-validation images would be "an artifact
nothing reads". That reading was wrong: `DESIGN.md:479` defines "full
training" at stage 3 as the 54,000 fit plus 6,000 validation composite,
`:492`'s compute cell corroborates it arithmetically (`~48-60k x 1008`,
where 48,000 = 0.8 x 60,000), and the ridge path cross-validates and
refits on all 60,000. The error is disclosed at the point it was made,
in `encode_stage3_local.py`'s own docstring, rather than edited away --
it was resolved in the direction that avoided re-work, which is exactly
the kind of choice that has to stay visible.

The rerun cost 11.3 minutes of local CPU.

## Two populations, two kinds of evidence

The regeneration spans images that already had an artifact and images
that did not, and the acceptance report keeps them apart deliberately.
A single "regeneration passed" would claim verification for a half of it
that nothing verified.

| population | n | status | judged how |
|---|---:|---|---|
| CNN fit side | 54,000 | reproduction | bit-exact against the prior artifact, joined by official index |
| locked validation | 6,000 | **new measurement** | fingerprinted at birth; tail reported, nothing compared |

## Part 1 -- the 54,000 reproduce bit-exactly

| array | shape | verdict |
|---|---|---|
| `thetas_505` | (54000, 505) float64 | **BIT-EXACT**, sha256 `1113cec3...bc7e85fc` both sides |
| `deltas` | (54000,) float64 | **BIT-EXACT**, sha256 `6a7753ff...740954f` both sides |

The match is on the aligned subset, joined by official KMNIST index --
`AUDIT_PROTOCOL.md` forbids positional-prefix comparison, and the new
artifact is in ascending official order, so the fit rows are scattered
through it: 53,985 of 54,000 sit at a different row than they did in the
baseline. `compare_stage3_regeneration.py` refuses an alignment that
turns out to BE a prefix, because a join that accepts that case cannot
be distinguished from never having joined.

An independent second check from the other direction: the aligned
subset's nonzero final-Delta count is **79**, exactly reproducing the
count this file already records for the 54,000-image run. Digests
agreeing while that count moved would have meant the comparison was not
looking at the rows it believed it was.

Bit-exactness is expected rather than lucky, and the reason is now
pinned by test rather than reasoned about: every per-image encode job
carries a constant seed and builds a fresh `default_rng(seed)`, so an
image's perturbation depends on the image and the seed and nothing else
-- not the chunk it landed in, not the worker count. Both runs used 9
workers on the same machine but different chunk sizes (5,400 then
6,000), which is precisely the situation CLAUDE.md principle 19 says not
to assume your way through.

## Part 2 -- the 6,000, measured for the first time

| role | n | final-Delta > 0 | rate | 95% CI (Clopper-Pearson) | max |
|---|---:|---:|---:|---|---:|
| all | 60,000 | 89 | 0.148% | [0.1191%, 0.1825%] | 2.468e-10 |
| fit | 54,000 | 79 | 0.146% | [0.1158%, 0.1823%] | 2.468e-10 |
| **validation** | **6,000** | **10** | **0.167%** | **[0.0800%, 0.3063%]** | **2.887e-13** |

`AUDIT_PROTOCOL.md` sets **no expected-agreement criterion** between the
two rates, and the intervals are why that matters: at n=6,000 the
interval is roughly three times wider than at n=54,000, so 0.167% and
0.146% are not distinguishable and a bare comparison of the percentages
would invite a conclusion the data does not support. Reported exactly as
the protocol requires -- numerator, denominator, split membership,
uncertainty -- and nothing inferred from it.

One descriptive observation, not a claim: the validation subset's worst
final-Delta is 2.887e-13, roughly three orders of magnitude below the fit
side's 2.468e-10. With ten nonzero values, that is a small-sample
observation about an extreme order statistic and nothing more. Both sit
far below the ODE solver's `rtol=1e-6`.

### The ARM/x86 stress set now has 89 cases, not 79

`AUDIT_PROTOCOL.md`'s companion section names "the 79 convergence-tail
stress cases", written when only the 54,000-image artifact existed.
`COMPANION_PROTOCOLS.md` states the construction as a rule rather than a
count -- every image with final-Delta > 0 in the regenerated artifact --
which is now **89**. Both documents are frozen and neither is wrong; the
rule is what governs, and the count moved because the population did.
Recorded here so that anyone building the set from the audit protocol
alone, and arriving at 79, can see why. The 500-case cap frozen in
`COMPANION_PROTOCOLS.md` before the count was known remains inert, as
expected.

## Result

**60,000 official KMNIST training images, 1,200 steps, 9 workers (Darwin
arm64, 10 cores), chunk 6,000.**

| quantity | measured |
|---|---:|
| encode wall-clock | **679.0s (11.3 min)** |
| per image | **11.32 ms** |
| upload (229 MB compressed, chunked) | 113.0s |
| non-finite theta / delta | 0 / 0 |
| final-Delta median / p95 / max | 0.0 / 0.0 / 2.468e-10 |

Per-image cost rose from 8.80 ms to 11.32 ms against the 54,000-image
run on the same machine. Not investigated; the plausible causes are
ordinary (thermal state, other load, the different chunk size), it
changes no result, and attributing it without measuring would be a
guess.

## Provenance -- what is now attached, and what the record shows

The artifact is published with a sidecar manifest
(`...npz.manifest.json`) carrying the payload digest, per-array
dtype/shape/SHA-256 for all seven arrays, and a fingerprint: commit
`12a8c46a`, the 18-file static-union-runtime source closure with each
file's digest, the environment, and the declared scientific config
(digest `768bf201...`). The closure was established BEFORE generation
and revalidated after: 0 modules imported that the fingerprint did not
already describe.

The recorded git state is worth reading rather than skipping, because
it is the first real exercise of a guard that was narrowed while it was
blocking this very run:

```
commit         : 12a8c46ad5aad152874b4a618f5b60784779f098
source closure : CLEAN -- every file committed at HEAD
working tree   : dirty elsewhere
  | M .gitignore
  | ?? .claude/claude2claude/DESKTOP_PROTOCOL.md
  | ?? .claude/claude2claude/c2c-mcp/deploy-proxy.sh
  | ?? .claude/claude2claude/c2c-mcp/run-c2c-mcp.sh
```

All four belong to an unrelated concurrent effort and none is in the
source closure, so none can reach this artifact. The whole-tree check
would have refused this run; the closure check names what actually
matters and records the rest. Detection is shown intact by test rather
than argued: a dirty closure file still halts and is named, staging is
not committing, and an untracked closure file is dirty rather than
silently clean.

## Code and artifacts

`encode_stage3_local.py` (`make stage2b-encode-stage3-local`) and
`compare_stage3_regeneration.py` (`make stage2b-compare-stage3`, read-
only). The acceptance report is
`results/stage3_regeneration_acceptance.json`. Object:
`stage2b/train/stage3/common/encoded_train_s1200.npz` (229.1 MB,
crc32c-verified), carrying `thetas_505`, `deltas`, `train_indices`,
`fit_indices`, `validation_indices`, `active_indices` and the run
summary. Rows are in ascending official index order, so row `i` is
official training image `i` -- the index arrays are stored anyway rather
than left implied.

## Next step

Phase B: a GPU-session driver that reads this artifact, regenerates
corruption and clean targets in-session, and runs evolution, ridge (with
the ladder's third real-data equivalence gate) and CNN training at full
scale. **Written and run since -- see "Phase B, the alpha floor, and the
amended grid" below; this paragraph is preserved as the state at the time
Phase A landed.** It should carry a spot-check that one image's
encoding re-derived in-session matches the stored array, with the
tolerance stated as ULP-level rather than exact, for the
cross-architecture reason recorded above.

It should also consume through `consume_validated` rather than
`download_file`. The contract is now built and this artifact publishes
under it, but no driver READS under it yet -- `ensure_artifact` still
does not call it, `force=True` still bypasses its trust point, and two
call sites still download directly. `NEGATIVE_PATH_EVIDENCE.md` records
that demand as covered-at-the-module-layer and not yet adopted.

**A decision Phase B's Makefile target has to make deliberately.** Two
definitions of "clean enough to run" now coexist and they disagree. The
ladder targets refuse on whole-tree `git status --porcelain`; the
fingerprint refuses on the source closure. This run is the case that
separates them: the closure check passed, the whole-tree check would
have refused. Leaving both in place means the guard that actually fires
is the coarse one -- the one this project concluded was the wrong
question -- while the sharper one sits behind it never reached. Phase B's
recipe should gate on the closure and say so in the recipe comment,
rather than inheriting the porcelain check by copying the stage-1 target.
Noted here rather than changed unilaterally, because it alters the
pre-flight on targets that spend money.

---

# Phase B, the alpha floor, and the amended grid

Phase B ran on 2026-08-07 (`STAGE3_OK`, 71.6 min, A100): the full 60,000
corpus through evolution, features, ridge and CNN. Zero solver failures
across 240,000 graph evolutions. The re-encode spot-check came in at 2.0
ULP against a 16.0 tolerance, consistent with the 3 ULP cross-architecture
maximum recorded above. Decision 2's equivalence extension passed on all
seven conditions at 12x the largest previously verified scale, worst
clipped-prediction difference 2.185e-12 against the frozen 1e-8.

What follows is the finding that came out of it, which is not about any of
that.

## Six of seven conditions selected the smallest alpha on the grid

At feasibility stage 2 (n=5,000) no condition sat at a grid edge. At
n=60,000, six of seven selected `ALPHA_GRID`'s minimum:

| condition | n=5,000 | n=60,000 (nine decades) |
|---|---|---|
| raw_505 | 1000.0 | **0.01 (floor)** |
| raw_784 | 1000.0 | **0.01 (floor)** |
| pre_evolution | 1000.0 | 1000.0 |
| T | 1.0 | **0.01 (floor)** |
| lattice | 1.0 | **0.01 (floor)** |
| rewired | 10.0 | **0.01 (floor)** |
| curr_random | 100.0 | **0.01 (floor)** |

Less regularization with more data is expected. A selection AT the
boundary is a different statement: the grid no longer brackets the
optimum, so those six values are a boundary, not a minimum.

This did not halt the run, correctly -- grid-edge selections are recorded
fact under the frozen plan, and no gate may be invented mid-ladder in
either direction. It fired a **pre-registered review item**: *"if several
conditions select the grid-minimum alpha, that is a reported fact
requiring scrutiny before Stage 4 -- not a halt, but a named review
item."* Nobody had to decide in the moment whether it mattered.

### The scrutiny step cost nothing, and split the conditions

The validation curves were already in the committed artifact. Relative
rise from the floor:

| condition | 0.01 -> 0.1 | 0.01 -> 1.0 | reading |
|---|---|---|---|
| raw_505 | 1.167e-07 | 1.284e-06 | plateaued |
| raw_784 | 1.060e-07 | 1.166e-06 | plateaued |
| pre_evolution | -6.227e-07 | -6.845e-06 | not pinned |
| **T** | **8.431e-03** | 2.043e-02 | still moving at the floor |
| **lattice** | **4.151e-03** | 1.145e-02 | still moving at the floor |
| **rewired** | **7.663e-03** | 1.749e-02 | still moving at the floor |
| **curr_random** | **6.195e-03** | 1.351e-02 | still moving at the floor |

The split fell along the treatment/control line: both raw baselines flat
to ~1e-7, all four evolved conditions moving four to five orders of
magnitude faster. The constraint bound asymmetrically across exactly the
comparison the readout exists to support.

## The amendment, frozen before any fitting

The reviewer ruling extended the grid four decades downward to
`{1e-6 .. 1e6}` -- thirteen values, exact decade spacing, all seven
conditions -- with a full re-run, no splicing, the equivalence check
re-run at production scale, and a halt for review if any condition
selected the new floor. Recorded with its provenance in `DESIGN.md`'s
Review history; it arrived by manual copy-paste from a ChatGPT session
rather than the audited channel, and a ruling's route is part of its
evidence.

**The procedure is one-shot by construction.** The exact-spacing clause
forbids interpolation or densification around an observed minimum, and a
pin at the new floor is named an anomaly rather than a trigger for
further extension. There is no second widening this procedure can
authorise -- which matters because a stopping rule chosen after seeing
results is not a stopping rule.

Two consequences neither the ruling nor the plan anticipated, both found
while implementing it:

**The write-once invariant enforces "do not splice" structurally.**
`ridge_cv.json` and `ridge_final.npz` are create-once LINEAGE artifacts;
`force=True` raises before `produce` runs. The re-run could not overwrite
the nine-decade tables and had to write new names.

**And that same invariant does nothing about silent REUSE.** The
nine-decade artifacts sat in the bucket with valid manifests, and
`ensure_json` passes no `expected_fingerprint` -- so the thirteen-decade
re-run would have hit `ensure_artifact`'s skip branch, accepted the
superseded results, recomputed nothing, and reported `STAGE3_OK`. That
failure satisfies "do not splice" **by accident**: nothing is spliced when
nothing is computed. Fixed by deriving the artifact name from the grid
itself (`ridge_cv_g13_88edf9ac`), so any grid change moves the name
automatically. Never-overwrite protects against clobbering history;
nothing about it protects against reading the wrong history.

## The amended-grid result

Re-run 2026-08-08, 48.5 min, thirteen decades, all seven conditions,
evolution/features/CNN consumed from cache and only the ridge recomputed.

| condition | selected | argmin | status |
|---|---|---|---|
| raw_505 | 1e-03 | 1e-06 | flat to 1e-10 over four decades; tie-break takes the largest |
| raw_784 | 1e-03 | 1e-06 | same |
| pre_evolution | 1e+03 | 1e+03 | interior |
| **rewired** | **1e-05** | 1e-05 | **interior minimum -- resolved** |
| **curr_random** | **1e-05** | 1e-05 | **interior minimum -- resolved** |
| **T** | **1e-06** | 1e-06 | **at the floor** |
| **lattice** | **1e-06** | 1e-06 | **at the floor** |

Equivalence passed on all seven at thirteen decades, worst difference
**5.421e-11** against the frozen 1e-8.

The extension resolved five of seven. `rewired` and `curr_random` have
genuinely higher MSE at 1e-6 than at 1e-5 (by 2.685e-04 and 3.805e-04) --
interior minima, not tie-breaks.

### What is established, and what is not

Stated in the reviewer's licensed formulations, because the distinctions
are load-bearing and each is easy to overstate in the direction that
flatters the hypothesis:

- For **T and lattice**, `1e-6` is the best **observed candidate on the
  frozen discrete grid**. Their continuous-domain ridge optima are **not
  bracketed**, and are not established to equal, lie below, or lie near
  `1e-6`. The grid also says nothing about locations **between** sampled
  decades.
- For **rewired and curr_random**, the frozen-grid selections are genuine
  interior minima at `1e-5`.
- The monotonically shrinking per-decade increments for T and lattice are
  legitimate descriptive evidence that the validation curve is
  **flattening** over the sampled low-alpha range. For T the 1e-6 -> 1e-5
  increment is **5.292e-05**, against **8.431e-03** at the old floor.
  This supports **deceleration**, **not localization** of the optimum.
- That flattening is **not** translated into a quantitative bound on the
  unseen optimum, nor into a claim that residual regularization bias is
  negligible.
- Any statement about direction of constraint is **training-side
  validation only**. Nothing here establishes the sign or magnitude of the
  official-test effect.
- Graph rankings involving T or lattice are explicitly
  **protocol-bounded**: they compare pipelines selected under the frozen
  finite alpha grid, not continuously optimally regularized graph
  conditions. A control beating a floor-pinned T is not a mechanistic
  superiority claim.

**The floor condition is a disclosure and a qualification on optimization
scope -- not a protocol defect.** The alpha -> 0, denser-grid and
continuous-optimum questions are post-confirmatory work.

## The gate that was frozen and never implemented

`DESIGN.md`'s amended procedure says, verbatim: *"HALT for review if any
production condition selects 1e-6."* That sentence was frozen hours before
the grid extension was implemented, **and the halt was never written into
the driver.** The re-run selected the floor on T and lattice and reported
`STAGE3_OK`.

That verdict records that **no such gate existed** -- not that one was
evaluated and cleared. It is preserved as such: the run report carries a
machine-readable annotation, and `read_run_report()` refuses to yield the
verdict without surfacing it.

The numerical artifacts from that run **remain admissible**. The defect
was in the readiness verdict and its enforcement, not in the reported
ridge computation.

The halt now exists (`floor_halt_reason()`), extracted as a pure function
so a test exercises the decision rather than the spelling of the code, and
confirmed to fail against `817ac08` -- the exact driver that produced the
report, which imports cleanly with `step7_ridge` intact, so the failure is
the absent gate rather than a broken module.

Recorded here rather than only in the process log because it is the
sharpest instance this project has of a class it has now hit six times in
a day: **the requirement exists in the document and not in the code.** The
document and the driver had the same author, in the same session, hours
apart. There was no handoff to blame, which is what rules out "be more
careful" as the remedy and motivated the binding-gate inventory now
required before the package.

---

# Stage 2B: Feasibility Ladder Stage 4 -- the official result

**Status: THE locked confirmatory result.** `DESIGN.md`'s ONE evaluation
on the official 10,000-image KMNIST test corpus, run under the locked
statistical procedure, evaluated once. Everything upstream of this
section -- stages 1 through 3 -- was feasibility work; stage 3 was
labeled "SMOKE OF THE MACHINERY ONLY... NOT A RESULT" throughout. This
section is the first and only place in Stage 2B where that label does
not apply.

**`STAGE4_OK`, `run_ladder_stage4.py`, commit `431d90a`.** Official
result: `stage2b/testsplit/stage4/common/official_result.json`.

## Result: T is the unique winner

**Primary test** (`d_i = MSE_i(T) - MSE_i(pre_evolution)`, active-support,
post-clip, 20,000-resample paired class-stratified bootstrap, `seed=42`):
95% CI **[-0.0046028, -0.0043002]**, mean **-0.0044509**, entirely below
zero. **Verdict: evolution improves reconstruction.** Uncorrected -- the
primary test sits outside both multiplicity families, per the locked
design.

**Denoising gate** (Level 2 of the hierarchical identity gate, evaluated
because the primary succeeded): `T` vs. identity (`clip(x_t)`,
active-support, post-clip), CI **[-0.1335739, -0.1328960]**, mean
**-0.1332334**, entirely below zero. **Gate passed** -- the "actual
denoising" claim is added to the primary reconstruction claim, not just
the weaker relative-improvement one. Context, reported independently and
outside the gate: `pre_evolution` vs. identity, CI [-0.1291310,
-0.1284354], mean -0.1287826 -- pre-evolution already denoises
substantially on its own; evolution adds a further, statistically
resolved margin on top.

**Family 1** (three controls vs. `pre_evolution`, Holm across three, all
favorable): `curr_random` mean -0.002872, raw/Holm p=1.851e-252/3.702e-252;
`rewired` mean -0.002179, raw/Holm p=1.395e-144/1.395e-144; `lattice` mean
-0.004073, raw and Holm p reported as `0.0` -- a float64 underflow of the
analytic t-tail (`t=-55.58`, df=9999), not an exact zero (CLAUDE.md
principle 6). All three Holm-rejected.

**Family 2** (six pairwise among the four evolved graphs, Holm across
six): all six Holm-rejected, every raw p underflowing or near it
(largest surviving p, `rewired_vs_curr_random`: 1.549e-30). Directional
picture -- `T` beats all three others (vs. `lattice` t=-8.74,
p=2.73e-18; vs. `rewired` t=-38.10; vs. `curr_random` t=-26.85);
`lattice` beats `rewired` and `curr_random` but loses to `T`;
`curr_random` beats `rewired` but loses to `T` and `lattice`; `rewired`
loses to everyone. Sign-flip robustness check (100,000 flips, `seed=42`,
studentized statistic) agrees in direction and significance on all six,
each landing at the Monte Carlo floor (`p=9.9999e-06`); reported per
DESIGN.md's own framing as robustness only, not a second corrected
family, and with the sign-exchangeability assumption stated per pair
rather than asserted once and applied silently.

**`one_graph_wins`: `unique_winner = "T"`.** `T` qualifies via the primary
rule (bootstrap interval entirely below zero) and outperforms each of the
three other evolved graphs after Family-2 Holm correction. This is
`DESIGN.md`'s named watched-for outcome #2 ("one evolved graph qualifies
per the branched rule AND outperforms each of the other three after
Family-2 correction"), realized rather than one of the other four named
outcomes.

## The caveat this result inherits from Stage 3, stated again rather than left implicit

Stage 3's amended-grid result (above) already states it for the training-
side number and it applies unchanged here, because stage 4 REFITS at the
exact frozen alpha stage 3 selected, not a re-selected one: **`T` and
`lattice` both sit at the ridge grid FLOOR (`alpha=1e-6`)**; `rewired` and
`curr_random` sit at genuine interior minima (`alpha=1e-5`);
`pre_evolution` at `alpha=1e+03`. `T`'s continuous-domain ridge optimum is
therefore **not established, and not known to equal, lie below, or lie
near `1e-6`** -- the grid says nothing about alpha values between sampled
decades, let alone below the smallest one.

What this does and does not license, stated in the same terms Stage 3
used: `T` beating `lattice` is a floor-vs-floor comparison, so that
particular result is not confounded by asymmetric regularization freedom
between the two. `T` beating `rewired` and `curr_random` compares a
floor-pinned condition against two conditions with room to move, and the
comparison is therefore of the four pipelines **as selected under the
frozen discrete grid**, not of the four graphs under continuously optimal
regularization each. A denser or extended grid could in principle move
`T`'s number in either direction; nothing here bounds by how much. The
primary test and the denoising gate, both stated in terms of `T` alone
against `pre_evolution` and against identity respectively, do not depend
on this comparison and are unaffected by it.

## CNN: retrained and verified, reported descriptively

Per `run_ladder_stage4.py`'s design (stage 3 persisted no trained
weights, only training histories and a summary -- see that file's module
docstring), the CNN was retrained from the same three fixed seeds on the
same locked 54,000/6,000 fit/validation split, then verified against
stage 3's persisted selection before being trusted for test-corpus
inference. **Reproduction confirmed**: seed 1, epoch 99, matching stage
3 exactly on both (the structural gate `cnn_reproduction_mismatch_reason`
requires exact equality on these two); `best_clipped_val_mse` differed
by **2.385e-07** -- reported, not gated, per the module's own refusal to
invent a numeric tolerance with no measured basis. This is now real,
repeated evidence (three separate real-GPU retrains across this run's
three attempts, differing by 9.328e-07 and 2.385e-07 respectively from
stage 3's own number) that this iterative, float32, cross-session
training procedure reproduces its seed/epoch selection reliably in
practice, though the module deliberately does not claim this as a proven
guarantee.

CNN test-corpus mean clipped MSE: **0.063069**. Reported descriptively
throughout this file and this driver, per DESIGN.md's own framing -- the
CNN is in neither statistics family and is not part of the inference this
section's verdict rests on.

**Stated plainly, because "T is the unique winner" above is easy to
over-read: the CNN's mean MSE is LOWER (better) than `T`'s, not higher.**
`T`'s mean clipped MSE (recovered from the identity-gate contrast:
identity's 0.198856 minus the gate's observed mean 0.133233) is
**0.065623**; the CNN's is **0.063069** -- a difference of 0.002554
(~3.9% relative) in the CNN's favor, the CNN winning on 5,814 of 10,000
images against `T`'s 4,186. `DESIGN.md` never places the CNN in either
statistics family or in any named watched-for outcome, so no corrected,
locked test exists between it and `T` or any ridge condition --
`one_graph_wins`'s verdict is scoped to the four ridge-based evolved-graph
conditions only, and correctly says nothing about the CNN. A
**descriptive-only** paired bootstrap computed after the fact for this
write-up (20,000 resamples, `seed=42`, NOT part of the locked design, not
Holm-corrected, not a second confirmatory family) puts the CNN-minus-T
gap's 95% CI at **[-0.00276, -0.00235]**, entirely below zero -- offered
as a magnitude estimate, not a second locked verdict. The honest claim
this section supports is narrower than "T is the best model overall": `T`
is established as the best-performing condition **among the ridge-based,
phase-representation conditions**, via the locked procedure; a
categorically different, nonlinear model class does numerically better on
the same corpus and was never tested against it.

## Descriptive baselines, official test corpus

| condition | mean clipped MSE |
|---|---:|
| `T` (recovered from the identity-gate contrast) | 0.065623 |
| `raw_505` | 0.198856 |
| `raw_784` | 0.198856 |
| rescaled identity (`clip(x_t_clip / sqrt(0.5), 0, 1)`) | 0.246952 |
| CNN | 0.063069 |
| identity (`clip(x_t)`, hierarchical-gate baseline) | 0.198856 |

Raw-pixel ridge and the identity baseline sit within rounding of each
other (0.198856 vs. 0.198856) -- no sign here that the phase
representation is reconstruction-lossy relative to doing nothing to the
corrupted pixels directly; named watched-for outcome #5 (raw-pixel ridge
dominating every phase-based condition) is not what was observed.
Corruption diagnostics, full 10,000-image test corpus: pre-clip MSE
0.515888 (505-support) / 0.513274 (784); post-clip 0.198856 / 0.196559.

## What it took to get a clean run

Three attempts, `commit`s `6e7f811` -> `1b2e58b` -> `431d90a`, roughly 44
minutes of A100 time in total (not the 72-hour/$3,000 scale this would
cost without the resumability contract every earlier ladder stage
already established -- the first two attempts' correctly-computed
artifacts were reused, not recomputed, by the attempt that finally
completed). Two real bugs, both caught by the first real execution rather
than by review, both fixed and pinned as regression tests before the next
attempt:

1. `parent_map` never threaded `allow_test_split` through to
   `read_manifest` -- every parent this driver records is a test-side
   object, unlike stage 3's `parent_map`, which never touches the test
   split. Surfaced at `2_test_corruption`, 42 seconds in, before any real
   cost was incurred.
2. `step7_test_cnn` called `clipped_validation_per_image_mse` directly on
   raw `(n, 28, 28)` arrays without `as_image_batch`'s channel-first cast
   -- the exact mistake that function's own docstring names and warns
   against. Surfaced 663.65 seconds into the CNN step, after a full,
   successful three-seed retrain -- confirming the retrain-and-verify
   design works, only to fail one call later on shape. Pinned with a real,
   executable regression test (`test_cnn_test_evaluation_runs_on_raw_corpus_arrays`)
   that actually runs JAX/equinox on CPU, not a static check -- this was
   the one bug static review could not have caught.

A third defect was caught and fixed before it could matter: `step11_report`
originally wrote `official_result` unconditionally, including on a FAIL
verdict, which would have let the first attempt's halted, pre-inference
run permanently occupy the one-shot slot and block every subsequent
attempt via `refuse_if_official_result_exists` -- including a correct
one. Fixed to gate on `verdict == OK_SENTINEL` before the second attempt
ran; confirmed working when the second attempt's own failure correctly
wrote no `official_result`.

## Code and artifacts

`run_ladder_stage4.py`, `tests/test_stage2b_ladder_stage4.py`. Official
result: `stage2b/testsplit/stage4/common/official_result.json` /
`.txt`. Per-attempt reports (all three, including the two failures, kept
as history): `stage4_report_20260809T033701Z`,
`stage4_report_20260809T034509Z`, `stage4_report_20260809T041415Z`.
Test-side intermediate artifacts (corpus, corruption, encoded phases,
evolved thetas per graph, features per condition, the ridge test MSEs,
the CNN reproduction and test evaluation) all under
`stage2b/testsplit/stage4/`.

## Status of the investigation

This closes DESIGN.md's feasibility ladder. The question the whole of
Stage 2B was built to ask -- does runtime oscillator-network evolution on
a learned topology improve single-step active-support denoising relative
to the same representation before evolution -- has a locked, confirmatory,
positive answer on the official held-out test corpus: yes, and the
stronger "actual denoising" claim holds too. What remains open, tracked
separately rather than folded into this result: whether a denser or
extended ridge grid would move `T`'s floor-pinned alpha (the caveat
above); the two companion protocols `AUDIT_PROTOCOL.md` names
(`COMPANION_PROTOCOLS.md`'s ARM/x86 propagation stress set and the
`ABS_CONV_EPS` sensitivity table -- the latter now run, see this file's
own section below; the former not yet started); none of these results
were part of this evaluation and none were required to be, per
`AUDIT_PROTOCOL.md`'s own scoping of what the confirmatory test itself
needs. **The 150-vs-1200 amendment-impact audit itself has since run --
see "Stage 2B: the amendment-impact audit" below, which supersedes the
"has not run" framing this paragraph previously carried.** And INFRA's
still-open finding that the CNN has no stated consumer
in `DESIGN.md`'s own text, which this result treats as settled in the
"descriptive comparator" reading rather than resolving the ambiguity
INFRA named -- sharpened by this section's own descriptive finding that
the CNN's mean MSE is numerically lower than `T`'s, since a design that
gave the CNN a stated consumer would have had to reckon with that
directly rather than reporting it as a footnote. **"T is the unique
winner" names the winner among the ridge-based, phase-representation
conditions -- the comparison `DESIGN.md`'s statistics families and
`one_graph_wins` actually run. It is not a claim that `T` is the
best-performing condition in this file, and the CNN's descriptive number
says it is not.**

# Stage 2B: the amendment-impact audit -- no trigger

`AUDIT_PROTOCOL.md`'s own core apparatus, run 2026-08-09 (`AUDIT_OK`, A100,
one attempt, ~29.4 minutes of GPU). Quantifies the representational impact
of the encoder-budget amendment (150 -> 1,200 steps) made after ladder
stage 1's gate failure -- a prospective, disclosed, post-failure amendment,
not a preregistered component of the original design. Per the protocol:
this is not a model-selection knob and the 1,200-step budget stays frozen
regardless of what follows; what can change is the scope of the claim.

## Result: none of the three triggers fired, in either alpha regime

| quantity | 150-step | 1,200-step | change |
|---|---|---|---|
| primary contrast (`T` vs. `pre_evolution`, fixed alpha) | -0.0052076071 | -0.0052073732 | +2.339e-7 |
| primary contrast (`T` vs. `pre_evolution`, reselected alpha) | -0.0052076071 | -0.0052073732 | +2.339e-7 |

Both regimes: `primary_sign_reversal = False`; all four
`graph_sign_reversals` (`T`, `lattice`, `rewired`, `curr_random`)
`= False`; every one of the six pairwise comparisons has
`order_reversed = False` (so `resolved` is `False` throughout --
`resolved` requires an order reversal in the first place, the frozen
threshold is what would matter had one occurred). `combined_triggered =
False` -- the explicit OR over both regimes that `AUDIT_PROTOCOL.md`
requires ("either alpha regime triggers review... a reversal seen under
fixed-alpha alone, or under reselected-alpha alone, is sufficient").

Reading the number: the change in the primary contrast is real and
measured, not zero, and sits roughly 2.7 orders of magnitude above the
frozen analytic resolution threshold (`4.604761e-10`) --
`2.339e-7 / 4.604761e-10 ~= 508` -- but the change itself is ~2.3e-7,
about 0.0045% of the contrast's own magnitude (-0.0052). A clean double
bound: well-resolved (real margin above the threshold, not a coin-flip
near it) AND scientifically immaterial (four-and-a-half-thousandths of a
percent of the quantity it changes) -- the two readings do not trade off
against each other here. The amendment moved the number by an amount the
audit can resolve, in a direction and scale that changes no sign, no
per-graph verdict, and no pairwise ordering. The 150-step and 1,200-step
representations tell the same qualitative story about which evolved
graph does best; the 1,200-step budget's own confirmatory result (the
stage-4 section above) is not put in question by this measurement.

## Stage-1/2 historical cross-check: the new OOF machinery reproduces trusted numbers

`gates.toml`'s `binding_gate.9bc6f9e3808a` (out-of-fold per-image MSEs
must reproduce the already-stored fold-aggregate values from the stage-1
and stage-2 runs, not merely agree with a within-run recomputation on
synthetic data). Run against each stage's OWN stored `ridge_cv.json` and
OWN pre-amendment nine-decade alpha grid (not the 1,200-step run's
thirteen-decade `ALPHA_GRID` -- the two grids have different column
counts, so using the wrong one fails on shape before comparing a single
value). All ten (stage x condition) checks passed, at or near float64
dust:

| stage | max abs diff across 5 conditions |
|---|---|
| 1 (n=1,000) | 1.388e-17 -- 2.776e-17 |
| 2 (n=5,000) | 0.0 (exact, 4 of 5 conditions) -- negligible |

This is the gate that discharges before the 60,000-image OOF ridge step
is trusted to mean anything: new machinery, pinned against numbers this
project already trusted, before being run at a scale ten times larger
than anything it had been checked against.

## Feature distances: real but small, and the amendment's own magnitude

150-vs-1200 distance per condition (gauge-fixed, reference node 363),
NOT the pre-vs-evolved distance within one budget -- the quantity that
answers "how much did the amendment change the representation", not
"how much does evolution change the representation":

| condition | cos/sin Euclidean, median | p95 | max |
|---|---|---|---|
| `pre_evolution` | 0.001761 | 0.005419 | 0.05987 |
| `T` | 0.001269 | 0.003926 | 0.03857 |
| `lattice` | 0.001286 | 0.004027 | 0.04247 |
| `rewired` | 0.000180 | 0.000588 | 0.01049 |
| `curr_random` | 0.000250 | 0.000812 | 0.01420 |

`pre_evolution` shows the largest median distance of the five -- the
raw encoding itself differs more between 150 and 1,200 steps than the
EVOLVED representations do, for three of the four graphs (`rewired` and
`curr_random`, the two most strongly synchronizing graphs per Stage 2A's
order-parameter measurements, show markedly smaller distances than
`pre_evolution`; `T` and `lattice` sit closer to `pre_evolution`'s own
scale). Consistent with graph evolution partially washing out
budget-dependent encoding differences for the graphs that synchronize
most, rather than amplifying them -- offered as a descriptive reading of
this table, not a claim this audit's own triggers test for.

## The scope statement the protocol requires in any write-up, stated verbatim

Per-budget fold-fitted `StandardScaler`s are retained (production
preprocessing). Fixed-alpha therefore isolates **the effect of alpha
reselection** -- it does **not** completely isolate raw representation
change. A shared-scaler comparison is optional secondary work, not
required, and must not be presented as the primary probe.

## What it took

One attempt, `AUDIT_OK` on the first real run. A prior attempt (same
commit, before `LADDER_STAGE=5` was added to `stage2b_gcs.py`'s
`LADDER_STAGES` validation tuple) failed at the first artifact write in
the 150-step evolution step, after ~163s of real evolution compute --
caught non-fatally once already (inside the sizing probe's own publish
step, which logs and continues) before failing fatally; fixed, pinned
with a regression test, and re-run. The session tore down cleanly both
times; no billing leak.

Total wall-clock 1,764.1s (~29.4 min): bootstrap 17.1s, load train-side
artifacts 54.6s, topologies 1.9s, consume stage 3's 1,200-step
thetas/features 313.2s, production alphas 9.1s, sizing probe 8.0s,
evolve the 150-step budget (4 graphs x 60,000 images) 307.8s, 150-step
features 460.3s, stage-1/2 cross-check 66.0s, the 60,000-image OOF ridge
(both alpha regimes, both budgets, 5 conditions) 507.5s, feature
distances 14.0s, trigger verdict 4.3s. The sizing probe's own projection
(516.8s for the OOF ridge step, measured from one JAX SVD at production
shape before anything expensive ran) came in within 2% of that step's
actual 507.5s -- the probe's methodology validated by the run it gated.

Production (1,200-step) alphas the fixed-alpha regime applied
identically to both budgets: `pre_evolution=1000.0`, `T=1e-6`,
`lattice=1e-6`, `rewired=1e-5`, `curr_random=1e-5` -- `T` and `lattice`
at the grid floor, the same caveat the stage-4 section above already
carries forward from Phase B's amendment.

## Code and artifacts

`run_audit.py`, `stage2b_audit.py`, `tests/test_stage2b_audit.py`,
`tests/test_stage2b_audit_driver.py`. Run report:
`stage2b/train/stage5/common/audit_report_20260809T192835Z.json` / `.txt`.
Per-condition, per-budget artifacts (150-step evolved thetas and
features, the 60,000-image OOF results in both alpha regimes, feature
distances, trigger verdict) all under `stage2b/train/stage5/`. The
1,200-step thetas and features are Phase B's own persisted artifacts
under `stage2b/train/stage3/`, consumed rather than re-evolved, per
`PHASE_B_PLAN.md`'s Decision 4.

## Status of the investigation

The amendment-impact audit is closed: no trigger fired, in either alpha
regime, on any of the three frozen conditions. The 150-vs-1200
encoder-budget amendment has a real, measured representational effect,
and it is too small to change the sign, per-graph verdict, or pairwise
ordering the stage-4 confirmatory result (and Phase B's own ridge
result) depend on. Still open at the time of the audit write-up: the
`ABS_CONV_EPS` sensitivity table (now run separately — see
`run_abs_conv_eps_sensitivity.py`) and the ARM/x86 propagation stress set.
Protocol 1 has since run; its account is the next section.


# Stage 2B Companion Protocol 1: ARM/x86 propagation stress set — PROTOCOL1_OK

COMPANION_PROTOCOLS.md's consequence rule specifies interpretation review
before Stage 4. Stage 4 has already run and is locked. A Protocol 1 result
here is therefore disclosed as post-hoc relative to that ordering, following
the same sequencing-deviation precedent already established for the
amendment-impact audit.

## Verdict

**`PROTOCOL1_OK`** — stage-5 halt did not fire. Every graph's
`max |Δ Delta_g|` is strictly below `CONTRAST_THRESHOLD = 4.604761e-10`.
Largest stage-5 value: `curr_random` at `1.830e-12` (~252× below threshold).

Run id: `20260810T124247Z`.
Report: `stage2b/train/stage3/common/protocol1_propagation_report_20260810T124247Z.json`
Frozen ridge: `stage2b/train/stage3/common/protocol1_ridge_frozen_20260810T124247Z.npz`

## Construction

| component | detail |
|---|---|
| A | **regenerated** (top-100 max-abs encoding discrepancy on provisional B∪C∪D); `component_a_source = "regenerated"` |
| B | `true_count = 89`, `cap = 500`, `cap_applied = false`, `n_used = 89` |
| C | class floor ≥20 via lowest official indices |
| D | 20/class, `seed = 42` |
| `n_stress` | **287** |
| `indices_refined` | **false** (provisional B∪C∪D equalled final after A re-injection — expected regenerate path) |
| `indices_sha256` | `5ebded9ea78da1f66aa826683828c0990fbd57ab3b0c9f2682f320fa9c11ead6` |

ARM stress encodings are an **index-join slice** of production
`encoded_train_s1200.npz` (authoritative Phase-A ARM encode) — not a second
ARM realization. x86 stress encodings used unmodified
`encode_stage3_local.encode_indices` on Colab x86_64.

## Platforms

| role | machine |
|---|---|
| ARM encode (production Phase A, sliced) | Darwin arm64 |
| x86 encode (this protocol) | Linux x86_64 (Colab) |
| propagate (evolve + frozen ridge + report) | Darwin arm64 |

## Five-stage maxima

Framing at every table: **adversarial upper bound on cross-architecture
propagation; not a corpus sample.** Component A is selected for maximal
encoding-stage divergence.

### Stage 1 — encoding

| quantity | max \|ARM − x86\| |
|---|---|
| `theta_505` | `4.441e-16` |

Adversarial upper bound on cross-architecture propagation; not a corpus sample.
Encoding-stage sanity gate (`> 1e-12` refuse) did not fire. Historical Phase-A
spot-check max was ~3 ULP; this stress-set max is consistent with that scale.

### Stage 2 — evolved features (per condition, dim 1008)

| condition | max \|ARM − x86\| |
|---|---|
| `pre_evolution` | `4.441e-16` |
| `T` | `1.769e-15` |
| `lattice` | `1.554e-15` |
| `rewired` | `1.554e-15` |
| `curr_random` | `1.332e-15` |

Adversarial upper bound on cross-architecture propagation; not a corpus sample.

### Stage 3 — prediction (frozen ridge, same `(fit, scaler)` both arches)

| condition | max \|ARM − x86\| |
|---|---|
| `pre_evolution` | `6.661e-16` |
| `T` | `4.610e-12` |
| `lattice` | `1.488e-11` |
| `rewired` | `1.711e-11` |
| `curr_random` | `3.576e-11` |

Adversarial upper bound on cross-architecture propagation; not a corpus sample.
One `fit_final` per condition at production alphas from
`ridge_final_g13_88edf9ac.npz`; never per architecture.

### Stage 4 — per-image clipped MSE

| condition | max \|ARM − x86\| |
|---|---|
| `pre_evolution` | `2.776e-17` |
| `T` | `9.975e-14` |
| `lattice` | `4.455e-13` |
| `rewired` | `5.483e-13` |
| `curr_random` | `1.830e-12` |

Adversarial upper bound on cross-architecture propagation; not a corpus sample.

### Stage 5 — Δ_g = MSE_evolved − MSE_pre (halt stage)

| graph | max \|Δ_g,ARM − Δ_g,x86\| | exceeds `4.604761e-10`? |
|---|---|---|
| `T` | `9.975e-14` | no |
| `lattice` | `4.455e-13` | no |
| `rewired` | `5.483e-13` | no |
| `curr_random` | `1.830e-12` | no |

Adversarial upper bound on cross-architecture propagation; not a corpus sample.

Halt rule (frozen): any graph **strictly greater than** threshold →
`PROTOCOL1_HALT`. Equality does not halt. None exceeded.

## Scope limitation

Component A is adversarial **for encoding-stage** discrepancy only
(`rank_discrepancy_indices` ranks on `theta_arm` vs `theta_x86`). Ranking on
post-evolution divergence would require evolving the full candidate population
first and would defeat a small stress subset. A clean `PROTOCOL1_OK`
establishes “no anomalous propagation on inputs adversarial for encoding
divergence”; it does **not** independently establish that evolution-stage
numerical sensitivity was adversarially stress-tested on its own terms.
Plausibly correlated (evolution’s inputs are the encodings) but not guaranteed.

## Artifacts

| kind | object |
|---|---|
| stress indices | `stage2b/train/stage3/common/protocol1_stress_indices.npz` |
| ARM stress encode | `stage2b/train/stage3/common/protocol1_encoded_stress_arm_s1200.npz` |
| x86 stress encode | `stage2b/train/stage3/common/protocol1_encoded_stress_x86_s1200.npz` |
| frozen ridge | `stage2b/train/stage3/common/protocol1_ridge_frozen_20260810T124247Z.npz` |
| report | `stage2b/train/stage3/common/protocol1_propagation_report_20260810T124247Z.json` |
| theta_T / features | under `stage2b/train/stage3/{pre_evolution,evolved_*}/protocol1_{theta_T,features}_{arm,x86}.npz` |

Driver: `run_arm_x86_propagation_stress.py`. Pure helpers:
`stage2b_audit.capped_positive_delta_indices`, `rank_discrepancy_indices`,
`max_abs_difference`, `evaluate_propagation_halt`, `propagation_stage_maxima`.
Tests: `tests/test_stage2b_arm_x86_propagation.py`. Make:
`stage2b-protocol1-arm-construct`, `stage2b-protocol1-x86-encode`,
`stage2b-protocol1-propagate`.

