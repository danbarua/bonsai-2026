Simplified Technical English version of `experiments/stage2b_denoising/FINDINGS.md`.

# Stage 2B: Feasibility Ladder Stage 1

**Status: this is a mechanical check only, exactly as `DESIGN.md` itself
frames it. This is NOT a scientific result, and it must not be read as
one.** Its job is only to confirm the pipeline runs correctly, start to
finish, at n=1,000 images, before scaling up further. The ridge and
statistics numbers in the final section below are only a check that the
machinery works, using training data the model has already seen — they
are not the locked confirmatory result, and they are labeled as such
throughout — following the same discipline Stage 2A used in its own
feasibility-stage findings.

## Scope

1,000 official KMNIST training images, drawn using the locked, nested,
stratified partition (random seed 42; the `stage1_indices` are a prefix
of the 5,000-image stage-2 draw). These images are corrupted using the
locked forward corruption process (seeded from
`SHA256(split:index:42)`, with `alpha_bar=0.5`), keyed against the
original dataset indices — this is verified directly, not just assumed:
recomputing `epsilon_for` from each drawn image's ORIGINAL index, and
re-deriving `forward_corrupt`, reproduces the test corpus bit-exactly,
at three spot-checked rows. This ran entirely on a Colab A100 GPU, via
`run_ladder_stage1.py`, which fetches one pinned commit of this code
repository rather than being uploaded together with its dependencies,
and writes every artifact through `ensure_artifact`, so that a session
that dies loses at most one step's worth of work.

## Part 1 — first run: an honest FAILURE, exactly as designed

The encoder-on-noisy-inputs gate's first run on real, majority-censored
KMNIST data (code commit `7723b96`, `ENCODER_STEPS=150`) failed the
pre-registered `rho <= 10` threshold:

| quantity | clean | noisy |
|---|---|---|
| median final-Delta | 2.177485e-07 | 3.698480e-05 |
| p95 final-Delta | 9.971726e-07 | 1.784018e-04 |
| non-finite phases / deltas | 0 / 0 | 0 / 0 |

**rho = 169.851**, against a threshold of 10 — roughly 17 times over the
limit. There were zero non-finite values anywhere: this was a genuine
ratio failure, not a numerical blow-up.

**What this failure was, and what it was not.** This original stage-1
failure was not itself caused by a numerical floor. It showed only that
150 iterations were not enough steps for noisy inputs, under the
original relative-ratio gate. The noisy median, 3.698e-05, and its 95th
percentile, 1.784e-04, both sit far above any numerical floor — seven
orders of magnitude above the `ABS_CONV_EPS` value of 1e-12 adopted
later, and nine or more orders of magnitude above the float64 numerical
noise Part 2, below, observes at 1e-14 to 1e-16. The numerical-floor
problem identified in Part 2 is a separate defect with a separate cause:
it is a property of the DIAGNOSTIC trajectory at 600 steps and above,
where both series have already decayed down into numerical noise. These
two problems are not the same failure, and neither one implies the
other.

What the gate's own summary reports is the median, the 95th percentile,
and the non-finite counts — it does not report a per-image count of
exactly-zero versus nonzero final-Delta values, and it does not report a
maximum value. The per-image `delta_clean` and `delta_noisy` arrays are
stored inside the `encoder_gate.npz` artifact, so these missing counts
can still be recovered from that artifact later.

Following `DESIGN.md`'s locked stop-gate rule, this halted the whole
stage immediately (result: `STAGE1_FAIL`, the session was torn down,
nothing kept billing); steps 5 through 10 never ran at all. This was
confirmed directly against the live storage bucket listing, before any
further work happened: 11 objects existed under `stage2b/train/stage1/`,
and none existed for `theta_T`, `features`, `ridge_cv`, `ridge_final`, or
`stats_smoke` — the claim that nothing downstream had run was actually
checked, not just assumed.

## Part 2 — diagnosis (not part of the locked pipeline)

This was investigated in `diagnose_encoder_gate_failure.py`, which runs
entirely on the CPU (`_local_converged_phases` has no GPU dependency, so
running this costs nothing in cloud fees). It rebuilds the exact stage-1
test corpus and its corruption locally, and checks this reconstruction
against the failed run's own reported identity-baseline MSE, bit-for-
bit, before trusting anything computed from it — confirmed exact, with a
relative difference of `0.000e+00`, and the independently recomputed
150-step row exactly reproduced the cloud run's own median deltas and
rho value, to full reported precision.

**Measurement 1, the convergence curve** (five step counts, computed on
the full 784-pixel grid):

| steps | median clean | p95 clean | median noisy | p95 noisy | rho |
|---|---|---|---|---|---|
| 75 | 8.663e-05 | 2.267e-04 | 1.298e-03 | 2.977e-03 | 14.98 |
| 150 | 2.177e-07 | 9.972e-07 | 3.698e-05 | 1.784e-04 | 169.9 |
| 300 | 1.538e-12 | 2.030e-11 | 2.945e-08 | 6.812e-07 | 1.915e4 |
| 600 | **0.0** | **0.0** | 1.776e-14 | 1.043e-11 | 17.76 |
| 1200 | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |

The noisy final-Delta value decays steadily, in a shrinking pattern: by
1,200 steps, both its median and its 95th-percentile value reach an
exact float64 zero — the same fixed point clean images reach. There is
no floor above a meaningful scale: on censored inputs, the encoder does
reach numerical convergence below any practically relevant tolerance, it
simply takes longer to get there.

**Two quantiles are not the whole distribution.** The median and the
95th percentile are the only two order statistics in this table, and
comparing median-to-median and p95-to-p95 ratios does not, by itself,
prove a shift across the WHOLE distribution — it only shows a shift at
these two specific points. Concretely, the 1,200-step row supports the
claim "the median and the 95th percentile are exactly zero across these
1,000 images," but it does not support the stronger claim "every single
one of these 1,000 images is exactly zero": no per-image count of
exactly-zero values, and no maximum value, was reported at this stage.
The 54,000-image measurement recorded further down in this document
shows that this stronger claim is actually false, at larger corpus
sizes. The per-image arrays are stored in the stage-1
`encoder_gate*.npz` artifacts, so these unreported counts remain
recoverable from the historical record.

**A second, independent problem is visible in the same table.** The rho
column does not move steadily in one direction (14.98, 169.9, 1.915e4,
17.76, 0.0), because the clean and noisy series cross their own float64
numerical floors at different step counts. At 600 steps, the clean
series' median had already reached exactly 0.0, while the noisy
series' median sat at 1.776e-14 — nine orders of magnitude below the
smallest meaningful measured value anywhere (2.177e-07) — yet the gate
reported **FAILURE at rho=17.76**, because `max(0.0, 1e-15)` silently
turned what should be a ratio test into an absolute test against the
1e-15 floor value. A ratio between two quantities that have both already
decayed into numerical noise only measures which one hit that noise
floor first, not whether the underlying process actually converged.

**Measurement 2, how far the state moved compared with the scale between
images** (noisy series, full 784-pixel grid, using the same reduction on
both sides — the maximum absolute wrapped difference):

- Movement from 150 to 600 steps, n=1,000: median **7.573e-04**, 95th
  percentile 4.702e-03
- Typical distance between different images at 150 steps, n=5,000
  pairs: median **1.806**, 95th percentile 2.146
- **Ratio: 0.0004**

The phase field has, for all practical purposes, already stopped moving
— by a factor of roughly 2,400 — relative to the scale that separates
one image from another, well before the final-Delta metric alone would
say so. This independently confirms Measurement 1's reading: this is
genuine, if slow, convergence, not a qualitatively different pattern on
noisy inputs.

## Part 3 — a disclosed post-lock amendment

Full details are in `DESIGN.md`'s "Encoder-on-noisy-inputs gate" section
and its Review history section. **The 1,200-step budget, and the
absolute-convergence rule, were both prospective amendments, made after
stage 1's failure and before any downstream confirmatory evaluation ran.
They were not part of the original, preregistered design.** These are
two separate changes, answering two separate failures:

1. **`ENCODER_STEPS` was raised from 150 to 1200**, applied uniformly at
   every encoding site, for both clean and noisy images identically.
   This answers the Part 1 failure: 150 iterations was simply not
   enough budget for noisy inputs. This follows the same pattern as
   Stage 2A's own precedent of raising `max_iter` from 1,000 to 10,000:
   halt honestly, diagnose the mechanism, amend with disclosure, and
   re-verify.
2. **The gate formula gained an absolute-convergence escape**: it now
   PASSES if `rho <= 10`, OR if both medians are already below
   `ABS_CONV_EPS=1e-12` (five or more orders of magnitude below the
   smallest meaningful measured Delta value, and well above the observed
   float64 numerical noise). The automatic failure on any non-finite
   value is unconditional and unaffected by this change. This answers
   the separate Part 2 defect — the ratio gate hitting a numerical floor
   at 600 steps and above — and it would NOT have rescued the Part 1
   failure, since both medians there sit far above 1e-12.

### Why S\* = 1,200 was chosen: three separate reasons, kept separate

**(a) The reasoning as it was actually given, at the time.** 1,200 was
the only step count, in the five-point scan, at which both reported
statistics — the median and the 95th percentile, on both the clean
series and the noisy series — sat at an exact float64 zero. At the time,
the team read this as the maximum possible distance from the fragile
crossover zone where clean and noisy cross their own numerical floors at
different step counts — the same zone that produced the spurious
FAILURE at 600 steps, at rho=17.76. No finer scan between 600 and 1,200
steps was run, on the reasoning that a smaller, untested value might
land inside that same fragile zone.

The decision rule that selected S\*=1200 was stated to give the same
verdict, not merely to be correct. An earlier, looser reading of the
rule — "some S\* brings noisy within 10x of clean at 150 steps" — was
caught and rejected before being used. That looser rule selects S\*=300,
which then immediately fails its own same-step re-run test, at
rho=1.915e4. The corrected rule instead compares same-step values, and
requires both series to have genuinely converged. This corrected rule
passes at S\*=1200 under EITHER the original or the corrected reading, so
the correction did not actually change which outcome was selected.

**(b) The more durable reasoning, identified in hindsight.** 600 steps
is the first measured budget that satisfies the amended absolute-
convergence rule: the noisy median is 1.776e-14 and the clean median is
exactly 0, both below `ABS_CONV_EPS=1e-12`, whereas 300 steps' clean
median of 1.538e-12 is not below it. 1,200 steps is a conservative,
twofold safety margin over this first passing budget, frozen before any
downstream model fitting happened. This reasoning can be derived
directly from the scan table above, and it does not rely on reason (a)'s
exact-zero premise at all.

**(c) Later evidence narrowed reason (a)'s premise.** Reason (a) rests
on exact zero being a genuine property of the fully converged encoder,
as the two order statistics reported by the n=1,000 scan appeared to
show at the time. This premise did not survive testing at larger scale,
and it degraded steadily as corpus size grew: stage 2's own n=5,000
encoding run already recorded a nonzero maximum value (2.22e-14), and
the 54,000-image encoding run recorded further down in this document
found 79 images with a nonzero final-Delta value, with a maximum of
2.468e-10. Reason (a) is preserved here as the historical record of what
was actually reasoned at the time, given the evidence available then —
not as a claim that still stands unmodified today. Reason (b) is
untouched by this correction: it turns only on the medians, which remain
exactly zero at every corpus size where a median has actually been
recorded, both at n=1,000 and at n=54,000.

A second, independent bug was found and fixed during the same
investigation, by tracing "every encoding site" through the actual code
call graph, rather than assuming the constant change alone was
sufficient: `_encode_one`'s returned theta value passed through
`stage2a_core.encode_and_restrict`, which has no `steps` parameter at
all, and is hardwired to `_local_converged_phases`'s own bare default
value of 150 — a Stage 2A convention, unrelated to Stage 2B, but load-
bearing for roughly 14 of Stage 2A's own already-verified pipeline
files. The final-Delta value correctly reflected the requested step
count, but the returned theta value silently did NOT — this was
invisible only because `ENCODER_STEPS` also happened to equal 150 at the
time, and it became a real defect the moment that stopped being true.
The exact same bug, independently, was also present in the driver's own
step-5 sanity check, which compared against this same hardwired
function. Both were fixed at the source: `_local_converged_phases` is
now called directly, at the caller's own requested step count,
everywhere in Stage 2B's pipeline.

A new working rule, `CLAUDE.md` principle 23, was added as a result: a
ratio gate between two quantities that each decay to a numerical floor
measures only which one hit that floor first, not whether the underlying
mechanism actually converged.

## Part 4 — re-run: the full pipeline completes

Code commit `32b6688`. The gate now passes robustly, not just marginally:

```
encoder-on-noisy-inputs gate: PASS
  median final-Delta clean : 0.000000e+00
  median final-Delta noisy : 0.000000e+00
  rho                      : 0 (threshold 10)
  absolute convergence     : True (both medians < 1.0e-12)
  non-finite phases/deltas : 0/0, 0/0
```

Both medians sit at exact float64 zero, exactly matching the earlier
diagnostic's own 1,200-step measurement. What the gate's summary reports
is the medians, the 95th-percentile values, and the non-finite counts
(0/0 on both sides); it does not report a per-image count of exactly-
zero versus nonzero final-Delta values, and it does not report a
maximum — a median of zero is not itself a claim about every individual
image; see the 54,000-image tail measurement further below. What this
particular run establishes is numerical convergence below any
practically relevant tolerance — not exact convergence for every single
image. Because the gate passed, the driver continued automatically
through steps 5 through 10, within the same run, exactly as the
amendment specified — a passing verdict satisfies the halt rule on its
own, with no separate authorization needed. **This is the first time any
Stage 2B code has run graph evolution, ridge model fitting, or the
statistics machinery against real data of any kind.**

**Step 5 (restrict)**: the new, corrected sanity check passed on this
real run — confirming that the earlier fix (comparing against a fresh
encoding at the gate's own actual step count, rather than the old,
stale, hardwired-150 reference) is correct in real production use, not
just inside the isolated unit test that first caught the bug.

**Step 6 (evolution)**: all four canonical graphs (`T`, `lattice`,
`rewired`, `curr_random`) ran through batched JAX evolution —
**0 solves failed, out of 1,000, for every graph** — and the CPU
reference cross-check on image 0 succeeded for all four graphs (using
the `RK45` solver on the first attempt, with no recovery step needed).
This step took 10.0 seconds total.

**Step 7 (features)**: all seven ridge conditions were built —
`pre_evolution`, `T`, `lattice`, `rewired`, `curr_random` (each 1008
values), `raw_505` (505 values), and `raw_784` (784 values). This step
took 4.4 seconds.

**Step 8 (ridge)**: cross-validation and the real-data ridge equivalence
check ran for the first time ever on non-synthetic features (every
prior equivalence check in this project had used synthetic data only).
All seven conditions passed **with an enormous margin**:

| condition | selected alpha | max abs pred diff | tol | alpha agrees |
|---|---:|---:|---:|:---:|
| raw_505 | 1000 | 6.928e-14 | 1e-8 | yes |
| raw_784 | 1000 | 7.511e-13 | 1e-8 | yes |
| pre_evolution | 1000 | 1.811e-13 | 1e-8 | yes |
| T | 100 | 8.159e-13 | 1e-8 | yes |
| lattice | 10 | 6.465e-13 | 1e-8 | yes |
| rewired | 1000 | 1.151e-12 | 1e-8 | yes |
| curr_random | 1000 | 6.568e-13 | 1e-8 | yes |

Every single difference is four or more orders of magnitude below the
1e-8 tolerance. The scaler-centering tolerance that scales with `n`
(`1e-9 * (n/1000)**0.5`, this project's most recent post-lock amendment
before this one) held, with a wide margin, at every condition —
`curr_random`, the condition this tolerance was specifically raised to
protect, sits at a `margin_ratio` of 0.076 (meaning `||mean(X)||` is at
about 7.6% of the tolerance), close to the 12.7x margin the amendment's
own derivation had predicted. Worth recording plainly, not smoothed
over: both the evolved and the pre-evolution conditions' fold condition
numbers are extreme (pre_evolution about 6e14, T and lattice about
1.2-1.6e14, rewired and curr_random about 2-7e13, compared against
raw_505 and raw_784's 14-490) — exactly the regime the JAX-SVD ridge
implementation was designed and stress-tested for, and the equivalence
numbers above show it holding correctly under real, not merely
synthetic, ill-conditioning.

**Step 9 (statistics smoke test)**: the full statistics machinery
(primary paired bootstrap, both Holm-corrected families, and the
branched winner rule) was exercised end-to-end against real, in-sample
ridge output, for the first time — no earlier test had exercised the
connection between ridge output and the statistics code before this
run. The output artifact's first line, quoted exactly, reads:
`SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE,
NON-INFERENTIAL, NOT A RESULT`. The mean per-image clipped MSE, in-
sample, is recorded for all conditions below, because the design allows
recording it — not because these particular n=1,000 in-sample numbers
support any scientific claim:

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

Every learned or raw-pixel condition beats the identity baseline by a
wide margin, exactly as expected of any fitted model compared against
"return the input unchanged." No ordering between the other conditions
here should be read as evidence of anything at all — in-sample MSE at
n=1,000, with no held-out split, is not the locked confirmatory design
(which uses a 20,000-resample paired bootstrap, against the official
10,000-image test set), and `DESIGN.md` explicitly scopes this step to
exercising the machinery only, not to producing any actual result.

## Runtime

**596.7 seconds (9.9 minutes) total**, measured end-to-end on the A100
GPU, not merely projected:

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

The encoder gate dominates, at 71% of the total wall-clock time — this
is expected, given 1,200 iterations of a local update per image, run
twice per image (an unavoidable byproduct of the `_encode_one` fix,
which is actually down from three passes before the fix). Every other
step finished in seconds.

## Code and artifacts

`stage2b_encoder_gate.py` (the gate, now amended), `run_ladder_stage1.py`
(the driver), `diagnose_encoder_gate_failure.py` (the investigation
script, diagnostic-only), and `stage_kmnist_inputs.py` (one-time input
staging). Tests: `tests/test_stage2b_encoder_gate.py`,
`tests/test_stage2b_ladder_stage1.py`. Every stage-1 artifact lives in
the public-read bucket `bonsai-2026-stage2b-cache`, under
`stage2b/train/stage1/`. The pre-amendment `encoder_gate.npz` file
(the 150-step FAILURE) remains alongside the post-amendment
`encoder_gate_s1200.npz` file (the PASS) as the historical record of the
first real run — it is not deleted, and it is not silently replaced.

## Next step

Feasibility ladder stage 2 (the 5,000-image development subset) —
described below.

# Stage 2B: Feasibility Ladder Stage 2

**Status: this is a mechanical check only, using the same framing as
stage 1.** Its job is to measure runtime and feature validity at five
times the scale, to check the production SVD's own condition-number
diagnostic, to check how the ridge grid behaves, to run the ladder's
second real-data ridge equivalence check, and to run the first CNN
training on real data. The CNN-versus-identity numbers and the in-sample
statistics numbers below are mechanical development reporting, not a
scientific result — `DESIGN.md` explicitly scopes this stage that way.

## Scope

5,000 official KMNIST training images, drawn using the same nested,
stratified draw that stage 1's 1,000 images are a prefix of (random seed
42) — this was checked explicitly, not merely trusted from the way the
draw was constructed, and further checked to be bit-exact against
stage 1's own cached corruption artifact, at the three shared prefix
rows. These images were corrupted using the same locked forward
corruption process. Graph topologies and staged KMNIST inputs were
reused directly from stage 1's cached objects, not rebuilt or re-staged.
This ran on a Colab A100 GPU, via `run_ladder_stage2.py`, which has the
same architecture as the stage-1 driver.

## A false start, fixed before any real cost was incurred

The first attempt crashed immediately, at the point the module itself
loaded, before `main()` even started: `NameError: name '__file__' is not
defined`. A refactor that imported `KMNIST_FILES` from
`run_ladder_stage1` at module-load time (to avoid duplicating this
dictionary) relied on `os.path.dirname(os.path.abspath(__file__))` to
locate that file — and this fails under
`mighty-colab exec -f script.py`: the file's own TEXT is transmitted
directly into an already-running kernel cell there, rather than being
run as a script or imported as a proper module, so `__file__` is never
even defined in that environment. Beyond that, `run_ladder_stage1.py`
does not even exist anywhere on the executed kernel's own filesystem
until `bootstrap_repo()` clones the repository — and that cloning
happens INSIDE `main()`, after every module-load-time statement has
already run. This was confirmed against the live session list, and
against the storage bucket, before any fix was even written: zero
objects existed under `stage2b/train/stage2/`, and no session was left
running — the crash cost only provisioning time and package-install
overhead, nothing more.

The fix was to revert back to plain duplication (matching stage 1's own
pattern), with two new, durable safety checks added directly as a
result of this failure: a static check confirming that neither driver
ever references `__file__` anywhere, and a dynamic check that runs each
driver's actual source code inside a namespace that has no `__file__`
key at all, exactly reproducing the real execution environment — which
an ordinary, import-based local test could not do, since Python's own
import system correctly sets `__file__` for a properly imported module.
This exact gap is how this bug reached a billing A100 GPU, uncaught by
every other local check that had run before it.

## Result: complete, all ten steps, on the first attempt after the fix

**`STAGE2_OK`.** Code commit `a84fac9`. Total wall clock:
**1,722.0 seconds (28.7 minutes)** — well under the 90-minute budget
reserved for this stage. There were no non-finite features anywhere, and
the corruption cross-stage spot-check matched stage 1's own cache bit-
exactly, at all three shared rows.

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

**Step 6 (evolution)**: all four graphs, **0 solves failed, out of
5,000, for every graph**; the CPU reference cross-check succeeded for
all four. **Step 8 (ridge)**: all seven conditions passed the ladder's
SECOND real-data equivalence check, with every difference four or more
orders of magnitude below the 1e-8 tolerance (ranging from 1.2e-14 to
1.1e-12), every alpha selection agreeing between JAX and sklearn, and no
condition landing at a grid edge. The scaler-centering tolerance that
scales with `n` held at every condition (see named item 2, below).

## Named items, described on their own terms

**(1) Measured encode cost, and the stage-3 projection.**
**218.28 milliseconds per image** at 1,200 steps, on a single worker,
at n=5,000 (1,091.4 seconds measured in total). Projected linearly to
stage 3's 54,000-image fit side, this gives: **11,787 seconds
(3.27 hours)** for encoding alone — by far the dominant cost in the
whole pipeline at that scale, and the number that should drive any
stage-3 planning decision about parallelizing this step, or otherwise
restructuring it. This is not validated at that scale; it is only a
linear projection from a single measurement, and it is stated as such.
This particular projection was made against a 54,000-image stage-3
population; the stage-3 populations are actually fixed differently — see
"Population roles" further below, in the Phase A section.

**(2) The `curr_random` centering margin, versus the amendment's own
prediction of about 0.075.** Measured: `margin_ratio = 0.0807`
(`||mean(X)|| = 1.804e-10` against a tolerance of `2.236e-09`, at this
fold's n=4,000 training rows) — this closely matches the amendment's own
derivation, and confirms it now holds at real n=5,000 scale, rather than
only at the n=1,000 rung it was first measured against. Every other
condition's margin sits between 0.0009 and 0.081 (`T` and `lattice` are
the tightest; `curr_random` remains the widest of the seven, consistent
with it being the condition the amendment's tolerance was specifically
set to protect).

**(3) The condition-number table, for all seven conditions**, taken
directly from the production SVD's own singular values (`fold_cond`),
with no second, separate `cond()` call:

| condition | mean fold cond(X) | alpha |
|---|---:|---:|
| raw_505 | 4.25 | 1000 |
| raw_784 | 5.34 | 1000 |
| pre_evolution | 171.0 | 1000 |
| curr_random | 1.49e6 | 100 |
| rewired | 1.87e6 | 10 |
| T | 4.78e6 | 1 |
| **lattice** | **4.82e8** | 1 |

Recorded here plainly, without further interpretation: `lattice`'s
condition number is roughly two orders of magnitude worse than `T`'s,
and `T` in turn is worse than `rewired` and `curr_random` by a factor of
roughly 3. All seven conditions passed the equivalence check regardless
— this diagnostic is purely descriptive, not a gate on its own, exactly
as it was designed to be.

**(4) CNN: best seed, best epoch, and validation MSE versus identity.**
The best seed was **0** (`best_epoch=94`, meaning it ran the full 100
epochs without ever early-stopping), reaching a **clipped validation
MSE of 0.063678**, against the identity baseline's own
**0.199770**, on the same locked 6,000-image validation partition — a
substantial, consistent margin. All three seeds landed within
0.0637-0.0642 of each other (seed 1: 0.064229, early-stopped at epoch
63 of 74; seed 2: 0.064156, early-stopped at epoch 81 of 92). The CNN's
total wall-clock time across all three seeds was 99.5 seconds. This is
labeled explicitly, matching the design's own framing: it is a mechanical
sanity check ("does the CNN beat doing nothing at all, on its own
validation data"), it is NON-INFERENTIAL, and it is not a locked
comparison — this is the first time this project has ever trained the
CNN against real data.

## Statistics smoke test (in-sample, non-inferential)

This step ran, and was not skipped: it was projected to take 54.3
seconds, based on stage 1's own measurement, and actually took 67.1
seconds — the linear projection understated the real cost by about 24%,
which is worth noting as a fact about the projection's own accuracy,
rather than as a concern (the decision threshold has 60 seconds of
margin below stage 1's own measured value, and this rung's real cost
still landed close to that threshold). The output artifact's first line,
quoted exactly, matches stage 1's:
`SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE,
NON-INFERENTIAL, NOT A RESULT`.

## Stage-3 projections, per pipeline stage (linear projections, not yet validated)

| stage | measured at n=5,000 | projected at n=54,000 (10.8x) |
|---|---:|---:|
| encode | 1,091.4s | 11,787s (3.27h) |
| evolution | 41.0s | 443.2s (7.4min) |
| ridge | 305.5s | 3,299.7s (55.0min) |
| CNN | 99.5s | 1,074.9s (17.9min), the basis for this differs — see the caveat below |

Every row here was projected against a 54,000-image stage-3 population.
The ridge cross-validation and final-fit corpus is actually the full
60,000-image training side (see "Population roles" below), so the
ridge row's own population is not actually the one that will be fitted.
This document does not state a revised projection here, because
rescaling one row's projection by a simple population ratio is exactly
the kind of cross-stage extrapolation this project has already been
caught out by before.

There is deliberately no single blended rate for all steps here. The CNN
projection is explicitly weaker than the others: CNN cost scales with
epochs times batches, not simply with `n`, and early stopping means the
epoch count itself is not fixed purely by corpus size — this row scales
wall-clock time linearly only as a first approximation, and it is not a
validated model of how CNN training cost actually scales.

## Code and artifacts

`run_ladder_stage2.py` (the driver). Tests:
`tests/test_stage2b_ladder_stage2.py`. Every stage-2 artifact lives in
the public-read bucket, under `stage2b/train/stage2/`; graph topologies
and KMNIST inputs are read directly from `stage2b/train/stage1/`
(reused, not duplicated).

## Next step

Feasibility ladder stage 3 — Phase A is complete at 54,000 images, and a
regeneration at 60,000 images is pending; see below.

# Stage 2B: Feasibility Ladder Stage 3, Phase A (encoding)

**Status: this covers Phase A only, and at the wrong population size —
this run encoded 54,000 images. It has since been SUPERSEDED by the
60,000-image regeneration recorded in the next section, which is the
authoritative Phase A artifact. Phase B (evolution, ridge, CNN) has NOT
run, and stage 3 has still produced no denoising number of any kind.**
This section is recorded here because it is a complete, measured unit of
work, with a durable saved artifact — and because this project's own
Part 4 lesson is that a result that is never written down does not
survive the session that produced it. Its numbers still stand: the
regeneration reproduced every one of them bit-exactly.

## The two-phase split, and why encoding moved off the GPU

Stage 2 measured encoding at 1,099 seconds, out of a total of 1,722
seconds — 64% of the whole run, and the only genuinely CPU-bound step in
the entire pipeline. Evolution, ridge, and the CNN are the steps that
actually use the A100 GPU. Running the encode step inside a paid GPU
cloud session would leave a metered A100 GPU idle for most of stage 3's
wall-clock time, at ten times stage 2's corpus size.

Stage 3 therefore splits into two phases: **Phase A** (test corpus,
corruption, encode, restrict) runs on local CPU cores, and writes only
the encoded array to GCS; **Phase B** reads that array, and regenerates
the corruption and clean target values itself, in-session, since both
are deterministic and cheap to compute. This means 218 MB crosses the
network boundary instead of 775 MB, for about 2.5 minutes of cloud CPU
time.

This is a disclosed post-lock amendment, not a quiet deviation from the
design — `DESIGN.md`'s "Computational strategy" section and its Review
history section both carry this disclosure. The rule this amends states
that generation happens "entirely in the cloud environment... never
round-tripped through local upload," and it names its own reason, in
the very same sentence: Stage 2A's Colab **session upload** size limit.
A direct write from the local machine to GCS never touches that upload
mechanism at all; it uses the same client library, and the same
chunked, checksum-verified transfer method in `stage2b_gcs.py`, that a
cloud-side write already uses, and `stage_kmnist_inputs.py` has already
moved KMNIST data this same way, for both earlier ladder rungs.

## Measured before acting on the decision: cross-chip-type reproducibility

Encoding on this Mac put the encoder on Apple Silicon chips for the
first time; stages 1 and 2 had both encoded on Colab's x86 chips. Rather
than simply assume the two chip types would produce equivalent results,
the team ran three comparisons on the same images first:

| comparison | result |
|---|---|
| same machine, encoding run twice | **bit-exact** |
| two different Colab sessions, same 1,000 images | **bit-exact**, matching on 784,000 of 784,000 coordinates |
| Mac (ARM chip) versus Colab (x86 chip), same images | 93.4% of coordinates identical; **maximum difference of 3 ULP** (4.441e-16), mean difference 0.07 ULP, maximum relative difference 3.98e-16 |

The middle row here matters, and it nearly went unchecked: an initial
assumption that "Colab's own hardware already varies too, so using a
local machine loses nothing extra" turns out to simply be false — two
separate Colab sessions produced byte-identical encodings of the same
images. The actual difference is specifically caused by the chip
architecture, not by which session ran the code.

This difference cannot grow larger over time. The encoder pulls values
toward a fixed point (a stable value it settles into): the measured
residual against the 1,200-step result falls from 8.370e-07 at 300
steps, to 8.062e-13 at 600 steps, to exactly 0.0 at 1,200 steps. Both
chip types settle on the SAME fixed point; only a minority of
coordinates land on a neighboring representable float64 value instead of
the identical one. Downstream, these phase values feed into an ODE
solver set to a relative tolerance of 1e-6, which is eight or more
orders of magnitude larger than this difference.

The team accepted this deliberately, with full disclosure, applying the
same standard the project already uses for accepting that GPU-run
logistic regression (using the cuML library) is not bit-for-bit
reproducible, and that this is fine at the level the project's actual
claims are made (see `docs/PROJECT_MEMORY.md` Part 4).

## Result

**54,000 fit-side images, 1,200 steps, 9 workers (Darwin, ARM64 chip,
10 cores in total).** This particular run encoded only the CNN fit side,
based on the reasoning that the CNN only ever consumes validation images
as raw, corrupted grids, and that ridge selects its alpha value using
internal cross-validation on the fit side alone, so an encoded
validation array would be an artifact nothing would ever actually read.
That reasoning turns out to be wrong for the ridge path specifically:
ridge cross-validation and the final refits both run on the full
60,000-image training side, so the 6,000 locked-validation images must
also be encoded. Phase A will be regenerated at 60,000 images; the
measurements below are specific to this original 54,000-image run.

**Population roles.** `DESIGN.md:479` defines "full training" at stage 3
as the 54,000 fit plus 6,000 locked validation images combined, and
`:492`'s compute-cost table reads "~48-60k x 1008", where 48,000 equals
0.8 times 60,000 — a five-fold cross-validation training portion of
60,000 images. The 54,000-versus-6,000 split governs only CNN weight
fitting and model selection: the 6,000 images are held out from CNN
gradient updates only, not from training-side analysis in general.

| role | n |
|---|---:|
| official training corpus | 60,000 |
| CNN weight-fit subset | 54,000 |
| CNN validation / model-selection subset | 6,000 |
| **ridge CV and final-fit corpus** | **60,000** |

This is frozen as Freeze 2 in `AUDIT_PROTOCOL.md`, which is the
authoritative source for these roles.

| quantity | measured |
|---|---:|
| encode wall-clock | **475.3s (7.9 min)** |
| per image | **8.80 ms** |
| upload (206 MB compressed, chunked) | 102.9s |
| non-finite theta / delta | 0 / 0 |

This was faster than the 9.5-minute projection made from an earlier
180-image sample — larger processing chunks amortize the overhead of
starting the process pool more efficiently. Against stage 2's Colab-CPU
measurement of 218.28 milliseconds per image on a single worker, this is
roughly a 25 times reduction in wall-clock time, of which about 3.4
times comes from faster per-core speed, and the rest from running
multiple cores in parallel.

## A scale-dependent finding: the convergence tail is not exactly zero

Stage 1 reported an exact float64 zero for final-Delta, at both the
median and the 95th percentile, across 1,000 images, and stage 2
measured a maximum of 2.22e-14 across 5,000 images. At 54,000 images,
the maximum is **2.468e-10** — four orders of magnitude larger, and only
visible at this larger scale. Neither earlier rung reported a count of
exactly-zero versus nonzero final-Delta values; both stages did store
their per-image arrays, however, so both of these counts are recoverable
from the artifacts, with no need to re-encode anything.

Population: **the 54,000 fit-side images of the official KMNIST training
split**, as encoded by this Phase A run. The 6,000-image locked
validation subset has never been encoded, so no tail measurement exists
for it, and none of the counts below apply to it. Nothing here touches
the test split at all.

| final-Delta, over the 54,000 encoded fit-side images | count |
|---|---:|
| exactly 0.0 | 53,921 (99.854%) |
| > 0 | **79 (0.146%)** |
| > 1e-13 | 9 |
| > 1e-12 | 4 |
| > 1e-10 | 2 |
| non-finite | 0 |

The median is 0.0, the 95th percentile is 0.0, and the maximum is
2.468e-10; the non-finite counts are 0 for both theta and delta. These
figures are derived from the counts above, rather than measured
directly: the 99th percentile is 0.0 (since only 79 of 54,000 values are
nonzero), and the 99.9th percentile lies somewhere in (0, 1e-13] — it
falls inside the nonzero tail, but only 9 values exceed 1e-13.

This is recorded here rather than smoothed over, with its consequences
stated precisely. It does not affect the encoder gate, since that gate
keys on the MEDIAN value (which is 0.0 here, so the absolute-convergence
escape fires regardless of this tail). It does not affect the pipeline
overall: the worst image sits 4,053 times below the ODE solver's
relative tolerance of 1e-6. What it does do is narrow stage 1's own
claim: "every one of 1,000 images" was never actually established by the
two order statistics stage 1 reported, and at a corpus 54 times larger,
a 0.146% tail of not-quite-fully-settled images appears. Two images, out
of 54,000, remain above 1e-10 even after 1,200 steps.

The standing wording used for what the encoder actually achieves is
therefore **numerical convergence below any practically relevant
tolerance**, never exact convergence as a universal claim about every
single image. Aggregate values sitting at zero do not mean every image
sits at zero, and this is the direct, in-project proof of exactly that.

## Code and artifacts

`encode_stage3_local.py`, run with `make stage2b-encode-stage3-local`
(local, CPU-only, provisions no cloud resources, bills nothing). This
composes the `corrupt_corpus` and `encode_with_final_delta_batch`
functions unchanged — the same numerics as both earlier rungs, only the
machine differs. Output: `stage2b/train/stage3/common/
encoded_fit_s1200.npz` (206.1 MB, checksum-verified), carrying the
encoded array for the 54,000 fit-side images, their per-image final-
Delta values, the fit indices and active support for self-description,
and the run summary. The object's file name carries the step count, so
a future change to `ENCODER_STEPS` will create a new object, rather than
silently resuming a stale one — this is the same self-invalidation
approach the stage-1 encoder-gate artifact already uses.

The 218 MB upload crossed `ensure_artifact`'s automatic-chunking
threshold of 64 MB, so the resumable, chunked upload path engaged
automatically, without the calling code having to ask for it. This
safeguard had been added speculatively, earlier in the same week; this
was its first real use on an artifact large enough to actually need it.

## Next step

Regeneration at 60,000 images — this is now done, and recorded in the
next section.

# Stage 2B: Feasibility Ladder Stage 3, Phase A regenerated at 60,000

**Status: this is now the authoritative Phase A artifact.
`stage2b/train/stage3/common/encoded_train_s1200.npz` supersedes
`encoded_fit_s1200.npz`, which stays in the storage bucket as the
baseline this run was checked against. Phase B has still NOT run, and
stage 3 has still produced no denoising number of any kind.**

This is also the first Stage 2B artifact of any kind to be published
with full provenance attached. Every earlier artifact carries none.

## Why this was rerun, and what that cost

The original Phase A encoded only the 54,000-image CNN fit side, based
on the reading that the 6,000 locked-validation images would be "an
artifact nothing reads." That reading turns out to be wrong:
`DESIGN.md:479` defines "full training" at stage 3 as the 54,000 fit
plus 6,000 validation images combined, `:492`'s compute-cost table
corroborates this arithmetically ("~48-60k x 1008", where 48,000 equals
0.8 times 60,000), and the ridge path both cross-validates and refits
using all 60,000 images. The error is disclosed exactly at the point it
was made, inside `encode_stage3_local.py`'s own documentation, rather
than simply edited away — it was originally resolved in the direction
that avoided extra rework, which is exactly the kind of choice that
needs to stay visible.

The rerun cost 11.3 minutes of local CPU time.

## Two populations, two kinds of evidence

This regeneration covers both images that already had an existing
artifact, and images that did not, and the acceptance report keeps these
two groups deliberately separate. A single "regeneration passed" message
would claim verification for half of the data that nothing had actually
verified.

| population | n | status | how it was judged |
|---|---:|---|---|
| CNN fit side | 54,000 | reproduction | checked bit-exact against the prior artifact, joined by official index |
| locked validation | 6,000 | **new measurement** | fingerprinted at the moment of creation; its tail is reported, with nothing to compare it against |

## Part 1 — the 54,000 reproduce bit-exactly

| array | shape | verdict |
|---|---|---|
| `thetas_505` | (54000, 505) float64 | **BIT-EXACT**, sha256 `1113cec3...bc7e85fc` on both sides |
| `deltas` | (54000,) float64 | **BIT-EXACT**, sha256 `6a7753ff...740954f` on both sides |

This match is checked on the aligned subset, joined by official KMNIST
index — `AUDIT_PROTOCOL.md` forbids comparing by row position alone, and
the new artifact is stored in ascending official order, so the fit rows
are scattered throughout it: 53,985 of the 54,000 sit at a different row
position than they did in the baseline artifact.
`compare_stage3_regeneration.py` refuses to accept an alignment that
turns out to actually BE a simple row-position prefix, because a join
that would accept that case cannot be told apart from never having
joined the data by index at all.

An independent, second check, coming from the opposite direction: the
aligned subset's nonzero final-Delta count is **79**, exactly matching
the count this document already records for the original 54,000-image
run. Two matching checksums, with that count having moved, would have
meant the comparison was not actually looking at the rows it believed it
was.

Getting an exact bit-for-bit match here is expected, not lucky, and the
reason for this is now pinned down by an automatic test, rather than
just reasoned about: every per-image encode job uses a constant seed and
builds a fresh `default_rng(seed)`, so a given image's own random
perturbation depends only on that image and the seed, and on nothing
else — not on which processing chunk it happened to land in, and not on
the number of workers used. Both runs used 9 workers on the same
machine, but with different chunk sizes (5,400, then 6,000) — this is
exactly the situation `CLAUDE.md` working rule 19 says the team must not
simply assume its way through.

## Part 2 — the 6,000 validation images, measured for the first time

| role | n | final-Delta > 0 | rate | 95% confidence interval (Clopper-Pearson) | max |
|---|---:|---:|---:|---|---:|
| all | 60,000 | 89 | 0.148% | [0.1191%, 0.1825%] | 2.468e-10 |
| fit | 54,000 | 79 | 0.146% | [0.1158%, 0.1823%] | 2.468e-10 |
| **validation** | **6,000** | **10** | **0.167%** | **[0.0800%, 0.3063%]** | **2.887e-13** |

`AUDIT_PROTOCOL.md` sets **no rule requiring these two rates to agree**,
and the confidence intervals show exactly why that matters: at n=6,000,
the interval is roughly three times wider than at n=54,000, so the
values 0.167% and 0.146% cannot actually be told apart statistically,
and a bare comparison of the two percentages alone would invite a
conclusion the underlying data does not support. This is reported
exactly as the protocol requires — numerator, denominator, which split
each image belongs to, and the uncertainty — with nothing further
inferred from it.

One purely descriptive observation, not a claim: the validation
subset's worst final-Delta value is 2.887e-13, roughly three orders of
magnitude below the fit side's own worst value of 2.468e-10. With only
ten nonzero values in total, this is a small-sample observation about
one extreme value, and nothing more than that. Both values sit far below
the ODE solver's relative tolerance of 1e-6.

### The ARM/x86 stress set now has 89 cases, not 79

`AUDIT_PROTOCOL.md`'s companion section names "the 79 convergence-tail
stress cases," a phrase written when only the 54,000-image artifact
existed. `COMPANION_PROTOCOLS.md` states this construction as a rule,
not as a fixed count — specifically, every image with a final-Delta
value greater than 0.0 — which is now **89** images. Both documents are
frozen, and neither one is wrong: the rule is what actually governs, and
the count changed because the underlying population changed. This is
recorded here so that anyone building this set directly from the audit
protocol's text, and arriving at 79, can see exactly why that number
moved. The 500-case cap frozen in `COMPANION_PROTOCOLS.md`, set before
this count was known, remains inert here, exactly as expected.

## Result

**60,000 official KMNIST training images, 1,200 steps, 9 workers
(Darwin, ARM64 chip, 10 cores in total), chunk size 6,000.**

| quantity | measured |
|---|---:|
| encode wall-clock | **679.0s (11.3 min)** |
| per image | **11.32 ms** |
| upload (229 MB compressed, chunked) | 113.0s |
| non-finite theta / delta | 0 / 0 |
| final-Delta median / p95 / max | 0.0 / 0.0 / 2.468e-10 |

Per-image cost rose from 8.80 milliseconds to 11.32 milliseconds,
compared with the 54,000-image run on the same machine. This was not
further investigated; the plausible causes are ordinary ones (the
machine's thermal state, other load on the machine, or the different
chunk size), it does not change any actual result, and attributing it to
one specific cause without measuring that cause directly would be
guessing.

## Provenance — what is now attached, and what the record shows

This artifact is published together with a sidecar manifest file
(`...npz.manifest.json`), carrying the full payload checksum, the data
type, shape, and SHA-256 checksum for each of the seven individual
arrays, and a fingerprint: code commit `12a8c46a`, the combined 18-file
static-and-runtime source list with each file's own checksum, the
environment, and the declared scientific configuration (checksum
`768bf201...`). This source list was established BEFORE generation, and
re-checked after: 0 modules were imported that this fingerprint did not
already describe.

The recorded state of the code repository is worth reading in full,
rather than skipping over, because it is the first real exercise of a
safety check that was itself being narrowed while it was actively
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

All four of these untidy items belong to an unrelated, concurrent piece
of work, and none of them is part of this driver's own source list, so
none of them can reach this artifact at all. A check that looked at the
entire code tree would have refused to run this job; the narrower
source-list check instead names exactly what actually matters, and
simply records everything else. This detection behavior is shown to work
by an actual test, rather than merely argued for: a dirty file inside
the source list still halts the run and is named directly; staging a
change with git is not the same as actually committing it; and an
untracked file that is part of the source list counts as dirty, rather
than being silently treated as clean.

## Code and artifacts

`encode_stage3_local.py` (run with `make stage2b-encode-stage3-local`)
and `compare_stage3_regeneration.py` (run with `make
stage2b-compare-stage3`, which only reads data). The acceptance report
is `results/stage3_regeneration_acceptance.json`. Object:
`stage2b/train/stage3/common/encoded_train_s1200.npz` (229.1 MB,
checksum-verified), carrying `thetas_505`, `deltas`, `train_indices`,
`fit_indices`, `validation_indices`, `active_indices`, and the run
summary. Rows are stored in ascending official-index order, so row `i`
is always official training image `i` — the index arrays are stored
explicitly anyway, rather than left as something merely implied.

## Next step

Phase B: a GPU-session driver that reads this artifact, regenerates the
corruption and clean target values itself, in-session, and runs
evolution, ridge (with the ladder's third real-data equivalence check),
and CNN training, all at full scale. **This has since been written and
run — see "Phase B, the alpha floor, and the amended grid" below; this
paragraph is preserved as the state of things at the time Phase A itself
landed.** It should carry a spot-check that one image's encoding, re-
derived in-session, matches the stored array, with the tolerance stated
as ULP-level rather than exact, for the cross-chip-type reason recorded
above.

It should also read data through `consume_validated`, rather than
through `download_file`. The contract for this is now built, and this
artifact is already published under it, but no driver actually READS
under it yet — `ensure_artifact` still does not call it,
`force=True` still bypasses its trust point entirely, and two call sites
still download data directly. `NEGATIVE_PATH_EVIDENCE.md` records this
demand as covered only at the module layer, and not yet actually
adopted in practice.

**A decision Phase B's own Makefile target has to make deliberately.**
Two different definitions of "clean enough to run" now exist side by
side, and they disagree with each other. The ladder targets refuse to
run based on a whole-tree `git status --porcelain` check; the
fingerprint system instead refuses based on the driver's own source
list. This particular run is the case that separates the two: the
source-list check passed, but the whole-tree check would have refused
to run this job. Leaving both checks in place means the one that
actually fires is the coarse, whole-tree one — the exact one this
project already concluded was asking the wrong question — while the
sharper, more precise check sits behind it, never actually reached.
Phase B's Makefile recipe should be written to gate on the source-list
check, and to say so directly in the recipe's own comment, rather than
inheriting the whole-tree porcelain check just by copying the stage-1
target. This is noted here rather than changed unilaterally, because
this change would alter the pre-flight check on targets that actually
spend money.

---

# Phase B, the alpha floor, and the amended grid

Phase B ran on 2026-08-07 (result: `STAGE3_OK`, 71.6 minutes of A100
GPU time): the full 60,000-image corpus, run through evolution,
features, ridge, and the CNN. There were zero solver failures across
240,000 total graph evolutions. The re-encode spot-check came in at 2.0
ULP, against a tolerance of 16.0 ULP, which is consistent with the
maximum of 3 ULP already recorded for cross-chip-type differences.
Decision 2's equivalence extension passed on all seven conditions, at 12
times the largest previously verified scale, with a worst clipped-
prediction difference of 2.185e-12, against the frozen tolerance of
1e-8.

What follows below is the actual finding that came out of this run,
which is not related to any of the above.

## Six of seven conditions selected the smallest alpha value on the grid

At feasibility stage 2 (n=5,000), no condition sat at a grid edge. At
n=60,000, six of the seven conditions selected `ALPHA_GRID`'s smallest
value:

| condition | n=5,000 | n=60,000 (nine decades) |
|---|---|---|
| raw_505 | 1000.0 | **0.01 (floor)** |
| raw_784 | 1000.0 | **0.01 (floor)** |
| pre_evolution | 1000.0 | 1000.0 |
| T | 1.0 | **0.01 (floor)** |
| lattice | 1.0 | **0.01 (floor)** |
| rewired | 10.0 | **0.01 (floor)** |
| curr_random | 100.0 | **0.01 (floor)** |

Needing less regularization as more data becomes available is expected.
A selection landing exactly AT the grid's boundary is a different
statement altogether: it means the grid no longer brackets the true
optimal value, so these six values represent a boundary, not a genuine
minimum.

This correctly did not halt the run — landing at a grid edge is recorded
as a fact under the frozen plan, and no new gate may ever be invented
mid-ladder, in either direction. It did, however, fire a
**pre-registered review item**: *"if several conditions select the
grid-minimum alpha, that is a reported fact requiring scrutiny before
Stage 4 — not a halt, but a named review item."* Nobody had to decide,
in the moment, whether this actually mattered.

### The extra scrutiny cost nothing, and split the conditions into two groups

The validation curves were already saved in the committed artifact.
Relative rise in error, moving away from the grid floor:

| condition | 0.01 -> 0.1 | 0.01 -> 1.0 | reading |
|---|---|---|---|
| raw_505 | 1.167e-07 | 1.284e-06 | plateaued |
| raw_784 | 1.060e-07 | 1.166e-06 | plateaued |
| pre_evolution | -6.227e-07 | -6.845e-06 | not pinned |
| **T** | **8.431e-03** | 2.043e-02 | still moving at the floor |
| **lattice** | **4.151e-03** | 1.145e-02 | still moving at the floor |
| **rewired** | **7.663e-03** | 1.749e-02 | still moving at the floor |
| **curr_random** | **6.195e-03** | 1.351e-02 | still moving at the floor |

This split falls exactly along the treatment-versus-control line: both
raw baselines are flat to within about 1e-7, while all four evolved
conditions are moving four to five orders of magnitude faster. So this
constraint bound in an unequal way, splitting exactly along the exact
comparison this whole readout method exists to support.

## The amendment, frozen before any fitting

The reviewer's ruling extended the alpha grid four decades further
downward, to `{1e-6 .. 1e6}` — thirteen values, keeping the exact decade
spacing, applied to all seven conditions — together with a full re-run,
no splicing of old data, the equivalence check re-run at production
scale, and a halt for review if any condition selected the new floor
value. This is recorded, with its full origin, in `DESIGN.md`'s Review
history section; it arrived by manual copy-paste from a ChatGPT session,
rather than through the project's usual audited review channel, and the
route a ruling travels by is itself part of its evidence.

**This procedure is built to allow only one attempt.** The exact-spacing
rule forbids any interpolation or denser searching around an observed
minimum, and landing exactly at the new floor value is named as an
anomaly, rather than treated as a trigger for extending the grid even
further. There is no way this procedure can authorize a second
widening — which matters, because a stopping rule chosen only after
seeing results is not actually a stopping rule at all.

Two consequences neither the original ruling nor the plan had
anticipated, both found while actually implementing it:

**The write-once rule enforces "do not splice" automatically, through
the structure of the storage system itself.** `ridge_cv.json` and
`ridge_final.npz` are create-once LINEAGE artifacts; passing
`force=True` raises an error before any new computation even runs. So
the re-run could not overwrite the nine-decade tables, and had to write
new file names instead.

**And that same rule does nothing to stop silent REUSE.** The
nine-decade artifacts sat in the bucket with valid manifests, and
`ensure_json` passes no `expected_fingerprint` argument, so the
thirteen-decade re-run would have hit `ensure_artifact`'s skip branch,
accepted the nine-decade results without complaint, recomputed nothing
at all, and reported `STAGE3_OK` anyway. This would have satisfied the
ruling's "do not splice" rule **only by accident**: nothing is spliced
together when nothing is actually computed. This was fixed by deriving
the artifact's file name directly from the grid itself
(`ridge_cv_g13_88edf9ac`), so any change to the grid automatically moves
the file name. Deriving names from content, and pinning expectations
with a fingerprint, are complementary safeguards: the first one stops
you from writing over history, the second one stops you from reading the
wrong history, and neither one covers what the other one does.

## The amended-grid result

Rerun on 2026-08-08, taking 48.5 minutes, across thirteen decades, for
all seven conditions, with evolution, features, and the CNN all reused
from cache, and only the ridge model itself recomputed.

| condition | selected | argmin | status |
|---|---|---|---|
| raw_505 | 1e-03 | 1e-06 | flat to within 1e-10, over four decades; the tie-break rule takes the largest value |
| raw_784 | 1e-03 | 1e-06 | same behavior |
| pre_evolution | 1e+03 | 1e+03 | an interior value, not at a grid edge |
| **rewired** | **1e-05** | 1e-05 | **an interior minimum — resolved** |
| **curr_random** | **1e-05** | 1e-05 | **an interior minimum — resolved** |
| **T** | **1e-06** | 1e-06 | **still at the grid floor** |
| **lattice** | **1e-06** | 1e-06 | **still at the grid floor** |

The equivalence check passed on all seven conditions at all thirteen
decades, with a worst difference of **5.421e-11**, against the frozen
tolerance of 1e-8.

This extension resolved five of the seven conditions. `rewired` and
`curr_random` have genuinely higher MSE at 1e-6 than at 1e-5 (higher by
2.685e-04 and 3.805e-04 respectively) — these really are interior
minima, not simple tie-breaks.

### What this establishes, and what it does not

The following is stated using the reviewer's own carefully licensed
wording, because these distinctions carry real weight, and each one is
easy to overstate in the direction that flatters the project's own
hypothesis:

- For **T and lattice**, `1e-6` is the best **observed candidate on the
  frozen, discrete grid**. Their true, continuous-domain ridge optimum
  values are **not bracketed** by this grid, and are not established to
  equal, to lie below, or to lie near `1e-6`. The grid also says nothing
  at all about values that fall **between** the sampled decades.
- For **rewired and curr_random**, the frozen-grid selections are
  genuine interior minima, at `1e-5`.
- The steadily shrinking per-decade increments seen for T and lattice
  are legitimate, descriptive evidence that the validation curve is
  **flattening out**, over the sampled low-alpha range. For T, the
  increment from 1e-6 to 1e-5 is only **5.292e-05**, against **8.431e-03**
  at the old floor. This supports **the curve slowing down**, and does
  **not** support **pinning down** where the true optimum actually is.
- This flattening is **not** translated into any quantitative bound on
  the unseen true optimum, and it is **not** translated into a claim
  that any leftover regularization bias is negligible.
- Any statement about the direction this constraint pushes results is
  **valid only for training-side validation data**. Nothing here
  establishes the sign or the size of the effect on the official test
  set.
- Any graph ranking that involves T or lattice is explicitly
  **bounded by the protocol used**: these rankings compare pipelines
  that were each selected under this one frozen, discrete alpha grid,
  not pipelines under continuously, optimally regularized versions of
  each graph. A control graph beating a floor-pinned T is not, on its
  own, a claim that one graph is mechanistically superior to another.

**The floor condition is a disclosure and a limit on the scope of what
was optimized — it is not a defect in the protocol itself.** Questions
about alpha values approaching zero, a denser grid, or a continuous
optimum are all follow-up work, to be done after this confirmatory
result, not before it.

## The gate that was frozen and never actually implemented

`DESIGN.md`'s amended procedure states, word for word: *"HALT for review
if any production condition selects 1e-6."* This exact sentence was
frozen only a few hours before the grid extension was actually
implemented, **and the halt itself was never written into the driver's
code.** The rerun selected the grid floor value for both T and lattice,
and still reported `STAGE3_OK`.

That reported verdict actually records that **no such gate existed at
all** — not that a gate was evaluated and cleared successfully. This is
preserved exactly as such: the run report carries a machine-readable
annotation marking this fact, and the function `read_run_report()`
refuses to return the verdict at all without also surfacing this
annotation.

The numerical results from that run **remain admissible**. The defect
was in the readiness verdict, and in how that verdict was enforced — not
in the reported ridge computation itself.

The halt now exists (`floor_halt_reason()`), pulled out as a pure
function specifically so that a test can exercise the actual decision
logic, rather than testing the exact wording of the code. This function
is confirmed to correctly fail against code commit `817ac08` — the exact
driver that produced the original report, which imports cleanly, with
`step7_ridge` fully intact, so the failure really is caused by the
missing gate, and not by some broken module.

This is recorded here, rather than left only in the process log, because
it is the sharpest example this project has of a class of mistake it
has now hit six times in a single day: **the requirement exists in the
document, and not in the code.** The document and the driver were
written by the same author, in the same session, only hours apart. There
was no handoff between people to blame, which is exactly what rules out
"be more careful" as the fix, and which motivated the binding-clause
inventory now required before the pre-test review package.

---

# Stage 2B: Feasibility Ladder Stage 4 — the official result

**Status: THIS IS the locked confirmatory result.** This is
`DESIGN.md`'s ONE evaluation on the official 10,000-image KMNIST test
corpus, run under the locked statistical procedure, evaluated exactly
once. Everything upstream of this section — stages 1 through 3 — was
feasibility work; stage 3 was labeled "SMOKE OF THE MACHINERY ONLY...
NOT A RESULT" throughout. This section is the first, and the only,
place in Stage 2B where that label does not apply.

**`STAGE4_OK`, `run_ladder_stage4.py`, code commit `431d90a`.** The
official result is stored at
`stage2b/testsplit/stage4/common/official_result.json`.

## Result: T is the unique winner

**Primary test** (`d_i = MSE_i(T) - MSE_i(pre_evolution)`, on the active
support, after clipping, on the official test set, using a
20,000-resample paired, class-stratified bootstrap, random seed 42): the
95% confidence interval is **[-0.0046028, -0.0043002]**, with a mean of
**-0.0044509**, entirely below zero. **Verdict: evolution improves
reconstruction.** This test is uncorrected — the primary test sits
outside both statistical multiplicity families, following the locked
design.

**Denoising gate** (Level 2 of the hierarchical identity gate, checked
because the primary test succeeded): `T` compared against the identity
baseline (`clip(x_t)`, on the active support, after clipping). The
confidence interval is **[-0.1335739, -0.1328960]**, with a mean of
**-0.1332334**. **The gate passed** — the stronger claim of "actual
denoising" is added to the primary reconstruction claim, not just the
weaker, relative-improvement claim on its own. Reported independently,
outside the gate, as context: `pre_evolution` compared against
identity, with a confidence interval of [-0.1291310, -0.1284354] and a
mean of -0.1287826 — the pre-evolution representation already denoises
substantially on its own, and evolution adds a further, statistically
confirmed margin on top of that.

**Family 1** (three controls, each compared against `pre_evolution`,
with Holm correction across all three, all favorable to evolution):
`curr_random`'s mean is -0.002872, with a raw p-value of 1.851e-252 and
a Holm-corrected p-value of 3.702e-252; `rewired`'s mean is -0.002179,
with a raw p-value of 1.395e-144 and a Holm-corrected p-value of
1.395e-144; `lattice`'s mean is -0.004073, with both the raw and the
Holm-corrected p-value reported as `0.0` — this is a float64 underflow
of the underlying analytic t-distribution tail (t=-55.58, degrees of
freedom=9999), not an exact, literal zero (see `CLAUDE.md` working
rule 6). All three of these controls were Holm-rejected (found
significant).

**Family 2** (six pairwise comparisons among the four evolved graphs,
with Holm correction across all six): all six were Holm-rejected, with
every raw p-value either underflowing to zero or very close to it (the
largest surviving p-value, for `rewired_vs_curr_random`, is 1.549e-30).
The overall directional picture: `T` beats all three other graphs (t=
-8.74, p=2.73e-18 versus `lattice`; t=-38.10 versus `rewired`; t=-26.85
versus `curr_random`); `lattice` beats both `rewired` and `curr_random`,
but loses to `T`; `curr_random` beats `rewired`, but loses to both `T`
and `lattice`; `rewired` loses to every other graph. The sign-flip
robustness check (100,000 random sign flips, seed 42, using a
studentized statistic) agrees in both direction and significance, on all
six comparisons, each one landing at the Monte Carlo floor
(p=9.9999e-06). This is reported, exactly as `DESIGN.md` frames it, as
robustness evidence only, not as a second, separately corrected family
— and with the sign-exchangeability assumption stated explicitly for
each individual comparison, rather than assumed once and then applied
silently everywhere.

**`one_graph_wins`: `unique_winner = "T"`.** `T` qualifies via the
primary rule (its bootstrap interval lies entirely below zero), and it
outperforms each of the three other evolved graphs, after Family-2 Holm
correction. This is exactly `DESIGN.md`'s named watched-for outcome
number 2 ("one evolved graph qualifies per the branched rule AND
outperforms each of the other three after Family-2 correction"), which
is the outcome that was actually realized, rather than any of the other
four named outcomes.

## The caveat this result inherits from Stage 3, stated again here rather than left implicit

Stage 3's amended-grid result, above, already states this caveat for the
training-side number, and it applies here unchanged, because stage 4
REFITS at the exact frozen alpha value stage 3 selected, rather than
re-selecting a new one: **both `T` and `lattice` sit at the ridge grid
FLOOR (`alpha=1e-6`)**; `rewired` and `curr_random` sit at genuine
interior minima (`alpha=1e-5`); `pre_evolution` sits at `alpha=1e+03`.
`T`'s true, continuous-domain ridge optimum is therefore **not
established, and it is not known to equal, lie below, or lie near
`1e-6`** — this grid says nothing about alpha values between the sampled
decades, let alone about values below the smallest sampled decade.

What this does and does not allow the team to conclude, stated using the
same terms Stage 3 used above: `T` beating `lattice` is a floor-versus-
floor comparison, so that particular result is NOT confounded by any
unequal freedom in regularization between the two. `T` beating `rewired`
and `curr_random`, on the other hand, compares a floor-pinned condition
against two conditions that had room to move freely, so this comparison
is only of the four pipelines **as they were actually selected under
this one frozen, discrete grid**, and not of the four graphs under
continuously optimal regularization for each. A denser or extended grid
could, in principle, move `T`'s own number in either direction; nothing
here places any bound on how much it might move. The primary test and
the denoising gate, both stated purely in terms of `T` alone, compared
against `pre_evolution` and against identity respectively, do not depend
on this particular comparison at all, and are unaffected by it.

## CNN: retrained and verified, reported descriptively

Following `run_ladder_stage4.py`'s own design (stage 3 saved no trained
model weights at all, only training histories and a summary — see that
file's own module documentation), the CNN was retrained from the same
three fixed random seeds, on the same locked 54,000/6,000 fit-versus-
validation split, and then checked against stage 3's saved selection
before being trusted to make predictions on the test corpus.
**Reproduction confirmed**: seed 1, epoch 99, exactly matching stage 3
on both of these values (the structural check
`cnn_reproduction_mismatch_reason` requires exact equality on these two
specific values); the `best_clipped_val_mse` value differed by
**2.385e-07** — this is reported, not gated on, following the module's
own deliberate refusal to invent a numeric tolerance with no measured
basis behind it. This is now real, repeated evidence — three separate
real-GPU retrainings, across this run's own three attempts, differing by
9.328e-07 and 2.385e-07 respectively from stage 3's own recorded number
— that this iterative, 32-bit-precision, cross-session training
procedure reliably reproduces its own seed and epoch selection in
practice, though the module deliberately stops short of claiming this
as a proven guarantee.

The CNN's mean clipped MSE on the test corpus is: **0.063069**. This is
reported descriptively throughout this document and throughout this
driver, per `DESIGN.md`'s own framing — the CNN belongs to neither
statistics family, and it is not part of the inference this section's
verdict actually rests on.

**Stated plainly, because "T is the unique winner" above is easy to
over-read: the CNN's own mean MSE is LOWER (better) than `T`'s, not
higher.** `T`'s own mean clipped MSE (recovered from the identity-gate
contrast: identity's 0.198856 minus the gate's own observed mean of
0.133233) is **0.065623**; the CNN's is **0.063069** — a difference of
0.002554 (about 3.9% relative), in the CNN's favor, with the CNN winning
on 5,814 of the 10,000 images, against `T`'s 4,186 wins. `DESIGN.md`
never places the CNN in either statistical family, or in any named
watched-for outcome, so no corrected, locked test actually exists
between it and `T`, or between it and any ridge condition —
`one_graph_wins`'s own verdict is scoped only to the four ridge-based
evolved-graph conditions, and it correctly says nothing at all about the
CNN. A **descriptive-only** paired bootstrap, computed after the fact
specifically for this write-up (20,000 resamples, seed 42, NOT part of
the locked design, not Holm-corrected, and not a second confirmatory
family), puts the CNN-minus-T gap's 95% confidence interval at
**[-0.00276, -0.00235]**, entirely below zero — this is offered only as
a rough estimate of magnitude, not as a second locked verdict. The
honest claim this section actually supports is narrower than "T is the
best model overall": `T` is established as the best-performing condition
**only among the ridge-based, phase-representation conditions**, through
the locked procedure; a categorically different, nonlinear model class
does numerically better on the same corpus, and it was never actually
tested directly against `T`.

## Descriptive baselines, official test corpus

| condition | mean clipped MSE |
|---|---:|
| `T` (recovered from the identity-gate contrast) | 0.065623 |
| `raw_505` | 0.198856 |
| `raw_784` | 0.198856 |
| rescaled identity (`clip(x_t_clip / sqrt(0.5), 0, 1)`) | 0.246952 |
| CNN | 0.063069 |
| identity (`clip(x_t)`, hierarchical-gate baseline) | 0.198856 |

Raw-pixel ridge and the identity baseline sit within rounding distance
of each other (0.198856 versus 0.198856) — there is no sign here that
the phase representation loses information for reconstruction, compared
with simply doing nothing at all to the corrupted pixels directly.
Named watched-for outcome number 5 (raw-pixel ridge dominating every
phase-based condition) is not what was actually observed here.
Corruption diagnostics, across the full 10,000-image test corpus:
pre-clip MSE of 0.515888 (505-pixel support) and 0.513274 (784-pixel
support); post-clip MSE of 0.198856 and 0.196559.

## What it took to get a clean run

Three attempts, code commits `6e7f811`, then `1b2e58b`, then
`431d90a`, taking roughly 44 minutes of A100 GPU time in total (not the
72-hour, roughly $3,000 scale this would have cost without the
resumability contract every earlier ladder stage had already
established — the first two attempts' correctly-computed artifacts were
reused, not recomputed, by the attempt that finally completed
successfully). There were two real bugs, both caught by an actual
execution run rather than by review, both fixed and pinned as regression
tests before the next attempt:

1. `parent_map` never threaded the `allow_test_split` flag through to
   `read_manifest` — every parent artifact this driver records is a
   test-side object, unlike stage 3's own `parent_map`, which never
   touches the test split at all. This surfaced at the
   `2_test_corruption` step, 42 seconds into the run, before any real
   cost had been incurred.
2. `step7_test_cnn` called `clipped_validation_per_image_mse` directly
   on raw `(n, 28, 28)` arrays, without first applying `as_image_batch`'s
   channel-first conversion — exactly the mistake that function's own
   documentation names and specifically warns against. This surfaced
   663.65 seconds into the CNN step, after a full, successful three-seed
   retrain had already completed — confirming that the retrain-and-
   verify design works correctly, only to then fail one call later, on a
   shape mismatch. This was pinned with a real, executable regression
   test (`test_cnn_test_evaluation_runs_on_raw_corpus_arrays`) that
   actually runs JAX and equinox code on the CPU, not a merely static
   check — this was the one bug that a static code review alone could
   not have caught.

A third defect was caught and fixed before it could actually matter:
`step11_report` originally wrote the `official_result` file
unconditionally, even on a FAIL verdict, which would have let the first
attempt's halted, pre-inference run permanently occupy the one-shot
result slot, blocking every subsequent attempt through
`refuse_if_official_result_exists` — including a correct one. This was
fixed to only write the file when `verdict == OK_SENTINEL`, before the
second attempt ran; this fix was confirmed working when the second
attempt's own failure correctly wrote no `official_result` at all.

## Code and artifacts

`run_ladder_stage4.py`, `tests/test_stage2b_ladder_stage4.py`. Official
result: `stage2b/testsplit/stage4/common/official_result.json` and
`.txt`. Per-attempt reports (all three, including both failures, kept
as history): `stage4_report_20260809T033701Z`,
`stage4_report_20260809T034509Z`, `stage4_report_20260809T041415Z`.
Test-side intermediate artifacts (the test corpus, its corruption, the
encoded phases, the evolved theta values per graph, the features per
condition, the ridge test MSE values, and the CNN reproduction and test
evaluation) all live under `stage2b/testsplit/stage4/`.

## Status of the investigation

This closes `DESIGN.md`'s feasibility ladder. The question this entire
Stage 2B investigation was built to answer — does runtime oscillator-
network evolution, on a learned graph topology, improve single-step
active-support denoising, compared with the same representation before
evolution — has a locked, confirmatory, positive answer, on the official
held-out test corpus: yes, and the stronger "actual denoising" claim
holds too. What remains open, tracked separately rather than folded into
this result: whether a denser or extended ridge grid would move `T`'s
own floor-pinned alpha value (the caveat described above); the two
companion protocols `AUDIT_PROTOCOL.md` names
(`COMPANION_PROTOCOLS.md`'s ARM/x86 propagation stress set, and the
`ABS_CONV_EPS` sensitivity table — the sensitivity table has now run,
see this document's own section below; the stress set has not yet
started); and none of these particular results were part of this
evaluation, nor were they required to be, following `AUDIT_PROTOCOL.md`'s
own scoping of what the confirmatory test itself actually needs. **The
150-vs-1200 amendment-impact audit itself has since run — see "Stage
2B: the amendment-impact audit" below, which supersedes the earlier "has
not run" framing this paragraph previously carried.** Also still open is
an unresolved question, raised by a review process called INFRA: the CNN
has no stated consumer anywhere in `DESIGN.md`'s own text, and this
result treats that as settled, in the "descriptive comparator" reading,
rather than actually resolving the ambiguity that review process named —
and this is sharpened further by this section's own descriptive finding
that the CNN's mean MSE is numerically lower than `T`'s, since a design
that had given the CNN a stated consumer would have had to reckon with
that fact directly, rather than reporting it as a footnote. **"T is the
unique winner" names the winner only among the ridge-based, phase-
representation conditions — the exact comparison `DESIGN.md`'s
statistics families and the `one_graph_wins` rule actually run. It is
not a claim that `T` is the best-performing condition in this whole
document, and the CNN's own descriptive number shows that it is not.**

# Stage 2B: the amendment-impact audit — no trigger fired

This runs `AUDIT_PROTOCOL.md`'s own core apparatus, on 2026-08-09
(result: `AUDIT_OK`, on an A100 GPU, one attempt, about 29.4 minutes of
GPU time). It measures the representational impact of the encoder-
budget amendment (raising the step count from 150 to 1,200), made after
ladder stage 1's gate failure — a prospective, disclosed, post-failure
amendment, not a preregistered part of the original design. Following
the protocol: this is not a tool for choosing between models, and the
1,200-step budget stays frozen no matter what this audit finds; what can
change is only the scope of the claim the project makes.

## Result: none of the three triggers fired, in either alpha regime

| quantity | 150-step | 1,200-step | change |
|---|---|---|---|
| primary contrast (`T` vs. `pre_evolution`, fixed alpha) | -0.0052076071 | -0.0052073732 | +2.339e-7 |
| primary contrast (`T` vs. `pre_evolution`, reselected alpha) | -0.0052076071 | -0.0052073732 | +2.339e-7 |

Under both alpha regimes: `primary_sign_reversal = False`; all four
`graph_sign_reversals` values (`T`, `lattice`, `rewired`,
`curr_random`) are `False`; and every one of the six pairwise
comparisons has `order_reversed = False` (so `resolved` is `False`
throughout — `resolved` only requires an order reversal to have
happened in the first place; the frozen threshold value would only have
mattered had one actually occurred). `combined_triggered = False` — this
is the explicit OR condition, across both alpha regimes, that
`AUDIT_PROTOCOL.md` requires ("either alpha regime triggers review... a
reversal seen under fixed-alpha alone, or under reselected-alpha alone,
is sufficient").

Reading this number carefully: the change in the primary contrast is
real and genuinely measured, not zero, and it sits roughly 2.7 orders of
magnitude above the frozen analytic resolution threshold
(`4.604761e-10`) — `2.339e-7 / 4.604761e-10` is approximately 508 — but
the change itself is only about 2.3e-7, which is about 0.0045% of the
contrast's own magnitude (-0.0052). This gives a clean double
conclusion: the change is well-resolved (a real margin above the
threshold, not sitting near a coin-flip close call) AND it is
scientifically immaterial (only four-and-a-half-thousandths of a percent
of the quantity it changes) — the two readings do not trade off against
each other here at all. The amendment moved this number by an amount the
audit is able to resolve, but in a direction and at a scale that changes
no sign, no per-graph verdict, and no pairwise ordering. The 150-step and
the 1,200-step representations tell the same qualitative story about
which evolved graph performs best; the 1,200-step budget's own
confirmatory result (the stage-4 section above) is not put into question
by this measurement.

## Stage-1/2 historical cross-check: the new out-of-fold machinery reproduces already-trusted numbers

This checks `gates.toml`'s `binding_gate.9bc6f9e3808a` (out-of-fold
per-image MSE values must reproduce the already-stored fold-aggregate
values from the stage-1 and stage-2 runs, not merely agree with a fresh
recomputation on synthetic data). This ran against each stage's OWN
stored `ridge_cv.json` file, and each stage's OWN pre-amendment
nine-decade alpha grid (not the 1,200-step run's later thirteen-decade
`ALPHA_GRID` — the two grids have different numbers of columns, so
using the wrong one fails on a shape mismatch before it can even compare
a single value). All ten (stage times condition) checks passed, at or
very close to float64 numerical noise:

| stage | max abs diff across 5 conditions |
|---|---|
| 1 (n=1,000) | 1.388e-17 -- 2.776e-17 |
| 2 (n=5,000) | 0.0 (exact, on 4 of 5 conditions) -- negligible |

This is the check that must pass before the 60,000-image out-of-fold
ridge step can be trusted to mean anything at all: new machinery, pinned
against numbers this project already trusted, before it was run at a
scale ten times larger than anything it had ever been checked against
before.

## Feature distances: real, but small, and matching the amendment's own scale

This shows the 150-versus-1200 distance, per condition (gauge-fixed at
reference node 363) — NOT the pre-evolution-versus-evolved distance
within a single budget. This is the quantity that answers "how much did
the amendment itself change the representation," not "how much does
evolution itself change the representation":

| condition | cos/sin Euclidean distance, median | p95 | max |
|---|---|---|---|
| `pre_evolution` | 0.001761 | 0.005419 | 0.05987 |
| `T` | 0.001269 | 0.003926 | 0.03857 |
| `lattice` | 0.001286 | 0.004027 | 0.04247 |
| `rewired` | 0.000180 | 0.000588 | 0.01049 |
| `curr_random` | 0.000250 | 0.000812 | 0.01420 |

`pre_evolution` shows the largest median distance of the five conditions
— the raw encoding itself differs more between 150 and 1,200 steps than
the EVOLVED representations do, for three of the four graphs
(`rewired` and `curr_random`, the two graphs that Stage 2A's own order-
parameter measurements found to synchronize most strongly, show markedly
smaller distances than `pre_evolution`; `T` and `lattice` sit closer to
`pre_evolution`'s own scale). This is consistent with graph evolution
partially washing out the budget-dependent differences from encoding,
for the graphs that synchronize the most — rather than amplifying those
differences. This reading is offered as a descriptive interpretation of
this table, and is not itself a claim this audit's own triggers actually
test for.

## The scope statement the protocol requires in any write-up, stated word for word

Per-budget, fold-fitted `StandardScaler`s are retained (matching
production preprocessing). This means fixed-alpha isolates only **the
effect of reselecting alpha** — it does **not** fully isolate the effect
of a raw representation change on its own. A comparison using one
shared scaler across both budgets is optional, secondary work, and it is
not required, and it must never be presented as the primary check.

## What it took

One attempt, resulting in `AUDIT_OK` on the first real run. A prior
attempt (same code commit, before `LADDER_STAGE=5` had been added to
`stage2b_gcs.py`'s `LADDER_STAGES` validation list) failed at the first
artifact write, during the 150-step evolution step, after about 163
seconds of real evolution computation — this was already caught non-
fatally once before, inside the sizing probe's own publish step (which
logs the problem and continues), before it then failed fatally. This was
fixed, pinned with a regression test, and rerun. The cloud session tore
down cleanly both times; there was no billing leak.

Total wall-clock time was 1,764.1 seconds (about 29.4 minutes):
bootstrap took 17.1 seconds, loading the training-side artifacts took
54.6 seconds, topologies took 1.9 seconds, reading stage 3's 1,200-step
theta values and features took 313.2 seconds, computing the production
alpha values took 9.1 seconds, the sizing probe took 8.0 seconds,
evolving the 150-step budget (4 graphs times 60,000 images) took 307.8
seconds, computing the 150-step features took 460.3 seconds, the
stage-1/2 cross-check took 66.0 seconds, the 60,000-image out-of-fold
ridge step (both alpha regimes, both budgets, 5 conditions) took 507.5
seconds, computing feature distances took 14.0 seconds, and the trigger
verdict took 4.3 seconds. The sizing probe's own projection (516.8
seconds for the out-of-fold ridge step, measured from one JAX SVD at
production shape, before anything expensive had actually run) came in
within 2% of that step's actual time of 507.5 seconds — the probe's own
methodology was validated by the very run it was gating.

The production (1,200-step) alpha values, which the fixed-alpha regime
applied identically to both budgets, were:
`pre_evolution=1000.0`, `T=1e-6`, `lattice=1e-6`, `rewired=1e-5`,
`curr_random=1e-5` — with `T` and `lattice` both sitting at the grid
floor, the same caveat the stage-4 section above already carries forward
from Phase B's own amendment.

## Code and artifacts

`run_audit.py`, `stage2b_audit.py`, `tests/test_stage2b_audit.py`,
`tests/test_stage2b_audit_driver.py`. Run report:
`stage2b/train/stage5/common/audit_report_20260809T192835Z.json` and
`.txt`. Per-condition, per-budget artifacts (the 150-step evolved theta
values and features, the 60,000-image out-of-fold results under both
alpha regimes, feature distances, and the trigger verdict) all live
under `stage2b/train/stage5/`. The 1,200-step theta values and features
are Phase B's own saved artifacts, under `stage2b/train/stage3/`, read
here rather than re-evolved, following `PHASE_B_PLAN.md`'s Decision 4.

## Status of the investigation

The amendment-impact audit is now closed: no trigger fired, in either
alpha regime, on any of the three frozen conditions. The 150-vs-1200
encoder-budget amendment has a real, measured representational effect.
This effect is too small to change the sign, the per-graph verdict, or
the pairwise ordering that the stage-4 confirmatory result (and Phase
B's own ridge result) depend on. At the time of this audit write-up,
two things were still open: the `ABS_CONV_EPS` sensitivity table (it
has now run separately — see `run_abs_conv_eps_sensitivity.py`), and
the ARM/x86 propagation stress set. Protocol 1 has since run. Its
account is in the next section.

# Stage 2B Companion Protocol 1: the ARM/x86 propagation stress set — result `PROTOCOL1_OK`

`COMPANION_PROTOCOLS.md`'s consequence rule calls for an interpretation
review before Stage 4 runs. Stage 4 has already run, and it is locked.
So this Protocol 1 result is disclosed here as happening after Stage 4,
not before it — following the same sequencing exception already set for
the amendment-impact audit above.

## Verdict

**Result: `PROTOCOL1_OK`.** The stage-5 halt did not fire. For every
graph, the largest observed `max |Δ Delta_g|` value (the biggest cross-
chip-type difference in the evolved-minus-pre-evolution contrast) is
strictly below the frozen threshold, `CONTRAST_THRESHOLD = 4.604761e-10`.
The largest stage-5 value measured was for `curr_random`, at `1.830e-12`
— about 252 times below the threshold.

Run ID: `20260810T151245Z`. An earlier report, `20260810T124247Z`,
existed as a JSON file, but it carried no process sentinel and no
manifest sidecar file. A sentinel is a special marker a script prints or
writes to show its own verdict; a manifest is a small file, published
next to an artifact, that records exactly which code produced it. This
run closes that gap. Later verification re-runs under the fixed driver
(for example, `20260810T151926Z`) confirm the same scientific result,
`PROTOCOL1_OK`, again. But `20260810T151245Z` remains the official run
ID that this write-up, and the binding-clause inventory's discharges in
`gates.toml`, both point to.

Report file:
`stage2b/train/stage3/common/protocol1_propagation_report_20260810T151245Z.json`

Frozen ridge file:
`stage2b/train/stage3/common/protocol1_ridge_frozen_20260810T151245Z.npz`

## How the stress set was built

| part | detail |
|---|---|
| A | **regenerated** for this run (the 100 images with the largest encoding-stage difference, found from the provisional set B∪C∪D); `component_a_source = "regenerated"` |
| B | `true_count = 89`, `cap = 500`, `cap_applied = false`, `n_used = 89` |
| C | the per-class floor of at least 20 images, filled using the lowest official indices |
| D | 20 images per class, seed `42` |
| `n_stress` (total stress-set size) | **287** |
| `indices_refined` | **false** (the provisional B∪C∪D set already equalled the final set, once A was added back in — this is the expected path when regenerating) |
| `indices_sha256` (checksum of the index list) | `5ebded9ea78da1f66aa826683828c0990fbd57ab3b0c9f2682f320fa9c11ead6` |

The ARM stress encodings are an **index-join slice** taken from the
production `encoded_train_s1200.npz` file (the authoritative Phase-A
ARM encoding) — they are not a second, separate ARM encoding run. The
x86 stress encodings used the unmodified
`encode_stage3_local.encode_indices` function, run on Colab's x86_64
chip.

## Platforms

| role | machine |
|---|---|
| ARM encoding (production Phase A, sliced) | Darwin, ARM64 chip |
| x86 encoding (this protocol) | Linux, x86_64 chip (Colab) |
| propagation (evolve + frozen ridge + report) | Darwin, ARM64 chip |

## Five-stage maximum differences

Every table below carries the same framing: **this is the maximum value
observed inside the 287-image provisional stress set. It is not a
measurement over the whole corpus.** Part A of the stress set was chosen
specifically to make the encoding-stage difference as large as possible.

### Stage 1 — encoding

| quantity | max \|ARM − x86\| |
|---|---|
| `theta_505` | `4.441e-16` |

This is the maximum observed inside the 287-image provisional stress
set; it is not a corpus-wide measurement. The encoding-stage sanity gate
(which refuses above `1e-12`) did not fire. The historical Phase-A
spot-check found a maximum of about 3 ULP; this stress-set maximum is
consistent with that same scale.

### Stage 2 — evolved features (per condition, dimension 1008)

| condition | max \|ARM − x86\| |
|---|---|
| `pre_evolution` | `4.441e-16` |
| `T` | `1.769e-15` |
| `lattice` | `1.554e-15` |
| `rewired` | `1.554e-15` |
| `curr_random` | `1.332e-15` |

This is the maximum observed inside the 287-image provisional stress
set; it is not a corpus-wide measurement.

### Stage 3 — prediction (frozen ridge model, same fit and scaler on both chip types)

| condition | max \|ARM − x86\| |
|---|---|
| `pre_evolution` | `6.661e-16` |
| `T` | `4.610e-12` |
| `lattice` | `1.488e-11` |
| `rewired` | `1.711e-11` |
| `curr_random` | `3.576e-11` |

This is the maximum observed inside the 287-image provisional stress
set; it is not a corpus-wide measurement. There is one fitted ridge
model (`fit_final`) per condition, taken from
`ridge_final_g13_88edf9ac.npz` at the production alpha values. The team
never fits a separate model per chip type.

### Stage 4 — per-image clipped MSE

| condition | max \|ARM − x86\| |
|---|---|
| `pre_evolution` | `2.776e-17` |
| `T` | `9.975e-14` |
| `lattice` | `4.455e-13` |
| `rewired` | `5.483e-13` |
| `curr_random` | `1.830e-12` |

This is the maximum observed inside the 287-image provisional stress
set; it is not a corpus-wide measurement.

### Stage 5 — Δ_g = MSE_evolved − MSE_pre (the halt stage)

| graph | max \|Δ_g,ARM − Δ_g,x86\| | exceeds `4.604761e-10`? |
|---|---|---|
| `T` | `9.975e-14` | no |
| `lattice` | `4.455e-13` | no |
| `rewired` | `5.483e-13` | no |
| `curr_random` | `1.830e-12` | no |

This is the maximum observed inside the 287-image provisional stress
set; it is not a corpus-wide measurement.

Halt rule (frozen): if any graph's value is **strictly greater than**
the threshold, the result is `PROTOCOL1_HALT`. A value exactly equal to
the threshold does not halt. No graph exceeded the threshold.

## Scope limitation

Part A of the stress set is adversarial (deliberately worst-case) **for
the encoding stage only**. The function `rank_discrepancy_indices` ranks
images by comparing `theta_arm` against `theta_x86` — the encoding
values, not anything downstream. Ranking by post-evolution difference
instead would need evolving the full candidate population first, which
would defeat the purpose of using a small stress subset.

A clean `PROTOCOL1_OK` result shows "no unusual propagation on inputs
chosen to stress the encoding stage." It does **not**, on its own, show
that the evolution stage's own numerical sensitivity was stress-tested
on its own separate terms. The two are plausibly related, because
evolution's inputs are the encodings, but this is not guaranteed.

## Artifacts

| kind | object |
|---|---|
| stress indices | `stage2b/train/stage3/common/protocol1_stress_indices.npz` |
| ARM stress encoding | `stage2b/train/stage3/common/protocol1_encoded_stress_arm_s1200.npz` |
| x86 stress encoding | `stage2b/train/stage3/common/protocol1_encoded_stress_x86_s1200.npz` |
| frozen ridge | `stage2b/train/stage3/common/protocol1_ridge_frozen_20260810T151245Z.npz` |
| report | `stage2b/train/stage3/common/protocol1_propagation_report_20260810T151245Z.json` |
| theta_T / features | under `stage2b/train/stage3/{pre_evolution,evolved_*}/protocol1_{theta_T,features}_{arm,x86}.npz` |

Driver script: `run_arm_x86_propagation_stress.py`. Pure calculation
functions: `stage2b_audit.capped_positive_delta_indices`,
`rank_discrepancy_indices`, `max_abs_difference`,
`evaluate_propagation_halt`, `propagation_stage_maxima`. Tests:
`tests/test_stage2b_arm_x86_propagation.py`. Make targets:
`stage2b-protocol1-arm-construct`, `stage2b-protocol1-x86-encode`,
`stage2b-protocol1-propagate`.

## Stage 2B Companion Protocol 2: the `ABS_CONV_EPS` sensitivity table — result `PROTOCOL2_OK`

This ran using `run_abs_conv_eps_sensitivity.py`, on local CPU. It
recomputes results from the stored final-Delta arrays (the diagnostic
pickle file, plus the ladder's `encoder_gate_s1200.npz` file). No
re-encoding happens.

**Result at the locked `ENCODER_STEPS=1200`: INVARIANT** (unchanged)
across `eps in {1e-10, 1e-11, 1e-12, 1e-13}`. No verdict flip happened.
The `HALT` condition did not fire. `ABS_CONV_EPS=1e-12` itself does not
change as a result.

### How this was built

This is a recomputation from the stored final-Delta values, not a new
encoding run. It uses `load_final_deltas` (the diagnostic source),
`load_ladder_encoder_gate_deltas`, and `merge_step_sources` (which
requires agreement of `1e-15` or better at the overlapping 1,200-step
point). It then calls `audit.sensitivity_table(...,
gate.evaluate_rho_gate)`, unmodified.

### Summary (from the published result file)

- Locked encoder step count: 1200
- Invariant at the locked step count: true
- Halt triggered: false
- Step sources at 1,200 steps: "diagnostic+ladder"; at other step
  counts: "diagnostic" only
- Ladder source object for 1,200 steps:
  `stage2b/train/stage1/common/encoder_gate_s1200.npz`

### The four justification factors (all four checked)

1. **float64 precision**: the observed numerical noise is 1e-14 to
   1e-16; 1e-12 sits above this band.
2. **Phase update scale**: the smallest meaningful measured final-Delta
   value is 2.177e-07 (clean images, 150 steps, stage 1); 1e-12 is five
   or more orders of magnitude below that.
3. **The encoder's own implementation**: the residual decays from
   8.370e-07, to 8.062e-13, to exactly 0.0, at 300, 600, and 1,200
   steps. It first crosses below 1e-12 somewhere between 300 and 600
   steps.
4. **Downstream feature sensitivity** (an analytic bound on the largest
   possible change to the cosine and sine features, given a phase
   residual):
   - at eps=1e-10: the bound is 1e-10 (0.0001 times the solver's
     relative tolerance of 1e-6; 0.405 times the largest measured
     production final-Delta of 2.468e-10)
   - at eps=1e-12: the bound is 1e-12 (1e-6 times the relative
     tolerance; 0.00405 times the largest measured production value)
   - at eps=1e-13: the bound is about 1e-13 (the bound decreases
     steadily; all values stay far below both the relative tolerance
     and the largest measured production value)
   - Method: `|cos(θ+ε)-cos(θ)| ≤ 2|sin(ε/2)| ≤ |ε|`. No full ODE
     re-evolution is needed to compute this bound.

### Artifacts

- Table file: `results/abs_conv_eps_sensitivity_table.json` (also
  published in the cloud-storage style, under `stage3/common`)
- A fingerprint is present, with source and configuration checksums.
- Sentinel: `PROTOCOL2_OK` (program exit code 0); the re-validation
  check passed.

Driver script: `run_abs_conv_eps_sensitivity.py`. Make target:
`stage2b-protocol2`. Tests:
`tests/test_stage2b_abs_conv_eps_sensitivity.py` (covering the merge
step, justification factor 4, the fingerprint-publishing check by
reading the code's syntax tree, the re-validation sentinel, and a
tier-2 check against the real diagnostic pickle file).
