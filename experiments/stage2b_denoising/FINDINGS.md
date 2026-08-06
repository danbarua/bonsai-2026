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

Feasibility ladder stage 2 (5,000-image development subset), per
`DESIGN.md`'s locked ladder -- see that document for what stage 2 adds
(runtime and feature-validity measurement at scale, the condition-number
diagnostic, the second ridge-equivalence pass, CNN development). Not yet
started.
