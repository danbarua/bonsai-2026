# Stage 2B: Does Runtime Oscillator Evolution Improve Single-Step Denoising?

**DESIGN STATUS: LOCKED.** Seven drafts, four external review rounds
(ChatGPT reviewer), one adversarial blind-spot review (Claude Fable 5),
one outsider peer review (Grok). Final sign-off received after three
execution corrections (intercept-aware ridge formula, SVD count, literal
optimizer configuration), all incorporated below. No further scientific
review round required. Changes after this point require an explicit,
disclosed post-lock amendment, per project convention.

**Task name, used consistently throughout**: single-step active-support
reconstruction under a fixed, majority-censored clipped-Gaussian
corruption. The unusual severity of the corruption is part of the
design's identity, not a diagnostic footnote.

## The question, precisely

> Does runtime graph evolution, on top of the same already-dynamically-
> encoded local phase state, improve single-step denoising prediction
> error, relative to the unevolved encoded state alone?

**Strongest supportable claim, stated in advance**: "Runtime graph
evolution improves linear reconstruction of clean intensities on the
fixed 505-pixel active support under one prespecified clipped-Gaussian
corruption level." "Actual denoising" is added to that claim only if
the hierarchical identity-baseline gate also succeeds. The two claims
are distinct and will not be conflated.

## Corpus construction: full-image noise, then encode, then restrict

```
x_0^784        = clean image, full 28x28, in [0,1]
x_t^784        = sqrt(alpha_bar_t) * x_0^784 + sqrt(1-alpha_bar_t) * epsilon,  epsilon ~ N(0,I)
x_t^784_clip   = clip(x_t^784, 0, 1)
theta_0^784    = encode(x_t^784_clip)          -- _local_converged_phases, full grid, unaltered
theta_0^505    = theta_0^784[active_indices]   -- restrict AFTER encoding
theta_T^505    = F_W^2.5(theta_0^505)          -- graph evolution, W = T or a control
```

Reconstruction target: `x_0` restricted to `active_indices` (505-dim).

## Corruption level: alpha_bar_t = 0.5, frozen, censoring profile stated as fact

Equal signal and noise coefficients before clipping (`sqrt(0.5)` each)
-- not "balanced signal-to-noise": KMNIST images are sparse,
non-centered, non-unit-variance, and clipping further distorts the
effective ratio.

**This corruption process is majority-censored, computed in advance**
(noise std 0.707):

| x_0  | P(clip below 0) | P(clip above 1) | total |
|------|-----------------|-----------------|-------|
| 0.00 | 0.500           | 0.079           | 0.579 |
| 0.25 | 0.401           | 0.122           | 0.523 |
| 0.50 | 0.309           | 0.180           | 0.489 |
| 0.75 | 0.227           | 0.253           | 0.480 |
| 1.00 | 0.159           | 0.339           | 0.498 |

Roughly half of all pixel values clip at every clean intensity (~54%
overall on a sparse-image proxy). Two stated consequences: (a) the
identity baseline is substantially stronger than naive intuition
suggests -- clipping alone corrects ~half of all noise excursions in
the correct direction for extreme-valued pixels; (b) background-
dominated MSE is heavily shaped by the clip-at-zero mechanism, making
the foreground/background breakdowns load-bearing.

**This level is frozen.** Adjusting it because measured clip rates
"look awkward" is post-hoc tuning this project refuses. Empirical clip
rates are reported (below-zero and above-one separately, 784 and 505
scopes, per class) as confirmation against this table, not discovery.

## Corruption RNG, exact values

```python
MASTER_SEED = 42
seed_bytes = SHA256(f"{split}:{index}:{MASTER_SEED}".encode()).digest()
seed = int.from_bytes(seed_bytes[:8], "little")
rng = numpy.random.Generator(numpy.random.PCG64(seed))
epsilon = rng.standard_normal(784, dtype=numpy.float64)
```

`split` is the literal string `"train"` or `"test"`. Not Python's
process-salted `hash()`. One realization per image, reused identically
across every condition. Inference is over the official test-image
sample under the prespecified one-independent-corruption-per-image
protocol; it does not estimate repeated-noise variability conditional
on a fixed image.

**Test-use scope**: the official KMNIST test images were used
extensively by Stage 2A and are not project-unseen. What is locked: the
prespecified Stage 2B corrupted test corpus, test features, model
predictions, and denoising scores are generated and inspected in one
final confirmatory evaluation only; no Stage 2B test-side result is
accessed during stages 1-3.

**Corruption diagnostics**: pre-clip `MSE(x_t, x_0)` and post-clip
`MSE(clip(x_t), x_0)`, each on both the 505 and 784 scopes. The
active-support post-clip value is the identity baseline used by the
hierarchical gate.

## Encoding pipeline

- Gauge: reference-node, `theta_ref` = node 363 (T's median-weighted-
  degree node; Stage 2A's final locked choice).
- Features: circular embedding (cos/sin per node), 1008-dim, the two
  trivially-constant reference-node columns dropped deterministically.
- Standardization: `StandardScaler`, fit per training fold only,
  applied identically to every condition including raw-pixel ridge
  inputs.
- Encoder RNG: seed 0 per image, primary; independent-per-image seeds
  as secondary robustness only.
- Inherited constants, literal: fold seed 42; canonical graphs -- T
  (learned, no seed), lattice (deterministic), rewired
  (`degree_preserving_rewire`, seed=0), curr_random
  (`generate_matched_sparsity_topology`, seed=0) -- Stage 2A's own
  canonical instances.

## Encoder-on-noisy-inputs gate (feasibility stage 1, executable)

`_local_converged_phases` was built and validated on clean, spatially-
smooth images only; clean-input convergence does not transfer to
majority-censored noisy inputs by assertion. At stage 1, on the same
1,000 images, record each image's final-iteration maximum absolute
phase update (final-Delta), encoding clean and noisy versions
separately.

**Gate**:

```
rho = median(Delta_noisy) / max(median(Delta_clean), 1e-15)
PASS iff rho <= 10
```

The `1e-15` floor is numerical protection, not a scientific threshold.
The 10x multiplier is arbitrary but pre-registered, locked before any
data exists. **Automatic failures, regardless of rho or the absolute-
convergence escape below**: any non-finite encoded phase; any non-finite
final-Delta. Both medians recorded in the stage-1 log regardless of
outcome; 95th-percentile final-Delta (noisy and clean) logged alongside
-- visibility for a passing-median-but-exploding-tail pattern, explicitly
not a second gate. Exceedance halts the stage pending investigation.

**Post-lock amendment (2026-08-06, after feasibility stage 1's first real
run): ENCODER_STEPS raised 150 -> 1200, and the gate formula gains an
absolute-convergence escape.** The gate's first execution on real,
majority-censored KMNIST FAILED honestly, as designed: rho=169.851
against the threshold of 10 (median final-Delta clean 2.177e-07, noisy
3.698e-05; zero non-finite values anywhere -- a clean ratio failure, not
a numerical blow-up). Investigated in
`experiments/stage2b_denoising/diagnose_encoder_gate_failure.py`
(diagnostic-only, no locked pipeline code touched by it), which
reconstructs the exact stage-1 corpus and corruption locally and verifies
that reconstruction bit-for-bit against the failed run's own reported
identity-baseline MSE before trusting anything computed from it
(confirmed exact, relative diff 0.000e+00).

Two measurements, pre-committed before either was run. (1) A convergence
curve across five step counts (75, 150, 300, 600, 1200): noisy final-
Delta decays geometrically all the way to EXACT float64 zero -- median
AND p95, every one of 1,000 images -- by 1,200 steps, the same fixed
point clean reaches. No floor above a meaningful scale exists; the
encoder converges on censored inputs, it is simply slower to. (2) Per-
image state drift from 150 to 600 steps (noisy) against the typical
between-image circular distance at 150 steps: median ratio 4e-4 -- the
phase field has, for all practical purposes, stopped moving relative to
the scale that distinguishes one image from another long before the
Delta metric says so. Both measurements independently support the same
reading: genuine, if slow, convergence, not a qualitatively different
regime on noisy inputs.

**A second, independent defect surfaced in the same investigation: the
ratio formula is unstable near either series' own numerical floor.** At
steps=600, clean's median had already hit exact 0.0 while noisy's sat at
1.776e-14 -- nine orders below the smallest meaningful final-Delta
measured anywhere (2.177e-07) -- yet the gate reported FAIL at rho=17.76,
because `max(0.0, 1e-15)` silently turned a RATIO gate into an ABSOLUTE
test against the 1e-15 floor. The full rho trajectory across the five
step counts (14.98, 169.9, 1.915e4, 17.76, 0.0) is non-monotone for
exactly this reason: it tracks which series crossed its own float64
floor first, not whether the mechanism converged. A threshold sitting
inside that crossover band is fragile by construction, independent of
where ENCODER_STEPS ends up.

**Fix, both parts disclosed together:**

```
rho = median(Delta_noisy) / max(median(Delta_clean), 1e-15)
PASS iff rho <= 10 OR (median(Delta_clean) < 1e-12 AND median(Delta_noisy) < 1e-12)
```

`ABS_CONV_EPS = 1e-12` follows the same margin logic as `RHO_THRESHOLD`
and `MEDIAN_FLOOR`: 5+ orders below the smallest meaningful measured
Delta (2.177e-07) and well above observed float64 dust (1e-14 to 1e-16),
so it cannot fire on a genuinely still-converging signal, only on values
already indistinguishable from numerical noise. It requires BOTH medians
below it, not either -- a lopsided case (one series converged, the other
still measurably moving) falls through to the ordinary rho test on its
own merits.

**The decision rule that selected S*=1200 is stated precisely because an
earlier, looser version of it was caught before being applied.** "Some
S* brings noisy final-Delta within 10x of clean-AT-150" is satisfied as
early as S*=300 -- which then fails its own re-run at rho=1.915e4,
because the real gate compares same-step (noisy(S*) vs. clean(S*)), not
against a stale reference from a different step count. The corrected
rule is same-step, both series required to have genuinely converged, and
it is **verdict-invariant at the chosen operating point**: S*=1200 passes
under either the original (flawed) reading or the corrected one, so the
correction is not what selected this outcome -- both readings agree here.
1200 is also the only step count in the five-point scan that passes
ROBUSTLY (both medians and both p95s exactly 0.0, maximum possible
margin) rather than sitting near the fragile crossover band a smaller,
untested S* might land in; no finer scan between 600 and 1200 was run --
encoder cost scales with steps, and robustness of the operating point was
judged worth more than shaving minutes off it. Revisit as its own
disclosed decision if measured full-scale cost ever makes that trade
different.

Every other stage-1 artifact (corpus, topologies, corruption, corruption
diagnostics) is encoder-independent and unaffected; only the encoder-gate
step recomputes. Its GCS object name carries the step count
(`encoder_gate_s{steps}.npz`), so the 150-step FAIL artifact remains in
the bucket, untouched, as the historical record of the first real run --
not deleted, not silently reused under stale semantics.

## Foreground mask

`m_ij_ink = 1[x_0,ij > 0.15]`, computed from each individual image's
own clean pixels on its 505 active-support coordinates. This is NOT the
construction-time class-mean threshold -- a different object sharing
the same numeric value. Zero-foreground images excluded from that
image's foreground breakdown, counted and reported. Averaging:
per-image first, then across images.

## Dtype rationale, per stage

| stage                 | dtype   | rationale |
|-----------------------|---------|-----------|
| ODE evolution         | float64 | required; established in Stage 1D's GPU verification (silent float32 truncation was a real caught failure) |
| Ridge SVD             | float64 | motivated: Stage 2A measured ~2e6 condition numbers on evolved-feature design matrices; float32's ~6e-8 precision x 2e6 conditioning => ~12% worst-case relative error; float64 => ~2e-10. Transfer to 2B's features is plausible, not established -- hence the stage-2 diagnostic below |
| Corruption generation | float64 | not numerically required; kept because the locked RNG spec produces it and spec reproducibility outweighs a trivial saving |
| Feature storage       | float64 | not required (3.5GB vs 1.8GB -- immaterial on target hardware); kept for uniformity |
| Sign-flip matrices    | int8    | +-1 needs no floating point; 6GB for all six contrasts vs 48GB at float64 |

**Stage-2 diagnostic (required)**: record `cond(standardized design
matrix)` per condition, one line each -- converts the float64-ridge
motivation from inherited story to checked table.

**Post-lock disclosure (2026-08-06, before ladder stage 3): encoded
features are bit-reproducible WITHIN a CPU architecture, not across
one.** Stage 3 splits execution into a local CPU encode phase and a
remote GPU phase (see "Computational strategy"), which put the encoder
on Apple Silicon for the first time; stages 1 and 2 both encoded on
Colab's x86. Measured directly on the same images rather than assumed,
before any stage-3 artifact was written:

| comparison | result |
|---|---|
| same machine (this Mac), encode run twice | bit-exact |
| two different Colab sessions, same 1,000 images | bit-exact, 784,000/784,000 coordinates |
| Mac (ARM) vs Colab (x86), same images | 93.4% coordinates identical; max 3 ULP (4.441e-16), mean 0.07 ULP, max relative 3.98e-16 |

The cross-architecture difference is at the float64 representation
limit, and it cannot amplify: the encoder is a contraction toward a
fixed point, measured residual against the 1,200-step result falling
8.370e-07 (300 steps) -> 8.062e-13 (600) -> exactly 0.0 (1,200). Both
platforms resolve the SAME fixed point; a minority of coordinates land
on an adjacent representable float64. Downstream the encoded phases feed
an ODE solver at `rtol=1e-6`, eight or more orders above this
difference, so no reported quantity in this design can turn on it.

Recorded rather than waved through, and scoped precisely: what is
guaranteed is within-architecture determinism, and stage 3's encoded
array in GCS is the artifact of record. Re-deriving it from scratch on a
different CPU architecture reproduces it to ~1 ULP, not bit-exactly.
Same standard, and the same reasoning, as this project's existing
acceptance that "cuML logistic regression on GPU is not bit-reproducible,
and that is fine at the level the claims are made"
(`docs/PROJECT_MEMORY.md` Part 4).

## Readout: multi-output ridge -- JAX SVD production path, sklearn as oracle

One multi-output ridge, shared `alpha` across all 505 outputs, grid
`{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6}`
(**amended post-lock after feasibility stage 3 -- see Review history**;
originally `{1e-2 .. 1e6}`), `fit_intercept=True`, inputs standardized
per-fold, targets unstandardized. Tie: mean validation MSE within `1e-10`
absolute; **larger alpha wins**. Folds:
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, identical
partition across all conditions.

**Decade spacing is exact and frozen.** Thirteen values, one per decade,
no interpolation and no densification around an observed minimum, ever.
This is a frozen procedural clause rather than a default: a grid that may
be refined near a minimum after the minimum has been seen has a stopping
rule chosen by looking at results, which is the shape the amendment below
exists to make impossible.

**Production implementation: direct JAX SVD-ridge, intercept-aware.**
Per (fold, condition): one thin SVD of the standardized training
features; all thirteen alphas evaluated from that single decomposition.
Because `fit_intercept=True` and the 505 target columns are left
unstandardized, the solve must center targets within the training fold
and restore the intercept from the general expression -- not assume it
away:

```
Y_tilde  = Y - mean(Y_train)                      # center each target column, training fold only
W_alpha  = V @ diag(s / (s^2 + alpha)) @ U.T @ Y_tilde
b_alpha  = mean(Y_train) - mean(X_train) @ W_alpha
```

(`mean(X_train)` is approximately zero after standardization, so
`b_alpha` will normally reduce to `mean(Y_train)` -- but the general
expression is what gets implemented, not the shortcut.)

**SVD count: 42, not 35.** 35 fold-level SVDs for model selection
(7 conditions x 5 folds, all 9 alphas reusing each decomposition),
followed by **7 full-training SVDs for the final ridge refits** at each
condition's selected alpha. Runtime accounting and cloud-job planning
use 42.

**The seven ridge conditions, and why the statistics families have
six keys.** The seven fitted conditions are `raw_505`, `raw_784`,
`pre_evolution`, `evolved_T`, `evolved_lattice`, `evolved_rewired`,
`evolved_curr_random`. The statistics operate on a DIFFERENT set of six
keys: `pre_evolution`, the four evolved graphs, and the identity
baseline. `raw_505` and `raw_784` are descriptive comparators only --
named watched-outcome 5 is about them -- and belong to neither
multiplicity family; the identity baseline belongs to neither family
either (it enters through the hierarchical gate above), and is fitted by
nothing, so it is not one of the 42 SVDs. Seven for compute accounting,
six for the statistics: neither number is a typo for the other.

**sklearn (`Ridge(solver="svd")`) is the verification oracle -- not in
the production path, never deleted.** **Equivalence gate, literal**: at
both the 1,000- and 5,000-image stages, JAX and sklearn must produce
(a) max absolute difference in clipped validation predictions <= 1e-8,
and (b) identical alpha selection. Any exceedance halts pending
investigation. Diagnostic only, not a second halt rule: max absolute
coefficient difference recorded at a looser tolerance for visibility --
prediction agreement is what matters for the endpoint.

**Post-lock amendment (before feasibility stage 1): the scaler-centering
guard's tolerance is n-dependent.** The per-fold guard on the
standardized training features halts when
`||mean(X_train_scaled)|| >= 1e-9 * (n / 1000) ** 0.5`, where `n` is the
row count of the matrix being checked -- the fold's training rows, not
the rung's nominal corpus size. It was previously a fixed `1e-10`.

Anchor and exponent come from different kinds of evidence, deliberately.
The anchor is measured: 1e-9 is a 12.7x margin over the worst value the
GPU spike recorded at n=1,000 on production-path evolved features
(`curr_random`, 7.87e-11 -- `README.md` holds the tables). The exponent
is mechanical: `||mean(X_scaled)||` grows because the mean of `n`
float64 values carries ~sqrt(n) accumulated rounding, amplified by
division by a small column standard deviation. 0.5 upper-bounds the
0.405 growth measured on `curr_random`, the binding condition, so its
margin widens with `n` rather than eroding -- 14.8x at the measured
n=5,000, ~18x projected at n=54,000. Two of the non-binding conditions
grew faster than sqrt(n) on the same two points and so narrow instead,
from margins three orders wide; `README.md`'s table carries every
condition's exponent. A fixed absolute tolerance on a sqrt(n)-growing
quantity halts eventually on any features at all, which is the failure
this replaces rather than a property of these features.

The guard keeps its detection power at both ends. 7.35e-9 at n=54,000
is four or more orders below the ~3e-4 level at which `||mean(X)||`
begins degrading the 1e-8 equivalence gate above (measured, `README.md`),
and about nine orders below the O(1) offset a genuinely broken scaler
produces. The exponent 0.66 fitted from the earlier CPU-evolved table is
superseded: it measures a different pipeline from the one the anchor
comes from.

**Model selection**: alpha chosen by clipped validation MSE
(`argmin_alpha MSE(clip(x_hat_alpha, 0, 1), x_0)`); ridge fitted
without output clipping; raw validation MSE diagnostic only.

## Prediction-range handling

Primary MSE after deterministic clipping to [0,1], every condition.
Raw unclipped MSE as secondary diagnostic; clipped-boundary fractions
reported per condition. Selection criteria (ridge alpha, CNN
checkpoint) use clipped validation predictions; training losses remain
raw -- the distinction is locked, not left to implementation.

## Identity baselines: hierarchical gate plus a descriptive rescaled variant

1. **Primary**: `evolved_T` vs. `pre_evolution`.
2. **Only if primary succeeds**: `evolved_T` vs. identity
   (`clip(x_t)`, active-support, post-clip) -- the denoising gate;
   never rescues a failed primary.
3. **Always reported independently**: `pre_evolution` vs. identity, own
   paired bootstrap interval -- context, outside the gate.

**Descriptive baseline (added at final review, not part of the gate)**:
the rescaled identity

```
x_hat_0_rescale = clip( x_t_clip / sqrt(0.5), 0, 1 )
```

uses only the known corruption coefficient, zero learned parameters --
it shows whether a model does more than undo the deterministic signal
attenuation. Reported descriptively alongside the other baselines;
the raw-pixel ridge remains the stronger learned comparator.

## Primary comparison and test

`d_i = MSE_i(evolved_T) - MSE_i(pre_evolution)`, active-support,
post-clip, official test set, evaluated once. 20,000-resample paired
class-stratified bootstrap, two-sided 95% percentile interval,
`seed=42`. Interval entirely below zero => evolution improves; entirely
above => pre-evolution wins; straddling => null, stated plainly.

(At n=10,000, the paired t-tests below are justified by the CLT even
for skewed per-image MSE differences -- stated once so assumption-
scrutiny applies symmetrically, not only to the sign-flip robustness
check.)

## Secondary comparisons: two families

- **Family 1 (three tests)**: lattice/rewired/curr_random vs.
  `pre_evolution`. Paired t-tests produce p-values; Holm across the
  three; bootstrap intervals alongside for effect size.
- **Family 2 (six tests)**: all pairwise comparisons among the four
  evolved graphs. Paired t-tests, Holm across the six. Sign-flip
  retained as robustness only: 100,000 flips, seed 42, studentized
  statistic `T_b = mean(s_b * d) / (SD(s_b * d) / sqrt(n))` matching
  the t-test's form, two-sided (`|T_b| >= |T_obs|`), +1 correction
  (`p = (1 + count) / 100001`), sign-exchangeability assumption stated
  wherever reported.
- The primary test sits outside both families, uncorrected.

**"One graph wins," branched per candidate**: T qualifies iff the
primary bootstrap interval lies entirely below zero. A control graph
qualifies iff its direction is favorable and its Family-1 Holm-adjusted
p < 0.05. A graph is the unique winner only if it qualifies by
whichever rule applies to it AND outperforms each of the other three in
the favorable direction after Family-2 Holm correction.

## CNN: equinox + optax, locked

**Framework**: equinox + optax -- both already project dependencies
(diffrax pulls in equinox; optax added previously), one framework end
to end, one RNG story, one GPU memory pool, zero new dependencies.
**PyTorch is the documented fallback-not-taken**: rejected because it
introduces a second deep-learning framework sharing one GPU with JAX,
whose default ~75% memory preallocation starves a co-resident torch
process absent explicit configuration -- a concrete known failure mode,
not a style preference. TensorFlow: rejected, no advantage at
second-framework cost.

**Architecture** (residual formulation, locked prospectively):

```
x_hat_0 = x_t_clip + f_psi(x_t_clip)
f_psi:  Conv2d(1,32,k=3,p=1) -> ReLU -> Conv2d(32,32,k=3,p=1) -> ReLU -> Conv2d(32,1,k=3,p=1), linear output
```

**9,857 trainable parameters** (computed:
`(1*32*9+32) + (32*32*9+32) + (32*1*9+1)`). Convolutional biases
enabled, consistent with that count. Zero padding for `padding=1`.
Scope-matched: masked active-support MSE for training and evaluation;
full-image MSE never reported for this model (279 output coordinates
receive no training signal under the mask). Linear final layer -- no
sigmoid -- so its unclipped diagnostic is categorically comparable to
ridge's; the shared clipping rule applies at evaluation only. CNN input
receives no normalization beyond the locked clipping to [0,1].

**Optimizer, literal** (no library defaults left implicit):

```python
optax.adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)
```

No weight decay. No gradient clipping. Loss: mean over batch images and
their 505 active coordinates (raw, unclipped, per the training/selection
distinction above).

**Training, literal**: batch 128, max 100 epochs, early-stopping
patience 10 on clipped active-support validation MSE, `min_delta=0.0`
with strict `<` improvement, best checkpoint restored. Validation
partition: `StratifiedShuffleSplit(n_splits=1, test_size=0.10,
random_state=42)`, created BEFORE feasibility subsets are drawn (the
5,000-image development subset comes only from the remaining 90%); at
stage 3, "full training" = 54,000 fit + 6,000 locked validation. Three
seeds (0,1,2), each jointly governing initialization, minibatch order,
and framework randomness; best-of-3 by clipped validation MSE,
training-derived only -- official-test performance never inspected
during selection.

## Computational strategy

| stage                 | shape                          | where         | working-out |
|-----------------------|--------------------------------|---------------|-------------|
| Corruption generation | 70k x 784 Gaussians            | CPU           | ~55M draws, vectorized numpy, <2s |
| Encoding              | 70k x 1200-iter local update   | CPU reference | ~4.6ms/image at 150 iters measured (2A); iteration count raised 150->1200 (see "Encoder-on-noisy-inputs gate" amendment) -- full-scale throughput at 1200 iters not yet measured, gated by that measurement before a firm estimate is stated; any port verified against reference first |
| Evolution             | 70k x 4 topologies             | GPU           | verified JAX (`evolve_on_graph_jax.py`), ~0.67ms/img/topology batched => ~3-4 min |
| Ridge                 | 42 SVDs (35 fold-level + 7 final refits, ~48-60k x 1008) | GPU | ~0.1 TFLOP each; seconds on A100, ~1 min each demonstrably viable on CPU |
| CNN                   | 9,857 params                   | GPU           | standard |
| Bootstrap/sign-flip   | chunked 512-4096 resamples/batch | GPU default | int8 signs (6GB all contrasts vs 48GB float64); ~0.8GB/contrast bootstrap; host loop over chunks permitted, per-resample Python loop is not; CPU demonstrably viable as fallback |

Generation, features, and statistics run entirely in the cloud
environment; artifacts pushed to Google Cloud Storage from within it --
never round-tripped through local upload (Stage 2A's 242MB-vs-~6-15MB
Colab upload limit, already hit once).

**Post-lock amendment (2026-08-06, before ladder stage 3): stage 3 runs
in two phases, and the encode phase runs locally.** The constraint above
is preserved in substance, and its own parenthetical states why it
exists: the failure it guards against is getting a large file INTO a
Colab session through that session's upload mechanism. A direct
local->GCS write never touches that mechanism -- it uses the same
`google-cloud-storage` client and the same chunked, resumable,
crc32c-verified transport in `stage2b_gcs.py` that a cloud-side write
uses, and `stage_kmnist_inputs.py` has moved KMNIST that way for both
prior rungs. The rule that is actually load-bearing is "feature
artifacts live in GCS, written through the verified transport", not
"every computation happens inside a Colab session".

What the split avoids is a real cost, not a hypothetical one: encoding
is the pipeline's one genuinely CPU-bound step -- 1,099s of stage 2's
1,722s total -- while evolution, ridge and the CNN are what use the
GPU. Running the encode inside a provisioned session leaves a metered
A100 idle for the majority of the run's wall-clock. Phase A (corpus,
corruption, encode, restrict) therefore runs on local CPU cores and
writes only the encoded array; Phase B (evolution, ridge, CNN) reads it
and regenerates corruption and clean targets in-session, both being
deterministic and cheap, so 218MB crosses the boundary instead of
775MB. `ensure_artifact`'s 64MB auto-chunk threshold engages on that
upload without the call site asking. See the dtype section's disclosure
for the cross-architecture reproducibility consequence, measured before
the decision was acted on.

Colab/GCP is a compute
RUNTIME for this implementation only -- not a commitment about
final deliverable format. The visuals/plots delivery format is an
explicitly DEFERRED decision, made once results exist, and is not
part of this locked design. Do not treat notebook packaging as a
pending or implied task at any point before then. GCS paths are
implementation details.
FINDINGS.md in-repo remains the record of note.

## Feasibility ladder

1. 1,000 official-training images -- mechanical correctness, encoder
   gate, ridge equivalence check (first pass).
2. Fixed 5,000-image development subset -- runtime, feature validity,
   ridge-grid behavior, condition-number diagnostic, ridge equivalence
   check (second pass), CNN development.
3. Full 60,000-image training side -- 54k fit / 6k locked validation,
   feature generation, model selection.
4. ONE locked evaluation on the official 10,000-image test corpus.

No Stage 2B test-side result accessed before stage 4.

## Named watched-for outcomes

1. All four graphs improve over pre-evolution; no pairwise difference
   detected (observational, not equivalence).
2. One evolved graph qualifies per the branched rule AND outperforms
   each of the other three after Family-2 correction.
3. Evolution does not help denoising, despite helping classification --
   a task-dependent dissociation.
4. The unevolved encoding alone carries most of the value; evolution
   adds little.
5. Raw-pixel ridge dominates every phase-based condition, evolved or
   not -- the phase representation itself is reconstruction-lossy. A
   priori plausible: the encoder was built to converge phases toward a
   target field for topology construction, never to preserve intensity
   information, and cos/sin of converged phases is a lossy nonlinear
   transform. If observed, a coherent, reportable dissociation from
   Stage 2A (representation useful for discrimination, harmful for
   reconstruction) -- a watched outcome, not a success criterion;
   success remains defined by the evolution-vs-encoded contrasts under
   the identity gate.

## What this does not do

No multi-step reverse sampling or generation. No timestep conditioning
or multi-noise-level training. No noise-target (epsilon) prediction.
Does not revisit Stage 2A's open items (#9, #10, #11).

## Review history

- Drafts 1-6: four ChatGPT review rounds (noise-ordering bug; ridge
  tie-break direction error; identity-gate hierarchy; two multiplicity
  families; literal RNG/fold/sign-flip values; chunked vectorization;
  CNN scope-matching; one-graph-wins branching for T).
- Fable-5 adversarial blind-spot review: majority-censored corruption
  computed in advance; raw-pixel-ridge named outcome; encoder-on-noise
  gate; per-stage dtype rationale; symmetric assumption-scrutiny; JAX
  SVD-ridge production path; equinox+optax framework lock.
- Grok outsider review: temperings on float64 transfer, oracle
  retention, framework choice as preference-not-constraint, and
  post-lock sigma-tuning refusal -- all incorporated.
- Final ChatGPT round: intercept-aware ridge formula; SVD count 42;
  literal optax configuration; rescaled-identity descriptive baseline;
  encoder-gate numerical completeness; task naming convention. LOCKED.
- Post-lock amendment, before feasibility stage 1: n-dependent
  scaler-centering tolerance, `1e-9 * (n / 1000) ** 0.5`, over a fixed
  `1e-10` a sqrt(n)-growing quantity outgrows; the seven ridge
  conditions and the distinct six-key statistics set, enumerated beside
  the SVD count.
- Post-lock amendment, computational strategy: Colab-notebook
  final-deliverable wording removed as an implied task; delivery format
  deferred, Colab/GCP a compute runtime.
- Post-lock amendment, after feasibility stage 1's honest FAIL
  (rho=169.851 at ENCODER_STEPS=150): raised to 1200, diagnosed by
  `diagnose_encoder_gate_failure.py` as genuine slow convergence to the
  same float64 fixed point, not a floor; gate formula gains an
  absolute-convergence escape (`ABS_CONV_EPS=1e-12`) after the same
  diagnostic exposed the ratio formula reading numerical dust as a real
  failure at steps=600; the decision rule that selected S*=1200 is
  stated as verdict-invariant under its own earlier, corrected defect,
  disclosed rather than silently fixed; encoder-gate artifact naming
  carries the step count so the 150-step FAIL remains in the bucket as
  history.
- Post-lock amendment, before feasibility stage 3: two-phase execution,
  local CPU encode writing only the encoded array to GCS and a remote
  GPU phase regenerating corruption and targets in-session -- the
  "generate in the cloud" convention's own stated rationale is the Colab
  session upload limit, which a direct local->GCS write on the already-
  verified transport never touches, and an in-session CPU encode would
  leave a metered A100 idle for most of the run; with it, the disclosed
  cross-architecture consequence, measured before the decision was acted
  on: encoded features are bit-reproducible within a CPU architecture
  (Colab-to-Colab bit-exact across sessions; this Mac bit-exact across
  runs) and agree to a maximum of 3 ULP across architectures, damped
  rather than amplified by the encoder's contraction to a fixed point.
- Post-lock amendment, after feasibility stage 3 Phase B: **the alpha
  grid extends downward by four decades to
  `{1e-6 .. 1e6}`**, thirteen values, original decade spacing preserved
  exactly, applied to **all seven conditions**.

  *Provenance*: ChatGPT review ruling, delivered via Dan's manual
  copy-paste from his ChatGPT chat rather than through the c2gpt channel
  — recorded because a ruling's route is part of its evidence, and this
  one did not travel the audited path. Operative text, verbatim:

  > "So my previous ruling stands unchanged: alpha in {1e-6, 1e-5, 1e-4,
  > 1e-3, 1e-2, 1e-1, 1, ..., 1e6} for all seven conditions, with the
  > exact extension frozen before fitting. Re-run the full ridge
  > procedure, do not splice lower-alpha points into the old tables,
  > re-run the production-scale JAX/sklearn equivalence checks, and halt
  > for review if any production condition selects 1e-6. One additional
  > execution detail should be frozen now: the amended grid should
  > preserve the original decade spacing exactly, so there is no later
  > ambiguity about interpolation or denser searching around an observed
  > minimum."

  *What prompted it*: Phase B's ridge selected the grid-MINIMUM alpha in
  six of seven conditions, where at stage 2 (n=5,000) none did. The
  scrutiny step cost nothing — the validation curves were already in the
  committed artifact — and showed the six split: both raw baselines are
  plateaued (relative MSE rise ~1e-7 across two decades, floor-pin
  cosmetic), while the four EVOLVED conditions rise 0.4-0.8% per decade
  off the floor with the argmin at the boundary. The constraint therefore
  bound asymmetrically across exactly the treatment/control line the
  readout exists to compare.

  *The frozen procedure, one-shot by construction*: a full re-run of the
  entire ridge procedure on the new grid (all seven conditions, all
  folds, all refits); **no splicing** of lower-alpha points into any
  existing table; the production-scale JAX/sklearn equivalence check
  re-run on the new grid at the frozen `1e-8` tolerance; and a **HALT for
  review if any production condition selects `1e-6`** — a pin at the new
  floor is a named anomaly and is never itself a trigger for further
  extension. Together with the exact-spacing clause above, the
  widen-look-widen loop is structurally unavailable rather than merely
  discouraged: there is no second extension this procedure can authorise.

  *Two consequences found while drafting this, neither anticipated by
  the ruling*:

  1. **The write-once invariant enforces "no splicing" by construction.**
     `ridge_cv.json` and `ridge_final.npz` are LINEAGE artifacts and
     create-once; `force=True` on either raises `WriteOnceViolation`
     before `produce` runs. The re-run therefore CANNOT overwrite the
     nine-decade results and must write new names carrying the grid
     identity, exactly as Phase A wrote `encoded_train_s1200` rather than
     overwriting `encoded_fit_s1200`. The nine-decade tables survive as
     history because the storage model refuses to destroy them.
  2. **`PROBE_JAX_SVD_COUNT = 42` is right as design accounting and wrong
     as a cost multiplier.** DESIGN.md's "SVD count: 42" counts 35
     fold-level plus 7 final refits — the PRODUCTION path. It omits
     `ridge_equivalence_check`'s own five SVDs per condition, which are
     real wall-clock. The true per-run count is 77 (7 x [5 CV + 5
     equivalence + 1 final]). The Phase B projection's 7.9% accuracy
     partly masked this, since the JAX leg is the small one. Disclosed
     rather than quietly corrected; the constant keeps its
     design-accounting meaning and the cost note uses 77.
