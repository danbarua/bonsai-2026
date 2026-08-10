Simplified Technical English version of `experiments/stage2b_denoising/DESIGN.md`.

# Stage 2B: Does Runtime Oscillator Evolution Improve Single-Step Denoising?

**DESIGN STATUS: LOCKED.** A locked design is a plan document the team
agrees not to change, except through a disclosed amendment. An amendment
is a change made to a locked document after it was locked, always
disclosed openly, never made silently. This design went through seven
drafts, four rounds of review by a ChatGPT reviewer, one review looking
specifically for blind spots (by a model called Claude Fable 5), and one
review by an outside reviewer (Grok). The team gave final sign-off after
three corrections to how the plan would be carried out: the exact ridge
formula that accounts for the model's intercept, the correct count of
SVD computations, and the exact optimizer settings. All three are already
included below. No further round of scientific review is required. Any
change after this point needs an explicit, disclosed, post-lock
amendment, following this project's standard rule.

**Task name, used consistently throughout this document**: single-step
active-support reconstruction under a fixed, majority-censored, clipped-
Gaussian corruption. Active support means the fixed set of 505 pixel
positions this stage restricts its analysis to, out of the full 784-pixel
image. Censoring, here, means a noisy pixel value lands outside the
allowed [0, 1] range and gets clipped back to the nearest edge, losing
the true noisy value. The unusually severe corruption used here is part
of the design's identity. It is not a side note.

## The question, precisely

> Does runtime graph evolution, on top of the same already-encoded local
> phase state, improve single-step denoising prediction error, compared
> with the unevolved encoded state alone?

Runtime graph evolution means running the oscillator network dynamics
forward in time on a graph, after the image is already encoded (turned
into phase values on the graph's nodes).

**Strongest claim the team will make, stated in advance**: "Runtime
graph evolution improves linear reconstruction of clean pixel values, on
the fixed 505-pixel active support, under one fixed level of clipped-
Gaussian corruption." The team will add the stronger claim "actual
denoising" to this only if the hierarchical identity-baseline gate
(defined below) also succeeds. These two claims are kept distinct. The
team will never blend them together.

## Building the test data: add noise to the full image, then encode, then restrict

```
x_0^784        = clean image, full 28x28, in [0,1]
x_t^784        = sqrt(alpha_bar_t) * x_0^784 + sqrt(1-alpha_bar_t) * epsilon,  epsilon ~ N(0,I)
x_t^784_clip   = clip(x_t^784, 0, 1)
theta_0^784    = encode(x_t^784_clip)          -- _local_converged_phases, full grid, unaltered
theta_0^505    = theta_0^784[active_indices]   -- restrict AFTER encoding
theta_T^505    = F_W^2.5(theta_0^505)          -- graph evolution, W = T or a control
```

The reconstruction target is the clean image `x_0`, restricted to the
505 active-support pixels.

## Corruption level: alpha_bar_t = 0.5, frozen, and the censoring profile stated as a known fact in advance

Both the signal coefficient and the noise coefficient are equal before
clipping, at `sqrt(0.5)` each. This is not the same as "balanced signal-
to-noise": KMNIST images are sparse, not centered around zero, and do
not have unit variance, and clipping further distorts the effective
signal-to-noise ratio.

**This corruption process censors most pixel values, and the team
computed this fact in advance** (with a noise standard deviation of
0.707):

| x_0  | P(clip below 0) | P(clip above 1) | total |
|------|-----------------|-----------------|-------|
| 0.00 | 0.500           | 0.079           | 0.579 |
| 0.25 | 0.401           | 0.122           | 0.523 |
| 0.50 | 0.309           | 0.180           | 0.489 |
| 0.75 | 0.227           | 0.253           | 0.480 |
| 1.00 | 0.159           | 0.339           | 0.498 |

Roughly half of all pixel values clip, at every clean pixel intensity
(about 54% overall, on a sparse-image estimate). This has two stated
consequences. First, the identity baseline (returning the clipped noisy
image unchanged) is much stronger than a first guess would suggest,
because clipping alone corrects about half of all noise excursions in
the correct direction for extreme pixel values. Second, background-
dominated mean squared error (MSE) is heavily shaped by this clip-at-
zero effect, which makes the foreground-versus-background breakdown of
results important, not optional.

**This corruption level is frozen.** Adjusting it after the fact because
measured clip rates "look awkward" is exactly the kind of post-hoc
tuning this project refuses to do. Empirical clip rates are reported
(below-zero and above-one separately, both at the 784-pixel and
505-pixel scope, per class) as a check against this table, not as a new
discovery.

## Corruption random-number generation, exact values

```python
MASTER_SEED = 42
seed_bytes = SHA256(f"{split}:{index}:{MASTER_SEED}".encode()).digest()
seed = int.from_bytes(seed_bytes[:8], "little")
rng = numpy.random.Generator(numpy.random.PCG64(seed))
epsilon = rng.standard_normal(784, dtype=numpy.float64)
```

Here, `split` is the literal string `"train"` or `"test"`, not Python's
own randomly-salted `hash()` function. Each image gets exactly one noise
realization, reused identically across every condition tested. Inference
runs over the official test-image sample, under the rule of exactly one
independent corruption draw per image; it does not estimate how much
results would vary under repeated noise draws on a fixed image.

**Test-data usage rule**: the official KMNIST test images were already
used extensively by Stage 2A, so they are not unseen by this project as
a whole. What is locked here is narrower: the Stage 2B corrupted test
corpus, its features, model predictions, and denoising scores are all
generated and inspected exactly once, in one final confirmatory
evaluation. No Stage 2B test-side result is accessed at any point during
stages 1 through 3.

**Corruption diagnostics**: the team reports the mean squared error (MSE)
before clipping, `MSE(x_t, x_0)`, and after clipping,
`MSE(clip(x_t), x_0)`, at both the 505-pixel and 784-pixel scope. The
active-support post-clip value is the identity baseline used by the
hierarchical gate, defined below.

## Encoding pipeline

- Gauge (a fixed reference point used to remove an arbitrary rotational
  freedom in the phase values): a reference node, `theta_ref` = node
  363. This is T's median-weighted-degree node, and is Stage 2A's final
  locked choice.
- Features: a circular embedding (cosine and sine per node), 1008
  values in total, with the two always-constant columns from the
  reference node dropped.
- Standardization: a `StandardScaler` is fit on the training fold only,
  and applied identically to every condition, including the raw-pixel
  ridge inputs.
- Encoder random-number seed: seed 0 per image, as the primary choice;
  independent per-image seeds are used only as a secondary robustness
  check.
- Inherited fixed values, stated exactly: fold seed 42; the four
  canonical graphs are T (learned, no seed), lattice (fixed,
  deterministic), rewired (built with `degree_preserving_rewire`, seed
  0), and curr_random (built with `generate_matched_sparsity_topology`,
  seed 0) — all four are Stage 2A's own canonical graph instances.

## Encoder-on-noisy-inputs gate (feasibility stage 1, an executable check)

`_local_converged_phases` is the function that encodes an image. It was
built and checked only on clean, spatially smooth images. The team does
not assume that convergence on clean images also holds for majority-
censored noisy images; it checks this directly. At stage 1, on the same
1,000 images, the team records each image's final change in phase value
at the last iteration (called final-Delta), encoding clean and noisy
versions separately.

**The gate rule**:

```
rho = median(Delta_noisy) / max(median(Delta_clean), 1e-15)
PASS iff rho <= 10
```

The `1e-15` value in the formula is only numerical protection against
dividing by zero; it is not a scientific threshold. The multiplier of 10
is an arbitrary but pre-registered value, fixed before any data existed.
**Automatic failures, regardless of the rho value or the absolute-
convergence escape described below**: any non-finite encoded phase value
(for example, infinity), or any non-finite final-Delta value. Both
median values are recorded in the stage-1 log regardless of the outcome.
The 95th-percentile final-Delta value, for both noisy and clean data, is
also logged alongside them — this gives visibility into a case where the
median passes but a small number of extreme values do not, and this
logging is explicitly not a second gate. If the gate fails, it halts the
whole stage until the team investigates.

**Post-lock amendment (2026-08-06, made after feasibility stage 1's
first real run): `ENCODER_STEPS` raised from 150 to 1200, and the gate
formula gains an absolute-convergence escape.** The gate's first run on
real, majority-censored KMNIST data FAILED honestly, exactly as the
design intended: `rho=169.851` against the threshold of 10 (median
final-Delta was 2.177e-07 for clean images and 3.698e-05 for noisy
images; zero non-finite values anywhere). This was a genuine ratio
failure, not a numerical blow-up. The team investigated this in
`experiments/stage2b_denoising/diagnose_encoder_gate_failure.py` (a
diagnostic script only; it does not touch any locked pipeline code). This
script rebuilds the exact stage-1 test data and corruption locally, and
checks that this rebuilt data matches the failed run's own reported
identity-baseline MSE exactly, before trusting anything computed from
it (confirmed exact, relative difference 0.000e+00).

The team ran two measurements, both planned in advance before either was
actually run. (1) A convergence curve across five step counts (75, 150,
300, 600, 1200): the noisy final-Delta value shrinks steadily all the
way down to an exact float64 zero — at both the median and the 95th
percentile, for every one of 1,000 images — by 1,200 steps, reaching the
same fixed point clean images reach. There is no floor above a
meaningful scale; the encoder does converge on censored inputs, it
simply takes longer. (2) The team also measured, per image, how much the
phase state moved between 150 and 600 steps for noisy images, and
compared this against the typical distance between different images'
phase states at 150 steps. The result: a median ratio of 4e-4 — meaning
the phase state has, for all practical purposes, stopped moving relative
to the scale that distinguishes one image from another, well before the
Delta metric alone would say so. Both measurements independently support
the same reading: this is genuine, if slow, convergence, not a
qualitatively different pattern on noisy inputs.

**A second, independent problem showed up in the same investigation:
the ratio formula in the gate becomes unstable near either series' own
numerical floor.** At 600 steps, the clean series' median had already
reached exactly 0.0, while the noisy series' median sat at 1.776e-14 —
nine orders of magnitude below the smallest meaningful final-Delta value
measured anywhere (2.177e-07) — yet the gate reported FAIL, at
`rho=17.76`, because `max(0.0, 1e-15)` silently turned what should be a
ratio check into an absolute check against the 1e-15 floor value. The
full rho sequence across the five step counts (14.98, 169.9, 1.915e4,
17.76, 0.0) does not move steadily in one direction, for exactly this
reason: it tracks which series crossed its own numerical floor first,
not whether the underlying process actually converged. A threshold that
sits inside this crossover zone is fragile by construction, no matter
where `ENCODER_STEPS` ends up.

**The fix, both parts disclosed together:**

```
rho = median(Delta_noisy) / max(median(Delta_clean), 1e-15)
PASS iff rho <= 10 OR (median(Delta_clean) < 1e-12 AND median(Delta_noisy) < 1e-12)
```

`ABS_CONV_EPS = 1e-12` is set using the same margin logic as the other
threshold values: it sits five or more orders of magnitude below the
smallest meaningful measured Delta value (2.177e-07), and well above the
observed float64 rounding noise (1e-14 to 1e-16). This means it cannot
fire on a signal that is genuinely still converging; it can only fire on
values already indistinguishable from numerical noise. The rule requires
BOTH medians to be below this value, not just one — a case where one
series has converged but the other is still measurably moving falls
through to the ordinary rho test, on its own merits.

**The decision rule that selected the final step count, S\* = 1200, is
stated precisely, because an earlier, looser version of that rule was
caught and rejected before being used.** The looser rule was: "some S\*
brings the noisy final-Delta within 10x of the clean value at 150
steps." This is already satisfied as early as S\*=300 — which then fails
its own re-run test at `rho=1.915e4`, because the real gate rule
compares the noisy and clean values at the same step count, not against
a stale reference value from a different step count. The corrected rule
compares same-step values, and requires both series to have genuinely
converged. Under either the original, flawed reading or the corrected
one, the final answer is the same: S\*=1200 passes. So the correction to
the rule did not cause this particular outcome; both readings agree
here. 1,200 steps is also the only step count in the five-point scan
where both medians AND both 95th-percentile values land exactly at 0.0
— the maximum possible safety margin — rather than sitting near the
fragile crossover zone a smaller, untested step count might land in. No
finer scan between 600 and 1,200 steps was run: encoder cost grows with
step count, and the team judged that a robust operating point was worth
more than saving a few minutes of runtime. If measured full-scale cost
ever changes this trade-off, the team will revisit it as its own
disclosed decision.

Every other stage-1 artifact — the test corpus, the graph topologies,
the corruption, and the corruption diagnostics — does not depend on the
encoder, so this change does not affect them; only the encoder-gate
step is recomputed. Its GCS object name carries the step count
(`encoder_gate_s{steps}.npz`), so the earlier 150-step FAILURE artifact
stays in the storage bucket, untouched, as the historical record of the
first real run. It is not deleted, and it is not silently reused under a
new meaning.

## Foreground mask

`m_ij_ink = 1[x_0,ij > 0.15]` is computed from each individual image's
own clean pixel values, on its 505 active-support coordinates. This is
NOT the same as a class-mean threshold computed once, in advance, from
many images — it is a different object that happens to share the same
numeric threshold value. Images with zero foreground pixels are excluded
from that particular image's foreground breakdown, and this exclusion
is counted and reported. Averages are computed per image first, and then
averaged across images.

## Data type used, per stage

| stage                 | data type | reason |
|-----------------------|---------|-----------|
| ODE evolution (running the oscillator dynamics forward in time) | float64 (64-bit precision) | required; established during Stage 1D's GPU checks, where silent float32 (32-bit) rounding was a real, caught failure |
| Ridge SVD | float64 | motivated by measurement: Stage 2A measured condition numbers around 2 million on evolved-feature design matrices; float32's precision of about 6e-8, multiplied by that conditioning, gives a worst-case relative error of about 12%; float64 gives about 2e-10 instead. Whether this transfers to Stage 2B's own features is plausible but not yet established, which is why the stage-2 diagnostic below exists |
| Corruption generation | float64 | not strictly required numerically; kept because the locked random-number recipe produces this precision anyway, and matching the exact specification is worth more than a small time saving |
| Feature storage | float64 | not strictly required (3.5GB versus 1.8GB, a difference that does not matter on the target hardware); kept for consistency across the pipeline |
| Sign-flip matrices | int8 (8-bit integer) | values of +1 or -1 need no floating-point precision at all; this uses 6GB for all six comparisons, versus 48GB if stored as float64 |

**Stage-2 diagnostic (required)**: the team records the condition number
of the standardized design matrix, per condition, one line each. This
turns the float64-for-ridge reasoning above from an inherited assumption
into a checked, measured table.

**Post-lock disclosure (2026-08-06, before ladder stage 3): encoded
features are bit-for-bit reproducible WITHIN one CPU chip type, but not
necessarily ACROSS different chip types.** Stage 3 splits its work into
a local CPU encoding phase and a remote GPU phase (see "Computational
strategy" below), which put the encoder on Apple Silicon chips for the
first time. Stages 1 and 2 had both encoded on Colab's x86 chips. The
team measured this directly, on the same images, before writing any
stage-3 artifact, rather than assuming it would be fine:

| comparison | result |
|---|---|
| same machine (this Mac), encoding run twice | bit-exact |
| two different Colab sessions, same 1,000 images | bit-exact, matching on 784,000 of 784,000 coordinates |
| Mac (ARM chip) versus Colab (x86 chip), same images | 93.4% of coordinates identical; maximum difference of 3 ULP (4.441e-16), mean difference 0.07 ULP, maximum relative difference 3.98e-16 |

(A ULP, unit in the last place, is the smallest possible difference
between two numbers as a computer stores them.)

The cross-chip-type difference sits right at the limit of what a
float64 number can represent, and it cannot grow larger over time: the
encoder pulls values toward a fixed point (a stable value it settles
into), and the measured difference from the 1,200-step result falls from
8.370e-07 at 300 steps, to 8.062e-13 at 600 steps, to exactly 0.0 at
1,200 steps. Both chip types settle on the SAME fixed point; only a
small minority of coordinates land on a neighboring representable
float64 value instead of the identical one. Further downstream, these
encoded phases feed into an ODE solver set to a relative tolerance
(`rtol`) of 1e-6, which is eight or more orders of magnitude larger than
this difference, so no number this design reports can depend on it.

The team recorded this rather than waving it through, and scoped it
precisely: what is guaranteed is reproducibility within one chip type.
Stage 3's encoded array, saved in GCS, is the official record. Re-
deriving that same array from scratch on a different chip type
reproduces it to within about 1 ULP, not bit-for-bit exactly. This is the
same standard, and the same reasoning, the project already applies to
its acceptance that GPU-run logistic regression (using the cuML
library) is not bit-for-bit reproducible, and that this is acceptable at
the level the project's claims are actually made (see
`docs/PROJECT_MEMORY.md` Part 4).

## Readout: multi-output ridge — a JAX-based SVD as the production method, sklearn kept as an independent check

The design uses one multi-output ridge model, sharing one alpha value
across all 505 output pixels, searched over the grid
`{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6}`.
(**This grid was amended after feasibility stage 3 — see the Review
history section below**; it originally ran only from `1e-2` to `1e6`.)
The model always fits an intercept (`fit_intercept=True`), inputs are
standardized per fold, and target values are left unstandardized. When
two alpha values tie (mean validation MSE within 1e-10 of each other),
the **larger** alpha value wins. Cross-validation uses
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, with the
identical partition of data used across all conditions.

**The decade spacing of the alpha grid is exact and frozen.** There are
thirteen values, one per decade, with no interpolation and no denser
searching around an observed minimum, ever. This is a frozen procedural
rule, not just a default setting: a grid that could be refined near a
minimum after that minimum is already known would have a stopping rule
chosen by looking at the results — exactly the pattern the amendment
described below exists to rule out.

**Production method: direct JAX-based SVD ridge, aware of the
intercept.** For each combination of cross-validation fold and
condition, the method computes one thin SVD of the standardized training
features, and reuses that single decomposition to evaluate all thirteen
alpha values. Because the model always fits an intercept, and the 505
target columns are left unstandardized, the solve must center the
targets within the training fold and recover the intercept using the
general formula below — not assume a shortcut:

```
Y_tilde  = Y - mean(Y_train)                      # center each target column, training fold only
W_alpha  = V @ diag(s / (s^2 + alpha)) @ U.T @ Y_tilde
b_alpha  = mean(Y_train) - mean(X_train) @ W_alpha
```

(`mean(X_train)` is approximately zero after standardization, so
`b_alpha` will normally reduce to `mean(Y_train)`. But the code always
uses the full general formula above, not this shortcut.)

**SVD count: 42, not 35.** There are 35 fold-level SVDs used for
choosing the model (7 conditions times 5 folds, with all 9 alpha values
reusing each decomposition), followed by 7 more SVDs, one per condition,
for the final full-training refit at each condition's chosen alpha
value. Runtime estimates and cloud job planning both use the number 42.

**The seven ridge conditions, and why the statistics use only six of
them.** The seven fitted ridge conditions are `raw_505`, `raw_784`,
`pre_evolution`, `evolved_T`, `evolved_lattice`, `evolved_rewired`, and
`evolved_curr_random`. The statistical tests, however, operate on a
DIFFERENT set of six keys: `pre_evolution`, the four evolved graphs, and
the identity baseline. `raw_505` and `raw_784` are descriptive
comparisons only — they support named watched-for outcome 5, described
below — and belong to neither statistical family. The identity baseline
also belongs to neither family; it enters only through the hierarchical
gate described below, and nothing is fitted for it, so it is not one of
the 42 SVDs. Seven conditions are used for computing cost; six keys are
used for statistics. Neither number is a mistake for the other.

**sklearn's `Ridge(solver="svd")` is kept as the independent verification
oracle. It is never part of the production path, and it is never
deleted from the code.** **The equivalence gate, stated exactly**: at
both the 1,000-image and 5,000-image stages, the JAX method and the
sklearn method must produce (a) a maximum absolute difference of no more
than 1e-8 in the clipped validation predictions, and (b) the identical
choice of alpha value. Any failure of either rule halts the run until
investigated. As a diagnostic only, not a second halt rule, the team
also records the maximum absolute difference in the fitted coefficients,
at a looser tolerance, purely for visibility — prediction agreement is
what actually matters for the design's endpoint.

**Post-lock amendment (made before feasibility stage 1): the scaler-
centering guard's tolerance now depends on `n`.** The per-fold check on
the standardized training features halts when
`||mean(X_train_scaled)|| >= 1e-9 * (n / 1000) ** 0.5`, where `n` is the
number of rows in the fold's training data being checked, not the
nominal size of the whole rung. This tolerance was previously a fixed
value of `1e-10`.

The anchor value and the growth exponent in this formula come from two
different kinds of evidence, deliberately. The anchor, 1e-9, is
measured: it is a 12.7x safety margin over the worst value the GPU
measurement recorded at n=1,000 on production-path evolved features
(`curr_random`, at 7.87e-11 — see `README.md` for the full tables). The
exponent is mechanical: `||mean(X_scaled)||` grows because the mean of
`n` float64 values accumulates roughly the square root of `n` in
rounding error, and this is amplified by dividing by a small column
standard deviation. The value 0.5 upper-bounds the measured growth rate
of 0.405 for `curr_random`, the condition the tolerance is set to
protect, so its safety margin widens as `n` grows rather than shrinking
— 14.8x at the measured n=5,000, and a projected 18x at n=54,000. Two of
the non-limiting conditions grew faster than the square root of `n` on
the same two data points, so their margins narrow instead, starting from
margins that are three orders of magnitude wide; `README.md`'s table
carries every condition's own growth exponent. A fixed absolute
tolerance on a quantity that grows like the square root of `n` will
eventually halt on any features at all, given enough data — this
tolerance formula replaces that failure mode rather than being a
property specific to these particular features.

This guard keeps its ability to detect real problems at both ends of
its range. The tolerance at n=54,000, which is 7.35e-9, sits four or
more orders of magnitude below the roughly 3e-4 level at which
`||mean(X)||` starts degrading the 1e-8 equivalence gate above (a
measured value, see `README.md`), and about nine orders of magnitude
below the size of offset a genuinely broken scaler would produce. The
growth exponent of 0.66, fitted from the earlier CPU-evolved table, is
superseded: it was measured on a different computation pipeline than the
one the anchor value comes from.

**Model selection**: the team chooses alpha by the clipped validation
MSE (`argmin_alpha MSE(clip(x_hat_alpha, 0, 1), x_0)`); the ridge model
itself is fitted without output clipping; raw, unclipped validation MSE
is reported only as a diagnostic.

## Handling the prediction range

The primary MSE is reported after deterministic clipping to the [0, 1]
range, for every condition. Raw, unclipped MSE is reported as a
secondary diagnostic. Clipped-boundary fractions (how often a prediction
actually hit the clip boundary) are reported per condition. Selection
criteria — the ridge alpha value, and the CNN checkpoint — both use
clipped validation predictions. Training losses stay raw, unclipped.
This distinction is locked into the design; it is not left up to
whoever implements the code.

## Identity baselines: a hierarchical gate, plus a separate descriptive rescaled version

1. **Primary comparison**: `evolved_T` against `pre_evolution`.
2. **Only checked if the primary comparison succeeds**: `evolved_T`
   against the identity baseline (`clip(x_t)`, active-support, after
   clipping). This is the denoising gate. It never rescues a failed
   primary comparison.
3. **Always reported, independently of the gate**: `pre_evolution`
   against the identity baseline, with its own paired bootstrap interval,
   reported as context, outside the gate.

**Descriptive baseline (added at the final review, not part of the
gate)**: the rescaled identity baseline,

```
x_hat_0_rescale = clip( x_t_clip / sqrt(0.5), 0, 1 )
```

uses only the known corruption coefficient, with zero learned
parameters. It shows whether a model does more than simply undo the
known, deterministic signal-shrinking effect of the corruption. It is
reported descriptively alongside the other baselines; the raw-pixel
ridge model remains the stronger learned comparison.

## The primary comparison and statistical test

`d_i = MSE_i(evolved_T) - MSE_i(pre_evolution)`, computed on the active
support, after clipping, on the official test set, evaluated exactly
once. The team runs a 20,000-resample paired, class-stratified bootstrap
test (a method that resamples the data many times to estimate how much a
result could vary by chance), producing a two-sided 95% confidence
interval, with random seed 42. If the interval is entirely below zero,
evolution improves results. If it is entirely above zero, pre-evolution
wins. If the interval straddles zero, the result is null, and the design
states this plainly.

(At n=10,000, the paired t-tests described below are justified by the
Central Limit Theorem, even for skewed per-image MSE differences. This
is stated once here so that this assumption gets the same scrutiny
everywhere it applies, not only in the sign-flip robustness check
below.)

## Secondary comparisons: two families

- **Family 1 (three tests)**: lattice, rewired, and curr_random each
  compared against `pre_evolution`. Paired t-tests produce p-values;
  Holm correction (a way of adjusting p-values so that running several
  tests together does not raise the false-positive rate) is applied
  across the three tests; bootstrap confidence intervals are reported
  alongside for effect size.
- **Family 2 (six tests)**: every pairwise comparison among the four
  evolved graphs. Paired t-tests are used, with Holm correction applied
  across the six tests. A sign-flip test is kept as a robustness check
  only: 100,000 random sign flips, seed 42, using a studentized test
  statistic `T_b = mean(s_b * d) / (SD(s_b * d) / sqrt(n))` that matches
  the shape of the t-test's own statistic, two-sided
  (`|T_b| >= |T_obs|`), with a +1 correction
  (`p = (1 + count) / 100001`). The assumption this test needs — that
  the sign of each difference could equally well have gone the other way
  — is stated wherever this test is reported.
- The primary test sits outside both statistical families and is not
  corrected for multiple comparisons.

**The "one graph wins" rule, split by which type of graph is being
checked**: T qualifies as a winner only if the primary bootstrap
interval lies entirely below zero. A control graph qualifies only if its
direction favors evolution, and its Family-1 Holm-corrected p-value is
below 0.05. A graph is the single, unique winner only if it qualifies by
whichever of these two rules applies to it, AND it beats each of the
other three graphs, in the favorable direction, after Family-2 Holm
correction.

## CNN: locked, built with equinox and optax

**Framework choice**: equinox and optax, both already dependencies of
this project — diffrax already pulls in equinox, and optax was already
added earlier — giving one framework end to end, one random-number
story, and one shared GPU memory pool, with zero new dependencies added.
**PyTorch is the documented alternative the team chose not to take.** The
team rejected it because it introduces a second deep-learning framework
sharing one GPU with JAX; PyTorch's default setting reserves about 75%
of GPU memory in advance, which would starve a co-resident JAX process
unless explicitly configured otherwise. This is a concrete, known
failure, not a matter of style preference. TensorFlow was also rejected:
it offers no advantage that would justify the cost of a second
framework.

**Architecture** (a residual design, fixed in advance):

```
x_hat_0 = x_t_clip + f_psi(x_t_clip)
f_psi:  Conv2d(1,32,k=3,p=1) -> ReLU -> Conv2d(32,32,k=3,p=1) -> ReLU -> Conv2d(32,1,k=3,p=1), linear output
```

**9,857 trainable parameters** (computed as:
`(1*32*9+32) + (32*32*9+32) + (32*1*9+1)`). Convolution bias terms are
enabled, consistent with this count. Padding is zero-padding for
`padding=1`. Training and evaluation are both scope-matched: the loss
uses only the masked active-support MSE; full-image MSE is never
reported for this model, because 279 of the output pixel positions
receive no training signal at all under the mask. The final layer is
linear, with no sigmoid activation, so its unclipped diagnostic can be
compared directly with ridge's own unclipped diagnostic; the shared
clipping rule is applied only at evaluation time. The CNN's input
receives no normalization beyond the locked clipping to the [0, 1]
range.

**Optimizer, stated exactly** (no library default value left unstated):

```python
optax.adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)
```

There is no weight decay and no gradient clipping. The loss is the mean
over batch images and their 505 active pixel coordinates (using the raw,
unclipped values, per the training-versus-selection distinction stated
above).

**Training, stated exactly**: batch size 128, a maximum of 100 epochs,
early stopping with a patience of 10 epochs on the clipped active-
support validation MSE, `min_delta=0.0` with a strict improvement
required, and the best checkpoint is restored at the end. The validation
split uses `StratifiedShuffleSplit(n_splits=1, test_size=0.10,
random_state=42)`, created BEFORE any smaller feasibility subset is
drawn (the 5,000-image development subset is drawn only from the
remaining 90%). At stage 3, "full training" means 54,000 images for
fitting plus 6,000 locked validation images. The model trains with three
random seeds (0, 1, 2), each controlling initialization, batch order,
and framework randomness together; the best of the three seeds is chosen
by clipped validation MSE, using only training-side information — the
official test-set performance is never inspected during this selection.

## Computational strategy

| stage                 | shape                          | where         | working-out |
|-----------------------|---------------------------------|---------------|-------------|
| Corruption generation | 70,000 by 784 Gaussian draws   | CPU           | about 55 million draws, using vectorized numpy, under 2 seconds |
| Encoding               | 70,000 images, 1200-iteration local update each | CPU reference | about 4.6ms per image at 150 iterations, as measured in Stage 2A; the iteration count was later raised from 150 to 1200 (see the "Encoder-on-noisy-inputs gate" amendment above); full-scale throughput at 1200 iterations was not yet measured at design time, and any implementation is checked against the reference implementation first |
| Evolution              | 70,000 images across 4 topologies | GPU           | verified with `evolve_on_graph_jax.py`, about 0.67ms per image per topology, batched, giving roughly 3-4 minutes total |
| Ridge                 | 42 SVDs (35 fold-level plus 7 final refits, on roughly 48,000-60,000 by 1008 matrices) | GPU | about 0.1 TFLOP each; a matter of seconds on an A100 GPU, and demonstrably viable on a CPU at roughly 1 minute each |
| CNN                   | 9,857 parameters               | GPU           | standard |
| Bootstrap and sign-flip tests | resamples processed in chunks of 512-4096 | GPU by default | signs stored as int8 (6GB for all comparisons versus 48GB at float64); about 0.8GB per comparison for the bootstrap; a host-level loop over chunks is allowed, but a per-resample Python loop is not; CPU is demonstrably viable as a fallback |

Generation of data, features, and statistics all run entirely inside the
cloud environment; the resulting artifacts are pushed to Google Cloud
Storage (GCS) directly from within it, and never routed back through a
local upload (Stage 2A already hit Colab's upload limit of roughly
6-15MB against a 242MB file once).

**Post-lock amendment (2026-08-06, before ladder stage 3): stage 3 runs
in two phases, and the encode phase runs locally.** The rule stated
above still holds in substance, and its own reasoning already explains
why: the failure it guards against is getting a large file INTO a Colab
session through that session's own upload mechanism. Writing directly
from the local machine to GCS never uses that mechanism at all — it uses
the same `google-cloud-storage` client, and the same chunked, resumable,
checksum-verified transfer method in `stage2b_gcs.py`, that a cloud-side
write already uses, and `stage_kmnist_inputs.py` has already moved
KMNIST data this way, for both earlier ladder rungs. The rule that
actually matters is "feature artifacts live in GCS, written through the
verified transfer method," not "every computation must happen inside a
Colab session."

What this split avoids is a real cost, not a hypothetical one: encoding
is the pipeline's one genuinely CPU-bound step — it took 1,099 seconds
of stage 2's 1,722-second total — while evolution, ridge, and the CNN
are the steps that actually use the GPU. Running the encode step inside
a paid cloud session would leave a metered A100 GPU idle for most of the
run. So Phase A (test corpus, corruption, encode, restrict) now runs on
local CPU cores and writes only the encoded array; Phase B (evolution,
ridge, CNN) reads that array, and regenerates the corruption and clean
target values itself, in-session, since both are deterministic and cheap
to compute. This means 218MB crosses the network boundary instead of
775MB. `ensure_artifact`'s automatic chunking threshold, 64MB, engages
on this upload without the calling code needing to ask for it. See the
data-type section above for the cross-chip-type reproducibility
consequence of this change, which the team measured before acting on
this decision.

Colab and Google Cloud are a compute RUNTIME for this implementation
only — this is not a commitment about the final delivery format. How
visuals and plots will eventually be delivered is an explicitly DEFERRED
decision, to be made once results exist, and it is not part of this
locked design. Notebook packaging must not be treated as a pending or
implied task at any point before that decision is made. GCS paths are
implementation details. `FINDINGS.md`, kept in this code repository,
remains the official record.

## Feasibility ladder

The feasibility ladder is a series of runs at growing size, used to
catch problems before spending money on the full-scale run.

1. 1,000 official training images — checks mechanical correctness, the
   encoder gate, and the ridge equivalence check (first pass).
2. A fixed 5,000-image development subset — checks runtime, feature
   validity, ridge-grid behavior, the condition-number diagnostic, the
   ridge equivalence check (second pass), and CNN development.
3. The full 60,000-image training side — 54,000 for fitting plus 6,000
   for locked validation — checks feature generation and model
   selection.
4. ONE locked evaluation on the official 10,000-image test corpus.

No Stage 2B test-side result is accessed at any point before stage 4.

## Named outcomes the team is watching for

1. All four graphs improve over pre-evolution, with no detectable
   pairwise difference between them. (This is an observation the team
   watches for, not a claim that the graphs are proven equivalent.)
2. One evolved graph qualifies under the branched rule described above,
   AND it outperforms each of the other three after Family-2 correction.
3. Evolution does not help denoising, even though it helps
   classification — a difference in outcome that depends on the task.
4. The unevolved encoding alone already carries most of the value;
   evolution adds little on top of it.
5. Raw-pixel ridge outperforms every phase-based condition, evolved or
   not — meaning the phase representation itself loses information
   needed for reconstruction. This is plausible in advance: the encoder
   was built to converge phase values toward a target field, for the
   purpose of building graph topologies, never to preserve pixel
   intensity information, and taking the cosine and sine of converged
   phase values is a lossy, nonlinear transformation. If this outcome is
   observed, it would be a coherent, reportable difference from Stage
   2A's finding (that the representation is useful for telling classes
   apart, but harmful for reconstructing them). This is one of the
   outcomes the team watches for, not a requirement for success; success
   is defined only by the evolution-versus-encoded comparisons under the
   identity gate.

## What this design does not do

This design does not include multi-step reverse sampling or generation.
It does not include conditioning on different timesteps or training
across multiple noise levels. It does not predict the noise itself
(sometimes called epsilon prediction). It does not revisit Stage 2A's
open items, numbered 9, 10, and 11 in that stage's own tracking.

## Review history

- Drafts 1 through 6: four rounds of review by ChatGPT, covering: a bug
  in the order noise was applied; an error in the direction of the ridge
  tie-break rule; the identity-gate hierarchy; the two multiplicity
  families; exact values for the random-number generator, folds, and
  sign-flip test; chunked vectorization; CNN scope-matching; and the
  one-graph-wins branching rule for T.
- Review by Claude Fable 5, looking specifically for blind spots:
  computing the majority-censored corruption profile in advance; adding
  the raw-pixel-ridge named outcome; the encoder-on-noise gate; the
  per-stage data-type reasoning; applying the same scrutiny to
  assumptions on both sides; the JAX SVD-ridge production path; and
  locking the equinox and optax framework choice.
- Review by Grok, as an outside reviewer: softened claims about float64
  transfer across chip types, kept the sklearn oracle, treated the
  framework choice as a preference rather than a hard constraint, and
  confirmed the refusal to tune the corruption level after seeing
  results — all of these were incorporated.
- Final round of ChatGPT review: the intercept-aware ridge formula, the
  SVD count of 42, the exact optax configuration, the rescaled-identity
  descriptive baseline, completing the encoder-gate numerical logic, and
  the task naming convention. LOCKED at this point.
- Post-lock amendment, before feasibility stage 1: the scaler-centering
  tolerance now depends on `n`, using the formula
  `1e-9 * (n / 1000) ** 0.5`, replacing a fixed `1e-10` value that a
  quantity growing like the square root of `n` would eventually outgrow;
  and the seven ridge conditions, together with the distinct six-key
  statistics set, were both written down explicitly, alongside the SVD
  count.
- Post-lock amendment, to the computational strategy: wording implying
  a Colab-notebook final deliverable was removed, since it was never an
  actual task; the delivery format is deferred, and Colab/GCP remains
  only a compute runtime.
- Post-lock amendment, after feasibility stage 1's honest FAILURE
  (`rho=169.851` at `ENCODER_STEPS=150`): the step count was raised to
  1200. `diagnose_encoder_gate_failure.py` diagnosed this as genuine
  slow convergence to the same float64 fixed point, not a hard floor.
  The gate formula gained an absolute-convergence escape
  (`ABS_CONV_EPS=1e-12`), after the same diagnostic exposed the ratio
  formula reading numerical noise as a real failure at 600 steps. The
  decision rule that selected S\*=1200 is stated as giving the same
  verdict under its own earlier, corrected version — disclosed rather
  than silently fixed. The encoder-gate artifact naming carries the step
  count, so the 150-step FAILURE remains in the storage bucket as
  history.
- Post-lock amendment, before feasibility stage 3: two-phase execution.
  A local CPU encode phase writes only the encoded array to GCS, and a
  remote GPU phase regenerates the corruption and target values itself,
  in-session. The "generate in the cloud" rule's own stated reason is
  the Colab session upload limit, which a direct local-to-GCS write, on
  the already-verified transfer method, never touches; and an in-session
  CPU encode would leave a metered A100 GPU idle for most of the run.
  Alongside this change, the team disclosed the cross-chip-type
  consequence it had measured before acting on the decision: encoded
  features are bit-for-bit reproducible within one CPU chip type
  (Colab-to-Colab is bit-exact across sessions; this Mac is bit-exact
  across runs), and they agree to within a maximum of 3 ULP across
  different chip types, a difference that the encoder's own convergence
  toward a fixed point damps down rather than amplifies.
- Post-lock amendment, after feasibility stage 3 Phase B: **the alpha
  grid extends downward by four decades, to `{1e-6 .. 1e6}`**, thirteen
  values in total, keeping the original decade spacing exactly, applied
  to **all seven conditions**.

  *Where this came from*: this was a ChatGPT review ruling, delivered by
  Dan manually copying and pasting it from his own ChatGPT chat, rather
  than through the project's usual audited review channel. This is
  recorded here because the route a ruling travels by is itself part of
  its evidence, and this one did not travel the usual audited path. The
  operative text, quoted exactly:

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

  *What led to this ruling*: at Phase B, the ridge model selected the
  smallest alpha value on the grid, in six of the seven conditions,
  where at stage 2 (n=5,000) none of them had. This extra scrutiny cost
  nothing to run, because the validation curves were already saved in
  the existing artifact, and it showed the six conditions split into two
  groups: both raw-pixel baselines were already flat (their MSE rose by
  only about 1e-7 across two decades — a cosmetic effect of sitting at
  the floor), while the four EVOLVED conditions rose by 0.4-0.8% per
  decade moving away from the floor, with their best value sitting right
  at the grid boundary. So this constraint bound in an unequal way,
  splitting exactly along the treatment-versus-control line the whole
  design exists to compare.

  *The frozen procedure, built to allow only one attempt*: a full re-run
  of the entire ridge procedure, on the new grid, for all seven
  conditions, all folds, and all refits; **no splicing** of lower-alpha
  points into any existing table; the production-scale JAX-versus-
  sklearn equivalence check re-run on the new grid, at the same frozen
  1e-8 tolerance; and a **HALT for review if any production condition
  selects `1e-6`** — landing at this new floor value is itself treated
  as a named anomaly, and it is never, on its own, a trigger to extend
  the grid further. Together with the exact-spacing rule above, this
  makes a repeating cycle of "widen the grid, look at results, widen
  again" structurally impossible, not just discouraged: this procedure
  provides no way to authorize a second extension.

  *Two consequences found while writing this up, which neither the
  original ruling nor the plan had anticipated*:

  1. **The write-once rule enforces "no splicing" automatically, by how
     the storage system works.** `ridge_cv.json` and `ridge_final.npz`
     are LINEAGE artifacts, meaning the code refuses to overwrite them
     once written. Passing `force=True` to either one raises a
     `WriteOnceViolation` error before any new computation even starts.
     So the re-run CANNOT overwrite the nine-decade results, and must
     write new file names that carry the new grid's identity — exactly
     as Phase A wrote `encoded_train_s1200` as a new name, rather than
     overwriting `encoded_fit_s1200`. The nine-decade tables survive as
     history because the storage system itself refuses to destroy them.
  2. **`PROBE_JAX_SVD_COUNT = 42` is correct as a design-accounting
     figure, but wrong as a cost multiplier.** `DESIGN.md`'s "SVD count:
     42" counts 35 fold-level SVDs plus 7 final refits — the PRODUCTION
     path only. It leaves out `ridge_equivalence_check`'s own five SVDs
     per condition, which take real wall-clock time. The true per-run
     count is 77 (7 conditions times the sum of 5 cross-validation runs,
     5 equivalence-check runs, and 1 final run). Phase B's cost
     projection was only 7.9% off partly because this error was masked
     — the JAX leg of the work is the smaller of the two legs. This is
     disclosed here rather than quietly corrected: the constant 42 keeps
     its original meaning as a design-accounting figure, and the cost
     note now uses 77 instead.
