Simplified Technical English version of `experiments/stage2b_denoising/AUDIT_PROTOCOL.md`.

# Stage 2B amendment-impact audit — FROZEN PROTOCOL

**Status: frozen. The team committed this document before any audit
result existed or was inspected.** A frozen document is one the team
agrees not to change once a result from it exists. An amendment-impact
audit is a check that measures how much a disclosed change to the locked
design (an amendment) actually changed the project's results. This
document's purpose is to fix, in advance, what the audit will measure,
what counts as a meaningful numerical difference, and what happens next
— so that none of these can be chosen after the fact, to fit whatever
result appears.

The team froze this document under the conditional agreement reached on
2026-08-06, following review rounds 1 through 8 by Claude Desktop and
ChatGPT, archived in `.claude/claude2gpt/archive/`.

## What this audit checks, and what it explicitly does not check

The team raised the encoder step-count budget from 150 to 1,200
iterations, after ladder stage 1's gate failed. This was a change made
after a failure, before any downstream confirmatory evaluation ran. It
was **not** part of the original design.

This audit measures how much that change affected the resulting
representation. **This audit is not a tool for choosing between models.**
The 1,200-step budget stays frozen no matter what number this audit
produces. What may change, depending on the result, is the SCOPE OF THE
CLAIM the project makes — never the budget itself.

## Sign convention (frozen)

    Delta_g = MSE_evolved_g - MSE_pre_evolution

**A negative value means improvement** (the evolved representation
reconstructs the image better than the pre-evolution representation).
This is stated explicitly here because every trigger condition below is
written in terms of the sign, and the ordering, of `Delta_g`.

## Which images this audit uses

This audit uses the full 60,000 official KMNIST training images, at both
step-count budgets, for both the local (pre-evolution) and evolved
feature comparisons.

The roles of each part of this population are frozen as Freeze 2:

| role | n |
|---|---|
| official training corpus | 60,000 |
| CNN weight-fit subset | 54,000 |
| CNN validation / model-selection subset | 6,000 |

The 6,000 validation images are held out only from **CNN weight
updates**. They are NOT held out from Stage 2B's training-side analysis
in general, and the ridge model's cross-validation and final fit both
use all 60,000 images. No test-split data is touched anywhere in this
audit.

Every comparison across artifacts uses the **official KMNIST image
index**, never a file's row position.

## The prediction basis: out-of-fold, under the frozen folds

Ridge MSE, at both step-count budgets, is computed from **out-of-fold
per-image predictions**. Out-of-fold means each image's prediction comes
from a model fold that did not include that image during training. This
uses the frozen five-fold split
(`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), for
`pre_evolution` and all four evolved conditions.

In-sample MSE, computed from a model refit on all the data at once, is
**descriptive only**, and it never feeds into a trigger. It is a weak and
potentially misleading way to measure the amendment's impact, because a
model refit can absorb a change in the underlying representation into
its own fitted coefficients, hiding the change.

The new machinery must pass a required cross-check: the per-fold mean
of the new per-image out-of-fold MSE values **must reproduce the already-
stored, fold-aggregate values** from the earlier stage-1 and stage-2
runs. This checks the new code against numbers the project already
trusts, rather than only against fresh expectations.

## Two alpha regimes (both reported; Freeze 3)

- **Fixed alpha** means: the alpha value is selected once, from the
  PRODUCTION 1,200-step representation, per condition, using the frozen
  five-fold procedure — then that same value is applied identically to
  both budgets' out-of-fold fits.
- **Reselected alpha** means: an independent alpha value is selected for
  each budget separately, using the same frozen procedure.

The audit reports the selected alpha value at both budgets under
reselection, and the production alpha value under the fixed regime.

**Disclosed, non-silent amendment: both regimes now use the THIRTEEN-
decade alpha grid.** `DESIGN.md`'s alpha grid was amended, after Phase B,
to run from `1e-6` to `1e6` — see `DESIGN.md`'s Review history section
for the full ruling, its origin, and the frozen one-time procedure that
carried it out. Nothing about this protocol's own logic changes because
of this: "the same frozen procedure" still means the same frozen
procedure, and the distinction between fixed and reselected alpha is
untouched. What changes is only which grid that procedure searches, at
both budgets, and for the production alpha value.

**Sequencing, unchanged and now important**: the amendment and the
ridge re-run both happen BEFORE the audit session runs. The audit's
fixed-alpha regime is defined as *the alpha value selected from the
production representation*, so running the audit against a production
alpha value chosen on the older, superseded nine-decade grid would
compare the two budgets through a constraint the amendment was meant to
remove — and the audit would then have to be redone. The audit reads
the re-run's own artifacts, not Phase B's original ridge output.

**An honest statement of scope, required in the results write-up**:
per-budget, fold-fitted `StandardScaler`s are kept, because that matches
the production preprocessing method. This means the fixed-alpha
comparison isolates only **the effect of reselecting alpha** — it does
*not* fully isolate the effect of the raw representation change on its
own. A comparison using one shared scaler across both budgets is
optional extra work, not required, and must never be presented as the
main check.

## Feature-distance measurements (frozen, gauge-fixed)

These are computed **after** applying the exact production gauge
(reference node 363, using the production phase-wrapping convention), at
both the pre-evolution and the evolved stage:

- per-image **maximum** wrapped circular distance, on gauged phase
  values
- per-image **root-mean-square (RMS)** wrapped circular distance, on
  gauged phase values
- per-image **Euclidean** distance, on the cosine/sine feature values

These are aggregated per image first, and then summarized across images
by median, 95th percentile, and maximum. Distances computed without the
gauge fix are a secondary diagnostic only, and never define a trigger.
No metric may be added to, or dropped from, this list after results
exist.

## The numerical resolution limit (Freeze 1) — computed analytically, not measured empirically

Let `d` be the largest recorded cross-implementation difference that is
independent of this audit: the largest `max_abs_clipped_pred_diff` value
across every condition in the stage-1 and stage-2 equivalence-check
artifacts.

    d = 1.151190e-12        (stage-1 `rewired`)

Predictions and targets are both clipped to the [0, 1] range, so
`|q - y| <= 1`. For a per-coordinate error `e` with `|e| <= d`:

    (q + e - y)^2 - (q - y)^2 = 2e(q - y) + e^2
    =>  |Delta MSE|  <=  2d + d^2

Carrying this bound through the audited statistics gives:

| statistic | terms | bound | value |
|---|---|---|---|
| single MSE | 1 | `2d + d^2` | 2.302381e-12 |
| graph contrast `Delta_g` | 2 | `4d + 2d^2` | 4.604761e-12 |
| pair ordering, shared `pre` term cancels algebraically | 2 | `4d + 2d^2` | 4.604761e-12 |
| pair ordering, four independent MSE terms | 4 | `8d + 4d^2` | 9.209522e-12 |

**A safety factor of `M = 100` is frozen here, before any audit result
exists.** The resulting operating thresholds are:

    contrast                 4.604761e-10
    pair ordering (4-term)   9.209522e-10

The team uses the 4-term bound wherever the shared `pre` term does not
cancel out algebraically in the implementation, and the 2-term bound
where it does. Whichever bound is used is stated in the results.

### Why the team rejected a threshold measured directly on the audited quantity

An earlier draft of this protocol proposed measuring the tolerance
directly, at the level of the quantity actually being audited: the
observed difference between the JAX and sklearn implementations in
`Delta_g`. That measured difference is **1.3878e-17**, with three of the
four graphs sitting at exactly 0.0.

The team rejected this approach, correctly, because of **cancellation**.
Both implementations' prediction errors are correlated on the same
images, so a contrast-level number can shrink to an artificially tiny
value even when the underlying errors are not that small. Matching the
level of the audited quantity does not make a cancellation-prone,
measured value into a reliable estimate of numerical resolution — and
three of the four graphs landing at exactly 0.0 is the *symptom* of this
cancellation, not reassurance that it is not happening. This measured
number is kept only as a **secondary check**.

The analytic derivation used instead is also what stops this from
becoming a "tuned tolerance": the bound follows only from the metric's
own algebra and a recorded value of `d`, neither of which depends on any
audit result.

**Deliberately NOT claimed here**: any advance claim that the audit's
trigger verdict would stay the same across orders of magnitude of the
safety factor `M`. Signal sizes measured at stage 2 do not establish what
effect sizes this audit will actually find. The team measures and
reports robustness **after** running the audit, not before.

## Triggers (all three, checked under EITHER alpha regime)

Any of the following conditions **triggers a renewed interpretation
review before Stage 4**:

1. **Primary-contrast sign reversal** — `Delta_T` changes sign between
   the two budgets.
2. **Improvement turning into deterioration on any graph** — any
   `Delta_g` changes sign between the two budgets.
3. **A numerically resolved pairwise order reversal** — any of the six
   pairwise graph comparisons reverses order between the two budgets, by
   more than the frozen threshold given above.

All six pairwise comparisons must be checked.

A reversal **smaller** than the threshold is treated as implementation
noise, and reported as such. A reversal **exceeding** the threshold is
called a *numerically resolved order reversal*, and it triggers review
automatically.

**Either alpha regime, on its own, is enough to trigger review.** The
alpha regime used determines the INTERPRETATION of a reversal — a
reversal under fixed alpha implicates both the representation change and
the alpha reselection together; a reversal only under reselected alpha
implicates the whole locked procedure's response to the amended
representation — but the regime does not determine WHETHER a review
happens at all.

The team assesses how much the trigger actually matters **during that
review**, not by building it into the numerical threshold in advance.

This trigger condition is not a hypothetical case: stage-2 cross-
validated contrasts already show both signs at once (`T` at -2.30e-3 and
`lattice` at -3.90e-3 are improving; `rewired` at +4.02e-3 and
`curr_random` at +2.38e-3 are worsening).

## Mechanical (automatic) checks

Mismatch checks halt the run automatically. Successfully completing all
the planned steps does **not**, by itself, create any additional
discretionary review checkpoint — review is triggered only by the
conditions listed above, or by the pre-Stage-4 review package, and never
simply because a run went smoothly.

## Companion protocols

These are two extra check plans, also frozen in advance, that this
protocol names as its partners.

- **ARM/x86 propagation stress set** — built deterministically from four
  parts: the images with the largest observed cross-chip-type encoding
  differences; the 79 **convergence-tail stress cases** (this name is
  used deliberately — an earlier description of these images as "basin-
  boundary candidates" was withdrawn as unsupported; a residual from the
  encoder pulling toward a fixed point is a different kind of quantity
  from being close to a boundary in the evolution dynamics); at least
  some images from every class; and a seeded, stratified random sample.
  The first comparison applies **identical, already-fitted ridge
  coefficients** to both the ARM-chip-derived and the x86-chip-derived
  features. This isolates the effect of numerical propagation from the
  effect of refitting a new model. The team reports the maximum
  differences at: the encoding step, the evolved features (per graph),
  the prediction, the per-image MSE, and the evolved-minus-pre-evolution
  contrast.
- **`ABS_CONV_EPS` sensitivity table** — the gate's verdict is
  recomputed from the stored final-Delta arrays, at
  `eps in {1e-10, 1e-11, 1e-12, 1e-13}`, along with a written
  justification for the value 1e-12, checked against four separate
  factors: float64 precision (observed numerical noise between 1e-14 and
  1e-16), the scale of the phase-update diagnostic itself (the smallest
  meaningful measured value is 2.177e-07), the encoder's own
  implementation, and how sensitive the downstream features are to this
  choice.
- **6,000-image final-Delta tail** — a new, independent measurement on
  images that had never been encoded before. The team reports the count
  of affected images, the total count, and which split (fit or
  validation) each belongs to, plus a descriptive comparison against the
   54,000-image rate (79/54,000 = 0.146%), with uncertainty. **There is
  no rule requiring the two rates to agree**: a different rate on the
  6,000 images is treated as a finding to report, not as a failure to
  reproduce a result.
