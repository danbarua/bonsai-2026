# Stage 2B amendment-impact audit — FROZEN PROTOCOL

**Status: frozen. Committed before any audit result exists or is
inspected.** Nothing in this document may be changed once an audit number
has been seen. Its purpose is to fix, in advance, what will be measured,
what the numerical resolution limit is, and what consequences follow —
so that none of those can be chosen after the fact to suit a result.

Frozen under the conditional consensus of 2026-08-06 (Claude Desktop and
ChatGPT review rounds 1-8, archived in `.claude/claude2gpt/archive/`).

## What this audits, and what it explicitly is not

The encoder budget was raised from 150 to 1,200 iterations after ladder
stage 1's gate failure — a prospective post-failure amendment, made
before downstream confirmatory evaluation, **not** a preregistered
component of the original design.

This audit quantifies that amendment's representational impact. **It is
not a model-selection knob.** The 1,200-step budget stays frozen
regardless of every number this audit produces. What may change as a
result is the SCOPE OF THE CLAIM, never the budget.

## Sign convention (frozen)

    Delta_g = MSE_evolved_g - MSE_pre_evolution

**Negative = improvement** (evolved reconstructs better than
pre-evolution). Stated explicitly because every trigger below is phrased
in terms of the sign and ordering of `Delta_g`.

## Population

Full 60,000 official KMNIST training images, both budgets, for local and
evolved feature comparisons alike.

Role labels, frozen (Freeze 2):

| role | n |
|---|---|
| official training corpus | 60,000 |
| CNN weight-fit subset | 54,000 |
| CNN validation / model-selection subset | 6,000 |
| ridge CV and final-fit corpus | **60,000** |

The 6,000 are held out from **CNN gradient updates only**. They are not
held out from Stage 2B training-side analysis, and the ridge path uses
all 60,000. No test-split data is touched anywhere in this audit.

All cross-artifact comparison is **by official KMNIST image index**,
never by positional prefix.

## Prediction basis: out-of-fold, under the frozen folds

Ridge MSE at both budgets is computed from **out-of-fold per-image
predictions** under the frozen five-fold partition
(`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), for
`pre_evolution` and all four evolved conditions.

In-sample MSE from a full refit is **descriptive only** and never feeds a
trigger: it is a weak and potentially misleading measure of amendment
impact, because a refit can absorb representational change into its own
coefficients.

Required cross-check on the new machinery: the per-fold mean of the new
per-image OOF MSEs **must reproduce the already-stored fold-aggregate
values** from the stage-1 and stage-2 runs. This pins new code against
numbers this project already trusts, rather than against fresh
expectations.

## Alpha regimes (both reported; Freeze 3)

- **Fixed alpha** := the alpha selected from the PRODUCTION 1,200-step
  representation, per condition, via the frozen five-fold procedure —
  then applied identically to both budgets' OOF fits.
- **Reselected alpha** := independent selection per budget, by the same
  frozen procedure.

Report the selected alpha at both budgets (reselected) and the
production alpha (fixed).

**Honest scope statement, required in the results write-up**: per-budget
fold-fitted `StandardScaler`s are retained, because that is the
production preprocessing. Fixed-alpha therefore isolates **the effect of
alpha reselection** — it does *not* completely isolate raw
representation change. A shared-scaler comparison is optional secondary
work, not required, and must not be presented as the primary probe.

## Feature-distance metrics (frozen, gauge-fixed)

Computed **after** applying the exact production gauge (reference node
363, production wrapping convention), at both the pre-evolution and the
evolved stage:

- per-image **max** wrapped circular distance on gauged phases
- per-image **RMS** wrapped circular distance on gauged phases
- per-image **Euclidean** distance on the cos/sin features

Aggregation is **per-image first**, then distribution: median, p95, max.
Ungauged distances are secondary diagnostic only and never
trigger-defining. No metric may be added or dropped after results exist.

## Numerical resolution limit (Freeze 1) — analytic, not empirical

Let `d` be the maximum recorded audit-independent cross-implementation
discrepancy: the largest `max_abs_clipped_pred_diff` over every
condition of the stage-1 and stage-2 equivalence artifacts.

    d = 1.151190e-12        (stage-1 `rewired`)

Predictions and targets are both clipped to [0,1], so `|q - y| <= 1`.
For a per-coordinate perturbation `e` with `|e| <= d`:

    (q + e - y)^2 - (q - y)^2 = 2e(q - y) + e^2
    =>  |Delta MSE|  <=  2d + d^2

Propagated through the audited statistics:

| statistic | terms | bound | value |
|---|---|---|---|
| single MSE | 1 | `2d + d^2` | 2.302381e-12 |
| graph contrast `Delta_g` | 2 | `4d + 2d^2` | 4.604761e-12 |
| pair ordering, shared `pre` cancelled algebraically | 2 | `4d + 2d^2` | 4.604761e-12 |
| pair ordering, four independent MSE terms | 4 | `8d + 4d^2` | 9.209522e-12 |

**Safety factor `M = 100`, frozen here, before any audit result.**
Operative thresholds:

    contrast                 4.604761e-10
    pair ordering (4-term)   9.209522e-10

Use the 4-term bound wherever the shared `pre` term is not cancelled
algebraically in the implementation; use the 2-term bound where it is,
and state in the results which was used.

### Why the empirical contrast-level number was rejected as primary

An earlier draft of this protocol proposed deriving the tolerance
empirically at the level of the audited quantity — the observed
JAX-vs-sklearn difference in `Delta_g`, which measures **1.3878e-17**,
with three of four graphs at exactly 0.0.

That was rejected, correctly, for **cancellation**: both implementations'
prediction errors correlate on the same images, so a contrast-level
aggregate can cancel to artificial tininess. Semantic-level matching does
not make a cancellation-prone empirical aggregate a reliable resolution
estimate — and three of four graphs sitting at exactly 0.0 is the
*symptom* of that cancellation, not reassurance about it. The empirical
number is retained as a **secondary check only**.

The analytic derivation is also what does the anti-"tuned tolerance"
work: the bound follows from the metric's algebra and a recorded `d`,
neither of which depends on any audit result.

**Deliberately NOT claimed here**: any prospective statement that the
trigger verdict is insensitive across orders of magnitude of `M`.
Stage-2 signal sizes do not establish unseen audit effect sizes.
Robustness is measured and reported **after** the audit, not asserted
before it.

## Triggers (all three, under EITHER alpha regime)

Any of the following triggers **renewed interpretation review before
Stage 4**:

1. **Primary-contrast sign reversal** — `Delta_T` changes sign between
   budgets.
2. **Improvement-to-deterioration on any graph** — any `Delta_g` changes
   sign between budgets.
3. **Numerically resolved pairwise order reversal** — any of the six
   pairwise graph comparisons reverses order between budgets by more
   than the frozen threshold above.

All six pairwise comparisons must be covered.

A reversal **smaller** than the threshold is implementation dust and is
reported as such. A reversal **exceeding** it is a *numerically resolved
order reversal* and triggers review automatically.

**Either alpha regime triggers review.** The alpha mechanism determines
the INTERPRETATION of a reversal — a fixed-alpha reversal implicates
representation plus reselection, a reselected-only reversal implicates
the complete locked procedure's response to the amended representation —
but it does not determine WHETHER review occurs.

Materiality is assessed **in that review**, not encoded retrospectively
into the numerical threshold.

The trigger is not hypothetical: stage-2 cross-validated contrasts
already show both signs (`T` -2.30e-3 and `lattice` -3.90e-3 improving;
`rewired` +4.02e-3 and `curr_random` +2.38e-3 worsening).

## Mechanical gates

Mismatch gates halt automatically. Successful completion of the planned
steps does **not** itself create an additional discretionary review
checkpoint — review is triggered by the conditions above, or by the
pre-Stage-4 package gate, and not merely by things having gone well.

## Companion protocols

- **ARM/x86 propagation stress set** — deterministic construction:
  largest observed cross-architecture encoding discrepancies; the 79
  **convergence-tail stress cases** (so named deliberately — the earlier
  "basin-boundary candidates" characterisation was retracted as
  unsupported, encoder-contraction residuals being a different
  dynamical quantity from evolution-flow boundary proximity); coverage of
  every class; and a seeded stratified random component. First
  comparison applies **identical frozen ridge coefficients** to
  ARM-derived and x86-derived features, isolating numerical propagation
  from refit variation. Report max differences at: encoding, evolved
  features per graph, prediction, per-image MSE, and the
  evolved-minus-pre contrast.
- **`ABS_CONV_EPS` sensitivity table** — gate verdict recomputed from the
  stored final-Delta arrays at eps in {1e-10, 1e-11, 1e-12, 1e-13}, with
  a written justification of 1e-12 against four axes: float64 precision
  (observed dust 1e-14..1e-16), the scale of the phase-update diagnostic
  (smallest meaningful measured value 2.177e-07), the encoder
  implementation, and downstream feature sensitivity.
- **6,000-image final-Delta tail** — an independent new measurement on
  never-encoded images. Report numerator, denominator and split
  membership, plus a descriptive comparison against the 54k rate
  (79/54,000 = 0.146%) with uncertainty. **There is no expected-agreement
  criterion**: a different proportion is a finding to report, not a
  reproducibility failure.
