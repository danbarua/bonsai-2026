# Pre-registration: the gauge-sensitivity comparison, run on Stage 2B

Written and committed BEFORE the comparison is run, and before any
circular-mean number exists. Its purpose is to fix the interpretations in
advance, so that whichever of the four outcomes lands, the reading was
chosen when it could still have gone the other way.

Authorised by Dan, 2026-08-12. Design reviewed in advance.

## Why this is being run at all

`DESIGN.md:198-206` (Stage 2A) pre-registered the reference-node gauge as
primary and circular-mean centering as the robustness check, and committed
that if the two "disagree materially on the confirmatory result, that
disagreement is reported as a finding about gauge sensitivity, not resolved
by picking whichever was significant."

The comparison was never run. `circular_mean_features` exists at
`stage2a_core.py:124`, is unit-tested for its output dimension only, and no
pipeline calls it.

Whether that omission could have mattered has since been measured
(`measure_gauge_offset.py`, committed): it is NOT a formality.

- No FIXED linear map relates the two feature sets. Relative residual
  8.854e-01. Both gauges rotate each `(cos, sin)` pair by a single angle,
  but that angle is `theta_ref` in one and the circular mean in the other,
  and their difference is PER IMAGE. An image-dependent rotation is not a
  fixed linear map, so no ridge readout can absorb it by reweighting.
- The offset does not collapse under synchronisation. On 200 real corrupted
  TRAIN images evolved through T at R(T)=0.9650, `theta_ref - mu` has mean
  +0.0473 rad, std 0.2659 rad, range [-0.4869, +0.8103].

**Why Stage 2B rather than Stage 2A.** Stage 2A is closed and locked, and
its confirmatory apparatus is not being reopened. Stage 2B uses the SAME
reference-node gauge, at the same node 363, throughout. So the live question
is not only "should Stage 2A disclose an unrun check" but "does the Stage 2B
result survive a change of vantage point". Nobody has shown the gauges DO
disagree on outcomes — only that nothing prevents it.

## Scope, and what this is not

- **TRAIN split only.** The test split is not touched.
- **RUN_SCOPED outputs, under new object names.** Nothing is written near
  the lineage `stage3/common` paths. The cached `theta_T` artifacts are
  consumed READ-ONLY, with pinned parents.
- **No new confirmatory statistic.** Every number here is DESCRIPTIVE and
  nominal. This is not a member of any corrected family, does not enter the
  evidence hierarchy, and does not alter the locked Stage 4 verdict.
- **Not a trigger.** No result here re-opens a closed stage or licenses a
  test-split evaluation.
- **The locked pipeline definition is unchanged.** The reference-node gauge
  was part of the frozen pipeline; this measures an alternative, it does not
  retroactively substitute one.

## What is held fixed

Everything except the gauge function. Any difference must be attributable to
the gauge alone.

| Held fixed | Value |
|---|---|
| Input states | the cached `stage3/*/theta_T.npz`, consumed read-only |
| Target `Y` | clean `images_01` restricted to the 505 active support |
| Stratifier | KMNIST train labels |
| Folds | `StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)` |
| Alpha grid | `stage2b_ridge.ALPHA_GRID`, all 13 values |
| Conditions | `pre_evolution`, `T`, `lattice`, `rewired`, `curr_random` |
| Gauge functions | imported from `stage2a_core`, never reimplemented |

`raw_505` and `raw_784` are excluded: they are pixel baselines with no phase
and therefore no gauge.

**Alpha is run BOTH ways**, because they answer different questions:

1. **Re-selected** — `cross_validate_alpha` unchanged. The honest
   pipeline-versus-pipeline comparison: each gauge gets the alpha its own CV
   would choose.
2. **Frozen at production values** — T/lattice/rewired/curr_random at 0.01,
   `pre_evolution` at 1000.0, read from the frozen `stage3/common/ridge_cv.json`.
   This isolates feature geometry from selection effects.

## The metric

**Per-condition mean clipped validation MSE**, the same quantity the stage-3
report ranks on. "Ranking" below means the ordering of the five conditions by
that number. Margins means the gaps between them.

## Registered interpretations — all four, fixed in advance

1. **Ranking preserved, margins similar.** A gauge-robust result. It
   strengthens the standing of the locked verdict, and WEAKENS the
   hypothesis that node 363's per-image deviation is itself part of the
   mechanism — the signal is readable from either vantage point.
2. **Ranking preserved, margins move materially.** The gauge affects readout
   efficiency but not ordering. Pure mechanism information; no threat to the
   result.
3. **Ranking changes.** The locked result stands untouched — the
   reference-node gauge was part of the frozen pipeline definition — and
   Stage 2A's own pre-registered rule applies one stage over: reported as a
   finding about gauge sensitivity, NOT resolved by picking a winner. What
   changes is the scope of the claim, which narrows honestly from "T's
   topology denoises best" to "T's topology, read from node 363, denoises
   best." A scope finding, not a refutation.
4. **Circular-mean better across the board.** Collective-phase subtraction is
   the cleaner representation, and any future work should carry the gauge as
   a DESIGNED factor rather than defaulting to either.

## The dimensionality asymmetry, registered as a caveat

The two gauges do not produce the same number of features:

- `reference_node_features` → **1008** = 2*505 - 2. The reference node's own
  `(cos, sin)` pair is deterministically `(1, 0)` and is dropped.
- `circular_mean_features` → **1010** = 2*505. Nothing is dropped.

Circular-mean therefore has a two-column capacity edge. It is tiny against
1008, but it is in the direction of favouring circular-mean, so **an epsilon
win for circular-mean must not be over-read**. Both are run exactly as
defined; neither is padded or trimmed to match, because changing either
definition would mean the thing measured is no longer the pre-registered
gauge.

## Companion measurement, with its prediction registered

**Per-condition `std(theta_ref - mu)`**, on the same states, computed with
the wrapped-angle handling from `measure_gauge_offset.py` (the raw difference
of two angles is discontinuous at the branch cut and its spread is
meaningless).

**Prediction, registered now so it can falsify the sync picture:** gauge
sensitivity should itself be ordered by R. Tighter lock means node 363 sits
closer to the collective mean, so the spread should DESCEND with ascending R:

> T > lattice > curr_random > rewired

**Caveat attached at registration:** node 363 is one node, and its degree
differs per graph, so mild deviations from the aggregate expectation are
tolerable. An INVERSION is not — that would mean something in the
synchronisation picture is wrong, which is worth knowing.

## A companion that is NOT being run, and why

The design conversation proposed also recomputing "the ρ=−1 accessible
target energy scalar" under the new gauge, described as the strongest
mechanism clue in the record.

**No such quantity exists in this repository.** Searched repo-wide across
every `FINDINGS.md` and every `.py`: there is no "accessible target energy",
and no ρ of −1. The only `rho` in Stage 2B is the encoder gate's ratio
(`stage2b_encoder_gate.py`, values like 169.851), which is an unrelated
quantity. The nearest real correlation in the record is Stage 2A's
`r = -0.170, p = 2.3e-27` between fold-0 `R(theta)` and projection onto the
smallest-singular-value direction — weak, negative, and not the same object.

Rather than guess a definition and compute something under a borrowed name,
it is left out and recorded here. If the intended quantity is identified, it
can be added as its own measurement.

## Verification gate, run before any circular-mean number is trusted

Before computing anything under the new gauge, the driver recomputes the
**reference-node** features from the same cached `theta_T` through the same
consume → restrict → feature path, and compares against the cached stage-3
`features` artifacts.

This is principle 16 aimed at this driver rather than at someone else's: a
gauge function verified in isolation says nothing about the glue around it —
artifact consumption, ordering, restriction, dtype. If the reproduction of
the KNOWN gauge does not match, no number from the UNKNOWN gauge means
anything, and the driver halts.

Expectation: byte-exact on x86 (same numpy code, same input bytes). On ARM,
~1 ULP differences are expected per this project's own measured propagation
table, so the gate compares with an explicit tolerance and REPORTS the
observed maximum difference either way.

## Halt conditions

The driver halts, loudly, on any of:

- the reference-node reproduction gate failing;
- any non-finite feature under either gauge;
- a scaler-centring margin (`min_col_std`) low enough to put the standardiser
  near its floor — checked at the n=1,000 smoke first, deliberately, so a
  floor surprise surfaces there rather than mid-run;
- any fold's alpha selection landing on a grid endpoint under the re-selected
  arm without that being reported.
