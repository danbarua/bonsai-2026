# Result: the gauge comparison, Stage 2B, n=60,000 train

> **RUN 1 OF 2. Read the caveats before the numbers.**
>
> **This run's frozen-alpha arm is mislabelled.** It consumed
> `ridge_cv.json`, which records a superseded 9-value grid whose floor was
> 0.01, rather than the live `ridge_cv_g13_88edf9ac.json`. So the arm below
> called "frozen at production" actually compares both gauges at 0.01. That
> is a valid common-alpha check and it is NOT the production comparison it
> claims to be. Run 2 fixes it.
>
> **This run's re-selected arm is sound**, and independently reproduced
> production's alpha selections exactly (T 1e-6, lattice 1e-6, rewired 1e-5,
> curr_random 1e-5, pre_evolution 1000).
>
> **These numbers were read before the pre-registration's amendments
> landed.** Disclosed in that document's Review History. The outcome
> classification below is therefore prose, not the decidable rule; the rule
> arrived afterwards and is applied to run 2 before reading.
>
> **Limitation carried by both runs:** the per-fold scaler diagnostics are
> discarded in favour of one full-corpus `min_col_std`.

Run 2026-08-12 on Colab L4, commit `f2d043d`, against the pre-registration
committed before it (`GAUGE_COMPARISON_PREREGISTRATION.md`). Driver:
`run_gauge_comparison.py`. Output: `results/gauge_comparison.json`.

TRAIN split only. Descriptive and nominal: no confirmatory statistic, no
member of any corrected family, no change to the locked Stage 4 verdict.

## The verification gate passed byte-exact

The locked reference-node gauge was recomputed from the cached `theta_T`
states through this driver's own consume path and compared against the
cached stage-3 `features.npz`:

| condition | max abs diff | byte-exact |
|---|---|---|
| pre_evolution, T, lattice, rewired, curr_random | **0.000e+00** | yes, all five |

Byte-exact on x86, as the pre-registration predicted (the ARM smoke gave
1.110e-16, ~1 ULP, also as predicted). The consume path is therefore not a
confound in anything below.

## The numbers

Per-condition mean clipped validation MSE, alpha re-selected by CV. `Δ%` is
circular-mean relative to reference-node; negative means circular-mean is
better.

| condition | R | offset std (rad) | reference-node | circular-mean | Δ% |
|---|---|---|---|---|---|
| T | 0.9660 | 0.2629 | 6.385452e-02 | 6.290824e-02 | **−1.48** |
| lattice | 0.9678 | 0.2311 | 6.453307e-02 | 6.368915e-02 | **−1.31** |
| curr_random | 0.9971 | 0.0136 | 6.600435e-02 | 6.578860e-02 | −0.33 |
| rewired | 0.9992 | 0.0313 | 6.676978e-02 | 6.640756e-02 | −0.54 |
| pre_evolution | 0.9116 | 0.4314 | 6.906190e-02 | 6.565832e-02 | **−4.93** |

The frozen-alpha arm agrees in every direction and ordering, so nothing
below is a selection artefact.

### Rankings

```
[re-selected] reference-node : T < lattice < curr_random < rewired < pre_evolution
[re-selected] circular-mean  : T < lattice < pre_evolution < curr_random < rewired
[frozen]      reference-node : T < lattice < curr_random < rewired < pre_evolution
[frozen]      circular-mean  : T < lattice < pre_evolution < curr_random < rewired
```

Ranking **not** preserved, in both arms, identically.

## Which registered outcome fired: BOTH 3 and 4, which was not anticipated

The pre-registration listed four outcomes as though they were exclusive.
Two of them fired together, and saying so is the honest report rather than
choosing whichever reads better.

- **Outcome 4 fired literally.** Circular-mean is better for *every*
  condition, in both alpha arms. Registered reading: collective-phase
  subtraction is the cleaner representation, and future work should carry
  the gauge as a DESIGNED factor rather than defaulting to either.
- **Outcome 3 fired by the letter.** "Ranking" was defined in advance as the
  ordering of the five conditions, and that ordering changed.

**But outcome 3's registered CONSEQUENCE did not occur, and this is the part
that must not be blurred.** Its wording anticipated the claim narrowing to
"T's topology, *read from node 363*, denoises best." That is not what
happened: **T is first under both gauges, in both arms**, and lattice is
second under both. T's primacy is not gauge-dependent.

What reordered is the *controls*, and entirely because of one condition:
`pre_evolution` moves from LAST to THIRD. Nothing else changes places.

So the claim that must narrow is not "T wins" but any claim that used the
full ordering — specifically, **"pre_evolution is worst" holds under the
reference-node gauge and NOT under circular-mean.** "Evolved beats
pre-evolution" survives for T and lattice under both gauges, and fails for
rewired and curr_random under circular-mean.

## The pattern is ordered by synchronisation, and it is coherent

The gain from changing gauge tracks the offset spread, which tracks R:

| condition | R | offset std | gain from circular-mean |
|---|---|---|---|
| pre_evolution | 0.9116 | 0.4314 | −4.93% |
| T | 0.9660 | 0.2629 | −1.48% |
| lattice | 0.9678 | 0.2311 | −1.31% |
| rewired | 0.9992 | 0.0313 | −0.54% |
| curr_random | 0.9971 | 0.0136 | −0.33% |

The reading this supports: the reference-node gauge penalises *poorly
synchronised* states, because one node's per-image deviation is noisiest
exactly where the collective phase is least defined. `pre_evolution` is the
least synchronised condition and gains most; the tightly-locked controls
barely move. Node 363's deviation behaves as added noise here, not as
carried signal — which weakens, rather than supports, the hypothesis that
the reference node's deviation is itself part of the readout mechanism.

## The dimensionality caveat, applied as registered

Circular-mean has 1010 features to reference-node's 1008 — a ~0.2% capacity
edge, in the direction of the winner. Registered instruction was that an
epsilon win for circular-mean must not be over-read.

Applied honestly: **curr_random (−0.33%) and rewired (−0.54%) are within the
range where two extra columns are a plausible contributor** and should not
be read as a gauge effect on their own. T (−1.48%), lattice (−1.31%) and
pre_evolution (−4.93%) are too large for that explanation. The ranking
change is driven by pre_evolution, which is in the second group.

## Companion: the prediction partly held, and inverted in the tail

Predicted (registered before the run): spread descends with ascending R, so

    T > lattice > curr_random > rewired

Observed:

    T (0.2629) > lattice (0.2311) > rewired (0.0313) > curr_random (0.0136)

The head of the ordering held: the two low-R graphs have by far the largest
spread, an order of magnitude above the others.

**The tail inverted, and the registration said an inversion is not
tolerable.** `rewired` has both higher R (0.9992 vs 0.9971) *and* higher
offset spread (0.0313 vs 0.0136) than `curr_random`, which is the opposite
of what tighter locking predicts. Reported as registered rather than
explained away. Mitigating context, offered as context and not as a rescue:
the two differ in R by 0.0021, both spreads are ~10x below the low-R graphs,
and node 363's degree differs per graph — but the direction is still wrong,
and the registration's point was that this is worth knowing.

## Also reported, as the pre-registration required

**Grid-endpoint selections.** Under the reference-node gauge, `T` and
`lattice` both select the LOWEST alpha on the grid (1e-6) — CV wanted less
regularisation than the grid offers, for the two best conditions under the
locked gauge. Under circular-mean both move off the endpoint (1e-5). No
other condition or arm hits an endpoint.

**Scaler centring margins.** `min_col_std` under circular-mean runs smaller
than under reference-node (e.g. curr_random 2.485e-05 vs 6.784e-05). No
halt threshold was breached and none was invented after the fact; recorded
because principle 23's territory is exactly a quantity drifting toward a
floor.

## What this does not do

It does not reopen Stage 2A, reopen Stage 2B, alter the locked Stage 4
verdict, or evaluate anything on the test split. The reference-node gauge
was part of the frozen pipeline definition and remains so; this measures an
alternative rather than substituting one.

Per Stage 2A's own pre-registered rule, applied one stage over: the
disagreement is REPORTED as a finding about gauge sensitivity, not resolved
by picking whichever gauge is more favourable.
