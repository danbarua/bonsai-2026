# Pre-registration: the gauge comparison, run on Stage 2A itself

Written and committed BEFORE any classifier is fitted under either gauge.
Authorised by Dan, 2026-08-12, after the Stage 2B run answered the prior
question ("could the gauges differ at all?") in the affirmative.

Companion to `experiments/stage2b_denoising/GAUGE_COMPARISON_PREREGISTRATION.md`
and its Review History. The decidable classification rule below is carried
over from that document's amendment 1 — this is its first use as a genuine
before-the-fact registration rather than a sighted application.

## What is different from the Stage 2B run

Stage 2B was cheap because its evolved states were already cached. **Stage
2A's are not.** No `scratch/` survives locally and nothing sits under a
`stage2a` prefix in the bucket, so its train-side states are being
REGENERATED from the committed pipeline: local encode of 60,000 KMNIST
training images, then evolution on four topologies.

The 2B states cannot be reused: 2B evolved CORRUPTED images for a denoising
readout, 2A evolves CLEAN images for a classification readout. Same
topologies, different inputs, different task.

| | Stage 2B run | This run |
|---|---|---|
| readout | ridge, alpha grid (13) | logistic, **C grid (9)**, 1e-4 … 1e4 |
| metric | mean clipped validation MSE | **validation accuracy** |
| folds | StratifiedKFold, FOLD_SEED | StratifiedKFold(5, shuffle, **seed 42**) |
| states | cached, consumed read-only | **regenerated** |

## Scope, and the constraint that is not negotiable

- **TRAIN SPLIT ONLY.** `stage2a-prepare-test` and `stage2a-evolve-test-gpu`
  are not run, and no path here reads the 10,000-image official test set.
  That evaluation was ONE-SHOT and is already spent on the locked result; a
  robustness check that consumed it would destroy the thing it is checking.
  So this answers "does the ordering hold under the other gauge, on train",
  and explicitly NOT "what is the test-set verdict under circular-mean".
- Stage 2A remains COMPLETE and LOCKED. This measures an alternative gauge;
  it does not substitute one, and it cannot alter the locked verdict.
- Descriptive and nominal. No new confirmatory statistic, no member of any
  corrected family.

## What is held fixed

Everything except the gauge function, which enters at a single call site
(`analyze_stage3_results_jax.py:71`). Both gauges are IMPORTED from
`stage2a_core` and neither is reimplemented (principle 16).

Same encoded states, same topologies, same labels, same folds, same seed,
same C grid, same classifier.

## The decidable rule, registered before any number exists

Carried over verbatim in structure from the 2B amendment, with the metric
inverted because **higher accuracy is better** where lower MSE was:

    M_g = ACC_cm(g) - ACC_ref(g)          per condition
    ranking = the exact condition order by validation accuracy, per gauge

**Class, in precedence order:**

- **(C)** ranking changed — any pairwise inversion
- else **(B)** preserved, margins material — max|M_g| >= theta
- else **(A)** preserved, margins similar

**Threshold.** theta is NOT carried over from 2B: that value (3.78e-4) was
an MSE margin and means nothing on an accuracy scale. Registered here
instead as **theta = 0.002 (0.2 accuracy points)**, chosen before the run as
roughly the resolution at which a 60,000-image five-fold CV accuracy
difference is worth discussing at all — approximately the binomial standard
error on a 12,000-image validation fold at ~90% accuracy
(sqrt(0.9*0.1/12000) ~ 0.0027). Chosen from the sampling geometry, not from
any observed difference.

**Direction flag, orthogonal to the class:** if all conditions' `M_g` share
a sign, report "uniform direction favouring X" ALONGSIDE the class, never
instead of it.

**Ties:** accuracy equal to 5 decimal places counts as preserved, so float
noise cannot manufacture an inversion.

## The dimensionality asymmetry, registered again

Unchanged from 2B and it applies identically here: `reference_node_features`
gives **1008** columns, `circular_mean_features` gives **1010**, because the
reference node's own constant `(1, 0)` pair is dropped. The two-column edge
favours circular-mean, so a uniform circular-mean win smaller than theta
carries that caveat.

## Registered expectation, so this can be wrong

From the 2B result: the gauge penalty scaled monotonically with (1 - R), and
node 363's deviation behaved as added noise rather than carried signal. If
that mechanism is a property of the states rather than of the denoising
task, then on 2A:

> **Prediction.** Circular-mean is >= reference-node for every condition
> (uniform direction), the gain is largest for the LEAST synchronised
> condition, and the evolved-graph ordering is preserved — with any ranking
> change, if one occurs, coming from `pre_evolution` moving up.

That is a real prediction and it can fail. 2A is a different readout on
different inputs: classification may be insensitive to a per-image rotation
that regression was sensitive to, in which case the honest result is
class (A) and the 2B mechanism does not transfer.

## What also gets revisited, for free

`FINDINGS.md:321-334` predicted the top of the C grid would stay
uninformative for evolved conditions at any sample size; at stage 3 both
`evolved_T` and `evolved_lattice` selected C=1000 and only the
non-convergence half was revisited (item 5 of the open Stage 2A
observations). This run re-selects C over the same grid, so it reports which
C each condition and gauge lands on, and whether any lands at a grid
endpoint.

## Halt conditions

- The regenerated states failing the pipeline's own go/no-go mechanical
  checks (solver failure rate, non-finite feature vectors).
- Non-finite features under either gauge.
- Any condition whose regenerated reference-node accuracy is inconsistent
  with the locked record, which would mean the REGENERATION — not the gauge
  — is what this run is measuring. This is the analogue of the 2B run's
  byte-exact reproduction gate, and it is weaker: the states are
  regenerated rather than consumed, so exact reproduction is not expected
  and the comparison is against the recorded values with the difference
  reported.
