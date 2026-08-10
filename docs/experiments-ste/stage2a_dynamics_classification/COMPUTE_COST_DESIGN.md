Simplified Technical English version of `experiments/stage2a_dynamics_classification/COMPUTE_COST_DESIGN.md`

# Stage 2A Follow-On: Full Compute-Cost Accounting, MLP Training vs. Oscillator Inference

*Second draft. It includes review round 1 (three important corrections,
four scope decisions) and the user's own hardware-fairness fix: measure
both approaches' training AND inference (prediction) on GPU, using
`cuml.accel`, instead of measuring one approach on CPU and the other on
GPU. This design is separate from the locked confirmatory result in
`FINDINGS.md`. It does not reopen or need to rerun that result. This
follows the project's usual practice: design first, get it reviewed,
then measure.
**Locked as of round 2** — reviewed and approved for measurement,
below.*

## A named, legitimate possible outcome, stated before measuring

**Flagged in round 2, worth saying plainly now, rather than letting it
read as a disappointing result later if that is what the numbers show**:
`Train_readout` (training the oscillator readout) is dominated by a 9x5
cross-validation grid search — a search over many more combinations than
fitting one MLP. So it is entirely possible that MLP training stays
clearly cheaper than oscillator training, even once both are measured on
GPU. If MLP inference (prediction) also turns out to be as trivial as
expected (a small matrix multiply), the honest, complete answer this
design produces may simply be: **no crossover exists at any plausible
deployment scale, full stop.** That is a real, reportable result on its
own terms. It is not a failure to find a crossover, and it is not a
reason to keep looking for a scale where the outcome flips.

## The asymmetry this design measures — corrected framing from round 1

**Round-1 correction, load-bearing**: the first draft's starting claim
("the oscillator approach: cheap training, expensive inference" versus
"the MLP approach: expensive training, cheap inference") does not match
the numbers this project has already measured. MLP training (on CPU
scikit-learn) took 6.3s (`H=13`) and 26.4s (`H=128`). The oscillator
readout's training (cross-validation search plus final refit, on CPU
scikit-learn, per `FINDINGS.md`'s confirmatory section) took **246.8
minutes** for the full 6-condition search. Training is not cheap for the
oscillator side. It is far more expensive than either MLP baseline's
training, at least under CPU scikit-learn. The genuinely open question
this design exists to answer is narrower, and more honest, than the
first draft's framing:

> Given that both approaches' training costs are already known (or, in
> this revised design, about to be measured symmetrically on GPU), does
> the oscillator's recurring per-image inference cost — re-solving an ODE
> on every prediction, not a one-time cost paid off over time — ever get
> outweighed by the MLP's simpler inference at realistic deployment
> volumes? And if so, at what volume?

The two approaches' cost *shapes* still differ, in the way the first
draft identified (one recurring cost that grows with `N`, one that does
not) — that structural point survives the correction. What is corrected
is the assumption about which side is "cheap," made before actually
measuring anything.

## GPU-fairness fix: measure both approaches on GPU, not one on each

**The user's own correction**: comparing an MLP measured on CPU
scikit-learn against an oscillator pipeline measured on GPU would break
this design's own stated rule ("use the same hardware for both
approaches wherever possible"). Both approaches' training and inference
are now measured under GPU acceleration — the oscillator side using the
already-verified JAX/`diffrax` GPU port, and the MLP side using
`cuml.accel` (NVIDIA RAPIDS' tool that speeds up scikit-learn code on
GPU without changing the code). This tool is already verified elsewhere
in this project (`CUML_ACCEL_FINDINGS.md`) to reproduce scikit-learn's
`LogisticRegression` results at full scale, 14.9 times faster.

**A real, unresolved uncertainty, not assumed away**: it is not
confirmed that `cuml.accel` supports
`sklearn.neural_network.MLPClassifier`. cuML's own historical focus is
classical machine learning (linear models, trees, clustering). If
`MLPClassifier` is not one of `cuml.accel`'s supported models, calling
it under `cuml.accel` would silently fall back to plain CPU
scikit-learn, with no error and no speedup — quietly breaking the "same
hardware" rule with no visible sign this happened. **Locked: the first
measurement in this design, before anything else, is a direct check** —
time an `MLPClassifier` fit of a comparable size under `cuml.accel`
against the same fit without it. If there is no real speedup,
`cuml.accel` does not speed up this model type, and that is reported
plainly as a real limitation, not hidden.

**Check 0 result: confirmed — `cuml.accel` does not speed up
`MLPClassifier`.** At `H=128`, on the full 60,000-image raw-pixel
training set, with identical `MLP_KWARGS` settings: 31.5s under plain
scikit-learn versus 30.9s with `cuml.accel.install()` active (a 1.02x
ratio — this is noise, not a speedup). Critically, both runs used
**identical `n_iter=32`** — direct evidence the "accelerated" call ran
the exact same CPU code path, not a partial or partial GPU dispatch.
This result is not ambiguous.

**Decision: option (a).** MLP training and inference are measured on CPU
scikit-learn throughout this design. The oscillator side stays on GPU.
The hardware-fairness rule above is honored everywhere it can be (the
check itself, and the oscillator side's own CPU-versus-GPU treatment),
except for this one model type, where it genuinely cannot be. This
imbalance is disclosed here rather than silently accepted. Option (b) (a
from-scratch, GPU-native MLP reimplementation) was considered and
deliberately not pursued. The MLP side is expected to be the
uninteresting, trivially cheap half of this comparison regardless of
which hardware it runs on. Taking on fresh reimplementation risk (this
project's own `CLAUDE.md` principle 16 territory) for a component
unlikely to change the shape of the answer was judged not worth it.

## What is already measured, and reused directly (no new work)

**`Train_readout`, per topology, is already fully known from existing
data — this alone answers round-1 concern #3 (the gap between an
aggregate number and a per-topology one), at no new cost.** This
combines the stage-3 data-generation numbers (`FINDINGS.md`) with the
already-measured `cuml.accel` per-condition cross-validation-search and
final-refit times (`CUML_ACCEL_FINDINGS.md`'s full 6-condition
replication). Those numbers are already GPU-measured, so no new
measurement is needed here to satisfy the GPU-fairness fix for this
term:

| topology | encode (shared) | evolve | feat_post | CV search (`cuml.accel`) | final refit (`cuml.accel`) | **Train_readout total** |
|---|---:|---:|---:|---:|---:|---:|
| T | 68.1s | 28.3s | 1.6s | 248.2s | 15.0s | **361.2s** |
| lattice | 68.1s | 27.3s | 1.6s | 200.7s | 10.3s | **308.0s** |
| rewired | 68.1s | 29.4s | 1.6s | 274.2s | 6.0s | **379.3s** |
| curr_random | 68.1s | 28.9s | 1.6s | 161.3s | 2.4s | **262.3s** |

(`feat_post`'s 6.4s total across all 4 topologies is split evenly here,
1.6s each, as a disclosed estimate. The original measurement did not
separate this out per topology, and it is small enough relative to the
other terms that an even split does not meaningfully change any of the
four totals.) `encode`'s 68.1s figure is genuinely shared work: it does
not depend on which topology is evolved. It appears once per topology's
own total (you cannot skip it even if you only want one topology), but
it is not counted twice across topologies in any combined total.

- **Oscillator batched-evolution throughput, corrected citation**: the
  first draft cited the 100-image sanity run's rate of about 0.67
  ms/image/topology. The full 60,000-image run is the larger, more
  representative sample (see `FINDINGS.md`'s stage-3 section) — its rate
  is **0.455-0.491 ms/image/topology, about 0.475 ms on average** — and
  this is the number this revision uses. This is still a throughput
  number (images processed per second in a batch), not a per-image
  latency number (see below for that).
- **MLP baseline training cost, CPU scikit-learn, already measured and
  simply never shown in `FINDINGS.md`'s baseline table** — this is not a
  new measurement, it is a documentation gap. `H=13`: 6.3s, 27
  iterations. `H=128`: 26.4s, 32 iterations. (Round-1 correction #1.)
  These remain useful as the CPU-side comparison point, once the new
  GPU numbers exist, per the GPU-fairness fix above.

## What needs new, direct measurement

### 0. Does `cuml.accel` actually speed up `MLPClassifier`? (new, first, gates everything else on the MLP side)

See "GPU-fairness fix," above. A direct, cheap timing check, run before
anything else in this design.

### 1. MLP training cost — resolved, no new measurement (option (a))

Already measured on CPU scikit-learn, already reported above: `H=13`
6.3s (27 iterations), `H=128` 26.4s (32 iterations), using identical
`MLP_KWARGS` settings to what actually produced the reported baseline
numbers. Per the check-0 decision, this is the number used in the cost
model. There is no GPU counterpart to measure.

### 2. Oscillator single-image inference latency — all 4 topologies, CPU and GPU

**This is still the measurement most likely to be silently gotten wrong
if it is not stated explicitly**: the existing throughput figures come
from batching many images through one GPU call (`vmap`). They say
nothing about how long a *single* new image takes on its own. Per the
user's follow-up, this is measured for **all four topologies** (T,
lattice, rewired, curr_random), not a subset — this resolves round-1
concern #4 (which topology to measure) by removing the need to choose.

- **Scope, stated explicitly**: encode
  (`_local_converged_phases`) plus restrict plus evolve plus gauge
  feature extraction plus linear-readout prediction, for one image, with
  no batching — this is the real single-prediction pipeline, not just
  the evolution step measured alone.
- **CPU path**: the plain numpy/scipy `stage2a_core.evolve_on_graph`
  (using `solve_ivp`), not the JAX-on-CPU path — this resolves round-1
  concern #9 (which of two existing CPU implementations to measure). The
  numpy/scipy path is what this project actually used for encoding and
  for evolution before the GPU port existed. JAX-on-CPU was always only
  a step toward GPU, never the actual deployed CPU path.
- **GPU path**: `evolve_on_graph_jax`, called with a batch of size 1 —
  this deliberately includes, rather than hides, the real per-call
  kernel-launch and dispatch overhead that a large batch normally
  spreads out over many images. This is reported honestly even if it is
  unfavorable to GPU at this batch size, which is fully expected (see
  this project's own `evolve_on_graph_jax.py` docstring for the same
  point made earlier).
- **Repeats and warm-up, stated explicitly** (round-1 concern #6): for
  the GPU path, one untimed warm-up call runs first (this excludes JIT
  compilation time, the same practice used for every other GPU timing in
  this project), then **100 repeated single-image timed calls**, with
  the mean and standard deviation reported, not just one measurement.
  For the CPU path, no warm-up is needed (there is no JIT step), but the
  same 100-repeat mean-and-standard-deviation reporting is used, since
  the ODE solver's step count genuinely depends on the input, and a
  single measurement would not represent that.

### 3. MLP single-image inference latency, CPU only (option (a))

CPU scikit-learn `predict_proba` on one image, using the same 100-repeat
mean-and-standard-deviation approach as item 2's CPU path (no warm-up
needed, no JIT). There is no GPU counterpart, per the check-0 decision —
this asymmetry with item 2's CPU-plus-GPU split is real and disclosed,
not silently matched. This is almost certainly trivial (a `784xH + Hx10`
matrix multiply), but it is measured directly rather than assumed, per
this project's own repeated lesson (`CLAUDE.md` principle 18: steps
assumed to be cheap have been wrong before).

## The cost model

Unchanged in structure from the first draft. `Train_readout` is now
fully known per topology (above), and both `Infer_*_per_image` terms
come from the new, GPU-fairness-checked single-image measurements:

```
Total_MLP(N)         = Train_MLP + N * Infer_MLP_per_image
Total_oscillator(N)  = Train_readout + N * Infer_oscillator_per_image
```

One curve exists per oscillator topology (four total), plotted against
each of the two MLP baselines. `N` is plotted on a log scale, from `N=10`
to `N=10^8`; the cost axis is also plotted on a log scale, given that
`Train_readout`'s range (about 260-380s) and `Train_MLP`'s range (on
GPU, likely much smaller, pending measurement 1) could span several
orders of magnitude once multiplied out against per-image costs at large
`N`. Report the break-even `N` (the point where the two lines cross)
directly, for each topology-and-MLP pair where such a point exists in
that range, or state plainly if one approach is cheaper at every
plausible scale for that pair.

**Report in two units, not one** (unchanged from the first draft): real,
measured wall-clock time as the primary measure — GPU throughout for the
oscillator side, CPU throughout for the MLP side (the check-0 result
above means true hardware parity is not available for this comparison;
this is reported as a disclosed limitation, not silently worked around);
and a rough theoretical FLOPs (floating-point operations) estimate as a
hardware-independent secondary check, which matters more here than it
otherwise would, given the hardware imbalance — it is the one measure in
this design that does not depend on which physical device either
approach happened to run on. This FLOPs estimate is reported as a range
or a measured average, given the oscillator ODE solver's genuinely
input-dependent step count, not as a single precise number.

## An important, explicit scope boundary: simulated dynamics, not a physical substrate

Unchanged from the first draft. Reservoir computing's traditional
physical-substrate appeal (an analog medium that evolves "for free," at
no simulation cost) is explicitly not the regime this design
investigates. Everything measured here is a numerically-simulated ODE
running on general-purpose GPU or CPU hardware, compared against another
numerically-computed approach. A genuinely different comparison would be
needed for dedicated neuromorphic or analog hardware — this is not
addressed here, and this measurement should not be read as informing
that question either way.

## What this design will and will not establish

**Will establish**: a genuine, measured, total-cost comparison as a
function of deployment scale, for this specific classification task, on
this specific (now GPU-symmetric where possible) hardware basis, per
topology — letting "which approach is cheaper" be answered as "cheaper
under these conditions, at this volume," not asserted from parameter
count, or from a single training-cost or inference-cost snapshot alone.

**Will not establish**: anything about tasks other than this
classification comparison; anything about a hypothetical dedicated
physical substrate; a general claim about "the oscillator approach" or
"the MLP approach" outside this specific measured setup. Whatever the
result, it is a data point about this comparison, scoped the same way
every other result in this project is scoped.

## Status

Second draft. Round-1 review is incorporated: three important
corrections (the motivating-asymmetry framing did not match already-known
numbers; there was a gap between an aggregate `Train_readout` number and
a per-topology one; and item 1 turned out to be a documentation gap
rather than a new measurement); and four scope decisions (which topology
to use for item 2, CPU-and-GPU symmetry for item 3, the repeat and
warm-up practice, and the throughput-citation update). The user's own
GPU-fairness fix (measure both approaches on GPU, using `cuml.accel`,
not one on CPU and one on GPU) is folded in throughout, including a new,
explicit first check — does `cuml.accel` even speed up `MLPClassifier`
— rather than assuming it does. All four oscillator topologies are now
in scope for item 2, per the user's follow-up, which removes the
topology-choice decision point entirely, rather than resolving it either
way.

Not yet implemented — ready for a further review round if anything here
still needs correcting, or for measurement to begin if this is
considered locked.
