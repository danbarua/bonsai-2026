# Stage 2A Follow-On: Full Compute-Cost Accounting, MLP Training vs. Oscillator Inference

*Second draft. Incorporates review round 1 (three load-bearing
corrections, four scoping decisions) and the user's own GPU-parity
adjustment (measure both approaches' training AND inference on GPU,
via `cuml.accel`, not CPU sklearn for one side and GPU for the other).
Standalone from, and does not reopen or depend on re-running, the
locked confirmatory result in `FINDINGS.md`. Follows this project's
established convention: design first, reviewed, then measured.
**Locked as of round 2** -- reviewed and approved for measurement,
below.*

## A named, legitimate possible outcome, stated before measuring

**Flagged in round 2, worth stating plainly now rather than letting it
read as a disappointing result later if it's what the numbers show**:
given `Train_readout` is dominated by a 9x5 CV grid search -- a
combinatorially larger procedure than fitting one MLP -- it is entirely
plausible that MLP training remains decisively cheaper than oscillator
training even once both are measured on GPU. If MLP inference also
turns out to be as trivial as expected (a small matrix multiply), the
honest, complete answer this design produces may simply be: **no
crossover exists at any plausible deployment scale, full stop.** That
is a real, reportable result on its own terms -- not a failure to find
one, and not grounds to keep looking for a scale where the picture
flips.

## The asymmetry this quantifies -- corrected framing from round 1

**Round-1 correction, load-bearing**: the first draft's motivating claim
("oscillator: cheap training, expensive inference" vs. "MLP: expensive
training, cheap inference") does not match the numbers this project has
already measured. MLP training (CPU sklearn) took 6.3s (`H=13`) and
26.4s (`H=128`); the oscillator readout's training (CV search + final
refit, CPU sklearn, `FINDINGS.md`'s confirmatory section) took **246.8
minutes** for the full 6-condition search. Training is not cheap for
the oscillator side -- it is dramatically more expensive than either
MLP baseline's training, at least under CPU sklearn. The genuinely open
question this design exists to answer is narrower and more honest than
the first draft's framing:

> Given both approaches' training costs are already known (or, per this
> revision, about to be measured symmetrically on GPU), does the
> oscillator's recurring per-image inference cost (re-solving an ODE on
> every prediction, not a one-time cost amortized away) ever get
> outweighed by the MLP's inference-side simplicity at realistic
> deployment volumes -- and if so, at what volume?

The two approaches' cost *shapes* still differ in the way the first
draft identified (one recurring cost that scales with `N`, one that
doesn't) -- that structural point survives the correction. What's
corrected is the assumption about which side is "cheap" anywhere in
this picture before actually measuring.

## GPU-parity adjustment: measure both approaches on GPU, not one on each

**The user's own correction**: comparing an MLP measured on CPU sklearn
against an oscillator pipeline measured on GPU would silently violate
this design's own stated principle ("same hardware for both approaches
wherever possible"). Both approaches' training and inference are now
measured under GPU acceleration -- the oscillator via the already-
verified JAX/`diffrax` GPU port, the MLP via `cuml.accel` (NVIDIA
RAPIDS' zero-code-change sklearn GPU acceleration, already verified
elsewhere in this project's `CUML_ACCEL_FINDINGS.md` to reproduce
sklearn's `LogisticRegression` results at full scale, 14.9x faster).

**A real, unresolved uncertainty, not assumed away**: `cuml.accel`'s
estimator coverage is not confirmed to include
`sklearn.neural_network.MLPClassifier`. cuML's own historical focus is
classical ML (linear models, trees, clustering) -- if `MLPClassifier`
isn't one of `cuml.accel`'s accelerated estimators, calling it under
`cuml.accel` would silently fall back to plain CPU sklearn underneath,
with no error and no speedup, quietly breaking the "same hardware"
premise without any visible signal that it happened. **Locked: the
first measurement in this design, before anything else, is a direct
check** -- time an `MLPClassifier` fit of comparable size under
`cuml.accel` against the same fit without it. If there is no
meaningful speedup, `cuml.accel` does not accelerate this estimator,
and that is reported plainly as a real limitation, not silently papered
over.

**Check 0 result: confirmed, `cuml.accel` does not accelerate
`MLPClassifier`.** `H=128`, full 60,000-image raw-pixel training set,
identical `MLP_KWARGS`: 31.5s plain sklearn vs. 30.9s with
`cuml.accel.install()` active (1.02x -- noise, not a speedup), and
critically **identical `n_iter=32` in both runs** -- direct evidence
the "accelerated" call executed the exact same CPU code path, not a
partial or marginal GPU dispatch. Not ambiguous.

**Decision: option (a).** MLP training and inference are measured on
CPU sklearn throughout this design; the oscillator side stays on GPU.
The hardware-parity principle above is honored everywhere it could be
(the check itself, and the oscillator side's own CPU-vs-GPU treatment)
except this one estimator, where it genuinely cannot be, and that
asymmetry is disclosed here rather than silently accepted. Option (b)
(a from-scratch GPU-native MLP reimplementation) was considered and
deliberately not pursued -- the MLP side is expected to be the
uninteresting, trivially-cheap half of this comparison regardless of
which hardware it runs on, and taking on fresh reimplementation risk
(this project's own `CLAUDE.md` principle 16 territory) for a component
unlikely to change the shape of the answer was judged not worth it.

## What's already measured, reused directly (no new work)

**`Train_readout`, per topology, fully resolved from existing data --
this alone answers round-1 concern #3 (the aggregate-vs-per-topology
scoping gap) at no new cost**, by combining the stage-3 data-generation
numbers (`FINDINGS.md`) with the already-measured `cuml.accel` per-
condition CV-search and final-refit times (`CUML_ACCEL_FINDINGS.md`'s
full 6-condition replication) -- itself already GPU-measured, so no
re-measurement is needed to satisfy the GPU-parity adjustment for this
term:

| topology | encode (shared) | evolve | feat_post | CV search (`cuml.accel`) | final refit (`cuml.accel`) | **Train_readout total** |
|---|---:|---:|---:|---:|---:|---:|
| T | 68.1s | 28.3s | 1.6s | 248.2s | 15.0s | **361.2s** |
| lattice | 68.1s | 27.3s | 1.6s | 200.7s | 10.3s | **308.0s** |
| rewired | 68.1s | 29.4s | 1.6s | 274.2s | 6.0s | **379.3s** |
| curr_random | 68.1s | 28.9s | 1.6s | 161.3s | 2.4s | **262.3s** |

(`feat_post`'s 6.4s total across all 4 topologies is split evenly here,
1.6s each, as a disclosed estimate -- the original measurement did not
isolate it per topology, and it is small enough relative to the other
terms that an even split does not materially affect any of the four
totals.) `encode`'s 68.1s is genuinely shared: it does not depend on
which topology is evolved, so it appears once per topology's own total
(you cannot skip it even if you only want one topology) but is not
double-counted across topologies in any aggregate.

- **Oscillator batched-evolution throughput, corrected citation**: the
  first draft cited the 100-image sanity run's ~0.67 ms/image/topology.
  The full 60,000-image run is the larger, more representative sample
  (`FINDINGS.md`'s stage-3 section) -- **0.455-0.491 ms/image/topology,
  ~0.475 ms average** -- and is the number this revision uses. Still a
  throughput number, not a per-image latency number; see below.
- **MLP baseline training cost, CPU sklearn, already measured and simply
  never surfaced in `FINDINGS.md`'s baseline table** -- not a new
  measurement, a documentation gap. `H=13`: 6.3s, 27 iterations.
  `H=128`: 26.4s, 32 iterations. (Round-1 correction #1.) These remain
  useful as the CPU-side comparison point once the new GPU numbers
  exist, per the GPU-parity adjustment above.

## What needs new, direct measurement

### 0. Does `cuml.accel` actually accelerate `MLPClassifier`? (new, first, gates everything else on the MLP side)

See "GPU-parity adjustment," above. A direct, cheap timing check, run
before anything else in this design.

### 1. MLP training cost -- resolved, no new measurement (option (a))

Already measured on CPU sklearn, already reported above: `H=13` 6.3s/27
iterations, `H=128` 26.4s/32 iterations, identical `MLP_KWARGS` to what
actually produced the reported baseline numbers. Per the check-0
decision, this is the number used in the cost model -- no GPU
counterpart exists to measure.

### 2. Oscillator single-image inference latency -- all 4 topologies, CPU and GPU

**Still the measurement most likely to be silently gotten wrong if not
stated explicitly**: the existing throughput figures come from batching
many images through one GPU call (`vmap`) -- they say nothing about how
long a *single* new image takes in isolation. Per the user's follow-up,
measured for **all four topologies** (T, lattice, rewired,
curr_random), not a subset -- resolves round-1 concern #4 (which
topology) by removing the need to choose.

- **Scope, explicit**: encode (`_local_converged_phases`) + restrict +
  evolve + gauge-feature extraction + linear-readout prediction, for
  one image, no batching -- the real single-prediction pipeline, not
  the evolution step in isolation.
- **CPU path**: the plain numpy/scipy `stage2a_core.evolve_on_graph`
  (`solve_ivp`), not the JAX-CPU-backend path -- resolves round-1
  concern #9 (ambiguity between two existing CPU implementations). The
  numpy/scipy path is what this project actually used for encoding and
  (pre-GPU-port) evolution; JAX-on-CPU was only ever an intermediate
  step toward GPU here, never the deployed CPU path.
- **GPU path**: `evolve_on_graph_jax`, called with a batch of size 1 --
  deliberately including, not hiding, the real per-call kernel-
  launch/dispatch overhead a large batch amortizes away. Reported
  honestly even if unfavorable to GPU at this batch size (fully
  expected to be, per this project's own `evolve_on_graph_jax.py`
  docstring precedent).
- **Repeats and warm-up, explicit** (round-1 concern #6): for the GPU
  path, one untimed warm-up call first (excludes JIT compilation, same
  discipline as every other GPU timing in this project), then **100
  repeated single-image timed calls**, reporting mean and standard
  deviation, not one measurement. For the CPU path, no warm-up needed
  (no JIT), but the same 100-repeat mean/std reporting, since ODE
  solver step count is genuinely input-dependent and a single draw is
  not representative.

### 3. MLP single-image inference latency, CPU only (option (a))

CPU sklearn `predict_proba` on one image, same 100-repeat mean/std
discipline as item 2's CPU path (no warm-up needed, no JIT). No GPU
counterpart, per the check-0 decision -- the asymmetry with item 2's
CPU+GPU split is real and disclosed, not silently matched. Almost
certainly trivial (a `784xH + Hx10` matrix multiply) -- measured
directly rather than assumed, per this project's own repeated lesson
(`CLAUDE.md` principle 18: assumed-cheap steps have been wrong before).

## The cost model

Unchanged in structure from the first draft, now with `Train_readout`
fully known per topology (above) and both `Infer_*_per_image` terms
coming from the new, GPU-parity single-image measurements:

```
Total_MLP(N)         = Train_MLP + N * Infer_MLP_per_image
Total_oscillator(N)  = Train_readout + N * Infer_oscillator_per_image
```

One curve per oscillator topology (four total) against each of the two
MLP baselines. Plot vs. `N` on a log-scale x-axis (`N=10` to `N=10^8`);
y-axis log-scale too, given `Train_readout`'s ~260-380s range and
`Train_MLP`'s (GPU, pending measurement 1) likely-much-smaller range
could span orders of magnitude once multiplied out against per-image
terms at large `N`. Report the break-even `N` explicitly for each
topology/MLP pair where one exists in that range, or state plainly if
one approach dominates at every plausible scale for that pair.

**Report in two units, not one** (unchanged from the first draft):
real, measured wall-clock as primary -- GPU throughout for the
oscillator side, CPU throughout for the MLP side (the check-0 result
above means true hardware parity isn't available for this comparison;
reported as a disclosed limitation, not silently worked around); a
rough theoretical FLOPs estimate as a hardware-independent secondary
cross-check, which matters more than it otherwise would given the
hardware asymmetry -- it's the one basis in this design that doesn't
depend on which physical device either approach happened to run on.
Reported as a range or measured-average given the oscillator ODE
solver's genuinely input-dependent step count, not a single precise
number.

## An important, explicit scope boundary: simulated dynamics, not physical substrate

Unchanged from the first draft. Reservoir computing's traditional
physical-substrate appeal (an analog medium evolving "for free," no
simulation cost) is explicitly not the regime this design investigates.
Everything measured here is a numerically-simulated ODE on
general-purpose GPU/CPU hardware, compared against another numerically-
computed approach. A genuinely different comparison would be needed for
dedicated neuromorphic/analog hardware -- not addressed here, and this
measurement should not be read as informing that question either way.

## What this does and will not establish

**Will establish**: a genuine, measured total-cost comparison as a
function of deployment scale, for this specific classification task, on
this specific (now GPU-symmetric) hardware basis, per topology --
letting "which approach is cheaper" be answered as "cheaper under these
conditions, at this volume," not asserted from parameter count or a
single training-cost or inference-cost snapshot alone.

**Will not establish**: anything about tasks other than this
classification comparison; anything about a hypothetical dedicated
physical substrate; a general claim about "the oscillator approach" or
"the MLP approach" outside this specific measured setup. Whatever the
result, it is a data point about this comparison, scoped the same way
every other result in this project is scoped.

## Status

Second draft. Round-1 review (three load-bearing corrections: the
motivating-asymmetry framing didn't match already-known numbers, the
`Train_readout` aggregate-vs-per-topology scoping gap, and item 1 being
a documentation gap rather than a new measurement; four scoping
decisions: topology choice for item 2, CPU/GPU symmetry for item 3,
repeat/warm-up discipline, and the throughput-citation update)
incorporated. The user's own GPU-parity adjustment (measure both
approaches on GPU via `cuml.accel`, not one on CPU and one on GPU) is
folded in throughout, including a new, explicit first check (does
`cuml.accel` even accelerate `MLPClassifier`) rather than assuming it
does. All four oscillator topologies now in scope for item 2, per the
user's follow-up, removing the topology-choice decision point entirely
rather than resolving it either way.

Not yet implemented -- ready for a further review round if anything
here still needs correcting, or for measurement to begin if this is
considered locked.
