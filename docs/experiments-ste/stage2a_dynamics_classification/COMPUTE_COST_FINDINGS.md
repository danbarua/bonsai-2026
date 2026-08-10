Simplified Technical English version of `experiments/stage2a_dynamics_classification/COMPUTE_COST_FINDINGS.md`

# Stage 2A Follow-On: Full Compute-Cost Accounting — Results

**Status: complete. This document answers the question
`COMPUTE_COST_DESIGN.md` (locked, round 2) asked. It is separate from
the locked confirmatory result in `FINDINGS.md`, and does not reopen
it.**

## Headline result, stated plainly

**No crossover exists at any plausible deployment scale. The oscillator
approach is strictly more expensive than either MLP baseline at every
value of `N`, from a single image to 100 million images, and the gap
grows with scale rather than shrinking.** This is exactly the outcome
named as legitimate in the design's round-2 review, before any
measurement took place. It is not a disappointing result — it is the
honest answer the numbers give.

At `N=1`: the oscillator (`evolved_T`, GPU evolution) costs 13.7 times
what `MLP_H128` costs. At `N=1,000,000`: 375.6 times. At
`N=100,000,000`: 551.8 times. Every algebraic break-even point
(oscillator versus either MLP, for any of the 4 topologies) works out to
a **negative** `N`. A negative break-even means there is no positive
deployment volume at which the oscillator's per-image cost ever becomes
cheaper, because it is not, in fact, cheaper per image. It costs more to
train, *and* it costs more to run, for every topology, against both
baselines.

## Check 0 (gating): `cuml.accel` does not speed up `MLPClassifier`

This was confirmed empirically before any other measurement, per the
design's own locked first step: `H=128` `MLPClassifier`, full
60,000-image raw-pixel training set, identical hyperparameters — 31.5s
under plain scikit-learn versus 30.9s with `cuml.accel.install()` active
(a 1.02x ratio — noise-level). Critically, both runs had **identical
`n_iter=32`** — direct evidence the "accelerated" call ran the exact
same CPU code path, not a partial GPU dispatch. The decision taken (the
design's option (a)): MLP training and inference are measured on CPU
scikit-learn throughout; the oscillator side stays on GPU. This hardware
imbalance is real and disclosed, not silently resolved — see "What this
does and does not establish," below, for what it does and does not
affect.

## `Train_readout`, per topology (already known, no new measurement)

Combining the stage-3 data-generation numbers (`FINDINGS.md`) with the
already-measured `cuml.accel` per-condition cross-validation-search and
final-refit times (`CUML_ACCEL_FINDINGS.md`'s full 6-condition
replication):

| topology | encode (shared) | evolve | feat_post | CV search (`cuml.accel`) | final refit (`cuml.accel`) | **Train_readout** |
|---|---:|---:|---:|---:|---:|---:|
| T | 68.1s | 28.3s | 1.6s | 248.2s | 15.0s | **361.2s** |
| lattice | 68.1s | 27.3s | 1.6s | 200.7s | 10.3s | **308.0s** |
| rewired | 68.1s | 29.4s | 1.6s | 274.2s | 6.0s | **379.3s** |
| curr_random | 68.1s | 28.9s | 1.6s | 161.3s | 2.4s | **262.3s** |

`Train_MLP`: `H=13` 6.3s (27 iterations), `H=128` 26.4s (32 iterations)
— CPU scikit-learn, already measured, never previously shown in
`FINDINGS.md`'s baseline table (this is a documentation gap corrected
here, not a new measurement, per round-1 review).

## Single-image inference latency — the new measurement this design exists to produce

**100 repeats per condition, mean and standard deviation reported, not
one draw.** The GPU path includes one untimed warm-up call (this
excludes JIT compilation time); the CPU path needs none (no JIT). Full
pipeline scope throughout: encode plus restrict plus evolve plus gauge
feature extraction plus linear-readout prediction, for one image, with
no batching.

### Oscillator (all 4 topologies)

| topology | CPU (numpy/scipy) | GPU (`evolve_on_graph_jax`, batch=1) |
|---|---|---|
| T | 322.33 +/- 4.72 ms | 29.61 +/- 0.31 ms |
| lattice | 321.79 +/- 3.92 ms | 29.30 +/- 0.30 ms |
| rewired | 281.98 +/- 2.90 ms | 29.67 +/- 0.25 ms |
| curr_random | 278.16 +/- 2.37 ms | 29.71 +/- 0.37 ms |

**A genuine, honestly-reported new finding, not expected going in**: on
CPU, `rewired` and `curr_random` (the two nearly fully-synchronized
topologies, per `FINDINGS.md`'s stage-3 R(theta) results) are noticeably
*faster* per single image (about 278-282ms) than `T` and `lattice`
(about 322ms) — roughly a 12-13% difference, well outside the roughly
1-2% measurement noise (standard deviation under 5ms in every case).
**On GPU, this difference nearly disappears** (29.3-29.7ms, under a 2%
spread across all four topologies). A plausible explanation: `scipy`'s
adaptive-step `solve_ivp` genuinely takes fewer or cheaper steps to
integrate a state that quickly settles into near-total synchronization,
while `diffrax`'s fixed-precision, GPU-batched integration is either
less sensitive to this effect at a single-item batch size, or the encode
step (identical across all four topologies, an unknown fixed share of
both totals) makes up enough of the GPU-path total to hide the spread.
This is reported as an observation, not investigated further — it is
outside this design's scope.

**An honest limitation, not hidden**: the CPU measurement ran on this
project's own machine's CPU. The "GPU path" ran encode, gauge, and
readout on the Colab virtual machine's own CPU, plus GPU evolution.
These are not measurements on identical CPU hardware — the roughly
9-11x CPU-versus-GPU-path speedup partly reflects genuine GPU-evolution
acceleration, and partly reflects however Colab's server CPU compares to
this machine's CPU. This design did not attempt to separate the two
effects.

### MLP (CPU only, per the check-0 decision)

| condition | CPU latency |
|---|---|
| MLP_H13 | 0.0525 +/- 0.0080 ms |
| MLP_H128 | 0.0534 +/- 0.0059 ms |

This is trivial, as expected, and it barely depends on `H`, confirming
rather than merely assuming this (per `CLAUDE.md` principle 18). **The
oscillator's cheapest single-image path (GPU, curr_random, 29.30ms) is
still about 550 times slower than the MLP's most expensive single-image
path (CPU, H128, 0.0534ms).**

*(Both the oscillator and MLP latency measurements use a classifier and
scaler fit on synthetic, randomly generated data of the correct shape,
not the real locked-C fit. This is a deliberate, disclosed choice:
`predict_proba`'s wall-clock cost depends only on the matrix dimensions,
not on the fitted parameter values or which `C` was used, so a real fit
was not needed to time this step honestly. This avoided re-running the
expensive real fits solely for a latency check.)*

## The cost model, on both hardware bases

**Every break-even `N` (oscillator versus either MLP, for any topology)
is negative** — confirmed algebraically, not just read off a chart:

| comparison | break-even N |
|---|---:|
| T (GPU) vs. MLP_H13 | -12,007 |
| T (GPU) vs. MLP_H128 | -11,327 |
| lattice (GPU) vs. MLP_H13 | -10,315 |
| lattice (GPU) vs. MLP_H128 | -9,628 |
| rewired (GPU) vs. MLP_H13 | -12,594 |
| rewired (GPU) vs. MLP_H128 | -11,916 |
| curr_random (GPU) vs. MLP_H13 | -8,632 |
| curr_random (GPU) vs. MLP_H128 | -7,954 |

A negative break-even means the two cost lines already crossed *before*
`N=0`, and they never cross again for any real deployment volume. The
oscillator line starts above the MLP line, and its slope (its per-image
cost) is also steeper, so the gap only grows.

**Representative total costs, `evolved_T` versus `MLP_H128`, using the
GPU-oscillator basis** (the basis this project would actually use if
deploying, since `cuml.accel` does not help the MLP side):

| N | oscillator (T, GPU) | MLP_H128 (CPU) | ratio |
|---:|---:|---:|---:|
| 1 | 361.23s | 26.40s | 13.7x |
| 1,000 | 390.81s | 26.45s | 14.8x |
| 1,000,000 | 29,971.20s (8.3 hr) | 79.80s | 375.6x |
| 100,000,000 | 2,961,361.20s (34.3 days) | 5,366.40s (1.5 hr) | 551.8x |

**Same-hardware basis (CPU versus CPU, the one comparison in this
design with no cross-machine confound)** — this shows an even starker
gap, since oscillator CPU inference (278-322ms) is about 10 times its
GPU-path figure:

| N | oscillator (T, CPU) | MLP_H128 (CPU) | ratio |
|---:|---:|---:|---:|
| 1 | 361.52s | 26.40s | 13.7x |
| 1,000 | 683.53s | 26.45s | 25.8x |
| 1,000,000 | 322,691.20s (89.6 hr) | 79.80s | 4,043.7x |
| 100,000,000 | 32,233,361.20s (373 days) | 5,366.40s (1.5 hr) | 6,006.5x |

![Total compute cost vs. deployment scale](results/compute_cost_vs_n.png)

*A log-log plot, `N=10` to `N=10^8`. All four oscillator/GPU curves
(colored) sit strictly above both MLP/CPU curves (black) at every value
of `N` — there is no crossing anywhere in, or beyond, the plotted range.*

## Rough FLOPs estimate — the hardware-independent cross-check

This is order-of-magnitude only, per the design's own framing (the
oscillator ODE solver's step count genuinely depends on the input; this
is not a fixed-shape computation the way a matrix multiply is):

| computation | ~FLOPs/image |
|---|---:|
| MLP_H13 forward pass | ~20,600 |
| MLP_H128 forward pass | ~203,300 |
| Oscillator encode (150 steps x 784 nodes x 5 trig-ops) | ~588,000 |
| Oscillator evolve, per RHS evaluation (505x505: diff+sin+sum) | ~765,000 |
| Oscillator evolve @ ~10 RHS evals (rough low end) | ~7,650,000 |
| Oscillator evolve @ ~100 RHS evals (rough high end) | ~76,500,000 |

Even at the low end of the RHS-evaluation-count range, the oscillator's
per-image FLOPs (encode plus evolve, about 8.2 million at the low
estimate) exceed `MLP_H128`'s (about 203,000) by roughly **40 times**.
At the high end, it is roughly **400 times**. This is the same
direction, and a similar order of magnitude, as the wall-clock result —
confirming the oscillator's higher inference cost is a **fundamental
property of the computation itself**, not an artifact of `diffrax` being
a less-optimized library than sklearn's or PyTorch's matrix-multiply
code. Both measures point the same way, independently.

## What this does and does not establish

**Does establish**: for this specific classification task, on this
specific (partly hardware-symmetric) measurement basis, across all four
prespecified topologies and both MLP baselines — the oscillator
approach's total compute cost is strictly higher than either MLP's, at
every deployment scale from 1 to 100 million images, with no crossover.
This is confirmed by two independent lines of evidence (measured
wall-clock time and a rough FLOPs estimate), not one.

**Does not establish**: anything about tasks other than this
classification comparison; anything about a hypothetical dedicated
physical substrate (per the design's own explicit scope boundary — none
of this measures or informs the "free, real-time physical dynamics"
regime that traditional reservoir computing sometimes assumes); a
general claim about "the oscillator approach" or "the MLP approach"
outside this specific measured setup. The CPU-versus-GPU-path hardware
imbalance (the MLP was never measured on GPU; the "GPU path" mixes this
machine's CPU-measured baseline against Colab's own CPU plus GPU) means
the *exact* multiplier at any given `N` should be read as approximate,
not precise to the last digit. But the qualitative conclusion (no
crossover, gap grows with scale) holds under both hardware bases
measured (CPU-versus-CPU, and GPU-oscillator-versus-CPU-MLP), so it is
not an artifact of which basis is used.

## Code

**Amended**: these scripts were originally left uncommitted, per this
project's convention for one-off GPU scripts. They were committed
alongside the reproducibility-gaps closure elsewhere in this project
(see `FINDINGS.md`'s "Reproducibility gaps" section), since these
specific scripts produced this document's headline numbers, the same
reasoning that applied to the confirmatory GPU driver scripts.
`measure_oscillator_cpu_latency.py` (CPU single-image latency, all 4
topologies), `prep_oscillator_latency_gpu_inputs.py` and
`measure_oscillator_gpu_latency.py` (the matching remote-session GPU
scripts — the first stages the small input files, the second runs on
the Colab GPU session itself; you cannot run it locally as it is, same
convention as `stage3_gpu_evolve.py`), `measure_mlp_cpu_latency.py` (MLP
single-image latency), `build_cost_model.py` (the cost-model analysis
and plot — this transcribes the already-verified numbers above rather
than re-deriving them from raw data files, as disclosed in its own
docstring). `check0_cuml_mlp.py` (the gating check) remains genuinely a
one-off script — a true one-time check that does not feed any number
reported here, unlike the others.

**Run order** (the CPU-side scripts run locally; the GPU pair needs a
`mighty-colab` session, see `README.md`'s "Reproducing the confirmatory
GPU evolution" for the general upload-and-run pattern):
```bash
uv run python experiments/stage2a_dynamics_classification/measure_oscillator_cpu_latency.py
uv run python experiments/stage2a_dynamics_classification/measure_mlp_cpu_latency.py
uv run python experiments/stage2a_dynamics_classification/prep_oscillator_latency_gpu_inputs.py
# upload the staged inputs + evolve_on_graph_jax.py, then on the remote session:
#   mighty-colab exec -s <session> -f measure_oscillator_gpu_latency.py
uv run python experiments/stage2a_dynamics_classification/build_cost_model.py
```

## Next step

None specified by this design — the question it was built to answer
(does a crossover exist, and where) has a complete, negative answer. Any
further extension (a genuinely GPU-native MLP for true hardware parity;
investigating the CPU-side synchronization-speed effect noted above)
would be a new, separate design decision, not a continuation of this
one.
