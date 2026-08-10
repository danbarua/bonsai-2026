Simplified Technical English version of `experiments/stage1d_topology_specificity_gpu/FINDINGS.md`.

# Stage 1D GPU Port: A JAX and A100 Test Run, and a Confirmed Bug in Delta_map

## What This Document Is About

The team built a new version of the simulator from Stage 1B.2 and Stage
1C. The new version uses JAX and diffrax. JAX is a Python library for
fast math. Diffrax is a JAX library that solves differential equations.
The original simulator function is `run_one_trial`, in the file
`stage1b2_core.py`. The new version runs on a GPU instead of a CPU. A
GPU is a computer chip built for fast parallel math.

The team ran the new version on an A100 GPU. An A100 is a type of GPU.
The team used a tool named `mighty-colab` to control the GPU session.

The Stage 1D pilot study tests topology specificity. Topology
specificity means: does the specific shape of a graph matter for the
result? The pilot study compares a lattice graph against
stochastic-control graphs. This study is in
`experiments/stage1d_topology_specificity/`. Its script
`analyze_stage1d.py` only runs on a CPU. A CPU run of the full pilot is
too slow, so the pilot only tested a small subset of the data.

The team wanted to answer one question: can the GPU version run the
full Stage 1D pilot fast enough to test all the data? This would
remove the CPU speed limit.

## What Is Confirmed to Work

A script named `verify_on_gpu.py` checks the new JAX version against
the real, trusted version. It compares `run_one_trial_jax_faithful.py`
(the new GPU version) against `run_one_trial` (the original,
numpy-based version). The check compares each output value one field
at a time. This is called a field-by-field comparison.

The check used 4 test cases. Each test case picks one node, one sign,
and one amplitude. The 4 nodes cover a range of node degree: low,
median, and high. Node degree means how many connections a node has.
The amplitudes cover all three amplitude regimes the project uses.

The check compared these output fields: `event_aligned_q`,
`event_aligned_r`, the tangent and residual versions of each, and the
matching fixed-time versions. Tangent response is the linear-guess
version of a result. Residual is the real result minus the tangent
response. See the project Glossary for the full definitions.

All fields matched closely. The largest difference was between 1e-6
and 1e-8. This is far smaller than the project's cross-solver
tolerance of 1e-4. A cross-solver tolerance is the largest difference
the project accepts between two different solver methods.

The value `tau_star` matched exactly between the two versions.
Tau_star marks a specific time point used in event alignment.

**The JAX port itself is correct.**

The script `real_pilot_benchmark.py` also checked simulation speed.
This speed check is real and trustworthy. The team ran all 37 pilot
trajectories on the A100 GPU. A trajectory is one full run of the
oscillator network from a starting point through time. The 37
trajectories break down as: 10 lattice trajectories, 9 rewired
trajectories, 9 hist_random trajectories, and 9 curr_random
trajectories. See the project Glossary for the meaning of `lattice`,
`rewired`, `hist_random`, and `curr_random`.

The GPU run used two JAX speed tools: `jax.vmap` and `jax.jit`. These
tools let JAX run many trials together and compile the code for speed.
The GPU run finished all 37 trajectories in 32.63 seconds.

The team also measured the same 37-trajectory simulation on an M1
CPU. The CPU run took 3660 seconds (61 minutes). The GPU run was 112
times faster than the CPU run, for the simulation step alone.

## The Bug: Delta_map Did Not Match Stage 1C's Stored Value

The script `real_pilot_benchmark.py` has a built-in check. It
recomputes Delta_map for one specific trajectory: T's own baseline
trajectory at seed=3000. T is the learned topology graph (see the
project Glossary). Delta_map is the project's main score for how
strongly a graph turns different starting kicks into different,
reliable spatial patterns. The recompute uses the real, trusted
functions in `analyze_stage1b2.py`. The check then compares this new
Delta_map value against Stage 1C's own stored, trusted value for this
same trajectory: 0.3505.

The GPU run reported a Delta_map of 0.2842 instead. This is a real,
meaningful difference. It is not random noise.

The team compared `real_pilot_benchmark.py`'s function
`build_432_batch()` against the real reference code. This comparison
found two candidate bugs.

**Bug 1: wrong replica-direction generation.** A replica direction is
a small, fixed nudge applied to a starting state (see the Glossary
entry for "replica"). The real function,
`generate_fixed_replica_directions()`, builds each direction like
this:

- Draw random values from a normal distribution:
  `rng.normal(0, 1, n)`.
- Remove the part of the direction that is just a global phase shift.
  A global phase shift moves every node by the same amount. It does
  not count as a real difference. The code removes it using a
  projection matrix: `P = I - ones(n,n)/n`.
- Scale the direction to unit length.

`build_432_batch()` did not do this. Instead, it drew values directly
from `rng.uniform(-1, 1, n)`. This is the wrong distribution. It skips
the projection step. It skips the unit-length scaling step. The team
used this wrong method for all 6 replica perturbation directions.

**Bug 2: dropped E_min validity gate.** The real function
`run_one_trial` has a validity rule. If a value called `E(tau)` never
goes above a threshold named `E_MIN`, the trial is not valid. When a
trial is not valid, `event_aligned_valid` is False. In that case,
`run_one_trial` returns `None` for every value in the
`event_aligned_*` group of results.

The new function, `run_one_trial_jax_faithful.py`, cannot return
`None`. It runs many trials together in one compiled, batched
computation, and `None` does not fit that structure. Instead, it
always computes a value for `event_aligned_q`. Its own code comment
says the caller must check `event_aligned_valid` and apply the E_min
rule.

`real_pilot_benchmark.py` did not do this check. It stored
`event_aligned_q` for every trial, valid or not. It never checked
`event_aligned_valid` before passing results to `analyze_stage1b2.py`.
The function `d_q()` in `analyze_stage1b2.py` only excludes a pair of
results when the value is exactly `None`. Since the value was never
`None`, invalid trials could leak into the result.

### Confirmed Cause (CPU Test, This Session)

A first attempt to find this bug happened earlier, on the GPU session
itself. The script was `diagnose_deltamap.py`, run through
`mighty-colab exec` on the A100. There was also a local copy,
`diagnose_deltamap_local.py`. That earlier attempt used a temporary
Claude Code session. The session ended before the work finished.
Nobody wrote up a conclusion. There is no results file and no findings
document, on the GPU machine or locally. A cached file
(`diagnose_deltamap_local.py`'s `.pyc` file) shows the script had only
just started loading its modules when the session ended.

This session redid the diagnosis from scratch, using
`diagnose_deltamap_local.py` in this repository. The test is a 4-way
factorial test. A 4-way factorial test tries every combination of two
yes/no choices. Here, the two choices are:

- Directions: correct or buggy.
- E_min gating: correct or buggy.

That gives 4 combinations. The test used the real numpy
`run_one_trial` for all 4 combinations, not the JAX version. The JAX
version was already confirmed to match the numpy version exactly (see
above). Using only the numpy version isolates the two candidate bugs.
It removes any chance that a difference between the numerical solvers
is the real cause.

The test ran in parallel, using `multiprocessing.Pool`. This follows
the project's standard method for this kind of test.

Results:

| Directions | Gating  | Pooled Delta_map |
|---|---|---|
| correct | correct | 0.3505 (exact match to Stage 1C's stored value) |
| buggy   | correct | 0.2851 |
| correct | buggy   | 0.3505 (no change) |
| buggy   | buggy   | 0.2851 (matches the GPU run's reported value of 0.2842) |

**The direction-generation bug, on its own, fully explains the
discrepancy.** The E_min-gating bug is real. It is worth fixing in the
JAX pipeline, for general robustness. But for this specific
trajectory, the E_min-gating bug had no effect on the result. All 432
trials for T at seed=3000 already passed the `event_aligned_valid`
check. So there were no invalid trials to leak into the average.

### The Fix

The fix has two parts.

Part 1: fix the direction-generation code. In `build_432_batch()`, in
both `real_pilot_benchmark.py` and any future GPU pilot script,
replace this code:

```python
rng_r = np.random.default_rng(replica_direction_seed)
directions = [rng_r.uniform(-1, 1, n) for _ in range(N_REPLICAS)]
```

with a call to the real reference function:

```python
directions = generate_fixed_replica_directions(n, replica_direction_seed, N_REPLICAS)
```

This function was already imported into these scripts from
`stage1b2_core`. It was simply never called.

Part 2: fix the E_min gating. This fix matters for general robustness,
even though it did not cause this specific bug. At the point where
results are stored, the code should check `event_aligned_valid`
first. If the trial is not valid, the code should store `None` for
`event_aligned_q`. This matches the rule the real numpy version
follows. Without this fix, the JAX port's raw output, which can
contain NaN values, could be stored for invalid trials instead.

## Update: The Fix Was Applied, Checked on CPU, Then Checked Again on GPU

The team applied both fixes directly to `build_432_batch()`. The
fixes cover both places this function is called in
`real_pilot_benchmark.py`: the main per-trajectory loop, and the
separate check on T's own seed=3000 trajectory.

Before running anything on a GPU, the team ran two checks.

Check 1: the team tested the fixed `build_432_batch()` on its own,
using the real `W_T` graph data. This test did not need diffrax. Only
the JAX trial-runner import needs diffrax. The test confirmed:

- Correct shapes: 432 trials, with 505-dimension states.
- Correct, unique values for node, sign, and amplitude across the
  trials.
- The first replica state, `theta0_b[0]`, exactly matched a
  separately, independently computed replica state built with
  `generate_fixed_replica_directions()`.
- The same value provably differed from the old, buggy state built
  with `uniform(-1, 1)`.

This confirms the fix is real. It did not accidentally have no
effect.

Check 2: this fixed code path is the same one already tested in the
4-way factorial table above. It is the same function, with the same
arguments, imported from the same `stage1b2_core` module. So no new
CPU recomputation of the full 432-trial Delta_map was needed. The team
already had this value in hand.

The team then set up a fresh A100 GPU session. The team confirmed the
old session was fully gone first: `mighty-colab sessions` showed no
sessions, and `adopt --orphanage` found nothing to recover.

The team re-ran `verify_on_gpu.py` unchanged, as a first step on the
fresh GPU. It reproduced the same precision as before: a largest
difference between 1e-6 and 1e-8, marked PASS. This confirmed the
fresh GPU environment works correctly.

The team then re-ran the corrected `real_pilot_benchmark.py` in full.
Results:

```
Total JAX GPU simulation time, all 37 trajectories: 32.67s
M1 baseline (real, measured, simulation stage only): 3660s (61 min)
Speedup vs real M1 baseline: 112.0x

Per-t_p Delta_map (T, seed=3000): {0: 0.39549, 0.833: 0.32147, 1.667: 0.33953, 2.5: 0.34557}
Pooled Delta_map (T, seed=3000, from this GPU run): 0.3505
(Stage 1C's own cached figure for T, seed=3000: 0.3505 -- exact match)
```

**Confirmed:** the corrected GPU pipeline reproduces Stage 1C's
stored Delta_map value exactly (0.3505 vs. 0.3505). The CPU-only
reproduction earlier in this document also got 0.3505. All three
values agree to about 7-8 significant figures. This level of
agreement matches the cross-solver precision already confirmed in
`verify_on_gpu.py`.

The fix did not meaningfully change simulation speed. The corrected
run took 32.67 seconds, close to the original buggy run's 32.63
seconds. This is expected. The bug was in what data the batch was
built from, not in how expensive the simulation itself was to run. So
the 112x speedup figure still stands.

Per-trajectory GPU compute time ranged from about 0.87 to 0.96
seconds. This excludes warm-up and compile time. The 37 trajectories
break down as:

- 10 lattice trajectories (seeds 3000 through 3090).
- 9 rewired trajectories: 3 realizations, with 3 trajectories each
  (seeds 3000, 3010, 3020).
- 9 hist_random trajectories: 3 realizations, with 3 trajectories
  each.
- 9 curr_random trajectories: 3 realizations, with 3 trajectories
  each.

This matches the pilot study's design exactly.

### Check for the Same Bug in Other Files

The team checked every other script in this folder for the same kind
of bug. The bug pattern is: writing a new, simplified, or wrong
version of a function instead of calling the real, already-imported
version. This check was requested. The team did not assume the other
files were clean.

Two files are clean: `verify_single_trial.py` and
`verify_vmap_batch.py`. Both correctly call
`generate_fixed_replica_directions()`.

Two files do not apply:
`extract_pre_computed_class0_construction.py` and
`extract_pre_computed_class0_lattice.py`. These are simple scripts
that load and re-save data. They contain no simulation logic.

One file is unrelated: `experiment.ipynb`. It is a Colab
session-information inspector.

One file is **not clean**:
`DO_NOT_USE_bonsai_real_pilot_gpu_benchmark.ipynb`. This file was
renamed from `bonsai real pilot gpu benchmark.ipynb` (see below for
why).

Cell 15 of this notebook defines a separate, simpler JAX
reimplementation, named `run_one_trial_jax`. It has its own inline
versions of `force_jacobian_jax` and `rhs`. This version does not
compute event alignment at all. It does not compute `tau_star`, `E`,
`C`, or the q, r, or residual values. It only returns the raw final
`(theta, delta)` state at `T_HORIZON`.

This notebook's own version of the batch-building function,
`build_432_batch_for_graph()`, has the identical direction-generation
bug: `rng_r.uniform(-1, 1, n)`.

A code comment in cell 17 reads: `M1_MINUTES_37_TRAJECTORIES = 61  #
Claude Code's first-run figure; re-check once the corrected run
lands`. This comment suggests the notebook was an in-progress
alternative version, or a rewrite. It was abandoned before it was
run. All the relevant cell outputs are empty. The notebook never
produced a reported number, correct or buggy.

This notebook predates `run_one_trial_jax_faithful.py`. It appears to
be superseded by it.

The team left this notebook as-is, rather than fixing it. The reason:
it is a materially different, less complete implementation. It is
missing the actual event-alignment logic that this whole pipeline
depends on. Deciding whether to finish, fix, or discard this notebook
is a scope decision. That decision belongs to whoever continues this
work next. It is not something to silently patch.

The team renamed the file. Its new name has a `DO_NOT_USE_` prefix.
Its old name was `bonsai real pilot gpu benchmark.ipynb`, in the same
directory. The new name stops the file from looking like a runnable
starting point to a future reader with no other context. The file's
contents were not otherwise changed.

## Status

The bug is fixed. The fix was checked on CPU, then checked again
end-to-end on GPU. The Stage 1D GPU pilot's Delta_map figures can now
be trusted.

The stale notebook mentioned above has been renamed with a
`DO_NOT_USE_` prefix. It was not fixed or deleted. Finishing it,
fixing it, or discarding it outright remains a scope decision for
whoever continues this thread.
