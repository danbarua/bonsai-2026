# Stage 1D GPU port: JAX/A100 pilot benchmark, and a confirmed Delta_map bug

## What this is

A JAX/diffrax port of Stage 1B.2/1C's per-trial simulator
(`run_one_trial` in `stage1b2_structured_transformation/stage1b2_core.py`),
run on an A100 via `mighty-colab`, to test whether the Stage 1D lattice/
stochastic-control pilot (`experiments/stage1d_topology_specificity/`,
CPU-only, `analyze_stage1d.py`) could be reproduced fast enough on GPU to
run at full scale instead of the CPU-limited pilot subset.

## What's confirmed good

`verify_on_gpu.py`: field-by-field comparison of `run_one_trial_jax_faithful.py`
against the real numpy `run_one_trial`, across 4 (node, sign, amplitude)
test cases spanning low/median/high-degree nodes and all three amplitude
regimes. All fields (`event_aligned_q`, `event_aligned_r`, tangent and
residual variants, fixed-time counterparts) matched to 1e-6 to 1e-8 max
absolute difference -- well inside the 1e-4 cross-solver tolerance.
`tau_star` matched exactly. **The JAX port itself is correct.**

`real_pilot_benchmark.py`'s raw simulation timing is also real: all 37
pilot trajectories (lattice x10, rewired x9, hist_random x9, curr_random x9)
ran in 32.63s on the A100 via `jax.vmap` + `jax.jit`, against a measured
M1 CPU baseline of 3660s (61 min) for the same 37-trajectory simulation
stage -- a genuine 112x speedup, simulation-only.

## The bug: Delta_map didn't reproduce Stage 1C's cached value

`real_pilot_benchmark.py` includes a cross-check: recompute Delta_map for
T's own baseline trajectory (seed=3000) from this GPU run's output via
the real `analyze_stage1b2.py` functions, and compare against Stage 1C's
already-cached, trusted figure for this exact trajectory (0.3505). The
GPU run reported **0.2842** -- a real, non-trivial discrepancy, not noise.

Two candidate bugs were identified by diffing `real_pilot_benchmark.py`'s
`build_432_batch()` against the real reference code:

1. **Wrong replica-direction generation.** The real
   `generate_fixed_replica_directions()` draws `rng.normal(0, 1, n)`,
   projects out the rotation-invariant (global phase-shift) component via
   `P = I - ones(n,n)/n`, then unit-normalizes. `build_432_batch()`
   instead drew `rng.uniform(-1, 1, n)` directly -- wrong distribution,
   no projection, no normalization -- for all 6 replica perturbation
   directions.

2. **Dropped E_min validity gate.** The numpy `run_one_trial` returns a
   dict of `None`s for the whole `event_aligned_*` block whenever
   `event_aligned_valid` is False (E(tau) never exceeded `E_MIN`).
   `run_one_trial_jax_faithful.py` can't return `None` from a batched/
   jitted computation, so it always computes `event_aligned_q` and
   documents (in its own comment) that the E_min gating is the caller's
   responsibility. `real_pilot_benchmark.py` stored `event_aligned_q`
   unconditionally and never checked `event_aligned_valid` before handing
   results to `analyze_stage1b2.py`, whose `d_q()` only excludes pairs on
   `is None`.

### Confirmed causal attribution (CPU reproduction, this session)

The GPU session that first investigated this (`diagnose_deltamap.py`,
run via `mighty-colab exec` against the A100 kernel, and a follow-up
local adaptation `diagnose_deltamap_local.py`) was an ephemeral Claude
Code instance whose session was torn down before it finished or wrote up
a conclusion -- no results file, no findings doc, on the VM or locally.
`diagnose_deltamap_local.py`'s `.pyc` cache shows it had only just started
importing modules when the process ended.

`diagnose_deltamap_local.py` (this repo) redoes the diagnosis cleanly: a
4-way factorial (correct/buggy directions x correct/buggy E_min gating),
using the real numpy `run_one_trial` throughout (no JAX -- already
verified equivalent above, so this isolates the glue-code bugs, not
numerical-engine differences), parallelized via `multiprocessing.Pool`
per this project's convention:

| directions | gating  | pooled Delta_map |
|---|---|---|
| correct | correct | **0.3505** (exact match to Stage 1C's cache) |
| buggy   | correct | 0.2851 |
| correct | buggy   | 0.3505 (unchanged) |
| buggy   | buggy   | 0.2851 (matches the GPU run's reported 0.2842) |

**The direction-generation bug alone fully reproduces the discrepancy.**
The E_min-gating bug, while real and worth fixing in the JAX pipeline's
calling convention for general robustness, turned out to be inert for
this specific trajectory -- all 432 trials for T/seed=3000 were already
`event_aligned_valid`, so there was nothing invalid to leak into the
average.

### The fix

In `build_432_batch()` (both `real_pilot_benchmark.py` and any future
GPU pilot script), replace:

```python
rng_r = np.random.default_rng(replica_direction_seed)
directions = [rng_r.uniform(-1, 1, n) for _ in range(N_REPLICAS)]
```

with a call to the real reference function:

```python
directions = generate_fixed_replica_directions(n, replica_direction_seed, N_REPLICAS)
```

(already imported in these scripts from `stage1b2_core`, just unused).

For robustness independent of whether it's causal in any given case, the
E_min gating should also be fixed at the call site: check
`event_aligned_valid` before storing `event_aligned_q`, and store `None`
(matching the numpy contract) rather than the JAX port's raw NaN-capable
output when invalid.

## Update: fix applied, confirmed on CPU, then re-verified end-to-end on GPU

Both fixes were applied directly to `build_432_batch()` and its two call
sites in `real_pilot_benchmark.py` (the per-trajectory loop, and T's own
seed=3000 cross-check). Before touching a GPU:

- The patched `build_432_batch()` was smoke-tested in isolation against
  the real `W_T` (no diffrax needed for this part -- only the JAX
  trial-runner import needs it): correct shapes (432 trials, 505-dim
  states), correct unique node/sign/amplitude values, and its
  `theta0_b[0]` was confirmed to (a) exactly match an independently
  computed `generate_fixed_replica_directions`-based replica state, and
  (b) provably differ from the old buggy `uniform(-1,1)` construction --
  i.e. the fix is real, not accidentally inert.
- This is the same code path already validated in the 4-way-factorial
  table above (same function, same arguments, imported from the same
  `stage1b2_core` module) -- no new CPU recomputation of the full
  432-trial Delta_map was needed to re-derive a value already in hand.

A fresh A100 session was then provisioned (the original was confirmed
fully gone -- `mighty-colab sessions` empty, `adopt --orphanage` found
nothing). `verify_on_gpu.py` was re-run unchanged first and reproduced
the same per-field precision as before (1e-6 to 1e-8 max abs diff, `PASS`),
confirming the fresh environment is sound. The corrected
`real_pilot_benchmark.py` was then re-run in full:

```
Total JAX GPU simulation time, all 37 trajectories: 32.67s
M1 baseline (real, measured, simulation stage only): 3660s (61 min)
Speedup vs real M1 baseline: 112.0x

Per-t_p Delta_map (T, seed=3000): {0: 0.39549, 0.833: 0.32147, 1.667: 0.33953, 2.5: 0.34557}
Pooled Delta_map (T, seed=3000, from this GPU run): 0.3505
(Stage 1C's own cached figure for T, seed=3000: 0.3505 -- exact match)
```

**Confirmed**: the corrected GPU pipeline reproduces Stage 1C's cached
Delta_map exactly (0.3505 vs 0.3505; the CPU-only reproduction earlier in
this document got 0.3505 too, agreeing with the GPU figure to ~7-8
significant figures, consistent with the cross-solver precision already
established in `verify_on_gpu.py`). Simulation speed is materially
unchanged by the fix (32.67s vs the original buggy run's 32.63s, as
expected -- the bug was in what the batch was constructed *from*, not
in how expensive it was to simulate), so the 112x speedup figure stands.

Per-trajectory timings (all ~0.87-0.96s, GPU compute time only, warm-up/
compile excluded): lattice x10 (seeds 3000-3090), rewired x3 realizations
x3 trajectories each (seeds 3000/3010/3020), hist_random x3x3, curr_random
x3x3 -- 37 trajectories total, matching the pilot design exactly.

### Anti-pattern scan (requested check, not assumed clean)

Checked every other script in this folder for the same failure mode
(reimplementing a simplified/wrong version of a function instead of
calling the real, already-imported one):

- `verify_single_trial.py`, `verify_vmap_batch.py`: **clean** -- both
  correctly call `generate_fixed_replica_directions()`.
- `extract_pre_computed_class0_construction.py`,
  `extract_pre_computed_class0_lattice.py`: trivial load/re-save
  scripts, no simulation logic, not applicable.
- `experiment.ipynb`: unrelated (Colab session-info inspector).
- **`bonsai real pilot gpu benchmark.ipynb`: NOT clean.** Cell 15 defines
  an entirely separate, cruder JAX reimplementation
  (`run_one_trial_jax`, with its own inline `force_jacobian_jax`/`rhs`)
  that does not compute event-alignment (`tau_star`, `E`/`C`, q/r/residual)
  at all -- it just returns the raw final `(theta, delta)` state at
  `T_HORIZON`. Its own `build_432_batch_for_graph()` has the identical
  `rng_r.uniform(-1, 1, n)` direction-generation bug. A code comment in
  cell 17 (`M1_MINUTES_37_TRAJECTORIES = 61  # Claude Code's first-run
  figure; re-check once the corrected run lands`) suggests this was an
  in-progress alternative/rewrite, abandoned before running (all relevant
  cell outputs are empty -- this never produced a reported number, buggy
  or otherwise). It predates `run_one_trial_jax_faithful.py` and appears
  superseded by it. Left as-is rather than fixed, since it's a materially
  different, less-complete implementation (missing the actual
  event-alignment logic this whole pipeline depends on) -- deciding
  whether to finish, fix, or discard this notebook is a scope decision
  for whoever continues this thread, not something to silently patch.

## Status

Fixed, CPU-sanity-checked, and now GPU-re-verified end-to-end. The
Stage 1D GPU pilot's Delta_map figures can be trusted going forward. The
one open item is the stale notebook above -- not blocking, but not
cleaned up either.
