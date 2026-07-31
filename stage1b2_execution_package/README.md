# Stage 1B.2 Execution Package — Ready to Run on Your M1 Max

This package implements the fully locked Stage 1B.2 design from
`bonsai_stage1b_pilot_findings.md` (Option A: three degree-stratified
nodes, 432 trials, corrected permutation test). Everything here has been
smoke-tested and unit-tested before packaging — see "What's already been
verified" below.

## Setup

```bash
pip install numpy scipy
```

That's the only dependency beyond the Python standard library.

## Running the experiment

```bash
cd stage1b2_execution_package
python3 run_stage1b2.py
```

This auto-detects your core count (`multiprocessing.cpu_count()`) and
uses `cores - 1` worker processes, leaving one core free for the system.
On an M1 Max (10 cores), expect ~9 workers.

**Progress and resumability**: the script checkpoints every 10 completed
trials to `stage1b2_results.pkl` and logs to `stage1b2_progress.log`. If
interrupted (Ctrl-C, sleep, crash), just re-run the same command — it
picks up from the checkpoint automatically, skipping any trial already
recorded.

**Expected runtime**: 432 trials total. Sequential cost was ~40s/trial in
testing; with ~9 parallel workers, expect roughly 30-50 minutes wall
clock, depending on thermal throttling and how many trials land on
efficiency vs. performance cores.

## Running the analysis

Once `run_stage1b2.py` reports "COMPLETE: 432/432 trials finished":

```bash
python3 analyze_stage1b2.py
```

This computes W, B_node, B_sign, B_amplitude, balanced B, and Delta_map
for each perturbation time and pooled, then runs the **corrected**
permutation test (10,000 permutations, independent per-replica label
shuffling) to get the one-sided Monte Carlo p-value for H0: Delta_map <= 0.

Results are printed and saved to `stage1b2_final_analysis.pkl`.

## What's already been verified before you run anything

1. **The core trial computation** (`stage1b2_core.py`) was smoke-tested
   on real data — confirmed to produce sensible tau*, E, C, q, r, and
   J_tan values for an actual (node, sign, amplitude) input.
2. **The full 432-trial specification builder** was confirmed to produce
   exactly 432 trial specs (3 nodes × 2 signs × 3 amplitudes × 4
   perturbation times × 6 replicas), and the worker function runs
   end-to-end without error on real specs.
3. **The permutation scheme was unit-tested on synthetic data**, exactly
   as the design's own pre-registration required before trusting it on
   real results:
   - Identical input maps (no real signal) → Delta_map = 0.000000
     exactly, as expected.
   - Maximally separated, replica-consistent maps (strong signal) →
     Delta_map = 0.8326, clearly positive, as expected.
   - Critically: the **corrected** permutation (independent per-replica
     shuffling) produces a genuinely varying null distribution
     (std ≈ 0.0125, not degenerate), and the observed signal in the
     separated case clearly exceeds the entire permutation distribution.
     This confirms the fix for the bug caught during calibration — an
     earlier draft of the permutation scheme ("common relabeling across
     all replicas") would have left Delta_map completely unchanged under
     permutation, making the test incapable of detecting anything. That
     bug is not present in this code.

## What this package does NOT do

- It does not touch the Stage 1B pilot's own closed results — those are
  historical record, referenced only for the amplitude choices (0.025
  tangent-consistent, 0.2 intermediate, 0.8 nonlinear) and node-degree
  stratification method.
- It reports results **by perturbation time as well as pooled** — per the
  locked design's explicit requirement not to let a pooled positive
  result conceal one informative time point sitting alongside three null
  ones. Check `stage1b2_final_analysis.pkl`'s `observed_by_tp` field for
  the per-time-point breakdown, not just the pooled `p_value`.
- It does not draw any conclusion about the class-0 topology *in
  general* — this is one baseline trajectory (seed=3000) with four
  repeated states along it, not four independent trajectories. A
  positive result supports "along this prespecified trajectory, nearby
  states implement an input-sensitive, locally reproducible spatial
  mapping" — not a claim about class 0 generally, which would need
  multiple independent baselines.
- It does not yet compare against graph controls (rewired/random/
  lattice) — per the capability-first design, that question is deferred
  until structured internal transformation itself is established.

## Files

- `stage1b2_core.py` — single-trial computation (tangent+finite
  integration, E/C/q/r/J_tan diagnostics, event-aligned and fixed-time
  evaluation)
- `run_stage1b2.py` — multiprocessing driver, checkpointed, resumable
- `analyze_stage1b2.py` — W/B/Delta_map computation and corrected
  permutation test
- `class0_constructions.pkl` — pre-built T topology for KMNIST class 0
  (extracted from the project's existing validated data; no need to
  regenerate)
- `graph_oscillator_field.py`, `spectral_readout.py`,
  `stage1b_taxonomy.py`, `stage1b_tangent_departure.py` — supporting
  modules carried over from the validated project codebase
