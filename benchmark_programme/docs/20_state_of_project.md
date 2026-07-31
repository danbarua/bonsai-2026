# Bonsai — State of the Project

*Compiled July 2026, from a repo last committed 2 April 2025 (~15 months dormant).*

## One-line summary

A Python (+ some hand-written C) research playground for phase-coupled oscillator networks, exploring three related but distinct model families, with a working benchmark suite that already reproduces your headline result: predictive coding on top of Kuramoto dynamics costs ~223x the compute but buys ~3x the coherence and real noise/occlusion robustness.

---

## What actually runs right now

- `tests/learning/benchmark_character_processing.py` runs clean, no fixes needed. Live output on this machine:

  | Metric | Hebbian Kuramoto | Predictive Hebbian |
  |---|---|---|
  | Converged | Yes (30 iters) | No (hit 1000-iter cap) |
  | Time | 17 ms | 3835 ms |
  | Mean coherence | 0.087 | 0.235 |
  | Final coherence | 0.088 | 0.238 |

  This is the core result the README's benchmark table refers to, and it reproduces.

- **141 tests passing, 28 failing, 5 erroring** out of ~174 collected. Not a rewrite situation — a project that was mid-refactor when you stopped.

---

## The three model families

You've actually built more than the README's two-model framing suggests. There are **three** distinct implementations under `models/`:

### 1. `models/hebbian/` — the baseline
`HebbianKuramotoOperator` (per the docstring, based on Bronski et al. 2017): standard Kuramoto phase coupling with Hebbian plasticity on the coupling weights (`w += dt * (mu * cos(Δphase) - alpha * w)`). Fast, simple, the "control" condition. A `minimalist.py` variant exists as a stripped-down single-function version — useful as a reference implementation when debugging the fancier ones.

### 2. `models/predictive/` — the novel contribution
`PredictiveHebbianOperator`: a genuinely more sophisticated design than the README lets on. It's not just "Hebbian + prediction bolted on" — it implements:
- Hierarchical, multi-layer predictive coding (Friston-style) using **complex-valued phase representations** (`exp(iθ)`) for predictions between layers, with circular error computed via `angle(exp(i(target - prediction)))` — correctly handling phase periodicity rather than naively subtracting angles.
- Both bottom-up prediction error and top-down error influencing phase updates simultaneously.
- Spectral (Xavier-style) weight initialization, spectral normalization of predictive weights via SVD, and analysis tooling: eigenvalue/condition-number tracking, "distance to theoretical fixed point," and a full system-energy decomposition (Hebbian energy + predictive-coding energy).
- This is a legitimately rich piece of code — more instrumentation and stability-engineering than a quick prototype would need. It reads like you were trying to make it *diagnosable*, not just runnable.

### 3. `models/akorn/` — the least mature, most experimental branch
Your own take on AKOrN (Artificial Kuramoto Oscillatory Neurons) — the arXiv line of work where oscillators are N-dimensional unit vectors rather than scalar phases. Two variants:
- `akorn_hebbian_kumaroto.py`: extends oscillators to `oscillator_dim`-dimensional vectors normalized onto a sphere, tracks an "equivalent Kuramoto state" for comparison against standard phase dynamics.
- `akorn_deluxe.py`: considerably more ambitious — config includes local/long-range connectivity, hub nodes, "resonance sensitivity," novelty detection, relationship learning/decay, stability windows. This looks like an attempt at a self-organizing, small-world-network topology on top of the oscillator dynamics, rather than a fixed all-to-all coupling.
- **This is where most of the current test failures live** (`test_akorne_deluxe.py`: 4 failures + 5 errors). Reads as the most recently-started, least-stabilized branch — the one you were actively extending when you stopped.

---

## Supporting infrastructure

- **`dynamics/`** — core types: `LayeredOscillatorState` (phases/frequencies/perturbations per layer) and the `StateMutation` protocol all three operators implement. Clean, generic, well-typed (uses `beartype` for runtime type checking throughout — a nice touch for research code where silent shape mismatches are the main source of bugs).
- **`maths/`** — `core.py`, `graphs.py` (graph Laplacian domain type — **currently has a bug**: normalized Laplacian construction fails its own validation, "rows must sum to zero," which cascades into several test failures), `spectral.py` (spectral decomposition / frequency-domain signal types).
- **`operators/gft_analysis.py`** — Graph Fourier Transform analysis: projects phase patterns onto the graph's eigenmodes using the Hebbian weight matrix as connectivity, tracks spectral gap as a modularity measure. This is a genuinely interesting analysis tool once the Laplacian bug is fixed — it would let you see whether the network is forming distinct "communities" of synchronized oscillators.
- **`cognitive-kernel/`** — hand-written C, not Python. `phase_locked_kernel.c` benchmarks phase-locked vs. "clone-and-shift" implementation strategies for a multi-channel oscillator field (16 layers × 256×256 grid × 4-dim state vectors per oscillator, 10,000 steps). `avx_clone_shift.c` is an AVX2-vectorized variant. You were pushing toward a fast native substrate for large-scale simulation — this is a separate performance track from the Python research code, not yet connected to it.
- **`scope/oscilliscope.py`** — a real-time Qt-based oscilloscope GUI (`pyqtgraph` + `torch`) for visualizing a "predictive coding kernel" live, with FFT analysis and trigger modes. This needs a display and GPU-capable torch; it's not something that runs headless, and it's not clear yet whether it's wired up to the Python models or the C kernel.

---

## Known issues / where things broke

1. **`maths/graphs.py` — Laplacian normalization bug.** Single root cause behind several test failures (`test_types.py`, downstream users of `GraphLaplacian`). Likely the highest-leverage first fix.
2. **`models/akorn/akorn_deluxe.py`** — the most work-in-progress piece; 9 of the 28 failing/erroring tests are here.
3. **Character-processing test duplication** — `tests/test_predictive_hebbian_character.py` and `tests/test_predictive_hebbian_character_backup.py` both exist and both fail the same three tests, suggesting a mid-refactor snapshot that never got cleaned up. Same pattern in `tests/learning/{hebbian,predictive}/test_character_processing.py` vs `tests/learning/utils/base_test.py`.
4. **Two edge-case failures in the stable baseline** (`test_hebian_kuramoto.py::TestHebbianKuramotoEdgeCases`) — worth a look since this is supposedly the most mature model.

---

## Open threads / ideas not yet converged on

- No connection yet between the C kernel (`cognitive-kernel/`) and the Python research models — two parallel implementation efforts at different levels of the stack.
- The oscilloscope GUI's relationship to the actual model code is unclear from a read-through alone; worth clarifying whether it was built against the C kernel, the Python predictive model, or is itself a semi-abandoned exploration.
- AKOrN-deluxe's topology ideas (hubs, long-range connections, novelty/relationship learning) point toward something like a small-world or scale-free connectivity structure replacing fully-connected coupling — this is the most conceptually novel unexplored direction in the repo, but also the buggiest.
- The GFT/spectral-gap analysis (`operators/gft_analysis.py`) is a real theoretical tool for asking "is the network segmenting into modules" — currently unusable because of the Laplacian bug, but potentially the most interesting *analysis* capability once fixed, independent of which model it's pointed at.

---

## Suggested entry points, if you want a place to restart

- **Lowest-effort, highest-leverage:** fix the Laplacian bug, get GFT analysis working, run it against the already-working predictive-Hebbian benchmark to see if coherence gains correspond to genuine modular structure.
- **Cleanup pass:** deduplicate the backup/duplicate test files before doing anything else, so failures are legible rather than triplicated.
- **Most novel unfinished idea:** stabilize `akorn_deluxe.py`'s topology mechanisms — this is where the project stops being "reimplementing Kuramoto variants" and starts being genuinely your own architecture.
