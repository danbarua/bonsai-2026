# Bonsai — Session Changelog (July 2026)

Patch: `bonsai_predictive_hebbian_fixes.patch`
Apply from repo root with: `git apply bonsai_predictive_hebbian_fixes.patch`
(or `git apply --check ...` first to dry-run against your current `main`)

## What changed

### 1. Removed dead diagnostic computation (`models/predictive/predictive_hebbian.py`)

`_analyze_weight_spectrum`, `_analyze_fixed_points`, and `_compute_system_energy` were being computed on **every iteration** — full eigendecompositions and SVDs — and then silently discarded, because `MetricsCollector.record_iteration` only keeps scalar (`int`/`float`/`bool`/`str`) fields from `last_delta`, and these three returned dicts/lists.

- Added `collect_diagnostics: bool = False` and `diagnostics_every: int = 1` to `PredictiveHebbianOperator`. Expensive diagnostics now only run if explicitly requested.
- Cheap coherence/error metrics (needed for the convergence check) still run every step, unconditionally.
- **Result: profiled 200-iteration run went from 1.127s → 0.329s (3.4x).** Full benchmark time ratio (Predictive/Hebbian) went from **223x → 5.7x**. The original 223x number was almost entirely instrumentation overhead, not model cost.

### 2. `eigvals` → `eigvalsh`

`within_layer_weights` are symmetric by construction (Hebbian target is `cos(phase_diff)`, which is symmetric). Swapped the general-purpose eigensolver for the symmetric-optimized one. Only matters when `collect_diagnostics=True`.

### 3. Removed redundant recomputation

- `_analyze_fixed_points` now reuses `target_weights` cached by `_update_hebbian_weights` (`self._last_target_weights`) instead of recomputing `phase_diffs`/`cos(phase_diffs)` from scratch.
- `_compute_system_energy` now takes `predictions` as a parameter and reconstructs `normalized_prediction = exp(i * predictions[i])` instead of redoing the `between_layer_weights @ lower_activity` projection — the prediction was already computed once per step in `_compute_hierarchical_predictions`.

### 4. Fixed the real bug: perturbations were never read

**`PredictiveHebbianOperator` never referenced `state.perturbations` anywhere.** Confirmed by grepping the whole file — every reference was to `phases`/`frequencies`/`layer_shapes`, never `perturbations`. Since initial phases are random and independent of the input character, this meant the model's entire trajectory (limit cycle, coherence, everything) was driven purely by internal dynamics, **completely blind to which character was presented.**

Contrast: `HebbianKuramotoOperator` does use perturbations (`phase_update_flat += state.perturbations[i].flatten()`).

Fix: added a sensory (bottom-up) error term at layer 0 in `_compute_combined_updates`, gated by which pixels are stimulated:

```python
if i == 0:
    stimulus_mask = (state.perturbations[i].flatten() != 0).astype(np.float64)
    sensory_error = np.angle(np.exp(1j * (0.0 - phases_flat))) * stimulus_mask
    pc_update += self.pc_error_scaling * self.pc_precision * sensory_error
```

Stimulated pixels are pulled toward a reference phase (0); unstimulated pixels are left to the existing internal dynamics. This is one reasonable design choice, not the only one — worth revisiting if you want a richer sensory-encoding scheme.

### 5. Seeded RNG in the benchmark script

`np.random.seed(42)` added at the `if __name__ == "__main__"` entry point of `benchmark_character_processing.py`. Previously unseeded, so every run — and every "converged"/"didn't converge" result — was non-reproducible.

### 6. New file: `tests/learning/utils/readout.py`

A windowed-average readout + nearest-centroid classifier, built because **no classification metric existed anywhere in the repo** prior to this (the existing character-processing tests are abstract stubs — `raise NotImplementedError` — measuring phase distance between final states, not recognition accuracy).

- `run_and_extract_feature`: runs a model for `T` iterations, returns the circular mean of `exp(iθ)` over the last `window` iterations of the input layer, as a real-valued `[real, imag]` feature vector. Window should be ≥ the dominant oscillation period.
- `build_templates` / `classify` / `evaluate_accuracy`: nearest-centroid classification against per-character templates, at a matched compute budget across models (so comparisons don't depend on one model claiming convergence and the other not).

## Key findings from this session (not yet in code, worth remembering)

1. **The convergence check has a false-positive mode.** A slowly-varying limit cycle passes through points where consecutive-step differences dip below any fixed threshold (near its peaks/troughs, where dθ/dt ≈ 0). Verified directly: baseline params reported "converged" at iteration 1479, but the trace's range afterward (0.075–0.407) was indistinguishable from before "convergence." The 10-step stability window isn't long enough relative to the ~123-iteration oscillation period to rule this out.

2. **Predictive Hebbian's coherence trajectory is a genuine limit cycle**, not slow convergence — dominant period ≈123 iterations (measured via FFT on a 5000-step, seed=42 trace of character 'A'), sustained amplitude (std ≈0.078 over the last 4000 of 5000 steps, not decaying).

3. **Forcing true convergence is possible** (separate the Hebbian vs. predictive-coding timescales — e.g. faster Hebbian learning rate + damped/slowed PC correction) but **trades away the coherence peaks**: converged fixed points topped out around 0.14, well below the oscillating baseline's peaks of 0.37–0.41. The interesting behavior and the stable behavior appear to be in tension.

4. **Once perturbations were wired in, clean classification accuracy went from exactly chance (0.143, 1-in-7) to 0.829.** Noise and occlusion robustness both hold up, and **occlusion degrades much more gracefully than equivalent-level noise** (0.60 accuracy at 30% occlusion vs. 0.286 at 30% noise) — mechanistically sensible, since occlusion just zeroes the sensory-error term for those pixels (leaving pattern completion to internal dynamics) while noise actively injects contradictory sensory error.

5. **Hebbian-Kuramoto scores at-or-below chance (0.086–0.171) on the same windowed-average readout**, despite correctly reading perturbations. Not yet explained — flagged as an open thread, not a solved problem. Worth checking whether Hebbian's coupling drives near-global synchronization regardless of input structure, which would collapse different characters to similar points under this particular readout.

## Test suite impact (read this before assuming "all fixed")

Fixing #4 (perturbations) genuinely changes the model's behavior, not just its correctness — so it doesn't just fix things, it also **breaks a handful of pre-existing tests that were unknowingly tuned against the old, input-blind dynamics.**

- `test_predictive_hebbian_basic.py` / `test_predictive_hebbian_edge_cases.py`: 3 tests expected `weight_spectrum`/`system_energy`/`fixed_point_analysis` unconditionally in `last_delta` (previously always computed, now gated behind `collect_diagnostics`). **Fixed** — updated these 3 tests to construct the operator with `collect_diagnostics=True`. Included in the patch.
- `test_predictive_hebbian_character.py` / `test_predictive_hebbian_character_backup.py`: 4 tests each now fail, e.g. `test_occlusion_handling` asserts `pred_similarity >= hebb_similarity * 0.9` comparing final-state phase similarity to a clean reference — a threshold that was calibrated against the old behavior and no longer holds now that the model actually reacts to occlusion (observed: 0.026 vs 0.040 on random occlusion, close but now on the wrong side of the inequality). **Not fixed in this patch** — these need someone to look at the actual numbers and decide what the right threshold/assertion is post-fix, not have it silently adjusted.
- `test_predictive_hebbian_all.py` / `test_predictive_hebbian_learning.py::test_combined_hebbian_predictive`: same category, not yet investigated in detail.

Net effect on the full suite (excluding `tests/learning/`, which was already broken pre-session): went from 13 failing / 5 erroring to **17 failing / 5 erroring** — 4 more failures, all in the "tuned against old behavior" category above, none of them new bugs as far as this session's investigation went. Worth a deliberate pass rather than assuming the patch is a strict improvement on every metric.

## Follow-up fixes (second round, after migrating to uv on the Mac Studio)

- **`system_energy` KeyError** in `test_predictive_hebbian_all.py` / `test_predictive_hebbian_learning.py::test_combined_hebbian_predictive`: same root cause as the 3 tests fixed earlier — expects `weight_spectrum`/`system_energy` unconditionally. Fixed by constructing that operator with `collect_diagnostics=True`.
- **`.gitkeep` + writing plots to `./plots`** (Dan's fix, not mine): resolved the `FileNotFoundError`s from empty output directories not surviving a fresh clone. This also *revealed* two previously-hidden bugs that never got far enough to run before:
  - **`SKLEARN_AVAILABLE` NameError** in `test_predictive_hebbian_character.py`: `SKLEARN_AVAILABLE` was set as a class attribute (`self.SKLEARN_AVAILABLE`) but referenced as a bare name inside a method — missing `self.`. Fixed.
  - **TSNE perplexity bug**: the embedding visualization checked `len(characters) >= 5` before using t-SNE, but never adjusted `perplexity` (default 30) to fit — t-SNE requires `perplexity < n_samples`, so this always failed once actually reached with 5 characters. Fixed with `perplexity=min(30, len(characters) - 1)`.
  - **`np.gradient` shape bug** in `_backup.py`: one hierarchy layer has a dimension of 1, below `np.gradient`'s minimum of 2 for its default `edge_order`. Guarded with a `min(phase_data.shape) >= 2` check.
- **`test_predictive_hebbian_character_backup.py` deleted.** Confirmed via diff (86 lines out of ~1000) that it was a near-duplicate of `test_predictive_hebbian_character.py` — same 4 test methods, no unique coverage — made strictly worse by two defects: a malformed `'A'` character matrix (5 rows before `.T` instead of 12, giving it a different shape than every other character and breaking the embedding step) and a missing `SKLEARN_AVAILABLE` guard. Removed rather than fixed, since the main file already covers the same ground correctly.
- **Migration note**: moving from conda (2013 Intel MacBook Pro) to `uv` (Apple Silicon Mac Studio) surfaced a real `numpy`/`beartype` incompatibility — `numpy==2.5.1` changed `NDArray` typing internals in a way `beartype==0.22.9` can't resolve (`BeartypeDecorHintNonpepNumpyException: ... ScalarT invalid`). Fixed by pinning `numpy<2.5` in `pyproject.toml`. Nothing to do with this session's model code — a pre-existing fragility in the dependency stack that the old conda-pinned environment happened to avoid.

## Suggested next steps, roughly in priority order

- Investigate the Hebbian below-chance result (likely a readout-mismatch issue, not a model issue, but unconfirmed).
- Decide deliberately whether the limit-cycle behavior is a feature or something to damp — option 2 from this session (characterize amplitude/period/mean as the actual descriptor, rather than converging) is unexplored but promising, especially combined with the trajectory-feature readout idea (option 3 in the earlier list, not yet built).
- Fix the `maths/graphs.py` Laplacian bug (separate, pre-existing issue, unrelated to this session's work) — still blocking ~9 of the pre-existing test failures.
- Consider extending the same "does it actually read the input" audit to `models/akorn/` — given how easy this bug was to miss in a hand-rolled predictive-coding implementation, worth checking rather than assuming.
- `tests/learning/utils/readout.py` (the windowed-average classifier from earlier this session) isn't wired into any formal test yet — it was an exploration tool. Worth turning into a real test with deliberately chosen pass/fail thresholds if the classification-accuracy framing is one you want to keep.
