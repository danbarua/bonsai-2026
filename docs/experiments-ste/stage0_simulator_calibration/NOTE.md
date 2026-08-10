Simplified Technical English version of `experiments/stage0_simulator_calibration/NOTE.md`.

This stage's original driver code lived in three files: `degree_preserving_rewiring.py`, `matched_sparsity_ablation.py`, and `graph_oscillator_field.py`.
The team moved this code into `src/bonsai/dynamics/`.
The versions in this stage's original tarball were older. Later versions of the same files, from Stage 1B.2's era, were strictly more complete.
At the time of the move, a file comparison (a diff) confirmed the move did not lose any code.

**That diff check confirms completeness. It does not confirm behavioral equivalence.** The team did not check behavioral equivalence for all three files at that time. The team corrected this by checking each file directly, one by one:

- `graph_oscillator_field.py`: verified. `tests/test_stage0_simulator_calibration.py` independently reproduces this stage's multistability claim, its stability and spectral-gap claim, and its RK45-vs-DOP853 claim, directly against the moved code.
- `degree_preserving_rewiring.py`: verified. The `degree_preserving_rewire` function reproduces the historical cached `rewired` construction in `class0_constructions.pkl`, byte-exact (matching exactly, down to the last bit), at seed=1. A sweep of seeds 0 through 9 confirmed this. See `tests/test_construction_driver.py`.
- `matched_sparsity_ablation.py` (the current edge-count-matched random construction): **not verified, and now confirmed to disagree** with the historical `random` construction.
  `generate_matched_sparsity_topology` does not reproduce the cached `random` construction, at any of 10 candidate seeds swept.
  The mismatch is structural. It is not a seed problem.
  The cached artifact has roughly half the learned topology's edge count. Its per-edge values are rescaled so its mean weighted degree matches the learned topology's own mean weighted degree exactly.
  `generate_matched_sparsity_topology` cannot produce these properties under any seed. It uses the same edge count as the learned topology, redistributes the learned topology's own values, and applies no rescaling.
  This is a deliberately different, intentional design. It is not a dropped step from the code move. See the function's own docstring.
  This is not a settled equivalence between the two constructions.
  `tests/test_construction_driver.py` has a Tier-2 test that checks this non-match directly. This keeps the mismatch documented, instead of silently assumed away.
  A separate reconstruction of the actual historical algorithm exists in `historical_matched_sparsity_random.py`. The team calls this reconstruction the historical half-edge random construction, coupling-budget normalized.
  This reconstruction is structurally verified: it uses the correct rescaling formula, an independently sampled support, and values drawn from the real topology's own weight pool.
  Its exact historical edge-count rule and its RNG seed remain unrecovered. See that module's docstring and `tests/test_historical_random_construction.py`.

See `FINDINGS.md` for this stage's actual results.
