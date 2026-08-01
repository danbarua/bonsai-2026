This stage's original driver code (`degree_preserving_rewiring.py`,
`matched_sparsity_ablation.py`, `graph_oscillator_field.py`) was
consolidated into `src/bonsai/dynamics/` -- the versions captured in
this stage's original tarball were superseded by later, strictly-more-
complete versions of the same files from Stage 1B.2's era, confirmed via
diff at consolidation time to not have lost any code. **That diff check
establishes completeness, not behavioral equivalence, and was not
uniformly verified as such** -- corrected here after actually checking,
file by file:

- `graph_oscillator_field.py`: verified -- `tests/test_stage0_simulator_calibration.py`
  independently reproduces Stage 0's multistability, stability/
  spectral-gap, and RK45-vs-DOP853 claims directly against the
  consolidated code, including `joint_tangent_matrix_response`, the same
  function this stage's own "Reproducing these results" section below
  names as its primary tool.
- `degree_preserving_rewiring.py`: verified -- `degree_preserving_rewire`
  reproduces the historical cached `class0_constructions.pkl`'s
  `rewired` construction byte-exact at seed=1 (confirmed by a 0-9 seed
  sweep; see `tests/test_construction_driver.py`).
- `matched_sparsity_ablation.py` (**current edge-count-matched random**):
  **not verified, and now confirmed to disagree** with the historical
  `random` construction. `generate_matched_sparsity_topology` does not
  reproduce the cached `random` construction at any of 10 candidate
  seeds swept, and the mismatch is structural, not a seed problem: the
  cached artifact has roughly half the learned topology's edge count,
  with per-edge values rescaled so its mean weighted degree matches the
  learned topology's own exactly -- properties this algorithm (same edge
  count as the learned topology, its own values redistributed, no
  rescaling) cannot produce under any seed. This is a deliberately
  different, intentional design (not a dropped consolidation step -- see
  its own docstring), not a settled equivalence -- see
  `tests/test_construction_driver.py`'s Tier-2 test, which asserts this
  non-match explicitly so it stays documented rather than silently
  assumed away. A separate reconstruction of the actual historical
  algorithm -- **historical half-edge random, coupling-budget
  normalized** -- lives in `historical_matched_sparsity_random.py`,
  structurally verified (correct rescaling formula, independently-
  sampled support, values from the real topology's own weight pool) but
  with its exact historical edge-count rule and RNG seed unrecovered
  (see that module's docstring and
  `tests/test_historical_random_construction.py`). Stage 1A's own
  T-vs-controls comparison (below) includes `random` as one of the three
  matched controls -- per the reviewer's guidance, re-verification of
  that specific comparison should use the historical reconstruction as
  the primary control (structurally closer to the original experiment),
  with the current edge-count-matched algorithm retained as a separate
  robustness check, not a replacement.

See `FINDINGS.md` for this stage's actual results; the code that
produced them now lives in the shared package.
