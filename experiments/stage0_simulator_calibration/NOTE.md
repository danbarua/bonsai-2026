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
  independently reproduces this stage's multistability, stability/
  spectral-gap, and RK45-vs-DOP853 claims directly against the
  consolidated code.
- `degree_preserving_rewiring.py`: verified -- `degree_preserving_rewire`
  reproduces the historical cached `class0_constructions.pkl`'s
  `rewired` construction byte-exact at seed=1 (confirmed by a 0-9 seed
  sweep; see `tests/test_construction_driver.py`).
- `matched_sparsity_ablation.py`: **not verified, and now confirmed to
  disagree**. `generate_matched_sparsity_topology` does not reproduce
  the cached `random` construction at any of 10 candidate seeds swept,
  and the mismatch is structural, not a seed problem: the cached
  artifact has roughly half the learned topology's edge count, with
  per-edge values rescaled so its mean weighted degree matches the
  learned topology's own exactly -- properties the current function's
  algorithm (same edge count as the learned topology, its own values
  redistributed, no rescaling) cannot produce under any seed. This is an
  open, unresolved discrepancy between whatever code originally built
  the historical `random` construction and the code now living in
  `src/bonsai/dynamics/`, not a settled equivalence -- see
  `tests/test_construction_driver.py`'s Tier-2 test, which asserts this
  non-match explicitly so it stays documented rather than silently
  assumed away.

See `FINDINGS.md` for this stage's actual results.
