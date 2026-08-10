Simplified Technical English version of `experiments/stage1a_infinitesimal_response/NOTE.md`.

This stage's original driver code was `degree_preserving_rewiring.py`,
`matched_sparsity_ablation.py`, and `graph_oscillator_field.py`. The team
consolidated this code into `src/bonsai/dynamics/`. The versions in this
stage's original tarball (packaged file) were superseded by later
versions of the same files, from Stage 1B.2's era. These later versions
are strictly more complete. At consolidation time, the team ran a diff
(a line-by-line comparison) and confirmed that no code was lost.

That diff check shows completeness. It does not show behavioral
equivalence, meaning it does not show the code still behaves the same
way. The original note did not make this distinction clearly for every
file. This note corrects that, after checking each file directly:

- `graph_oscillator_field.py`: verified. The test file
  `tests/test_stage0_simulator_calibration.py` independently reproduces
  Stage 0's claims directly against the consolidated code. These claims
  are: multistability, stability and spectral gap, and agreement between
  RK45 and DOP853. This includes `joint_tangent_matrix_response`, the
  same function that `FINDINGS.md`'s "Reproducing these results" section
  names as its main tool.
- `degree_preserving_rewiring.py`: verified. The function
  `degree_preserving_rewire` reproduces the `rewired` construction from
  the historical cached file `class0_constructions.pkl`. The
  reproduction is byte-exact at seed=1. The team confirmed this with a
  sweep of seeds 0 through 9. See `tests/test_construction_driver.py`.
- `matched_sparsity_ablation.py` (the current edge-count-matched random
  construction): **not verified, and now confirmed to disagree** with
  the historical `random` construction.

  The function `generate_matched_sparsity_topology` does not reproduce
  the cached `random` construction, at any of 10 candidate seeds tested.
  The mismatch is structural. It is not a seed problem.

  The cached artifact has roughly half the learned topology's edge
  count. Its per-edge values are rescaled so its mean weighted degree
  matches the learned topology's own mean weighted degree exactly. The
  current algorithm cannot produce these properties under any seed. The
  current algorithm uses the same edge count as the learned topology. It
  redistributes its own values. It applies no rescaling.

  This is a deliberately different, intentional design. It is not a
  dropped step from the consolidation. See the module's own docstring.
  This is not a settled equivalence between the two constructions. See
  the Tier-2 test in `tests/test_construction_driver.py`. That test
  explicitly asserts this non-match, so the difference stays documented
  rather than silently assumed away.

  A separate reconstruction of the actual historical algorithm lives in
  `historical_matched_sparsity_random.py`. This reconstruction is called
  the historical half-edge random, coupling-budget normalized
  construction. The team verified it structurally: it uses the correct
  rescaling formula, an independently-sampled support, and values from
  the real topology's own weight pool. But its exact historical
  edge-count rule and its RNG seed remain unrecovered. See that module's
  docstring and `tests/test_historical_random_construction.py`.

  Stage 1A's own T-vs-controls comparison, in `FINDINGS.md`, includes
  `random` as one of the three matched controls. Per the reviewer's
  guidance, any re-verification of that specific comparison should use
  the historical reconstruction as the primary control. This
  reconstruction is structurally closer to the original experiment. The
  team should keep the current edge-count-matched algorithm too, but
  only as a separate robustness check, not as a replacement.

See `FINDINGS.md` for this stage's actual results. The code that
produced these results now lives in the shared package,
`src/bonsai/dynamics/`.
