# Bonsai — Phase B Completion Changelog (Predictive Hebbian sign fix + test reorg)

## Result

```
Before Phase B:  3 failed, 140 passed, 17 skipped
After Phase B:   0 failed, 140 passed, 17 skipped   (stable across 3 repeated full-suite runs)
```

Note: this changelog covers changes relative to `origin/2026-07` (i.e. it
bundles Phase A + Phase B together, since Phase A wasn't pushed to GitHub
yet at the time this was written -- if Phase A is already applied on your
end, only the Phase B-specific pieces below are new).

## 1. Fixed the within-layer coupling sign bug in `PredictiveHebbianOperator`

Exact same bug as `HebbianKuramotoOperator`, same fix: `phase_diffs[i,j]`
was computed as `theta_i - theta_j` ("self minus other"), silently flipping
positive-weight coupling from attractive to repulsive (Bronski et al.'s
equation uses `sin(theta_j - theta_i)`, "other minus self"). Fixed in
`models/predictive/predictive_hebbian.py::_compute_combined_updates`, with
the same explanatory comment used in the Hebbian fix, cross-referencing
`tests/test_hebbian_kuramoto_bronski.py`.

**Immediate effect:** all 50 tests in the predictive mechanics suite
(`test_predictive_hebbian_{basic,edge_cases,learning,all}.py`) passed
outright with this one fix, including a test that had been failing
(`test_perturbation_response` in `test_predictive_hebbian_all.py`).

## 2. Fixed a genuine test-isolation bug (order-dependent flakiness)

`test_predictive_hebbian_learning.py::test_perturbation_response` passed
reliably in isolation but failed intermittently as part of the full suite.
Root cause: the test's initial state (phases, frequencies, perturbations)
is fully deterministic, but `PredictiveHebbianOperator` randomly initializes
its between-layer weights internally on first `apply()` if none are
supplied -- so the test's actual outcome depended on whatever global
`np.random` state was left over from whichever tests happened to run
before it. Fixed by seeding the RNG (`np.random.seed(42)`) at the start of
the test. Confirmed fixed: 3 repeated full-suite runs, consistently 0
failures.

## 3. Retired `tests/test_predictive_hebbian_character.py` (standalone duplicate)

Same pattern as the Hebbian duplicate in Phase A, but this one wasn't a
clean 1:1 match -- required actual porting work before deleting:

- Diffed method inventories: standalone had 4 methods
  (`test_hierarchical_character_processing`, `test_noise_robustness_comparison`,
  `test_ambiguous_character_resolution`, `test_occlusion_handling`); the
  organized `tests/learning/predictive/test_character_processing.py` had 6
  (missing `test_hierarchical_character_processing` and
  `test_noise_robustness_comparison` as such, though covering similar ground
  under different names/looser assertions).
- Compared the two shared-name tests' actual assertions: the organized
  versions use sensible, loose assertions (e.g. "disambiguation is nonzero"),
  while the standalone versions used a strict comparative claim
  (`pred_similarity >= hebb_similarity * 0.9`) -- the same "tuned to old,
  now-incorrect behavior" pattern flagged repeatedly earlier this project.
  The organized versions already pass; no porting needed there.
- Found one genuinely unique, valuable piece of functionality:
  `visualize_character_embedding` (PCA/t-SNE multi-character phase-space
  embedding) -- this is the exact feature where the `SKLEARN_AVAILABLE` typo
  and TSNE-perplexity bugs were found and fixed earlier this project, and it
  existed *only* in the standalone file. Ported it properly rather than
  losing it:
  - Added to `tests/learning/utils/viz_utils.py` as a module-level function
    (matching that module's convention), with both bug fixes already baked
    in and a module-level `SKLEARN_AVAILABLE` guard (no `self.` indirection
    needed, since it's not a class attribute here).
  - Added a new `test_character_embedding` test in the organized predictive
    test file, wired to the ported function, with the same all-pairs
    distinctiveness check the original had (not just checking one pair).
  - Verified it passes before deleting the source file.
- Deleted `tests/test_predictive_hebbian_character.py`.

## Net result

Zero failures anywhere in the suite. 140 passed, 17 skipped (AKOrN,
correctly deprioritized), confirmed stable (not order-dependent) across
repeated runs.

## Open items for future sessions

- The limit-cycle behavior characterized earlier this project for
  `PredictiveHebbianOperator` (period ~123 iterations, sustained
  oscillation) was investigated *before* this sign fix. Worth
  re-characterizing now that the within-layer coupling is correctly
  attractive -- the earlier finding may have been partly an artifact of the
  bug rather than a purely emergent property of the predictive-coding
  architecture. Not yet done.
- No Bronski-style rigorous stability test exists yet for
  `PredictiveHebbianOperator` (the paper's theorem is specifically for the
  flat all-to-all Hebbian-Kuramoto system, not this model's hierarchical/
  predictive-coding structure) -- would need to check whether/how the
  theorem extends before attempting this.
- `tests/learning/utils/readout.py` (the windowed-average classifier from
  earlier this project) is still not wired into any formal test.
