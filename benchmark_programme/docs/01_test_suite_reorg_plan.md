# Bonsai Test Suite — Audit & Reorganization Plan

## Current inventory

**Foundation (models/hebbian):**
- `tests/test_hebian_kuramoto.py` (52K) — original ad hoc test suite for
  `HebbianKuramotoOperator`. Coherence thresholds, step counts, no direct
  connection to the paper. 2 pre-existing failures
  (`test_fixed_point_stability_with_small_frequency_differences` has its own
  separate bug — a fixed-point weight formula missing the `mu` factor;
  `test_perturbation_response` is a boundary-exact assertion artifact).
- `tests/test_hebbian_kuramoto_bronski.py` (16K, new this session) — verifies
  the model directly against Bronski et al.'s equations and stability
  theorem. 8/8 passing. Confirmed to actually catch the coupling-sign bug
  (4/8 fail if the bug is reintroduced).

**Character processing — Hebbian (duplicated):**
- `tests/test_character_processing.py` (32K) — standalone
  `TestHebbianKuramotoCharacterProcessing(unittest.TestCase)`, own inline
  helpers, no shared base class.
- `tests/learning/hebbian/test_character_processing.py` (14K) — same class
  name, but inherits `CharacterProcessingBaseTest` and uses the shared
  `tests/learning/utils/` infrastructure (`character_utils.py`,
  `viz_utils.py`, `metrics_utils.py`).
- These have drifted: 2-3 tests differ in pass/fail state between them after
  the sign fix (`test_noisy_character` fails in the top-level file but not
  the `learning/` one).

**Character processing — Predictive (duplicated, same pattern):**
- `tests/test_predictive_hebbian_character.py` (52K) — standalone, own inline
  `visualize_*` methods and helpers (this is the file where the
  `SKLEARN_AVAILABLE` typo and TSNE perplexity bugs were found earlier this
  project — bugs specific to *this* copy, not the `learning/` one).
- `tests/learning/predictive/test_character_processing.py` (19K) — shared
  base class + utils, same pattern as the Hebbian side.
- `tests/test_predictive_hebbian_character_backup.py` — already deleted
  earlier this project (confirmed near-duplicate with a malformed character
  matrix and missing guard).

**Predictive Hebbian mechanics (not duplicated — only exists flat):**
- `tests/test_predictive_hebbian_basic.py`,
  `_edge_cases.py`, `_learning.py`, `_all.py` (loads the first three) — ad
  hoc mechanics tests, analogous in spirit to `test_hebian_kuramoto.py` but
  for the predictive model. No Bronski-style equations/stability
  verification exists for this model yet, since it still has the unfixed
  sign bug (next phase, per agreed plan).

**Shared infrastructure:**
- `tests/learning/utils/base_test.py` — `CharacterProcessingBaseTest`, a
  **design flaw**: it inherits `unittest.TestCase` directly and defines its
  shared test methods (`test_single_character_processing` etc.) as
  `raise NotImplementedError(...)` stubs meant to be overridden. Because it
  IS a `TestCase`, pytest collects and runs it directly too, in addition to
  every subclass — those 4 stub methods always fail, in every file that
  imports it (currently: 3 files x 4 stubs = 12 permanent failures that
  aren't testing anything, just artifacts of the base-class design).
- `tests/learning/utils/character_utils.py`, `viz_utils.py`,
  `metrics_utils.py` — genuinely shared, reusable, no issues found.
- `tests/learning/utils/readout.py` — the windowed-average classifier built
  earlier this project, not yet wired into any formal test.

**Deprioritized:**
- `tests/test_akorne_deluxe.py` — now correctly `@unittest.skip("...")`'d,
  17 tests visibly skipped (was previously silently disappearing tests
  entirely due to a bare-decorator bug, now fixed).

**Unrelated / already correct:**
- `tests/test_types.py` — cleaned up this project (removed shadow duplicate
  domain types, now tests the real `maths.core`/`maths.graphs` module code).
  1 bug found and fixed (Laplacian normalization check).

## Problems, in priority order

1. **The base-class collection bug is free, permanent noise** — 12 failures
   that will exist forever regardless of model correctness, actively
   obscuring genuine signal in the test summary. Highest-value, lowest-risk
   fix.
2. **Duplicate flat-vs-organized files for both Hebbian and Predictive
   character processing** — real maintenance burden, and a demonstrated
   history of bugs being fixed in one copy and not the other (this project
   found at least 3 examples: `SKLEARN_AVAILABLE`, TSNE perplexity, and the
   already-deleted `_backup.py`'s malformed matrix). The `tests/learning/`
   structure is clearly the intended target design (shared utilities, base
   class, organized by model). The flat top-level files look like the
   pre-refactor originals that were never removed.
3. **`test_hebian_kuramoto.py`'s ad hoc tests now partially overlap/conflict
   with `test_hebbian_kuramoto_bronski.py`'s rigorous ones** — some old tests
   are checking things the new suite checks more precisely and correctly;
   others check genuinely different things (edge cases: NaN handling,
   overflow, perturbation response, clustering) that are still valuable and
   don't have Bronski-style equivalents.
4. **Predictive Hebbian's test suite is entirely provisional** — it still
   has the unfixed sign bug, so any reorganization there is premature until
   that's fixed and reverified, per the agreed order (foundation first).

## Proposed plan

### Phase A — Hebbian model (do now, foundation is verified)

1. **Fix `CharacterProcessingBaseTest`** so it isn't directly collectible:
   convert it to a plain mixin (doesn't inherit `unittest.TestCase`) that
   concrete test classes multiple-inherit alongside `unittest.TestCase`, or
   set `__test__ = False` on the base class (pytest respects this
   convention). Removes 12 permanent no-signal failures in one small change.
2. **Retire `tests/test_character_processing.py`** (the standalone flat
   file) in favor of `tests/learning/hebbian/test_character_processing.py`
   (the organized one) — after first diffing them carefully for any
   genuinely unique test cases worth porting over before deletion (learned
   from the `_backup.py` precedent: check before deleting, don't assume).
3. **Triage `test_hebian_kuramoto.py` test-by-test** against
   `test_hebbian_kuramoto_bronski.py`: mark for removal anything that's now
   strictly subsumed by a more rigorous equivalent; keep and possibly
   rename/reorganize anything genuinely distinct (edge cases, numerical
   robustness, the `test_synchronization_clusters` test which is a great,
   distinct, real-world-flavored complement to the analytical Bronski
   tests). Fix the 2 remaining pre-existing failures as part of this pass
   (the `mu`-formula bug and the boundary-exact assertion) rather than
   deleting them, since their underlying intent is sound.
4. **Fix the 2 new post-sign-fix failures**
   (`test_frequency_vs_perturbation`, `test_processing_stability`) as part
   of the same triage — both look like they need their assumptions
   corrected (uniform frequency shouldn't disrupt synchronized structure;
   the stability threshold is marginally miscalibrated) rather than being
   deleted outright.

### Phase B — Predictive Hebbian model (next, after Phase A settles)

1. Fix the within-layer coupling sign bug (same fix pattern as Hebbian).
2. Re-run and re-characterize the limit-cycle behavior — determine whether
   it's a genuine emergent property or was partly an artifact of the bug.
3. Write the Predictive-Hebbian equivalent of
   `test_hebbian_kuramoto_bronski.py`, if the paper's stability theory
   extends to the hierarchical/predictive-coding case (needs checking — the
   paper's theorem is specifically for the flat all-to-all Hebbian-Kuramoto
   system, not this model's hierarchy).
4. Apply the same duplicate-file cleanup
   (`test_predictive_hebbian_character.py` vs
   `tests/learning/predictive/test_character_processing.py`).
5. Decide whether to formalize `tests/learning/utils/readout.py` into a real
   test with deliberate thresholds.

### Untouched / deferred

- `test_akorne_deluxe.py` stays skipped as-is.
- `tests/test_types.py`, `maths/graphs.py` — already correct, no further
  action needed.

## Sequencing recommendation

Do Phase A steps 1-2 first (quick, safe, high-signal cleanup with no model
risk), then step 3-4 together as a single focused pass over
`test_hebian_kuramoto.py`, since triaging and fixing naturally happen in the
same sweep once you're looking at each test's actual intent.
