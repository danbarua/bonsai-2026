# tests/

Pins specific, already-documented claims -- a `FINDINGS.md` number, a
construction's match against a historical artifact, a guard's behaviour,
a Makefile/workflow agreement. Not a unit-test mirror of `src/bonsai/`:
most modules there have no dedicated test file, and that is by design.

## Three categories, not two

The root `AGENTS.md` describes a two-tier convention (self-contained
Tier 1, artifact-gated Tier 2). That describes the historical research
tests accurately and does not cover the whole suite. What is actually
here:

1. **Always-run structural/contract tests.** Synthetic or static input:
   Stage 2A/2B module contracts, CI and Cloud Build configuration,
   provenance hooks, review tooling, Makefile agreement. The majority of
   the suite, and none of it touches `datasets/` or cached artifacts.
2. **Skip-gated historical/artifact tests.** 16 files carry
   `pytest.mark.skipif`. Every gate is a filesystem-existence,
   importability, or executable-lookup check -- **no test gates on an
   environment variable.** Gated on: gitignored `.pkl`/`.npz` artifacts,
   `datasets/kmnist/`, optional JAX/diffrax, or `bash`/`claude` on PATH.
3. **Slow / live-infrastructure tests.** Only three files carry
   `@pytest.mark.slow`: `test_stage1b_pilot.py`,
   `test_stage2b_gcs_roundtrip.py`, `test_provenance_live_registration.py`.
   Slow does not imply historical reproduction, and historical
   reproduction does not imply slow -- the two labels are independent.

## Traps

- **`test_stage2b_gcs_roundtrip.py` is marked slow but is not
  skip-gated on credentials or network.** Deselected by default, so it
  is invisible until someone runs `-m slow` -- at which point it fails
  rather than skips when infrastructure is absent. Runs with `-s`
  deliberately: its wire-level evidence is most of its value.
- **Optional-dependency handling is inconsistent.** One test in
  `test_stage2a_pipeline.py` skips on JAX importability;
  `test_stage2b_cnn.py` imports JAX at module scope and only skips its
  dataset-dependent tests.
- **No `conftest.py` exists anywhere in the repo.** Fixtures are
  file-local and duplicated by design. `tests/fixtures/` holds JSON
  data, not pytest fixtures. `tests/_makefile.py` is a shared parser
  helper, not collected.
- **`TEST_FILES` (Makefile:296) and `STAGE2B_TEST_FILES` (Makefile:311)
  are hand-maintained narrowings.** `STAGE2B_TEST_FILES` has already
  under-covered once, invisibly, because it was verified with
  `pytest tests/` -- a glob. Verifying a narrowing with the broader form
  proves nothing about the narrowing (root principle 21). Adding a test
  file means adding it to the relevant list, or asserting the list
  against a derived set.

## Before believing a green run

A test that cannot fail on the bug it names is worse than none. Mutate
the implementation and confirm the test actually breaks -- three
vacuous tests were caught this way in Stage 2B alone, each written by
the agent that also wrote the code. Catalogue and taxonomy:
`docs/VACUOUS_TESTS.md`.

New permutation schemes and chunked RNG draws get this treatment
mandatorily, not optionally: see root principles 10 and 19 for the two
failure modes that produce plausible p-values from a broken test.

## Running

`make test` (whole suite, slow deselected), `make stage2a-test`,
`make stage2b-test`. All from the repository root. `-rs` is set on the
suite targets so every skip prints its reason -- an absence has to
announce itself rather than being inferred from a changed total.
