# Lattice construction handoff

## Code to add to the project
- `lattice_construction.py` -> place at `src/bonsai/dynamics/lattice_construction.py`
- `test_lattice_construction.py` -> place at `tests/test_lattice_construction.py`
  - Before use, change the import line from `from lattice_construction import build_lattice_topology`
    to `from bonsai.dynamics.lattice_construction import build_lattice_topology`
  - Also update the two Path references at the bottom of the file
    (`RAW_TOPOLOGIES`, `HISTORICAL_ARTIFACT`) to point wherever you keep the two
    historical pkls below, if not directly alongside the test file.

## Historical data (for independent verification only -- keep local, do not commit)
- `kmnist_class_topologies_200.pkl` -- raw 784x784 per-class KMNIST topologies
  (all 10 classes), never previously handed over; only existed in Claude's sandbox.
- `stage1a_all_classes.pkl` -- the full historical {class: {constructions, n_active}}
  cache (T/rewired/random/lattice for all 10 classes) that
  `build_lattice_topology` was reverse-engineered from and verified against.

With both files present next to `test_lattice_construction.py` (or with the
paths in the test adjusted to point at wherever you place them), all 8 tests
should pass, including the Tier 2 historical byte-exact check. Without them,
Tier 2 skips and the 6 self-contained Tier 1 tests still run and pass on
their own.
