# Bonsai: Oscillator-Network Research Project

**First thing to do in any session: read `docs/PROJECT_MEMORY.md` in
full.** It is the living, current-state record of this project --
what's established, what's open, what methodology this project holds
itself to, and what the active frontier is. Nothing below duplicates
its content; this file is a map to get you there and to the rest of the
repository, not a substitute for reading it.

## What this project is

An investigation into coupled-oscillator dynamics evolved over
topologies learned from image populations (MNIST and variants), asking
first whether the resulting representations are useful for
classification (closed, largely negative -- see below), and now whether
the dynamics themselves perform structured computation independent of
anything exported as a static feature (open, active, currently at its
strongest positive result to date).

## Directory structure

Two philosophies, deliberately split, because the two halves of this
project are different kinds of thing: one is finished science that just
needs to stay reproducible, the other is active, cumulative development
where a shared, hardened library actually earns its cost.

- **`docs/PROJECT_MEMORY.md`** -- the living project memory. Read this
  first, always. Update it when findings materially change.
- **`benchmark_programme/`** -- the CLOSED benchmark-feature programme
  (does oscillator dynamics produce useful exported classifier
  features? -- answered, largely negatively, see PROJECT_MEMORY.md Part
  1). Deliberately isolated, per-milestone snapshots, not a shared
  library: each numbered subfolder (`00_...` through `39_...`, name
  matching its findings doc) is a frozen decant of exactly the code that
  produced that milestone's result, with no cross-dependencies on other
  subfolders or on `src/bonsai`. This is intentional -- these results
  are settled, and isolated snapshots protect against exactly the
  failure mode this project has hit before: a later "improvement" to
  shared code silently changing what a historical experiment would now
  produce if rerun. Some of this code depends on the OLD `bonsai` repo's
  supporting modules (`dynamics.oscillators`, `maths.*`) which are not
  part of this project -- these are historical record, not standalone-
  runnable code. `benchmark_programme/docs/` holds the sequential
  findings documents themselves (00-39, read in order for full history;
  numbering is a best-effort chronological reconstruction from session
  history, not verified against exact per-file timestamps). Folders with
  no dedicated code tarball (pure planning/narrative docs) contain just
  a `NOTE.md` explaining that and pointing to where the real work landed.
- **`experiments/`** -- the ACTIVE dynamics-as-computation lineage
  (Stage 0 through Stage 1B.2 and beyond -- see PROJECT_MEMORY.md Part
  3). Unlike `benchmark_programme`, this is cumulative: each stage
  builds on the last, and stage-specific driver/analysis scripts here
  import shared, evolving code from `src/bonsai` rather than duplicating
  it. `stage0_simulator_calibration/` and `stage1a_infinitesimal_response/`
  contain only `FINDINGS.md` and a `NOTE.md` -- their original code was
  consolidated into `src/bonsai` after confirming (via diff, not
  assumption) that the shared version was a strict superset.
  `stage1b_pilot/` and `stage1b2_structured_transformation/` contain the
  stage-specific scripts (taxonomy, trial grid, analysis) that are
  reused nowhere else, plus each stage's own `FINDINGS.md` and a
  `results/` subfolder for cached `.pkl` artifacts (gitignored,
  regenerable).
- **`src/bonsai/`** -- the shared, actively-developed package
  underlying the `experiments/` lineage: `dynamics/` (the oscillator
  simulator and graph-construction/control utilities),
  `data/` (`mnist_loader.py`), `stats/` (the generic permutation-test
  framework and tangent-departure diagnostics). Editable-installed via
  `uv pip install -e .` so `from bonsai.dynamics.graph_oscillator_field
  import ...` resolves properly.
- **`datasets/`** -- MNIST, Fashion-MNIST, KMNIST, notMNIST, all in a
  consistent IDX format (dash-separated filenames, `.gz` where the
  original distribution was compressed, uncompressed where it wasn't --
  `src/bonsai/data/mnist_loader.py` handles both transparently). Load
  with `load_mnist(data_dir, gz=True/False)`. **Note: this PyCharm
  project does not have the cached intermediate pickle files
  (topology models, etc.) that exist in Claude's own sandbox
  environment** -- regenerating those from the datasets here, via the
  code in `experiments/` and `benchmark_programme/`, is an open,
  useful stress test of whether this repository is genuinely
  self-sufficient for a fresh agent.
- **`tools/notMNIST-to-MNIST/`** -- a third-party conversion tool
  (cloned repo, has its own `.git`) used to produce notMNIST's
  MNIST-format files. Reference only; the conversion has already been
  done and the output lives in `datasets/notmnist/`.
- **`tarballs/`** -- the original packaged deliverables (one per major
  milestone across the whole project) that `benchmark_programme/` and
  parts of `experiments/` were decanted from. Kept for provenance; not
  needed for day-to-day work once decanted. A newer sub-pattern,
  `tarballs/*_handoff/` (e.g. `lattice_construction_handoff/`,
  `random_control_handoff/`), holds small recovery/verification
  packages -- a design-doc excerpt plus historical cached data, used to
  reconstruct or verify already-committed `src/bonsai/` code against a
  known result -- distinct from the frozen milestone deliverables above.
  The large binary data in these (`.pkl`/`.npz`) is gitignored and kept
  local only; only the small provenance document gets committed. This
  data is for verification of existing, committed code, not a shortcut
  for generating new, independently-unreproducible claims -- see
  `docs/PROJECT_MEMORY.md` Part 2, principle 16.
- **`tests/`** -- pytest verification of specific, already-documented
  quantitative claims (a `FINDINGS.md` claim, a construction's byte-exact
  match against a historical cached artifact), not a general test suite
  for `src/bonsai/` in the ordinary sense. Two-tier convention,
  established starting with `test_stage0_simulator_calibration.py`: Tier
  1 is self-contained structural tests on small synthetic data, always
  run; Tier 2 is historical-artifact verification, using
  `pytest.mark.skipif` to skip cleanly when the needed local-only data
  (gitignored `.pkl`/`.npz` files, or `datasets/*/` itself) isn't
  present, rather than failing or fabricating a result. Slow, full-grid
  reproduction checks (minutes, not seconds) are tagged
  `@pytest.mark.slow` and excluded by default
  (`pytest -m "not slow"`); omit that flag for the full suite.

## Running things

- Python environment: `uv`-managed `.venv` at the project root, with
  `src/bonsai` editable-installed (`uv pip install -e .`) so
  `from bonsai.dynamics... import ...` resolves. Use `.venv/bin/python`,
  not bare `python3` -- the latter won't have scipy/numpy/tqdm/the
  bonsai package installed.
- Existing named run configurations may reference stale paths from
  before this project's restructuring -- prefer creating a fresh one
  from `filePath`+`line` against the actual script you want to run
  (e.g. `experiments/stage1b2_structured_transformation/analyze_stage1b2.py`)
  rather than trusting an old configuration name.
- Long-running computation should be launched with `waitForExit=false`
  and monitored via `get_console_output` (or, for a terminal-launched
  process, `execute_terminal_command` in a **fresh, non-reused** terminal
  window -- reusing a window that has a foreground process running in
  it will kill that process).
- This machine has 10 CPU cores. Scripts in `experiments/` already use
  `multiprocessing.Pool` with `max(1, cpu_count()-1)` workers and
  `SeedSequence.spawn` for independent per-worker random streams where
  Monte Carlo correctness depends on it -- follow this pattern for any
  new parallelized analysis rather than writing a single-threaded loop
  and parallelizing later as an afterthought (this has happened twice
  in this project already). The generic version of this machinery lives
  in `src/bonsai/stats/permutation.py` -- import and reuse it rather
  than writing new permutation-test code from scratch.

## Where the project actually is right now

Two closed programmes, one open one:

1. **Benchmark-feature programme (closed)**: does oscillator dynamics
   produce useful exported classifier features? E and R (the two
   candidate features) were both investigated exhaustively and both
   retired from the benchmark feature vector -- not because they don't
   work, but because generic, non-oscillator controls matched or beat
   them on every test that was run. Full detail in `PROJECT_MEMORY.md`
   Part 1, and `benchmark_programme/docs/22` through `38`.

2. **Dynamics-as-computation programme (open, active)**: does the
   oscillator system perform structured computation through its own
   evolving state, independent of any exported feature? Currently at
   **Level 2 of 3 established, across ten independent baseline
   trajectories** on the KMNIST class-0 learned topology (Stage 1B.2
   established it on one prespecified trajectory, following a
   three-round correction process to close a subtle information-leakage
   bug in the source-node-exclusion diagnostic; Stage 1C then confirmed
   it generalizes across nine further independent trajectories, mean
   Delta_map=0.3296, CV~5.2%, all ten significant at the Monte Carlo
   floor). Full detail in `PROJECT_MEMORY.md` Part 3, and
   `experiments/stage1b2_structured_transformation/FINDINGS.md` /
   `experiments/stage1c_trajectory_generalization/FINDINGS.md`.

   **The two remaining open questions, in priority order, are the
   natural next work** (trajectory generalization is resolved, per
   Stage 1C, and is no longer one of them): does learned topology T show
   any advantage over the matched graph controls (rewired/random/
   lattice) already built and validated for this exact purpose in the
   E/R/Stage-1A work -- Stage 1A's own re-verification (raw-scale and
   log-scale) found no such advantage survives its robustness battery
   for historical-random or current-random, and remains genuinely
   inconclusive for rewiring; this question is about T-vs-controls
   specifically within the Stage 1B.2/1C mapping design, not a repeat of
   Stage 1A -- and whether the structured mapping can be connected to
   any externally defined task.

## Methodological discipline this project holds itself to

Summarized in `PROJECT_MEMORY.md` Part 2 (17 numbered principles) --
read them before designing any new experiment. The short version: verify
before trusting, prefer narrower and more precisely scoped claims over
more impressive ones, correct multiplicity within prespecified families,
unit-test any new statistical machinery on synthetic data before running
it on real results, and treat a second independent review (human or AI)
as a first-class part of the method, not a formality.

# IntelliJ MCP Server Companion
This project is open in Pycharm IDE (IntelliJ IDEA platform). 
Call `get_mcp_companion_overview` to discover available tools and how to use them.


