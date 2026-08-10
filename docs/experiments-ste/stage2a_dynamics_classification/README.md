Simplified Technical English version of `experiments/stage2a_dynamics_classification/README.md`

# Stage 2A: Dynamics as Computation, Classification

This directory holds the active Stage 2A work. Stage 2A asks one question:

Does runtime graph evolution add classification value? Graph evolution runs
on top of a phase state. The phase state is already encoded by local
dynamics.

A "phase state" is a value on a torus, tracked by an oscillator (a unit that
cycles regularly), which "wraps around" like an angle. "Graph evolution"
means the oscillator network changes its state over time by following the
graph's connections.

The design is locked. See `DESIGN.md`. The confirmatory result is in. See
`FINDINGS.md`. Confirmatory means: this is the one official, pre-planned
test, not an exploratory check.

Several follow-on documents extend this work:
- `JAX_CLASSIFIER_PORT_FINDINGS.md`
- `CUML_ACCEL_FINDINGS.md`
- `COMPUTE_COST_DESIGN.md` and `COMPUTE_COST_FINDINGS.md`

Read `FINDINGS.md` first. It states what the team found. This file explains
how to run the code.

## Reading order

1. `DESIGN.md` — the locked design. Read this before any code.
2. `FINDINGS.md` — everything measured, in the order it happened. It ends
   with the locked confirmatory result and later follow-up work.
3. `JAX_CLASSIFIER_PORT_FINDINGS.md`, `CUML_ACCEL_FINDINGS.md`,
   `COMPUTE_COST_DESIGN.md`, and `COMPUTE_COST_FINDINGS.md` — separate
   follow-on threads. Each document stands on its own.

## What is in this directory

**Rule for new files**: when someone adds a new script, they must add a
one-line note about it here, in the same commit. This directory has been
checked for completeness three times already. Each check found files that
were not documented. Keeping this section current is cheaper than finding
the gap again later.

The list below is organized by thread of work, not by file-name pattern.
A simple pattern search (like all files starting with `diagnose_`) missed
two files before. So this list names every file directly.

1. **Core pipeline code** (other scripts import this; it does not run on
   its own): `stage2a_core.py` (encodes, evolves, and extracts gauge
   features), `stage2a_classifier.py` (the locked cross-validation,
   standardization, and classifier steps), `stage2a_stats.py` (the
   confirmatory statistics: paired bootstrap, McNemar's test, and Holm
   correction), `stage2a_pipeline.py` and `stage2a_pipeline_jax.py` (run
   the pipeline on CPU and on GPU in batches), `stage2a_topologies.py`
   (builds the 4 graphs used in the confirmatory result), `stage2a_paths.py`
   (finds the correct file paths for scratch data), `evolve_on_graph_jax.py`
   (the verified GPU evolution code).
2. **Feasibility and confirmatory driver scripts** (these produced the
   locked result): `run_feasibility_stage1.py` and
   `run_feasibility_stage2.py` (small-scale checks that the pipeline
   works correctly), `run_feasibility_stage3_encode.py` and
   `stage3_gpu_evolve.py` (encode and evolve all 60,000 training images
   on GPU), `run_official_test_encode.py` and `stage4_gpu_evolve.py` (the
   one and only encode and evolve step for the 10,000 official test
   images), `run_confirmatory_evaluation.py` (runs the locked confirmatory
   analysis), `run_posthoc_graph_pairwise.py` (an after-the-fact,
   Holm-corrected comparison between graphs), `generate_artifact_manifest.py`
   (see below).
3. **Checks that support the confirmatory result**:
   `diagnose_stage2_convergence.py`,
   `diagnose_stage2_convergence_hypotheses.py`,
   `diagnose_rewired_currrandom_synchronization.py`,
   `diagnose_topology_synchronizability.py`, `analyze_stage3_results.py`
   and `analyze_stage3_results_jax.py` (combine data and run the
   classifier; one is plain Python, one uses JAX to speed up part of the
   work), `verify_analyze_stage3_results_jax.py` (checks the JAX version
   against the plain version), `verify_stage2a_pipeline_equivalence.py`.
4. **JAX classifier port thread** (see `JAX_CLASSIFIER_PORT_FINDINGS.md`):
   `stage2a_classifier_jax.py`, `verify_stage2a_classifier_jax.py`,
   `diagnose_classifier_jax_grad_norm_calibration.py`. This work is a
   side investigation. No reported result uses it. See that document for
   why.
5. **cuML cross-check thread** (see `CUML_ACCEL_FINDINGS.md`): this
   thread has no dedicated driver script for the main cross-validation
   check. It reuses `analyze_stage3_results.py` and
   `run_confirmatory_evaluation.py`, run under `cuml.accel` (NVIDIA's
   tool that speeds up scikit-learn code without changing it), using
   this project's own unchanged code. One exception exists with its own
   driver script: the class-0 support audit's GPU classifier runs. See
   thread 7 below.
6. **Compute-cost thread** (see `COMPUTE_COST_DESIGN.md` and
   `COMPUTE_COST_FINDINGS.md`): `measure_oscillator_cpu_latency.py`,
   `prep_oscillator_latency_gpu_inputs.py` and
   `measure_oscillator_gpu_latency.py` (the matching GPU-session
   version), `measure_mlp_cpu_latency.py`, `build_cost_model.py` (builds
   the cost model and its plot). ("Latency" here means the time to
   process one item, not a batch.)
7. **Class-0 support audit thread** (folded into `FINDINGS.md`; no
   separate document): `run_class0_support_audit.py` (part 1: checks how
   much ink is kept, local and free to run), `run_class0_support_audit_classify.py`
   (part 2: fits two baseline classifiers; it has an optional `--cuml`
   flag, but that flag combined with a local
   `stage2a_paths.scratch_root()` read has only run on local CPU
   scikit-learn so far), `class0_support_audit_classify_gpu.py` (part
   2's actually-tested remote GPU driver script — see
   `CUML_ACCEL_FINDINGS.md`'s "Code" section for why this is a separate
   file rather than the same script with a flag).
8. **Topology and correlation plotting thread**: `visualize_topologies.py`,
   `visualize_normalized.py`, `plot_ink_correlation.py` and
   `diagnose_ink_correlation.py` (the table behind the plot),
   `plot_decomposed_correlation.py` and `diagnose_decomposed_correlation.py`
   (same pattern).

**`generate_artifact_manifest.py`**: produces `results/ARTIFACT_MANIFEST.json`.
This file records SHA256 hashes, data sizes, image order, graph hashes,
and the `C` values (regularization strengths) the confirmatory run
actually used. Run this after you reproduce the pipeline. It lets you
check your own output matches the numbers behind the reported result. It
only covers the locked confirmatory pipeline's own files (the stage3 and
stage4 saved data). It does not cover the follow-on threads' own result
files.

**`results/`**: a cache folder. Git ignores it. It holds large `.pkl` and
`.npy` files that you can regenerate. It also holds a small number of
files that are committed to the project on purpose: plots, and
`ARTIFACT_MANIFEST.json`.

**`scratch/`**: a local, git-ignored folder for large intermediate files
from encoding and GPU evolution. `stage2a_paths.py` sets this as the
default location. You can override it with `STAGE2A_SCRATCH_ROOT`.

## How to run things

Every Stage 2A step — local CPU steps, GPU evolution, checking output
files, and the test suite — runs through the root-level `Makefile`. The
Makefile is the single source of truth for the actual commands. This
section only points you to the right targets; it does not repeat them.

**Local versus remote, in short**: every script in this directory either
runs locally (CPU, free, no `mighty-colab` session needed) or runs on a
remote GPU session. The Makefile's own `##@`-grouped sections show which
is which. Run `make stage2a-help` to list every target, grouped exactly
that way: local feasibility, data-prep, and analysis targets come first;
GPU-evolution and GPU-classify targets come after, each marked "bills
while running." Do not guess this split from file names alone.

```bash
make stage2a-help   # from the repository root — lists every target, grouped by pipeline stage
```

All targets assume you have the project's `uv`-managed virtual
environment and `datasets/kmnist/` on your machine. They find their own
paths using `git rev-parse --show-toplevel` (see the Makefile's header
comment). Run everything from the repository root. You do not need `cd`.

**Typical order**, matching `stage2a-help`'s own grouping:

1. `stage2a-feasibility1` and `stage2a-feasibility2` — optional checks
   that the pipeline works correctly. Not confirmatory. See
   `FINDINGS.md`'s "Feasibility Stage 1" and "Feasibility Stage 2"
   sections.
2. `stage2a-prepare-train` and `stage2a-prepare-test` — local, CPU steps.
   They encode the official KMNIST splits. **`stage2a-prepare-test` is
   the ONLY place this project touches the test-set images or labels for
   the locked confirmatory analysis. Do not run it just to try it out.**
3. `stage2a-evolve-train-gpu` and `stage2a-evolve-test-gpu` — GPU
   evolution steps, run through `mighty-colab`. **These bill while
   running.** Read the "GPU evolution" section below before you run
   these.
4. `stage2a-analyze` — selects the classifier model. **Takes about 4
   hours on CPU scikit-learn.** Read Result 3 in `FINDINGS.md` before you
   run this unattended.
5. `stage2a-confirm` — the locked confirmatory evaluation.
6. `stage2a-posthoc` — the after-the-fact, graph-to-graph comparison.
   Takes seconds. It reuses saved per-image losses. No new GPU time.
7. `stage2a-manifest` and `stage2a-verify` — regenerate the artifact
   manifest and compare it against the committed one.
8. `stage2a-class0-audit` and `stage2a-class0-classify` — the class-0
   support audit. This ran after the main result, at the request of
   external review. See `FINDINGS.md`'s "class-0 confound" sections.
   `stage2a-class0-classify-gpu` runs part 2 under `cuml.accel` on a
   `mighty-colab` GPU session instead. **This bills while running.** See
   the Makefile comment above it, and `CUML_ACCEL_FINDINGS.md`, for why
   this uses a separate driver script (`class0_support_audit_classify_gpu.py`)
   instead of the local script with a flag.

Override the scratch location if you do not want the default
(`experiments/stage2a_dynamics_classification/scratch/`):

```bash
export STAGE2A_SCRATCH_ROOT=/path/to/your/scratch/dir
```

**Follow-on threads** (the JAX classifier port, the NVIDIA cuML
cross-check, and compute-cost accounting) mostly do not appear in the
Makefile. Each thread describes its own reproduction steps in its own
findings document: `JAX_CLASSIFIER_PORT_FINDINGS.md`'s "Files" section,
`CUML_ACCEL_FINDINGS.md`'s "Code" section, and `COMPUTE_COST_FINDINGS.md`'s
"Code" section. One exception: the class-0 support audit's GPU classifier
runs are wired into `make stage2a-class0-classify-gpu` (see "What is in
this directory," thread 7 above, and `CUML_ACCEL_FINDINGS.md`'s "Code"
section for why that one has a dedicated driver when the rest of the
cuML thread does not).

## GPU evolution

The GPU-evolution step uses [`mighty-colab`](https://pypi.org/project/mighty-colab/)
(a command-line tool for Google Colab) to start an A100 GPU session,
upload input data, run the evolution step remotely, and download the
resulting `theta_T` states (`theta_T` is the phase state after graph
evolution). `stage3_gpu_evolve.py` and `stage4_gpu_evolve.py` are the
exact scripts that run on the remote GPU. You cannot run them locally as
they are. They read and write files under `/content/...`, which is the
remote session's own file layout. `mighty-colab` is a pinned dependency
(see `pyproject.toml`'s `[dependency-groups].gpu`, the official PyPI
release). The command `uv run --group gpu`, which the Makefile calls
internally, installs it into this project's own `.venv` the first time
you use it. A clean checkout of the project needs no separate install
step.

**Full sequence** (training set; swap `-train-`/`stage3` for
`-test-`/`stage4` to run the official test set instead):
```bash
make stage2a-prepare-train      # local, CPU
make stage2a-evolve-train-gpu   # bills while running
make stage2a-verify             # regenerate + diff the artifact manifest
```
`make stage2a-verify` checks the artifact, graph, and array SHA256
hashes. A clean diff on those fields confirms your regenerated output is
byte-for-byte the same as the original. **The `environment.git_commit_sha`
field will always differ once you are on a later commit than the one
that made the committed manifest. A difference in that field alone is
not a reproduction failure.** Only the hash fields matter. (`make
stage2a-verify`'s own output repeats this note.)

**Upload size limit**: a single large data file (the original 250MB
Stage 3 `theta0` and topologies package) hit the upload service's size
limit. This is why `stage2a-prepare-train` splits its output into 12
files (`theta0_chunk_00.npy` through `theta0_chunk_11.npy`, about 20MB
each). `stage2a-prepare-test` does not need to split its output, because
Stage 4's combined package is only about 48MB total, under the limit.
Files under about 50MB upload fine as one transfer.

## Public artifact cache (GCS)

The large cached files behind the confirmatory result and its follow-on
threads (encoded features, GPU-evolved states, the 4 topology adjacency
matrices, and the confirmatory results file) are also stored in a
public, read-only Google Cloud Storage bucket:

```
gs://bonsai-2026-stage2a-cache/
├── stage3_train/          # 60,000-image official training set artifacts
├── stage4_test/           # 10,000-image official test set artifacts
├── results/               # small results pkls (go/no-go, classifier conditions, confirmatory results)
└── class0_support_audit/  # the 6 .npy inputs to class0_support_audit_classify_gpu.py (raw505/encoded784 x train/test, labels)
```

You need no authentication for HTTPS access. This is useful for pulling
data directly into a fresh Colab session — a transfer between two cloud
services is much faster than uploading from a local machine over a home
internet connection.

```python
import urllib.request
urllib.request.urlretrieve(
    "https://storage.googleapis.com/bonsai-2026-stage2a-cache/stage3_train/stage3_gpu_results.pkl",
    "/content/stage3_gpu_results.pkl",
)
```

Or use `gcloud` or `gsutil` locally, run from the repository root
(`$STAGE2A_DIR` matches the Makefile's own variable,
`experiments/stage2a_dynamics_classification` by default):
```bash
gcloud storage cp gs://bonsai-2026-stage2a-cache/stage3_train/*.pkl "$STAGE2A_DIR/scratch/stage3_train/"
```

**Why this bucket exists**: re-uploading these files from a local machine
to a fresh Colab session, every time a new GPU experiment was needed,
became real, repeated work. Each upload of 1-2GB took tens of minutes
over a typical home connection, and had to be split into dozens of small
transfers to stay under the upload size limit. Pulling from GCS instead
is one fast download, entirely on Google's own network. The bucket is
public because this repository is public, and the source data (KMNIST)
is a public academic dataset. Nothing in the cached files reveals
anything beyond what the public code and public data already show. The
confirmatory findings in `FINDINGS.md` already state the actual
scientific claims in full. If you plan to reuse this same public-bucket
pattern for a dataset or result where that reasoning does not hold, see
the project's own reasoning about this choice — it is not written down
elsewhere, only discussed in the conversation thread that made this
decision.

**Checking your own regenerated files match**: run `make stage2a-verify`
from the repository root. It regenerates the manifest and compares it
against the committed one. As above, ignore `environment.git_commit_sha`
in that comparison. Compare the artifact, graph, and array SHA256 fields
specifically.

## Two kinds of reproducibility: replaying artifacts, and rebuilding from raw data

**Amended by external review, then fully fixed, not just documented.**
These are two different claims about reproducibility. Both are now
supported.

- **Replaying the cached artifacts.** Given the cached intermediate
  files (from the GCS bucket above, or from your own earlier local run),
  every later step — classifier cross-validation, the confirmatory
  bootstrap and McNemar test, the after-the-fact pairwise comparison, and
  the artifact manifest — reruns from those files and reproduces the
  reported numbers. `tests/test_stage2a_stats.py`'s
  `test_frozen_primary_effect_matches_findings_md` test (Tier 2) checks
  this directly. Most of the "How to run things" section above uses this
  path, once the encoding and GPU-evolve files already exist.

- **Rebuilding fully from raw data** (starting from nothing but the
  public KMNIST dataset and the public code). Before a recent fix,
  `stage2a_topologies.build_all_topologies()` depended on a file from an
  earlier stage of the project,
  `experiments/stage1b2_structured_transformation/results/class0_constructions.pkl`.
  Git ignores this file. It is not committed, and it is not part of the
  Stage 2A GCS bucket above (that bucket mirrors Stage 2A's own files,
  not Stage 1B2's). The old code depended on it in two places: it read
  the `lattice` graph directly from that cache instead of building it
  fresh, and it needed the cache present just to check `T` (the learned
  topology) against it, even though building `T` itself never needed the
  cache.

  **Both problems are now fixed.** `T` now calls
  `build_and_verify_T(require_historical_verification=False)`. This
  still checks the result against the cache if the cache happens to be
  present, but no longer stops with an error if it is missing. `lattice`
  is now rebuilt directly, using `build_lattice_topology`
  (`src/bonsai/dynamics/lattice_construction.py`). The `rewired` and
  `curr_random` graphs never depended on the cache. They were always
  built fresh from `T`.

  **Checked directly, not assumed**: with `class0_constructions.pkl`
  present locally, the new from-scratch method was compared, value by
  value, against what the old cache-reading method returned. `T` and
  `lattice` match to within float64 machine precision (the largest
  difference is `2.22e-16`). This is the same precision level this
  project's other from-scratch rebuilds already reach. It is not
  bit-for-bit identical, because one side is computed fresh now and the
  other side was computed once, in the past, on different hardware and
  library versions. The team also confirmed the historical file is now
  genuinely optional, not just optional in theory: they pointed the
  cache path at a file that does not exist, and `build_all_topologies()`
  still succeeded, producing the same `T` and `lattice` (to that same
  precision). Both checks are now permanent tests:
  `tests/test_stage2a_topologies.py::test_from_scratch_reconstruction_matches_cache_backed_values`
  (Tier 2, skipped cleanly if the cache is not present locally to compare
  against — but the other tests in that file, and the pipeline itself,
  no longer need it).

## Testing

```bash
make stage2a-test
```

This project uses a two-tier test convention (see the root `CLAUDE.md`
file). Tier 1 tests run on self-contained, made-up data. They always
run. Tier 2 tests are skipped cleanly, not failed, when the local-only
cached files they need are not present. `test_stage2a_stats.py`'s
`test_frozen_primary_effect_matches_findings_md` is a Tier 2 test. It
recomputes the primary bootstrap result from
`results/stage4_confirmatory_results.pkl`, if that file is present, and
checks it still matches the numbers stated in `FINDINGS.md`. This test
catches a future code change that might silently alter the statistic.
`test_stage2a_classifier.py`, `test_stage2a_pipeline.py`, and
`test_stage2a_paths.py` are Tier 1 only (made-up data, no cached-file
dependency). `test_stage2a_topologies.py` is Tier 2 only.
`build_all_topologies()` needs real KMNIST data to rebuild from. There
is no meaningful made-up-data version of that check. This test no longer
needs the `class0_constructions.pkl` historical file for its main checks
(see "Two kinds of reproducibility" above). One test inside the file
still uses that file, when present, as a direct regression check: it
confirms the from-scratch rebuild matches what the old, artifact-backed
path used to return. That one test skips independently if the file is
absent.

This test suite deliberately does not cover the class-0-audit,
compute-cost, and cuML-accel threads. These are lower priority for a
research codebase. This is a scope decision, not an oversight.
