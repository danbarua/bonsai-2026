# Stage 2A: Dynamics-as-Computation, Classification

This directory holds the active Stage 2A investigation: does runtime
graph evolution, on top of an already-dynamically-encoded phase state,
add classification value a linear readout can use? The design is
locked (`DESIGN.md`), the confirmatory result is in (`FINDINGS.md`),
and several follow-on threads (`JAX_CLASSIFIER_PORT_FINDINGS.md`,
`CUML_ACCEL_FINDINGS.md`, `COMPUTE_COST_DESIGN.md` /
`COMPUTE_COST_FINDINGS.md`) extend from it. **Read `FINDINGS.md` first**
for what's actually been found; this file is about how to run things.

## Reading order

1. `DESIGN.md` -- the locked design, read this before any code.
2. `FINDINGS.md` -- everything measured, in the order it happened,
   ending with the locked confirmatory result and its post hoc
   follow-ups (test-set reuse disclosures, the graph-to-graph pairwise
   comparison).
3. `JAX_CLASSIFIER_PORT_FINDINGS.md`, `CUML_ACCEL_FINDINGS.md`,
   `COMPUTE_COST_DESIGN.md` / `COMPUTE_COST_FINDINGS.md` -- standalone
   follow-on threads, each self-contained.

## Directory contents

**Convention, going forward**: any new script gets a one-line mention
here, in the same commit that creates it. This directory has already
been through three reproducibility/organization passes, each catching
real files that had gone undocumented since the last one -- cheaper to
keep this section current as files are added than to keep
re-discovering the gap later.

Organized by thread, not by glob (`diagnose_*.py`/`verify_*.py`/
`analyze_*.py` alone previously let two files go unnoticed as orphans --
enumerated explicitly here instead):

1. **Core pipeline code** (imported by other scripts, not run
   standalone): `stage2a_core.py` (encode/evolve/gauge features),
   `stage2a_classifier.py` (the locked CV/standardization/classifier
   procedure), `stage2a_stats.py` (the confirmatory statistical
   machinery -- paired bootstrap, McNemar, Holm correction),
   `stage2a_pipeline.py` / `stage2a_pipeline_jax.py` (CPU and
   GPU-batched pipeline orchestration), `stage2a_topologies.py` (the 4
   confirmatory-expansion graphs), `stage2a_paths.py` (scratch-artifact
   path resolution), `evolve_on_graph_jax.py` (the verified GPU
   evolution kernel).
2. **Feasibility/confirmatory drivers** (the locked-result lineage):
   `run_feasibility_stage1.py` / `run_feasibility_stage2.py`
   (small-scale mechanical validation), `run_feasibility_stage3_encode.py`
   + `stage3_gpu_evolve.py` (full 60,000-image training-set encode + GPU
   evolve), `run_official_test_encode.py` + `stage4_gpu_evolve.py` (the
   official 10,000-image test set's one-and-only encode + GPU evolve),
   `run_confirmatory_evaluation.py` (the locked confirmatory analysis
   itself), `run_posthoc_graph_pairwise.py` (the post hoc, Holm-corrected
   graph-to-graph comparison), `generate_artifact_manifest.py` (below).
3. **Diagnostics feeding the confirmatory thread**:
   `diagnose_stage2_convergence.py`, `diagnose_stage2_convergence_hypotheses.py`,
   `diagnose_rewired_currrandom_synchronization.py`,
   `diagnose_topology_synchronizability.py`, `analyze_stage3_results.py`
   + `analyze_stage3_results_jax.py` (the phase-2 combine-and-classify
   step, numpy and JAX-batched-postprocessing variants) +
   `verify_analyze_stage3_results_jax.py` (verifies the latter against
   the former), `verify_stage2a_pipeline_equivalence.py`.
4. **JAX classifier port thread** (`JAX_CLASSIFIER_PORT_FINDINGS.md`):
   `stage2a_classifier_jax.py`, `verify_stage2a_classifier_jax.py`,
   `diagnose_classifier_jax_grad_norm_calibration.py` -- investigative,
   not used for any reported result; see that doc for why.
5. **cuML cross-check thread** (`CUML_ACCEL_FINDINGS.md`): no dedicated
   driver script -- reuses `analyze_stage3_results.py`,
   `run_confirmatory_evaluation.py`, and
   `run_class0_support_audit_classify.py` under `cuml.accel` via a
   `--cuml` flag / `cuml.accel.install()` call, on this project's own
   unmodified selection/fitting code.
6. **Compute-cost accounting thread** (`COMPUTE_COST_DESIGN.md` /
   `COMPUTE_COST_FINDINGS.md`): `measure_oscillator_cpu_latency.py`,
   `prep_oscillator_latency_gpu_inputs.py` + `measure_oscillator_gpu_latency.py`
   (remote-session GPU counterpart), `measure_mlp_cpu_latency.py`,
   `build_cost_model.py` (the cost-model analysis and plot).
7. **Class-0 support audit thread** (folded into `FINDINGS.md`, no
   standalone doc): `run_class0_support_audit.py` (part 1: retained-ink
   statistics, local/free), `run_class0_support_audit_classify.py`
   (part 2: the two baseline classifier fits, `--cuml` optional).
8. **Topology/correlation visualization thread**: `visualize_topologies.py`,
   `visualize_normalized.py`, `plot_ink_correlation.py` +
   `diagnose_ink_correlation.py` (the correlation-table companion
   feeding the plot), `plot_decomposed_correlation.py` +
   `diagnose_decomposed_correlation.py` (same pattern).

**`generate_artifact_manifest.py`**: produces `results/ARTIFACT_
MANIFEST.json` -- SHA256 hashes, dimensions, image ordering, graph
hashes, and the selected `C` values the confirmatory run actually
consumed. Run after reproducing the pipeline to verify your own
artifacts match the ones behind the reported numbers. Scoped to the
locked confirmatory pipeline's own artifacts (stage3/stage4 pkls) only
-- does not cover the post hoc threads' own result pkls.

**`results/`**: gitignored cache of `.pkl`/`.npy` artifacts (large,
regenerable) plus a small number of genuinely committed outputs
(plots, `ARTIFACT_MANIFEST.json`) that belong in the reproducible
record.

**`scratch/`**: gitignored local scratch directory for the large
encode/GPU-evolve intermediate artifacts (`stage2a_paths.py`'s
default location, overridable via `STAGE2A_SCRATCH_ROOT`).

## Reproducing the pipeline locally

All scripts assume the project's `uv`-managed virtualenv (`uv run
python ...`, not bare `python3` -- see the root `CLAUDE.md`) and
`datasets/kmnist/` present locally.

```bash
# Stage 3: encode the full 60,000-image official training set (CPU, ~70s)
uv run python experiments/stage2a_dynamics_classification/run_feasibility_stage3_encode.py

# Official test set: encode the 10,000-image official test set (CPU, ~25s)
# -- the ONE place this project touches test-set images/labels for the
# locked confirmatory analysis. Do not run this speculatively.
uv run python experiments/stage2a_dynamics_classification/run_official_test_encode.py
```

Both produce a `theta0` batch + topologies package that then needs GPU
evolution (`stage3_gpu_evolve.py` / `stage4_gpu_evolve.py`) -- see
"Reproducing the confirmatory GPU evolution," below, for the exact
remote-session sequence. Once both splits have their GPU-evolved
`theta_T` cached locally (`scratch/stage3_train/stage3_gpu_results.pkl`,
`scratch/stage4_test/stage4_gpu_results.pkl`):

```bash
# Feasibility stage 3 model selection (classifier CV, ~4hr on CPU sklearn
# -- see FINDINGS.md's Result 3 before running this unattended)
uv run python experiments/stage2a_dynamics_classification/analyze_stage3_results.py

# The locked confirmatory evaluation itself
uv run python experiments/stage2a_dynamics_classification/run_confirmatory_evaluation.py

# Post hoc graph-to-graph pairwise comparison (seconds -- reuses the
# confirmatory run's already-saved per-image losses, no new GPU time)
uv run python experiments/stage2a_dynamics_classification/run_posthoc_graph_pairwise.py

# Artifact manifest (hashes, dimensions, selected C, frozen headline numbers)
uv run python experiments/stage2a_dynamics_classification/generate_artifact_manifest.py
```

**Small-scale feasibility stages** (mechanical validation only, not
confirmatory -- see `FINDINGS.md`'s "Feasibility Stage 1"/"Feasibility
Stage 2" sections; each is self-contained, no prior encode/GPU step
needed):
```bash
uv run python experiments/stage2a_dynamics_classification/run_feasibility_stage1.py   # 1,000 images, CPU, ~1 min
uv run python experiments/stage2a_dynamics_classification/run_feasibility_stage2.py   # 5,000 images, CPU, ~6 min
```

**Class-0 support audit** (post hoc, per external review's request --
see `FINDINGS.md`'s "class-0 confound" sections; part 1 is free/local,
part 2 needs the same cached training-set artifacts as the main
pipeline above and optionally accepts `--cuml` to fit under `cuml.accel`
on a GPU session rather than local sklearn):
```bash
uv run python experiments/stage2a_dynamics_classification/run_class0_support_audit.py
uv run python experiments/stage2a_dynamics_classification/run_class0_support_audit_classify.py            # local sklearn
uv run python experiments/stage2a_dynamics_classification/run_class0_support_audit_classify.py --cuml      # cuml.accel, remote GPU session
```

**Follow-on threads** (JAX classifier port, NVIDIA cuML cross-check,
compute-cost accounting) are not reproduced from this section --
each owns its reproduction details directly in its own findings doc
(`JAX_CLASSIFIER_PORT_FINDINGS.md`'s "Files" section,
`CUML_ACCEL_FINDINGS.md`'s "Code" section, `COMPUTE_COST_FINDINGS.md`'s
"Code" section), not centralized here.

Override the scratch location if you don't want the default
(`experiments/stage2a_dynamics_classification/scratch/`):

```bash
export STAGE2A_SCRATCH_ROOT=/path/to/your/scratch/dir
```

## Reproducing the confirmatory GPU evolution

**Amended by external review**: the single generic example previously
here (`theta0_batch.npy`, `topologies.pkl`) does not match either
driver's actual expected filenames -- following it as written would not
reproduce the run without inspecting the source first. Split into two
exact, separate workflows below, training and official-test, each with
real filenames and shapes throughout.

The GPU-evolution step used [`mighty-colab`](https://pypi.org/project/mighty-colab/)
(a Colab CLI/MCP wrapper) to provision an A100 session, upload inputs,
run the evolution kernel remotely, and download the resulting `theta_T`
states. `stage3_gpu_evolve.py` / `stage4_gpu_evolve.py` are the exact
scripts that ran on the remote kernel -- not runnable locally as-is
(they read/write `/content/...`, the remote session's filesystem
convention).

### Workflow A: Stage 3, training-set evolution (60,000 images)

**1. Local preparation.** `run_feasibility_stage3_encode.py` (already
run in "Reproducing the pipeline locally," above) produces one combined
package, `scratch/stage3_train/stage3_gpu_upload.pkl`
(`{"theta0_batch": (60000, 505) float64, "topologies": {4 x (505, 505)
float64}}`, ~250MB). That single pickle is too large for the upload
endpoint (see "Upload size limit," below), so a separate script splits
it into the exact files the remote driver expects:
```bash
uv run python experiments/stage2a_dynamics_classification/prepare_stage3_gpu_upload.py
```
Produces, in `scratch/stage3_train/`: `theta0_chunk_00.npy` through
`theta0_chunk_11.npy` (12 files, `(5000, 505)` float64 each, ~20MB
each) and `stage3_topologies.pkl` (`{4 x (505, 505)} float64`, ~8MB) --
verified to reassemble byte-identical to the original before the script
reports success, not just "the right shape."

**2. Upload + GPU execution:**
```bash
mighty-colab sessions   # check for orphaned sessions first, they bill while running
mighty-colab new -s stage3-evolve --gpu A100
mighty-colab reinstall -s stage3-evolve jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8

mighty-colab upload -s stage3-evolve evolve_on_graph_jax.py /content/evolve_on_graph_jax.py
mighty-colab upload -s stage3-evolve scratch/stage3_train/stage3_topologies.pkl /content/stage3_topologies.pkl
for i in 00 01 02 03 04 05 06 07 08 09 10 11; do
  mighty-colab upload -s stage3-evolve scratch/stage3_train/theta0_chunk_$i.npy /content/theta0_chunk_$i.npy
done

mighty-colab exec -s stage3-evolve -f stage3_gpu_evolve.py
```

**3. Output, download, stop:**
```bash
# stage3_gpu_evolve.py writes /content/stage3_gpu_results.pkl
mighty-colab download -s stage3-evolve /content/stage3_gpu_results.pkl scratch/stage3_train/stage3_gpu_results.pkl
mighty-colab stop -s stage3-evolve   # GPU sessions are billed while running
```

**4. Verify** your regenerated artifact matches the one behind the
reported numbers -- regenerate the manifest and compare the
`stage3_gpu_results.pkl` entry's `sha256` against the committed
`results/ARTIFACT_MANIFEST.json`:
```bash
uv run python experiments/stage2a_dynamics_classification/generate_artifact_manifest.py
git diff --stat experiments/stage2a_dynamics_classification/results/ARTIFACT_MANIFEST.json
```
A clean diff (or a diff touching only expected fields, e.g. a
regeneration timestamp if one is added later) confirms byte-identical
regeneration; a changed `sha256` for `stage3_gpu_results.pkl`
specifically means your regenerated evolution output differs from the
one the reported numbers are based on.

### Workflow B: Stage 4, official-test-set evolution (10,000 images)

**1. Local preparation.** `run_official_test_encode.py` (already run
above -- the one and only place this project touches test-set
images/labels for the locked confirmatory analysis; do not run this
speculatively) already produces the exact files the remote driver
expects, no separate chunking step needed at this smaller scale:
`scratch/stage4_test/stage4_gpu_upload_topologies.pkl`
(`{"topologies": {4 x (505, 505)} float64}`, ~8MB) and
`scratch/stage4_test/stage4_theta0_test.npy` (`(10000, 505)` float64,
~40MB -- small enough for a single-file upload).

**2. Upload + GPU execution:**
```bash
mighty-colab sessions
mighty-colab new -s stage4-evolve --gpu A100
mighty-colab reinstall -s stage4-evolve jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8

mighty-colab upload -s stage4-evolve evolve_on_graph_jax.py /content/evolve_on_graph_jax.py
mighty-colab upload -s stage4-evolve scratch/stage4_test/stage4_gpu_upload_topologies.pkl /content/stage4_gpu_upload_topologies.pkl
mighty-colab upload -s stage4-evolve scratch/stage4_test/stage4_theta0_test.npy /content/stage4_theta0_test.npy

mighty-colab exec -s stage4-evolve -f stage4_gpu_evolve.py
```

**3. Output, download, stop:**
```bash
# stage4_gpu_evolve.py writes /content/stage4_gpu_results.pkl
mighty-colab download -s stage4-evolve /content/stage4_gpu_results.pkl scratch/stage4_test/stage4_gpu_results.pkl
mighty-colab stop -s stage4-evolve
```

**4. Verify**, same pattern as Workflow A -- `generate_artifact_manifest.py`,
compare the `stage4_gpu_results.pkl` entry's `sha256` against the
committed `results/ARTIFACT_MANIFEST.json`.

**Upload size limit, worth knowing in advance**: a single large pickle
(the original 250MB Stage 3 `theta0`+topologies package) hit the
transfer endpoint's size limit -- this is why Workflow A's chunking step
exists and Workflow B's doesn't need one (Stage 4's combined package is
only ~48MB total, under the limit that broke Stage 3's single-file
upload). Small files (under ~50MB) upload fine as a single transfer.

**`cuml.accel`, if extending into that territory** (see
`CUML_ACCEL_FINDINGS.md`): install via
`mighty-colab reinstall -s my-session --requirement cuml_requirements.txt`
where the requirements file contains:
```
--extra-index-url=https://pypi.nvidia.com
cuml-cu12
```
Activate with `import cuml.accel; cuml.accel.install()` **before**
importing `sklearn` (or anything that imports it) -- `cuml.accel`
patches sklearn's estimator classes at import time. Verified to
accelerate `LogisticRegression`; verified **not** to accelerate
`MLPClassifier` (silent CPU fallback, no error) -- check any new
estimator directly before trusting it, per `CUML_ACCEL_FINDINGS.md`'s
own check-0 precedent, rather than assuming coverage.

## Public artifact cache (GCS)

The large cached artifacts behind the confirmatory result and its
follow-on threads (encoded features, GPU-evolved states, the 4
topology adjacency matrices, the confirmatory results pickle) are
mirrored in a public, read-only Google Cloud Storage bucket:

```
gs://bonsai-2026-stage2a-cache/
├── stage3_train/    # 60,000-image official training set artifacts
├── stage4_test/     # 10,000-image official test set artifacts
└── results/         # small results pkls (go/no-go, classifier conditions, confirmatory results)
```

Public HTTPS access, no authentication needed -- useful for pulling
data directly into a fresh Colab session (cloud-to-cloud transfer,
much faster than uploading from a local machine over a residential
connection):

```python
import urllib.request
urllib.request.urlretrieve(
    "https://storage.googleapis.com/bonsai-2026-stage2a-cache/stage3_train/stage3_gpu_results.pkl",
    "/content/stage3_gpu_results.pkl",
)
```

Or via `gcloud`/`gsutil` locally:
```bash
gcloud storage cp gs://bonsai-2026-stage2a-cache/stage3_train/*.pkl ./scratch/stage3_train/
```

**Why this exists**: re-uploading these artifacts from a local machine
to a fresh Colab session every time a new GPU experiment was needed
became real, repeated friction (each ~1-2GB upload took tens of
minutes over a typical residential connection, chunked into dozens of
small transfers to stay under the direct-upload size limit). Pulling
from GCS instead is a single fast download, entirely on Google's own
network. The bucket is public because this repository itself is
public and the source data (KMNIST) is a public academic dataset --
nothing in the cached artifacts discloses anything not already
derivable from the public code + public data, and the confirmatory
write-up in `FINDINGS.md` already states the actual scientific claims
in full. See the project's own reasoning on this (this thread's
conversation, not otherwise documented) if replicating the pattern for
a dataset or result where that reasoning wouldn't hold.

**Verifying your own regenerated artifacts match**: run
`generate_artifact_manifest.py` and compare SHA256 hashes against
`results/ARTIFACT_MANIFEST.json`'s committed values.

## Artifact replay vs. full raw-data regeneration

**Amended by external review, then closed (not just documented).**
These are two different reproducibility claims. Both are now supported.

- **Artifact replay.** Given the cached intermediate artifacts (via the
  GCS bucket above, or your own prior local run), every downstream step
  -- classifier CV, the confirmatory bootstrap/McNemar, the post hoc
  pairwise comparison, the artifact manifest -- reruns from those
  artifacts and reproduces the reported numbers. This is what
  `tests/test_stage2a_stats.py`'s `test_frozen_primary_effect_matches_
  findings_md` (Tier 2) checks directly, and what most of "Reproducing
  the pipeline locally," above, actually exercises once the
  encode/GPU-evolve artifacts exist.

- **Full raw-data regeneration (from nothing but public KMNIST + public
  code).** `stage2a_topologies.build_all_topologies()` previously
  depended on a cross-stage historical artifact,
  `experiments/stage1b2_structured_transformation/results/
  class0_constructions.pkl` -- gitignored, not committed, not part of
  the Stage 2A GCS bucket above (that bucket mirrors Stage 2A's own
  artifacts, not Stage 1B2's) -- in two places: `lattice` was read
  directly from that cache rather than reconstructed, and `T`'s own
  reconstruction hard-required the cache present just to verify against
  (even though the construction itself never needed it). **Both fixed**:
  `T` now calls `build_and_verify_T(require_historical_verification=
  False)` (still verifies against the cache opportunistically if
  present, skips the check rather than raising if not); `lattice` is
  now reconstructed via `build_lattice_topology`
  (`src/bonsai/dynamics/lattice_construction.py`) directly. `rewired`/
  `curr_random` were already fully cache-independent (derived from
  freshly-reconstructed `T`, never from the cache).

  **Verified directly, not assumed**: with `class0_constructions.pkl`
  present locally, the new from-scratch path was compared elementwise
  against what the old cache-reading path returned -- `T` and `lattice`
  both match to float64 machine epsilon (max diff `2.22e-16`, same
  precision this project's other from-scratch reconstructions already
  establish, not literally bit-identical since one side is recomputed
  fresh and the other was computed once, historically, on different
  hardware/library versions). Separately confirmed the historical
  artifact is now genuinely optional, not just theoretically so: the
  cache path was monkeypatched to a nonexistent file and
  `build_all_topologies()` still succeeded, producing the same `T`/
  `lattice` (to that same tolerance). Both checks are now a permanent
  regression test,
  `tests/test_stage2a_topologies.py::test_from_scratch_reconstruction_matches_cache_backed_values`
  (Tier 2, skipped if the cache isn't present locally to compare
  against -- but the other tests in that file, and the pipeline itself,
  no longer need it).

## Testing

```bash
uv run pytest tests/test_stage2a_core.py tests/test_stage2a_stats.py tests/test_stage2a_classifier.py tests/test_stage2a_pipeline.py tests/test_stage2a_topologies.py tests/test_stage2a_paths.py -v
```

Two-tier convention (this project's established pattern, see the root
`CLAUDE.md`): Tier 1 tests are self-contained on synthetic data, always
run. Tier 2 tests are skipped cleanly (not failed) when the local-only
cached artifacts they check against aren't present --
`test_stage2a_stats.py`'s `test_frozen_primary_effect_matches_
findings_md` is Tier 2, recomputing the primary bootstrap from
`results/stage4_confirmatory_results.pkl` if present and asserting it
still matches `FINDINGS.md`'s stated numbers, catching a future
refactor that silently changes the statistic. `test_stage2a_classifier.py`,
`test_stage2a_pipeline.py`, and `test_stage2a_paths.py` are Tier 1 only
(synthetic data, no cached-artifact dependency).
`test_stage2a_topologies.py` is Tier 2 only -- `build_all_topologies()`
needs real KMNIST data to reconstruct from; there's no meaningful
synthetic version of that check. It no longer additionally needs the
`class0_constructions.pkl` historical artifact (see "Artifact replay
vs. full raw-data regeneration," above) -- one test within the file
does still use it, when present, as a direct regression check that the
from-scratch reconstruction matches what the artifact-backed path used
to return, and skips independently if it's absent.

Deliberately not covered by this test suite: the class-0-audit,
compute-cost, and cuML-accel threads -- lower priority for a research
codebase, noted as a scope decision rather than an oversight.
