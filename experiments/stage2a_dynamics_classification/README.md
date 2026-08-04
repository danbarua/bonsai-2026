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

- **Pipeline code**: `stage2a_core.py` (encode/evolve/gauge features),
  `stage2a_classifier.py` (the locked CV/standardization/classifier
  procedure), `stage2a_stats.py` (the confirmatory statistical
  machinery -- paired bootstrap, McNemar, Holm correction), `stage2a_
  pipeline.py` / `stage2a_pipeline_jax.py` (CPU and GPU-batched
  pipeline orchestration), `stage2a_topologies.py` (the 4 confirmatory-
  expansion graphs), `evolve_on_graph_jax.py` (the verified GPU
  evolution kernel).
- **Stage drivers**: `run_feasibility_stage1.py` /
  `run_feasibility_stage2.py` (small-scale mechanical validation),
  `run_feasibility_stage3_encode.py` + `stage3_gpu_evolve.py` (full
  60,000-image training-set encode + GPU evolve),
  `run_official_test_encode.py` + `stage4_gpu_evolve.py` (the official
  10,000-image test set's one-and-only encode + GPU evolve),
  `run_confirmatory_evaluation.py` (the locked confirmatory analysis
  itself), `run_posthoc_graph_pairwise.py` (the post hoc, Holm-corrected
  graph-to-graph comparison).
- **Diagnostics**: `diagnose_*.py`, `verify_*.py`, `analyze_*.py`
  scripts -- each self-documenting, referenced by name from the
  `FINDINGS.md` section they support.
- **`generate_artifact_manifest.py`**: produces `results/ARTIFACT_
  MANIFEST.json` -- SHA256 hashes, dimensions, image ordering, graph
  hashes, and the selected `C` values the confirmatory run actually
  consumed. Run after reproducing the pipeline to verify your own
  artifacts match the ones behind the reported numbers.
- **`results/`**: gitignored cache of `.pkl`/`.npy` artifacts (large,
  regenerable) plus a small number of genuinely committed outputs
  (plots, `ARTIFACT_MANIFEST.json`) that belong in the reproducible
  record.
- **`scratch/`**: gitignored local scratch directory for the large
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

Override the scratch location if you don't want the default
(`experiments/stage2a_dynamics_classification/scratch/`):

```bash
export STAGE2A_SCRATCH_ROOT=/path/to/your/scratch/dir
```

## Reproducing the confirmatory GPU evolution

The GPU-evolution step used [`mighty-colab`](https://pypi.org/project/mighty-colab/)
(a Colab CLI/MCP wrapper) to provision an A100 session, upload the
`theta0` batch + topologies, run the evolution kernel remotely, and
download the resulting `theta_T` states. `stage3_gpu_evolve.py` /
`stage4_gpu_evolve.py` are the exact scripts that ran on the remote
kernel -- they are not runnable locally as-is (they read/write
`/content/...`, the remote session's filesystem convention).

**Pattern for spinning up a fresh test run on a Colab A100** (adjust
session names freely; `mighty-colab` also exposes these as MCP tools if
you're driving this from an agent rather than a shell):

```bash
# 1. Check for orphaned sessions first (they bill while running)
mighty-colab sessions

# 2. Provision
mighty-colab new -s my-session --gpu A100

# 3. Install pinned deps (single atomic install+restart if the
#    package/version might already be imported in a live kernel;
#    `install` alone is enough on a genuinely fresh session)
mighty-colab reinstall -s my-session jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8

# 4. Upload the evolution kernel + your theta0/topologies package
mighty-colab upload -s my-session evolve_on_graph_jax.py /content/evolve_on_graph_jax.py
mighty-colab upload -s my-session path/to/theta0_batch.npy /content/theta0_batch.npy
mighty-colab upload -s my-session path/to/topologies.pkl /content/topologies.pkl

# 5. Run the driver (stage3_gpu_evolve.py / stage4_gpu_evolve.py, or your
#    own script following the same chunked-batch pattern -- see either
#    script's own comments for why CHUNK_SIZE=1000: vmap materializes a
#    (batch, n, n) diff tensor per RHS evaluation, which does not fit in
#    GPU memory for a full 60,000-image batch at once)
mighty-colab exec -s my-session -f stage3_gpu_evolve.py

# 6. Download results, stop the session (GPU sessions are billed while running)
mighty-colab download -s my-session /content/stage3_gpu_results.pkl ./stage3_gpu_results.pkl
mighty-colab stop -s my-session
```

**Upload size limit, worth knowing in advance**: a single large pickle
(the original 250MB `theta0`+topologies package) hit the transfer
endpoint's size limit. The working pattern is to shard large arrays
into ~20MB `.npy` chunks (`theta0_chunk_00.npy`, `theta0_chunk_01.npy`,
...), upload each separately, and reassemble with `np.concatenate` on
the remote side -- `stage3_gpu_evolve.py`'s own inline comment shows
the reassembly. Small files (under ~50MB) upload fine as a single
transfer.

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

## Testing

```bash
uv run pytest tests/test_stage2a_core.py tests/test_stage2a_stats.py -v
```

Two-tier convention (this project's established pattern, see the root
`CLAUDE.md`): Tier 1 tests are self-contained on synthetic data, always
run. Tier 2 tests are skipped cleanly (not failed) when the local-only
cached artifacts they check against aren't present --
`test_stage2a_stats.py`'s `test_frozen_primary_effect_matches_
findings_md` is Tier 2, recomputing the primary bootstrap from
`results/stage4_confirmatory_results.pkl` if present and asserting it
still matches `FINDINGS.md`'s stated numbers, catching a future
refactor that silently changes the statistic.
