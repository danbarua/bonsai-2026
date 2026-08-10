Simplified Technical English version of `experiments/stage2b_denoising/README.md`.

# Stage 2B: Dynamics-as-Computation, Denoising

This folder holds the Stage 2B investigation. Stage 2B asks one question.
Does runtime graph evolution improve single-step denoising prediction
error? Runtime graph evolution means running the oscillator network
forward in time on a graph, after the image is already encoded. Encoding
means turning an image into phase values on the graph's nodes. Denoising
means predicting the original clean pixel values from a corrupted, noisy
image. The question compares the evolved state against the encoded state
alone, before any evolution.

The design for this test is locked. A locked design is a plan document
the team has agreed not to change, except through a disclosed amendment.
An amendment is a change made to a locked document after it was locked.
The team always discloses each amendment. The team never edits a locked
document silently. The locked design lives in `DESIGN.md`.

**Current status lives in `FINDINGS.md`. This file does not restate it.**

The team learned this rule the hard way. This file's status line once said
"Phase B not written" for a full day, after Phase B had already run twice.
A review of the project found the same fact stated in three documents,
three different ways. Only `FINDINGS.md` was correct. So the fix is not a
better sentence in this file. The fix is one authoritative source, with
every other document pointing to it. Restating a fact that changes over
time is how copies come to disagree with each other. The newest commit is
not always the most current claim, either: the file with the older,
wrong statement had a *later* commit date, because that commit touched
unrelated lines.

Two facts do not change over time, so they belong here, not in
`FINDINGS.md`:

- No confirmatory statistic has been computed on the test split.
- No confirmatory statistic may be computed on the test split before the
  Stage 4 gate.

Read `DESIGN.md` first. It is the one authoritative source for every
constant, gate, and statistical rule this code follows. Seven drafts and
six rounds of review produced it, before any code was written.

## Reading order

1. `DESIGN.md` — the locked design. Read this before any code. Any change
   after the lock needs an explicit, disclosed amendment.
2. `../../CLAUDE.md` — the working rules this stage must follow.
   Principles 16, 19, and 20 were each applied, or added, during Stage
   2B's work.
3. This file — how to run things, and which safety checks not to bypass.

## Directory contents

**Rule, inherited from Stage 2A**: every new script gets one line in this
section, in the same commit that creates the script.

### The pipeline modules

Each module below can be tested on its own.

- **`stage2b_corruption.py`** — builds the locked forward corruption. It
  turns a clean image into a noisy one using `SHA256(split:index:42)` to
  seed a PCG64 random-number generator, producing one noisy version per
  image. It clips noisy values back into range, reports clip rates
  against the design's analytical table, computes the rescaled-identity
  baseline (a simple description, not a fitted model), and computes
  `clip_rate_agreement`. This compares the measured clip rate against the
  table's predicted rate, using an exact binomial Monte Carlo tolerance:
  `SE = sqrt(sum p_i(1-p_i))/N`, with a default of 5 standard errors. This
  check is a diagnostic. It does not block anything.
- **`stage2b_encoder_gate.py`** — the encoder-on-noisy-inputs gate. This
  gate checks whether encoding still works correctly on noisy images, not
  just on clean ones. It computes the final change in phase value (called
  final-Delta) per image, for both clean and noisy encodings. It computes
  `rho = median_noisy / max(median_clean, 1e-15)`. The gate PASSES if
  `rho <= 10`, or if both medians are already below the absolute-
  convergence floor (`ABS_CONV_EPS=1e-12`). This floor exists because a
  ratio between two quantities that are both at the edge of what a
  computer can measure tells you only which one hit that edge first, not
  whether the process actually converged. Any non-finite value (for
  example, infinity) is an automatic failure, no matter what the ratio
  says. `ENCODER_STEPS=1200` — the team raised this number from 150 after
  the gate's first real run FAILED on noisy KMNIST data. See `DESIGN.md`'s
  "Encoder-on-noisy-inputs gate" amendment, and
  `diagnose_encoder_gate_failure.py`, for the full story.
- **`stage2b_ridge.py`** — the multi-output ridge readout. Ridge is a
  prediction method that fits a model to data while resisting
  overfitting. This module runs one thin SVD (a mathematical
  decomposition) per cross-validation fold on a JAX implementation aware
  of the model's intercept term. It reuses that one SVD for all nine
  alpha values. Alpha is a number that controls how strongly ridge
  resists overfitting; a higher alpha means stronger resistance. The
  module also keeps sklearn's `Ridge(solver="svd")` as an independent
  check, and records the scaler-centering margin on every fold. A scaler
  centers each feature so it has mean zero; the margin measures how close
  to zero the team actually achieved.
- **`stage2b_stats.py`** — the confirmatory statistics. This includes the
  primary paired bootstrap test (a method that resamples data many times
  to estimate how much a result could vary by chance), the two Holm-
  corrected test families (Holm correction adjusts p-values so that
  running more tests together does not raise the false-positive rate),
  the studentized chunked sign-flip test (a test that flips the sign of
  paired differences at random to build a null distribution), the
  branched one-graph-wins rule, and a descriptive ranking that is
  explicitly not a statistical claim.
- **`stage2b_cnn.py`** — the locked CNN denoiser (a convolutional neural
  network, a kind of learned model), built with the equinox and optax
  libraries. It has 9,857 parameters, a number the team asserts by test.
  It has one shared masking function, and a training loop that keeps the
  raw training loss separate from the clipped loss used for model
  selection.
- **`stage2b_partition.py`** — builds the validation split, and the
  nested stratified draw used to pick smaller test corpora. The 1,000-
  image corpus is a prefix of the 5,000-image corpus.
- **`stage2b_conditions.py`** — holds the full list of condition names in
  one place: the statistics keys (`pre_evolution` plus the four graphs),
  the object-path segments (`evolved_T` and the rest), and the mapping
  between them. This means no script has to reinvent which name goes
  where. This module depends on nothing else.
- **`stage2b_gcs.py`** — moves artifacts to and from Google Cloud Storage
  (GCS), a cloud file-storage service. It builds object paths, guards the
  test-split objects, provides `ensure_artifact` (a function that skips
  redoing a step if its output already exists and checks out), and
  supports chunked, resumable uploads that pick up again after a process
  dies. It verifies content on every transfer. It also carries the
  sidecar-manifest system: `publish_manifest` writes a small manifest
  file next to each artifact, and `consume_validated` is the one approved
  way to read an artifact — it checks the manifest first. Because
  `ensure_artifact`'s skip branch itself calls `consume_validated`, a
  resumed step is always validated. An object simply existing in storage
  is never enough proof, on its own, that the step is safe to resume
  from. Passing a `fingerprint=` argument makes publishing the artifact
  and its manifest one atomic action. Artifacts written before this
  contract existed can still be read, through a `require_manifest=False`
  option that names, at the call site, exactly which stage predates the
  contract.
- **`stage2b_fingerprint.py`** — defines what "the same code produced
  this result" means in this project. A fingerprint is the combined list
  of source files a script actually uses, found both by reading its
  imports and by watching what it imports while running, then hashed.
  The team builds this fingerprint before a run, and checks it again
  after the run. `ConsumePolicy` decides which fields must match for a
  given kind of artifact, because Stage 2 legitimately reuses Stage 1's
  topologies under a different code commit. `array_manifest` records a
  data type, shape, and SHA-256 checksum for each individual array in a
  file. This module also runs a pre-flight check for GPU jobs:
  `--check-closure <driver.py>` exits with a non-zero code, and names any
  file in that driver's own source list that differs from the current
  commit. Uncommitted changes elsewhere in the project are recorded in
  the manifest, but do not block a run, because the actual run uses one
  fixed, pinned commit and cannot see those other changes anyway.

### The feasibility ladder

The feasibility ladder is a series of runs at growing size, used to catch
problems before spending money on the full-scale run.

- **`run_ladder_stage1.py`** — the stage-1 driver, running on 1,000
  images. It runs on a Colab cloud computer with a GPU. It fetches one
  pinned commit of this code repository and the staged KMNIST input
  files, rather than being uploaded together with its dependencies. It
  combines the modules above; it does not reimplement any of them. Every
  artifact it produces goes through `ensure_artifact`, so a session that
  dies loses at most one step's worth of work.
- **`stage_kmnist_inputs.py`** — copies the four KMNIST data files into
  the cloud storage bucket, once, from this local machine. It moves them
  from the local machine to GCS because the `datasets/` folder is not
  part of the code repository, so the cloud driver's copy of the code
  does not include it.
- **`encode_stage3_local.py`** — stage 3, Phase A. It corrupts and
  encodes all 60,000 official training images on this machine's own CPU
  cores, and writes only the resulting 505-value-per-image encoded array
  to GCS, for the GPU phase to read later. This step is split out because
  encoding is the one CPU-bound step in the whole pipeline. Running it
  inside a paid cloud GPU session would leave that GPU idle for most of
  the run. This script reuses the `corrupt_corpus` and
  `encode_with_final_delta_batch` functions unchanged, so the numbers
  match earlier stages exactly; only the machine differs. Rows in the
  output are stored in ascending official-index order. The result is
  published with a fingerprint manifest.
- **`compare_stage3_regeneration.py`** — the acceptance test for the
  regenerated data. The 54,000 images the earlier Phase A run already
  encoded must come back bit-exact, matched by their official KMNIST
  index rather than by position in the file. The additional 6,000
  validation images are new: they are reported with their own final-
  Delta results, and there is no rule requiring them to match anything
  earlier. This script only reads data. It writes nothing to the bucket.
- **`run_ladder_stage2.py`** — the stage-2 driver, running on a 5,000-
  image development set. It has the same structure as the stage-1
  driver: pinned-commit fetch, every artifact through `ensure_artifact`.
  It reuses stage 1's topologies and staged KMNIST inputs directly,
  instead of rebuilding them. It checks its own corruption output is
  bit-exact against stage 1's cached result. It adds the production
  SVD's own condition-number diagnostic (a number showing how sensitive
  a calculation is to small input errors), reports on how the ridge grid
  behaves, runs the ladder's second real-data ridge equivalence gate, and
  runs the first CNN training on real data, early-stopped on the locked
  6,000-image validation set, keeping the best of three random seeds.
- **`run_ladder_stage3.py`** — the stage-3 Phase B driver, running on the
  full 60,000-image training corpus. It has no encode step: Phase A
  already encoded the corpus locally, and this driver reads
  `encoded_train_s1200.npz` through the validated read path, checking its
  manifest identity and spot-checking one re-encoded image at ULP
  tolerance rather than requiring an exact match. A ULP, unit in the
  last place, is the smallest possible difference between two numbers as
  a computer stores them. This is the first driver to write under the
  fingerprint contract, so every artifact it creates carries full
  provenance and a record of its parent artifacts. A sizing probe in this
  step measures the cost of the JAX leg and the sklearn leg of the work
  separately, and halts if either exceeds a budget fixed before the run.
  Phase values (theta) and features are saved per graph and per condition,
  because the amendment-impact audit consumes those exact saved objects
  later.
- **`run_ladder_stage4.py`** — the stage-4 driver. This is `DESIGN.md`'s
  ONE locked evaluation on the official 10,000-image KMNIST test corpus.
  It ran on 2026-08-09 (result: `STAGE4_OK`). It took three attempts; the
  first two attempts caught and fixed two real bugs. This is the
  official, locked confirmatory result; the full account is in
  `FINDINGS.md`'s stage-4 section. The command `make stage2b-ladder-
  stage4` refuses to run unless a separate environment variable,
  `STAGE4_RELEASE_CONFIRMED=1`, is also set. `AUDIT_PROTOCOL.md` states
  that the test corpus is "evaluated once"; the driver enforces this
  itself with a one-shot lock, not just by stating the rule. This is the
  only file allowed to pass `allow_test_split=True` for real scientific
  work. `smoke_stage2b_gcs.py` also uses this same setting, but only as a
  named exception, because it is a hand-run check of the data-transport
  system, not a real driver. The stage-4 driver refits ridge fresh from
  stage 3's saved TRAINING features, at the fixed production alpha value.
  This produces a deterministic SVD solve that matches exactly, by
  construction. It retrains the CNN from the same three fixed seeds on
  the same locked split, and checks that this retraining reproduces stage
  3's saved `(best_seed, best_epoch)` choice before trusting it to make
  predictions on the test corpus. This check is needed because stage 3
  saved only its training history and a summary, not the fitted ridge
  coefficients, scaler, or model weights themselves. The driver refuses
  to compute the official result a second time if it already exists.

- **`run_audit.py` / `stage2b_audit.py`** — the amendment-impact audit
  driver and its calculations. An amendment-impact audit is a check that
  measures how much a disclosed change to the locked design actually
  changed the project's results. This driver has the same bootstrap-
  fetched-commit structure as the stage-3 and stage-4 drivers, rather
  than being a local command-line tool. It re-runs the pipeline ONLY at
  the 150-step encoder budget; the 1,200-step side reuses Phase B's own
  saved evolved-feature artifacts rather than recomputing them (see
  `PHASE_B_PLAN.md`'s Decision 4). It touches no test-split data
  anywhere. It is a diagnostic tool by design, with no one-shot lock:
  every artifact resumes normally, like stage 3's does, unlike stage 4's
  single locked confirmatory result. A sizing probe, measuring this
  driver's own cost pattern (pure JAX work, no sklearn leg, 100 fold-
  level SVDs), gates the 60,000-image out-of-fold ridge step before it
  runs. Before trusting the new out-of-fold calculation machinery at full
  scale, this driver reproduces stage 1's and stage 2's own saved fold-
  aggregate results, using each stage's own pre-amendment nine-value
  alpha grid. It ran on 2026-08-09 (result: `AUDIT_OK`, about 29.4
  minutes of GPU time, one real attempt after an earlier attempt caught a
  genuine bug: `stage2b_gcs.py`'s `LADDER_STAGES` validation list had
  never been extended to include this driver's own `LADDER_STAGE=5`
  value; the team fixed this and re-ran). No trigger fired in either
  alpha regime. The full account is in `FINDINGS.md`'s audit section. The
  command `make stage2b-audit` refuses to run unless
  `STAGE2B_AUDIT_RELEASE_CONFIRMED=1` is also set.

- **`run_abs_conv_eps_sensitivity.py`** — runs `COMPANION_PROTOCOLS.md`
  Protocol 2. This recomputes the encoder gate's verdict from stored
  per-image final-Delta arrays, with no re-encoding, at
  `eps in {1e-10, 1e-11, 1e-12, 1e-13}`, across every step count that
  `diagnose_encoder_gate_failure.py`'s diagnostic file covers. This runs
  entirely on the CPU, with no network use. It calls
  `stage2b_audit.sensitivity_table` and
  `stage2b_encoder_gate.evaluate_rho_gate` unmodified — it never
  reimplements them. Its one input file is reproducible from committed
  code (`diagnose_encoder_gate_failure.py`); this input file is not
  committed to the code repository and is regenerated on demand. Run it
  directly with: `uv run python run_abs_conv_eps_sensitivity.py`.

- **`generate_stage2b_artifact_manifest.py`** — produces the committed
  `ARTIFACT_MANIFEST.json` file. This lists GCS object paths, payload
  SHA256 checksums, the producing code commit, and the frozen headline
  numbers behind Stage 2B's two locked results (stage 4's official
  confirmatory result, and stage 5's amendment-impact audit). This lets
  anyone check provenance from a fresh copy of the code repository, with
  no GCS credentials and no need to re-run anything. It mirrors
  `stage2a_dynamics_classification/generate_artifact_manifest.py`, but
  reads already-published GCS manifests instead of hashing local files,
  because that is where Stage 2B's artifacts actually live. Long arrays
  — for example the primary test's 20,000 bootstrap resamples — are
  shortened to a count only, to keep the manifest file short; the full
  arrays remain in the GCS artifacts this file points to. Run it with
  `make stage2b-generate-artifact-manifest`.

- **`gate_corpus.py`** — fixes the exact list of documents the binding-
  clause inventory covers. A binding clause is a sentence in one of Stage
  2B's frozen protocol documents that commits the project to doing
  something, rather than just explaining a decision or reporting a
  result. The binding-clause inventory is a checklist file that records
  which sentences are true commitments and how each one is checked. This
  script checks its fixed list of documents against the `.md` files
  actually present on disk, in both directions, before handing the list
  to `tools/gates/gate_inventory.py`. Run it with `make stage2b-gate-
  inventory`. It exits with a non-zero code while any clause has no
  recorded disposition yet — which is the normal state until the
  inventory work is complete.

### Frozen protocols

Each of these was committed before any of its own numbers existed.
Nothing in them may change once a result has been seen.

- **`AUDIT_PROTOCOL.md`** — the 150-vs-1200 amendment-impact audit
  protocol. It fixes the sign convention, the roles of each image
  population, the out-of-fold prediction method, both alpha regimes, the
  analytic numerical-resolution limit, and all three review triggers.
- **`COMPANION_PROTOCOLS.md`** — the two companion protocols
  `AUDIT_PROTOCOL.md` names as partners to itself: the ARM/x86 stress
  test (a deterministic, worst-case set of test images, using frozen
  ridge coefficients across both computer chip types), and the
  `ABS_CONV_EPS` sensitivity table. Each has its own pre-committed halt
  condition.
- **`STAGE3_PLAN.md`** — the stage-3 plan of record, and its five frozen
  decisions.

### Plans

Unlike the frozen protocols above, these plans can still be revised.

- **`PHASE_B_PLAN.md`** — Phase B's step-by-step structure, and the
  decisions that shaped it. These include: the manifest as the true
  commit point (the payload file is written first, under a precondition;
  the exact write it produced is recorded; the sidecar manifest is
  written second, recording that exact write; anyone reading the data
  first checks the producer against what they expect, then reads that
  exact version, never just "whatever has this name now"); the 60,000-
  image equivalence check, added as a new cautious extra check whose
  failure is still a hard stop; the encoder gate, treated as a device
  used only at stage 1; and the amendment audit, sequenced after Phase B,
  against Phase B's own saved artifacts. This plan also records the
  measured data-retention finding: version history is turned off on the
  storage bucket, so artifacts are meant to be unchangeable by policy,
  and a replaced version of one is treated as a halt-worthy problem.

### Evidence documents

- **`NEGATIVE_PATH_EVIDENCE.md`** — a table linking each of the five
  demanded negative-path checks (proof that a safety check really stops
  a specific failure) to the test that proves it, what that test checks,
  where a guard was confirmed by deliberately breaking it, and where the
  coverage is narrower than what was demanded. Which demands are covered
  is stated in that document, not repeated here. This paragraph used to
  name stale-artifact refusal as the one demand not yet covered, and it
  stayed wrong for a day after that changed — the same problem as the
  status header earlier in this file.

  One fact here does not change over time, so it is stated here: the
  project reports coverage as narrower than the demand wherever it
  really is narrower. It never counts a mechanism that merely exists as
  proof that the team has adopted it.

### Diagnostics

Diagnostic scripts are not part of the locked pipeline. Following Stage
2A's convention, `diagnose_*.py` scripts investigate a problem; they do
not change anything themselves.

- **`diagnose_encoder_gate_failure.py`** — investigated feasibility
  stage 1's first encoder-gate FAILURE (`rho=169.851` at
  `ENCODER_STEPS=150`). It runs entirely on the CPU, locally —
  `_local_converged_phases` has no GPU dependency, so running this
  script costs nothing in cloud fees. It regenerates the exact stage-1
  test corpus, and checks it matches the failed cloud run's own reported
  numbers bit-for-bit before trusting anything computed from it. Its
  findings are now recorded in `DESIGN.md`'s "Encoder-on-noisy-inputs
  gate" amendment.

### Cloud-side and manual scripts

- **`colab_gcs_roundtrip_probe.py`** — the plain Python script that the
  round-trip test runs on the Colab cloud machine itself. It is not a
  notebook, and it is never run on the local machine.
- **`stage2b_verify_gpu.py`** — runs `DESIGN.md`'s ridge equivalence gate
  on a real GPU, at both ladder scales, using synthetic test matrices
  built to be as poorly conditioned as the real ones. It refuses to pass
  if run on a CPU fallback, or if 64-bit precision is not actually
  active on the device. Uploaded and run with `make stage2b-verify-gpu`.
- **`stage2b_verify_cnn_gpu.py`** — compares the CNN's forward pass
  (32-bit floating point) between a CPU and a GPU, at the compiler's
  default numerical precision and at a fixed precision, because reduced-
  precision math could shift the validation number that early stopping
  reads. Uploaded and run with `make stage2b-verify-cnn-gpu`.
- **`smoke_stage2b_gcs.py`** — a manually-run check against the real
  storage bucket. It checks both directions of file transfer, a chunked
  upload resumed partway through, the content checksum the real cloud
  service records for a composite object, and both delete-refusal
  checks. It is deliberately not collected or run by pytest.

## Running things

Every Stage 2B action runs through the root-level `Makefile`. The
Makefile is the one authoritative source for the actual commands. This
section is a map to the Makefile's targets, not a copy of them.

```bash
make help            # from the repository root -- every target, grouped
make stage2b-test    # the fast suite -- no network, no cloud
```

The team adds ladder targets to the Makefile only as each rung is
actually run, not ahead of time. `stage2b-stage-inputs` puts KMNIST data
in the bucket once. `stage2b-ladder-stage1`, `-stage2`, and `-stage3` run
their own rungs of the ladder. The Makefile is the true list of what
exists — this paragraph once said "later rungs have none yet" while two
of them already existed.

## Cloud execution: scripts, not notebooks

Stage 2B runs plain Python scripts on Colab cloud machines, using the
`mighty-colab` tool. Colab is only a compute service here, nothing more.
How results and visuals eventually get delivered is a decision the team
has deferred; it is not a pending task right now.

Artifacts move to GCS directly from the cloud environment. They are
never routed back through a local upload — Stage 2A already hit Colab's
upload size limit doing that once. The storage bucket is called
`bonsai-2026-stage2b-cache`. It is public-read, so anyone reading from it
needs no credentials; writing to it needs a service-account key.

The bucket name is not written directly into any script.
`stage2b_gcs.bucket_name()` reads it from the `BONSAI_GCS_BUCKET`
environment variable, falling back to a default value in the module if
that variable is not set. The Makefile declares that same default in one
place, and exports it to every target that reaches GCS. This means
pointing a run at a different bucket is as simple as running
`make stage2b-smoke-gcs BONSAI_GCS_BUCKET=other`, with no code edit.
`tests/test_stage2b_gcs_makefile.py` checks that the Makefile and the
module still agree on the bucket name.

`stage2b_gcs.py` imports the `google.cloud.storage` library lazily,
meaning only inside the functions that actually need it. This choice
matters, not just for style: it is what lets the whole module and its
tests run in an environment where that library is not installed and
there is no network connection. Three tests enforce this rule directly:
two block the `google` library from loading at all, and the third checks
that nothing under `google.` ever loads. The team must never move that
import to the top of the file, and the same rule applies to the
`google_crc32c` import.

## What a downloaded artifact is guaranteed to be

Every GCS file transfer verifies its content by default, in both
directions. The check uses `crc32c`, a checksum method that GCS computes
for every object it stores, including composed objects. This one method
works for both upload paths: a composed object has no `md5_hash` value,
so a system that checked MD5 instead would behave differently depending
on which upload method produced the file, and a downloader has no way to
know which method was used.

- A **download** is verified while it is still a temporary `.part` file,
  so a file with wrong content never reaches its real destination. The
  temporary file is deleted too. Any already-correct file already at the
  destination is left alone. This check covers a transfer that finished
  with the wrong content; a separate, existing check (an atomic rename)
  covers a transfer that stopped partway through.
- An **upload** is compared against the local file once it lands in the
  bucket, and a file that fails this check is deleted from the bucket.
  `ensure_artifact` treats an object's existence as proof that its step
  is done, so a wrong object must not be left in place making that false
  claim.
- An object with **no checksum recorded** causes a
  `ChecksumMissingError`. The code never treats "could not be checked" as
  the same as "checked and fine."
- Setting `verify_content=False` turns this check off, visibly, at the
  place it is used. Scientific runs should never have to remember to ask
  for correctness; correctness is the default.

The local side of this check uses the `google-crc32c` library, which is
a required dependency of `google-cloud-storage` and so is present
wherever a real file transfer happens. In this local environment,
neither library is installed, so the code falls back to a slower,
pure-Python version of the same checksum. This keeps the check running
under injected fake test buckets, instead of silently doing nothing. This
slow fallback never runs on a real large file transfer. The function
`checksum_backend()` reports which version is actually in use.

Whether the real cloud service always fills in a `crc32c` value for a
composed object is the one thing no local test can settle.
`smoke_stage2b_gcs.py` checks this directly, against the real bucket,
and reports what it finds.

## Measured before the ladder: what sets the centering tolerance

`assert_scaler_centered` is a check on the ridge input data. It halts a
run when `||mean(X_scaled)|| >= mean_x_tol_for(n)`, where the tolerance
is `1e-9 * (n / 1000) ** 0.5` and `n` is the number of rows in the matrix
being checked (see `DESIGN.md`, "Readout"). The measurements below are
what set this tolerance. They come from real corrupted, encoded, and
evolved features (the worst cross-validation fold, per condition), using
the CPU version of graph evolution:

| condition | n=300 | n=1,000 | fitted exponent |
|---|---|---|---|
| pre_evolution | 4.34e-14 | 8.00e-14 | 0.51 |
| lattice | 5.18e-13 | 8.60e-13 | 0.42 |
| T | 5.52e-13 | 1.17e-12 | 0.62 |
| rewired | 8.80e-12 | 1.94e-11 | 0.66 |
| curr_random | 3.65e-11 | 8.07e-11 | 0.66 |

**Measured at stage-2 scale, on GPU-evolved features** (a roughly
2-minute spike of GPU time: 5,000 images encoded locally in 3.7 seconds
using 9 CPU cores, then evolved under all four graphs, using the checked
`evolve_on_graph_jax` function, at 2.3 seconds per graph; all 5,000
solves reported success):

| condition | n=1,000 | n=5,000 | fitted exponent | vs `mean_x_tol_for(5000)` = 2.24e-9 |
|---|---|---|---|---|
| pre_evolution | 8.00e-14 | 1.66e-13 | 0.45 | 13,470x margin |
| lattice | 7.13e-13 | 1.76e-12 | 0.56 | 1,270x margin |
| T | 1.08e-12 | 3.24e-12 | 0.68 | 690x margin |
| rewired | 1.72e-11 | 3.94e-11 | 0.51 | 57x margin |
| **curr_random** | 7.87e-11 | **1.51e-10** | **0.405** | **14.8x margin** |

`curr_random` is the condition that decides the tolerance's design, and
0.405 is the growth rate the tolerance's own exponent needs to stay
above. It does: the tolerance's exponent of 0.5 upper-bounds this growth,
so the safety margin gets wider, not narrower, as `n` grows. The margin
is 12.7x at n=1,000, 14.8x at n=5,000, and a projected 18x at n=54,000.
The next-closest condition, `rewired`, is projected to reach about
1.3e-10 at n=54,000, against a tolerance of 7.35e-9 — about a 55x margin.
Both projections extend each condition's own measured growth rate past
the largest corpus size anyone has actually measured, which is 5,000.

`T` and `lattice` grow faster than the square root of `n` on these two
points (growth rates of 0.68 and 0.56), so their margins narrow rather
than widen — from 690x and 1,270x at n=5,000 to a projected roughly 450x
and 1,100x at n=54,000. Neither comes close to the limit, and neither is
the condition the tolerance was set against in the first place.

The *order* in which conditions rank here comes from how strongly each
graph synchronizes its nodes. Stage 2A measured order parameters
(a measure of synchronization) of 0.997 for `rewired` and 0.991 for
`curr_random`. This means those two graphs drive node phases to values
that barely depend on the input image, so their feature columns barely
vary. The *growth with n*, on the other hand, is not a sign of a growing
problem: a growth rate near 0.5 is ordinary floating-point rounding
error, since the mean of `n` values accumulates roughly the square root
of `n` in rounding, and this gets amplified when divided by a small
column standard deviation. That is why the tolerance itself uses this
same 0.5 growth rate, instead of a rate fitted to any one dataset: a
fixed absolute tolerance on a quantity that grows like the square root of
`n` will eventually fail for any features at all, given enough data.
The 0.66 growth rate in the first table above belongs to a different
computation pipeline than the one the tolerance's anchor value comes
from.

**What the guard protects, and how much room there is.** The tolerance
value is not arbitrary. sklearn's `Ridge(fit_intercept=True)` centers `X`
internally, while the JAX implementation centers only `Y`. The two
methods agree, to `DESIGN.md`'s 1e-8 equivalence gate, *because*
`||mean(X)||` is negligible. Measured at Stage 2B's real data shape
(n=1,000, p=1,008, t=505), by deliberately injecting a mean offset and
comparing both methods:

| `\|\|mean(X)\|\|` | JAX-vs-sklearn prediction difference |
|---|---|
| 3.2e-9 | 4.9e-14 |
| 3.2e-7 | 1.4e-13 |
| 3.2e-5 | 1.3e-09 |
| 3.2e-4 | 1.3e-07 — breaches the 1e-8 gate |

So the two methods' agreement holds up to roughly 1e-4. This is what
fixes the tolerance's value from above: 7.35e-9 at n=54,000 sits four or
more orders of magnitude below the level where `||mean(X)||` starts
costing anything, and about nine orders of magnitude below the size of
offset a genuinely broken scaler would produce. The guard has room on
both sides.

One thing the tolerance does not address: a feature column that is
nearly, but not exactly, constant is divided by its own tiny scale during
scaling, and produces `||mean(X_scaled)||` around 9.7e-7. This is about
400 times above the n=5,000 tolerance, and it is a different mechanism
from ordinary float rounding. `assert_scaler_centered`'s own
documentation records these measured boundaries. The check still halts
in this case, and this remains an open issue.

## Guards you must not route around

Three checks below exist because the locked design depends on them, not
because they are tidy.

1. **Test-split corruption.** `corrupt_image` and `corrupt_corpus` raise
   a `PermissionError` unless `allow_test_split=True` is explicitly
   passed. Only stage 4 may pass this.
2. **Test-side GCS objects** live under their own storage path, need the
   same explicit opt-in, and are additionally refused at every ladder
   stage except stage 4.
3. **`delete_prefix`** refuses to delete anything outside `stage2b/`
   under any circumstance. It refuses to delete a non-test path without
   an explicit force flag. And — the case that actually matters — it
   checks the objects it actually *matched*, not just the text of the
   path prefix. The string `"stage2b/t"` is not, by plain text
   comparison, under the test-data path, yet it matches every object
   under the test path too. A check based on the string alone would have
   let this delete request through.

If a guard blocks something, that means the guard is working correctly.
The correct response is a disclosed amendment to `DESIGN.md`, never a
keyword argument that bypasses the check.

## Testing

```bash
make stage2b-test              # the fast suite, ~40s, no network
make stage2b-test-roundtrip    # real Colab+GCS round trip; bills while running
make test                      # the whole repository suite
```

| file | covers |
|---|---|
| `test_stage2b_gcs.py` | transport, guards, chunked resumable upload, content verification, the sidecar manifest and validated read path |
| `test_stage2b_fingerprint.py` | the combined static-and-runtime import list, refusals for a dirty code tree and for re-check failures, per-kind read policies, per-array payload manifests |
| `test_stage2b_negative_path_evidence.py` | that every test `NEGATIVE_PATH_EVIDENCE.md` cites still exists under that name |
| `test_stage2b_encode_stage3_local.py` | that the encoding result does not depend on chunk size, that corruption is keyed by index, the exact binomial tail report, and the population roles |
| `test_stage2b_compare_stage3.py` | the regeneration join: by official index, never by position in a file |
| `test_stage2b_cnn.py` | architecture, shared masking, training loop |
| `test_stage2b_stats.py` | sign-flip test, Holm-corrected families, the winner rule |
| `test_stage2b_ridge.py` | the SVD ridge implementation versus the sklearn oracle, alpha selection, the tolerance that scales with `n` |
| `test_stage2b_partition.py` | split ordering, the nested stratified draw |
| `test_stage2b_corruption.py` | random-number determinism, clip rates against the design table |
| `test_stage2b_encoder_gate.py` | the rho gate, handling of non-finite values |
| `test_stage2b_gcs_roundtrip.py` | credential-gate checks, plus the one slow real round trip |
| `test_stage2b_contracts.py` | rules that cross module boundaries, which no single module's own tests can see |
| `test_stage2b_gcs_makefile.py` | that the Makefile and the module agree on the bucket name; that every GCS-touching script has a Makefile target |
| `test_stage2b_ladder_stage1.py` | the stage-1 driver's constants, call sites, and Makefile agreement |
| `test_stage2b_ladder_stage2.py` | the stage-2 driver's constants, call sites (including the CNN's dependencies), and Makefile agreement |
| `test_stage2b_ladder_stage3.py` | the stage-3 driver's constants, the sizing probe's projections and halt paths, the pre-contract read paths, and Makefile agreement |
| `test_stage2b_ladder_stage4.py` | the stage-4 driver's constants, the CNN reproduction check, and (derived by reading every `.py` file's syntax tree) that the test-split opt-in exists in exactly one place, checked both ways |
| `test_stage2b_gate_corpus.py` | which documents the binding-clause inventory covers, checked against what is actually on disk, both directions |
| `test_stage2b_audit.py` | the pure amendment-audit calculations: index alignment, gauge phases, trigger verdicts, the stress-set construction, the sequencing-gate guard |
| `test_stage2b_audit_driver.py` | the audit driver's constants, the pinned pre-contract checksum table, the sizing probe's projections and halt paths, and the fixed-versus-reselected alpha combination's completeness and its break-confirmation |
| `test_stage2b_abs_conv_eps_sensitivity.py` | the `ABS_CONV_EPS` sensitivity table's invariance check and halt rule, plus a check that skips cleanly (rather than failing) when the real diagnostic data file is absent |
| `test_stage2b_artifact_manifest.py` | the artifact-manifest generator's list shape, its break-confirmation for stripping long lists, and a real-bucket check against the committed manifest |

This table is the complete list of what `make stage2b-test` runs, with
one exception: the slow round trip. It deliberately does not list test
counts. A count copied by hand from a derived number went stale four
separate times before the team removed it — three rows were stale at
once in one case, and in another case it went stale from a code merge,
where nobody wrote a wrong number and the number became wrong anyway.
Following working rule 21, the team either derives such a list or checks
it against the true list; it does not duplicate a count. `make stage2b-
test` prints the true count when it runs, so this file's list of test
files is what is checked here, and the counts are not repeated.

`make stage2b-test` prints every skipped test with its reason (using the
`-rs` flag). This is because a plain `uv sync` command once removed the
`google-crc32c` library, which `stage2b_gcs.py` is designed to fall back
away from gracefully. The test that checks this fallback against the
real library simply became a skip. Nothing failed outright; the total
test count moved by one, and nobody compares counts by eye. An absence
has to announce itself; it cannot be safely inferred from a total number.

The `google-crc32c` dependency is now declared in the project's `dev`
dependency group, so it is present wherever the test suite runs. It is
deliberately not placed in the `gpu` dependency group: this test target
can run inside continuous integration (CI), and `tools/ci/ci_targets.py`
forbids putting a dependency group into a CI-runnable target, precisely
to stop cloud command-line tools from being pulled into an environment
that has no credentials.

The round trip test is the only test that leaves this machine. It starts
a CPU cloud session, writes an object to GCS from inside it, and reads
that object back here twice — once with credentials, and once without
any. This is because "readable from outside the session" and "readable
with no credentials at all" are different claims, and only the second
read actually exercises the public-read setting. This test runs with the
`-s` flag on purpose, so its step-by-step output is shown. Its evidence
is most of its value (working rule 20), and a plain green PASS result
alone would not show what actually happened on the network.

## Learnings worth carrying forward

Things this stage's work produced that will matter beyond this stage:

- **Chunked random-number draws are not automatically the same random
  stream** (`CLAUDE.md` working rule 19). `Generator.integers`, at sub-
  64-bit widths, buffers bits internally. This means a chunked sign-flip
  test and an unchunked one can silently produce different results, and
  both can look like plausible p-values. `Generator.random` does not have
  this problem. The safeguard is a test that sweeps different chunk
  sizes, not just a code comment.
- **A manually-checked behavior should become an automatic test**
  (working rule 20). The team first confirmed the public-read setting on
  the bucket by hand. It is now checked by an automatic test.
- **A test that cannot fail on the bug it names is worse than no test at
  all.** Three separate agents building this stage each found a test in
  their own work that could never fail correctly: a masking check that
  left 782 coordinates unchecked, a best-checkpoint test whose test data
  never actually changed, and a winner-rule test that recomputed its own
  expected answer from the same data it was supposed to check. All three
  were found by deliberately breaking the underlying code and confirming
  the test actually failed. Do this check before trusting a green test
  suite.
- **Uploads and downloads can fail in different ways.**
  `download_file` was already safe against a mid-transfer crash, using a
  temporary `.part` file and an atomic rename. Uploads were not safe this
  way, and this is exactly the direction that matters most when a
  temporary cloud session is pushing large amounts of data out.
- **A transfer completing is not the same as a transfer being correct.**
  A transfer that finishes, and a transfer that has the right content,
  are two separate properties. The `.part`-file safeguard only ever
  guaranteed the first one. Data that arrives whole but wrong is the
  failure that silently produces bad numbers, instead of an error
  message.
