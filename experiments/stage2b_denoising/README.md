# Stage 2B: Dynamics-as-Computation, Denoising

This directory holds the Stage 2B investigation: does runtime graph
evolution, on top of the same already-dynamically-encoded local phase
state, improve single-step denoising prediction error relative to the
unevolved encoded state alone? The design is locked (`DESIGN.md`).

**Status: built, not yet run.** Every component below exists and is
tested; no feasibility-ladder stage has been executed, and there is no
`FINDINGS.md` yet because there are no findings. Nothing here has
produced a number about denoising. Read `DESIGN.md` first — it is the
authoritative spec for every constant, gate and statistical rule this
code implements, and it was locked through seven drafts and six review
rounds before any of this was written.

## Reading order

1. `DESIGN.md` — the locked design. Read before any code. Post-lock
   changes require an explicit, disclosed amendment.
2. `../../CLAUDE.md` — the methodological discipline this stage is held
   to. Principles 16, 19 and 20 were each either applied or added
   during Stage 2B's construction.
3. This file — how to run things, and what not to route around.

## Directory contents

**Convention, inherited from Stage 2A**: any new script gets a one-line
mention here, in the same commit that creates it.

**The pipeline modules**, each independently testable:

- **`stage2b_corruption.py`** — the locked forward corruption
  (`SHA256(split:index:42)` → PCG64 → one realization per image),
  clipping, clip-rate diagnostics against the design's analytical
  censoring table, and the rescaled-identity descriptive baseline.
- **`stage2b_encoder_gate.py`** — the encoder-on-noisy-inputs gate:
  final-Δ per image for clean and noisy encodings,
  `rho = median_noisy / max(median_clean, 1e-15)`, PASS iff `rho <= 10`, with non-finite
  values as automatic failures regardless of rho.
- **`stage2b_ridge.py`** — the multi-output ridge readout. Intercept-aware
  JAX SVD (one thin decomposition per fold, all nine alphas reused from
  it), sklearn `Ridge(solver="svd")` retained as the verification
  oracle, and the scaler-centering margin recorded on every fold.
- **`stage2b_stats.py`** — the confirmatory statistics: primary paired
  bootstrap, the two Holm families, the studentized chunked sign-flip,
  the branched one-graph-wins rule, and a descriptive ranking that is
  explicitly not an inferential claim.
- **`stage2b_cnn.py`** — the locked equinox+optax residual denoiser
  (9,857 parameters, asserted), its single shared masking primitive, and
  the training loop with its raw-loss / clipped-selection split.
- **`stage2b_partition.py`** — the validation split and the nested
  stratified ladder draw (the 1,000 is a prefix of the 5,000).
- **`stage2b_conditions.py`** — the condition vocabulary in one place:
  the statistics keys (`pre_evolution` plus the four graphs), the object-
  path segments (`evolved_T` and the rest), and the mapping between them,
  so no driver has to reinvent which spelling belongs where. Depends on
  nothing.
- **`stage2b_gcs.py`** — artifact transport: object paths, the
  test-split guards, idempotent `ensure_artifact`, chunked checkpointed
  upload that resumes after a process death, and content verification on
  every transfer.

**The feasibility ladder:**

- **`run_ladder_stage1.py`** — the stage-1 driver (n=1,000). Runs on a
  Colab runtime, fetching one pinned commit of this repo and the staged
  KMNIST inputs rather than being uploaded with its dependencies. It
  composes the modules above and implements none of them; every artifact
  goes to GCS through `ensure_artifact`, so a dead session resumes having
  lost at most one step.
- **`stage_kmnist_inputs.py`** — stages the four KMNIST IDX files into the
  bucket, once, from here. The only Stage 2B upload that goes local → GCS,
  because `datasets/` is gitignored and so absent from the driver's clone.

**Cloud-side and manual scripts:**

- **`colab_gcs_roundtrip_probe.py`** — the plain Python script the
  round-trip test executes *on* the Colab runtime. Not a notebook, and
  not run locally.
- **`stage2b_verify_gpu.py`** — runs `DESIGN.md`'s ridge equivalence gate
  on a real GPU, at both ladder scales, on synthetic ill-conditioned
  matrices shaped like the real ones. Refuses to pass on a CPU fallback
  or with x64 not realised on the device. Uploaded and run by
  `make stage2b-verify-gpu`.
- **`stage2b_verify_cnn_gpu.py`** — compares the CNN's float32 forward
  pass CPU versus GPU, at XLA's default precision and pinned, because
  reduced-precision convolutions would move the validation metric early
  stopping reads. Uploaded and run by `make stage2b-verify-cnn-gpu`.
- **`smoke_stage2b_gcs.py`** — a manually-run smoke check against the
  real bucket: both round trips, a chunked upload resumed mid-transfer,
  the content digest the real service records for the resulting
  composite object, and both delete refusals. Deliberately not collected
  by pytest.

## Running things

Every Stage 2B operation runs through the root-level `Makefile`, which
is the single source of truth for the actual commands. This section is
a map to the targets, not a copy of them.

```bash
make help            # from the repository root -- every target, grouped
make stage2b-test    # the fast suite -- no network, no cloud
```

Ladder targets get added as each rung is actually driven, not written
speculatively. Stage 1 has them: `stage2b-stage-inputs` puts KMNIST in
the bucket once, and `stage2b-ladder-stage1` runs the rung. Later rungs
have none yet.

## Cloud execution: scripts, not notebooks

Stage 2B runs **plain Python scripts** on Colab runtimes via
`mighty-colab`. Colab is a compute runtime here, nothing more; how
results and visuals eventually get delivered is a deferred decision and
not a pending task.

Artifacts move to GCS **from within the cloud environment**, never
round-tripped through a local upload — Stage 2A already hit Colab's
upload ceiling doing that. Bucket `bonsai-2026-stage2b-cache` is
public-read, so a consumer needs no credentials; writing needs the
service-account key.

The bucket is not written into any script. `stage2b_gcs.bucket_name()`
resolves it from `BONSAI_GCS_BUCKET`, falling back to the module
default, and the `Makefile` declares that same default in one place and
exports it to every target that reaches GCS — so pointing a run at a
different bucket is `make stage2b-smoke-gcs BONSAI_GCS_BUCKET=other`
with no edit. `tests/test_stage2b_gcs_makefile.py` asserts the Makefile
and the module still agree.

`stage2b_gcs.py` imports `google.cloud.storage` **lazily**, inside the
functions that need a client. This is load-bearing, not stylistic: it
is what lets the whole module and its tests run in an environment
where the package is not installed and there is no network. Three tests
enforce it structurally in subprocesses — two block `google` via a
`sys.meta_path` finder, the third asserts nothing under `google.` enters
`sys.modules`. Don't hoist that import, and don't hoist the
`google_crc32c` one either.

## What a downloaded artifact is guaranteed to be

Every GCS transfer verifies content, by default, in both directions.
The check is on `crc32c` — the checksum GCS computes for **every**
object it stores, a composed one included. That choice is what makes one
verification path serve both upload routes: a composite object carries
no `md5_hash`, so an MD5-verifying consumer would behave differently
depending on whether `upload_file` or `upload_file_chunked` produced
what it is reading, and a downloader has no business knowing which.

- A **download** is verified while it is still the `.part` sidecar, so a
  file whose bytes are wrong never reaches the destination path. The
  sidecar goes too, and any good file already at that path is left
  alone. The atomic rename covers a transfer that *stopped*; this covers
  one that *finished with the wrong contents*.
- An **upload** is compared against the local file once it lands, and an
  object that fails is deleted. `ensure_artifact` reads an object's
  existence as proof its step is done, so one known to be wrong must not
  sit there making that claim.
- An object carrying **no checksum raises** `ChecksumMissingError`.
  Nothing is treated as fine merely because it could not be checked.
- `verify_content=False` opts out, at the call site, visibly. Science
  runs should not have to remember to ask for correctness.

Computing the local side uses `google-crc32c`, a hard dependency of
`google-cloud-storage` and therefore present wherever a real transfer
happens. Where neither is installed — this local environment — a
pure-Python CRC32C stands in so the check still runs under the injected
fake buckets instead of silently becoming a no-op. It is slow, and it is
never on a real gigabyte transfer's path; `checksum_backend()` reports
which one is live.

Whether the real service populates `crc32c` for a composed object is the
one part of this no unit test can settle. `smoke_stage2b_gcs.py` asks it
directly, against the real bucket, and reports what came back.

## Measured before the ladder: what sets the centering tolerance

`assert_scaler_centered` halts when
`||mean(X_scaled)|| >= mean_x_tol_for(n)`, which is
`1e-9 * (n / 1000) ** 0.5` on the row count of the matrix it is handed
(`DESIGN.md`, "Readout"). The measurements below are what that tolerance
is derived from. Measured on real corrupted-encoded-evolved features
(worst CV fold, per condition), CPU evolution path:

| condition | n=300 | n=1,000 | fitted exponent |
|---|---|---|---|
| pre_evolution | 4.34e-14 | 8.00e-14 | 0.51 |
| lattice | 5.18e-13 | 8.60e-13 | 0.42 |
| T | 5.52e-13 | 1.17e-12 | 0.62 |
| rewired | 8.80e-12 | 1.94e-11 | 0.66 |
| curr_random | 3.65e-11 | 8.07e-11 | 0.66 |

**Measured at stage-2 scale, on GPU-evolved features** (a ~2-minute A100
spike: 5,000 images encoded locally in 3.7 s on 9 cores, evolved under
all four graphs via the verified `evolve_on_graph_jax` kernel at 2.3 s
per graph, all 5,000 solves reporting success):

| condition | n=1,000 | n=5,000 | fitted exponent | vs `mean_x_tol_for(5000)` = 2.24e-9 |
|---|---|---|---|---|
| pre_evolution | 8.00e-14 | 1.66e-13 | 0.45 | 13,470x margin |
| lattice | 7.13e-13 | 1.76e-12 | 0.56 | 1,270x margin |
| T | 1.08e-12 | 3.24e-12 | 0.68 | 690x margin |
| rewired | 1.72e-11 | 3.94e-11 | 0.51 | 57x margin |
| **curr_random** | 7.87e-11 | **1.51e-10** | **0.405** | **14.8x margin** |

`curr_random` is the binding condition and 0.405 is the growth the
tolerance's exponent has to dominate. It does: 0.5 upper-bounds it, so
the margin widens with `n` — 12.7x at n=1,000, 14.8x at n=5,000, ~18x
projected at n=54,000. `rewired`, the next-closest, projects to ~1.3e-10
at n=54,000 against a 7.35e-9 tolerance, ~55x. Both projections
extrapolate each condition's own measured exponent past the largest
corpus anyone has measured, which is 5,000.

`T` and `lattice` do grow faster than sqrt(n) on these two points (0.68
and 0.56), so their margins narrow rather than widen — from 690x and
1,270x at n=5,000 to ~450x and ~1,100x projected at n=54,000. Neither
comes near binding, and neither is the condition the anchor was set
against.

The *ordering* is the synchronization mechanism — Stage 2A measured
order parameters of 0.997 (`rewired`) and 0.991 (`curr_random`), so
those graphs drive node phases nearly image-independent and their
cos/sin columns barely vary. The *growth* is not a worsening pathology:
an exponent near 0.5 is ordinary floating-point accumulation, the mean
of n values carrying ~sqrt(n) rounding, amplified by division by a small
column std. That is why the tolerance carries the same exponent rather
than a fitted one — a fixed absolute tolerance on a sqrt(n)-growing
quantity fires eventually for any features at all, and the first table's
0.66 belongs to a different pipeline from the anchor's.

**What the guard protects, and how much room there is.** The tolerance
is not arbitrary: sklearn's `Ridge(fit_intercept=True)` centres `X`
internally while the JAX path centres only `Y`, so the two agree to
DESIGN.md's 1e-8 equivalence gate *because* `||mean(X)||` is negligible.
Measured at Stage 2B's real shape (n=1,000, p=1,008, t=505), injecting a
mean offset and comparing both paths:

| `\|\|mean(X)\|\|` | JAX-vs-sklearn prediction difference |
|---|---|
| 3.2e-9 | 4.9e-14 |
| 3.2e-7 | 1.4e-13 |
| 3.2e-5 | 1.3e-09 |
| 3.2e-4 | 1.3e-07 — breaches the 1e-8 gate |

So the equivalence gate survives until roughly 1e-4. That is what fixes
the tolerance from above: 7.35e-9 at n=54,000 sits four or more orders
below the level at which `||mean(X)||` starts costing anything, and
about nine orders below the O(1) offset a genuinely broken scaler
produces. The guard has room on both sides.

One thing the n-dependent tolerance does not address: a
nearly-but-not-exactly-constant feature column is divided by its tiny
scale and produces `||mean(X_scaled)||` around 9.7e-7 — some 400x above
the n=5,000 tolerance, and a different mechanism from float
accumulation. `assert_scaler_centered`'s docstring carries the measured
boundaries. It still halts there, and that remains open.

## Guards you must not route around

Three of these exist because the locked design's integrity depends on
them, not because they are tidy:

1. **Test-split corruption** (`corrupt_image` / `corrupt_corpus`) raises
   `PermissionError` unless `allow_test_split=True`. Only stage 4 may
   pass it.
2. **Test-side GCS objects** live under their own prefix, need the same
   opt-in, and are additionally refused at any ladder stage but 4.
3. **`delete_prefix`** refuses anything outside `stage2b/`
   unconditionally, refuses a non-test prefix without an explicit force,
   and — the case that actually matters — checks the objects it
   *matched*, not just the prefix string. `"stage2b/t"` is not under the
   test root by string comparison yet matches the entire test side; the
   string checks alone would have passed it.

If a guard is in the way, that is the guard working. The remedy is a
disclosed amendment to `DESIGN.md`, not a keyword argument.

## Testing

```bash
make stage2b-test              # the fast suite, ~40s, no network
make stage2b-test-roundtrip    # real Colab+GCS round trip; bills while running
make test                      # the whole repository suite
```

| file | covers |
|---|---|
| `test_stage2b_gcs.py` | transport, guards, chunked resumable upload, content verification |
| `test_stage2b_cnn.py` | architecture, shared masking, training loop |
| `test_stage2b_stats.py` | sign-flip, Holm families, winner rule |
| `test_stage2b_ridge.py` | SVD ridge vs sklearn oracle, alpha selection, the n-dependent centering tolerance |
| `test_stage2b_partition.py` | split ordering, nested stratified draw |
| `test_stage2b_corruption.py` | RNG determinism, clip rates vs the design table |
| `test_stage2b_encoder_gate.py` | rho gate, non-finite handling |
| `test_stage2b_gcs_roundtrip.py` | credential-gate checks, plus the one slow round trip |
| `test_stage2b_contracts.py` | cross-module contracts no single module's tests can see |
| `test_stage2b_gcs_makefile.py` | Makefile and module agree on the bucket; every GCS-touching script has a target |
| `test_stage2b_ladder_stage1.py` | the ladder driver's constants, call sites and Makefile agreement |

This table is the whole of what `make stage2b-test` runs, and the one
exclusion is the slow round trip. It carries no test counts, deliberately:
per-file counts are a hand-maintained copy of a derived number, and this
one drifted four separate times before it was removed — three rows stale at
once in one instance, and in another it went stale from a *merge*, where
nobody wrote a wrong number and the numbers became wrong anyway. Principle
21's rule is derive the set or assert it matches; the count is derived by
`make stage2b-test`, which prints it, so the file list is what is asserted
here (`tests/test_stage2b_ladder_stage1.py`) and the numbers are not
duplicated.

The round trip is the only test that leaves this machine. It provisions
a CPU runtime, writes an object to GCS from it, and reads that object
back here **twice** — once authenticated, once anonymously, because
"readable from outside the session" and "readable without credentials"
are different claims and only the second one exercises the public-read
grant. It runs with `-s` deliberately: its step-by-step evidence is most
of its value (principle 20), and a bare green PASS would record that the
assertions held without showing what happened on the wire.

## Learnings worth carrying forward

Things this stage's construction produced that outlive it:

- **Chunked RNG draws are not automatically the same stream**
  (`CLAUDE.md` principle 19). `Generator.integers` at sub-64-bit widths
  buffers bits, so a chunked and an unchunked sign-flip diverge silently
  and both return plausible p-values. `Generator.random` does not. The
  guard is a test sweeping chunk sizes, not a comment.
- **Hand-verified functionality becomes an executable test**
  (principle 20). The public-read grant was confirmed interactively
  first; it is now an assertion.
- **A test that cannot fail on the bug it names is worse than none.**
  Three separate agents building this stage found vacuous tests in
  their *own* work — a 2×2 masking probe that left 782 coordinates
  unchecked, a best-checkpoint test whose fixture never diverged, a
  winner-rule test that recomputed its expectation from the same dict.
  All three were caught by mutating the implementation and checking the
  test actually broke. Do that before believing a green suite.
- **Uploads and downloads fail asymmetrically.** `download_file` was
  already death-safe via a `.part` sidecar and `os.replace`; uploads
  were not, which is the direction that matters when an ephemeral Colab
  session is pushing gigabytes out.
- **Atomicity is not correctness.** A transfer that completes and a
  transfer that is right are separate properties, and the `.part`
  sidecar only ever established the first. Bytes that arrive whole and
  wrong are the failure mode that produces numbers instead of an error.
