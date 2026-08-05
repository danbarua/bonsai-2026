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
- **`stage2b_gcs.py`** — artifact transport: object paths, the
  test-split guards, idempotent `ensure_artifact`, chunked checkpointed
  upload that resumes after a process death, and content verification on
  every transfer.

**Cloud-side and manual scripts:**

- **`colab_gcs_roundtrip_probe.py`** — the plain Python script the
  round-trip test executes *on* the Colab runtime. Not a notebook, and
  not run locally.
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
make stage2b-test    # the fast suite: 451 tests, no network, no cloud
```

The feasibility ladder itself has no targets yet, because no ladder
stage has been run. They get added as each rung is actually executed —
not written speculatively against a pipeline nobody has driven.

## Cloud execution: scripts, not notebooks

Stage 2B runs **plain Python scripts** on Colab runtimes via
`mighty-colab`. Notebooks are deferred to the end of the project.
`DESIGN.md` contains a stale line calling a Colab notebook the "final
deliverable" — it is known-stale and should not be acted on; correcting
it is a documentation amendment nobody has needed badly enough to make
yet.

Artifacts move to GCS **from within the cloud environment**, never
round-tripped through a local upload — Stage 2A already hit Colab's
upload ceiling doing that. Bucket `bonsai-2026-stage4a-cache` is
public-read, so a consumer needs no credentials; writing needs the
service-account key.

`stage2b_gcs.py` imports `google.cloud.storage` **lazily**, inside the
functions that need a client. This is load-bearing, not stylistic: it
is what lets the whole module and its 137 tests run in an environment
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

## Measured before the ladder: the centering guard will fire

`assert_scaler_centered` halts when `||mean(X_scaled)|| > 1e-10`, and its
own docstring flags the tolerance as an open question. Measured on real
corrupted-encoded-evolved features (worst CV fold, per condition):

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

| condition | n=1,000 | n=5,000 | vs 1e-10 |
|---|---|---|---|
| pre_evolution | 8.00e-14 | 1.66e-13 | 602x margin |
| lattice | 7.13e-13 | 1.76e-12 | 57x margin |
| T | 1.08e-12 | 3.24e-12 | 31x margin |
| rewired | 1.72e-11 | 3.94e-11 | 2.5x margin |
| **curr_random** | 7.87e-11 | **1.51e-10** | **FIRES** |

So it is not a projection: `curr_random` exceeds the guard at ladder
stage 2, and `rewired` sits 2.5x from it there — meaning a tolerance
widened only far enough for `curr_random` at stage 2 would halt again on
`rewired` at stage 3.

The ordering is the synchronization mechanism — Stage 2A measured order
parameters of 0.997 (`rewired`) and 0.991 (`curr_random`), so those
graphs drive node phases nearly image-independent and their cos/sin
columns barely vary. But the *growth* is not a worsening pathology: an
exponent near 0.5 is ordinary floating-point accumulation, the mean of n
values carrying ~sqrt(n) rounding, amplified by division by a small
column std. A fixed absolute tolerance on a sqrt(n)-growing quantity
fires eventually for any features at all.

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

So the equivalence gate survives until roughly 1e-4, five orders beyond
the worst value the ladder is projected to produce. The guard as set
will halt on a number that causes no harm to the thing it exists to
protect.

This is recorded, not acted on. `stage2b_ridge.py` states plainly that
the tolerance "is a locked-design question, not an implementation
choice", so changing it is a disclosed amendment to `DESIGN.md` rather
than an edit. The measurement exists so that decision is made from
numbers before the ladder runs, instead of from a halt in the middle of
stage 2.

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
make stage2b-test              # 451 fast tests, ~30s, no network
make stage2b-test-roundtrip    # real Colab+GCS round trip; bills while running
make test                      # the whole repository suite
```

| file | tests | covers |
|---|---|---|
| `test_stage2b_gcs.py` | 137 | transport, guards, chunked resumable upload, content verification |
| `test_stage2b_cnn.py` | 76 | architecture, shared masking, training loop |
| `test_stage2b_stats.py` | 66 | sign-flip, Holm families, winner rule |
| `test_stage2b_partition.py` | 49 | split ordering, nested stratified draw |
| `test_stage2b_ridge.py` | 47 | SVD ridge vs sklearn oracle, alpha selection |
| `test_stage2b_corruption.py` | 35 | RNG determinism, clip rates vs the design table |
| `test_stage2b_encoder_gate.py` | 24 | rho gate, non-finite handling |
| `test_stage2b_gcs_roundtrip.py` | 18 | 17 fast credential-gate checks + 1 slow round trip |

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
