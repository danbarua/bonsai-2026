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
  test-split guards, idempotent `ensure_artifact`, and chunked
  checkpointed upload that resumes after a process death.

**Cloud-side and manual scripts:**

- **`colab_gcs_roundtrip_probe.py`** — the plain Python script the
  round-trip test executes *on* the Colab runtime. Not a notebook, and
  not run locally.
- **`smoke_stage2b_gcs.py`** — a manually-run smoke check against the
  real bucket, including both delete refusals. Deliberately not
  collected by pytest.

## Running things

Every Stage 2B operation runs through the root-level `Makefile`, which
is the single source of truth for the actual commands. This section is
a map to the targets, not a copy of them.

```bash
make help            # from the repository root -- every target, grouped
make stage2b-test    # the fast suite: 432 tests, no network, no cloud
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
is what lets the whole module and its 118 tests run in an environment
where the package is not installed and there is no network. Two tests
enforce it structurally in subprocesses — one blocks `google` via a
`sys.meta_path` finder, the other asserts nothing under `google.` enters
`sys.modules`. Don't hoist that import.

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
make stage2b-test              # 432 fast tests, ~35s, no network
make stage2b-test-roundtrip    # real Colab+GCS round trip; bills while running
make test                      # the whole repository suite
```

| file | tests | covers |
|---|---|---|
| `test_stage2b_gcs.py` | 118 | transport, guards, chunked resumable upload |
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
