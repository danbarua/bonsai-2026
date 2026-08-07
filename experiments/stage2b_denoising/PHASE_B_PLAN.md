# Stage 2B ladder stage 3, Phase B — plan of record

**Status: plan, not a frozen protocol.** The frozen documents
(`DESIGN.md`, `AUDIT_PROTOCOL.md`, `COMPANION_PROTOCOLS.md`, and
`STAGE3_PLAN.md`'s five freezes) are untouched by this and govern
wherever they speak. Execution — the sizing probe onward — waits on Dan's
explicit release.

Reviewed by Claude Desktop 2026-08-07; the four decisions that review
settled are recorded below at the points they bind.

## Why this document exists

`STAGE3_PLAN.md` item 6 is the whole of what was frozen about Phase B:

> **Phase B** — stage-2 driver structure plus kappa_alpha / rank /
> coefficient norms / CNN preservation.

Everything else is new specification, first reviewed here. That is worth
stating plainly because the alternative — treating a consolidating
summary as carrying the authority of the document it summarises — is a
failure this stage's history has now recorded four times, most recently
in the review round that produced this plan.

---

## What Phase B is

A GPU-session driver that reads Phase A's 60,000-image encoded artifact,
regenerates corruption and clean targets in-session, and runs evolution,
ridge and CNN training at full scale. It produces no test-split quantity
and reads no test-split object.

It inherits two contracts that did not exist when stage 2 ran:

- **`consume_validated` is the only consume.** `ensure_artifact`'s skip
  branch is a validated consume, so every resumed step checks what it
  resumes from. Enforced by an AST check over every stage2b script.
- **The pre-flight is closure-keyed.** A GPU target refuses when a file in
  the driver's own import closure differs from HEAD, and proceeds when the
  tree is dirty elsewhere. `test_every_repo_fetching_gpu_target_runs_the_closure_check`
  is derived, so this target inherits the guard on the day it is written.

---

## Decision 1 — Freeze 4: generation preconditions, and semantic naming declared satisfied

Freeze 4 was consented with two clauses that had not been implemented:
content-addressed naming, and race-free publication. Resolved as **(a) +
(c)**, and disclosed to ChatGPT in parallel rather than letting the
pre-test package be the first mention of a partially-unimplemented
freeze.

**(a) Generation preconditions — implemented.** Every write that lands an
artifact carries a server-side compare-and-set: `if_generation_match=0`
to create, the observed generation to replace. A check performed here
followed by a write is not a precondition — the gap between them is the
race. Scope, stated as a test rather than a comment: the guard covers the
composed destination and the manifest sidecar; the chunked route's *parts*
deliberately carry none, because a resumed upload rewriting a damaged part
is an overwrite by design and guarding it would break the resume the route
exists for. The race Freeze 4 names is two writers replacing each other's
artifact, and the artifact is the destination.

The failure mode is live rather than theoretical: the resumability story
explicitly contemplates a second session resuming after the first is
*presumed* dead, and "presumed dead but still uploading" is exactly what
`stop`'s checked exit status exists for.

**(c) Content-addressed naming — NOT satisfied by (a) alone.** The
satisfied-in-substance claim was rejected on review and replaced by a
stronger requirement: **the manifest is the commit point.** Implemented as
such.

**Write order** — payload first under its own precondition; the generation
that write produced is captured; the sidecar is written second, under its
own precondition, recording that exact generation.

**Consume order** — manifest read first; the producer verified against the
consumer's expected fingerprint *before any payload bytes move*; then the
payload fetched **at the committed generation**, never latest-by-semantic-
name; then the digest recomputed on the fetched bytes.

**Crash window** — a payload with no manifest is **UNCOMMITTED**. Not
merely "fails a validity rule": a session that died between the two writes
leaves bytes no consumer may accept, and the next run must not read them
as a completed step.

**Competing writer** — loses on a precondition, never silently replaces a
committed artifact. Both writes are guarded, payload and sidecar alike.

Pinning is not redundant with the digest, and there is a test that shows
why: a replacement carrying the **same bytes** at a new generation
satisfies `payload_sha256` and fails the pin. The digest answers "are
these the right bytes"; the generation answers "is this the object that
was committed", and only the second is structural.

### Retention, measured rather than assumed

Generation-pinning protects against races but not against later
re-fetchability unless the bucket retains superseded generations. Measured
against `bonsai-2026-stage2b-cache` on 2026-08-07, two ways — the bucket's
declared config via `gcloud`, and live probes on scratch objects deleted
afterwards:

| question | finding | how established |
|---|---|---|
| can a current generation be pinned and read? | **yes** — 5,366 bytes at an explicit generation | live probe |
| is object versioning enabled? | **no** | `gcloud storage buckets describe` returns no `versioning` key, and a pinned read of a superseded generation 404s |
| does the service refuse a stale precondition? | **yes** — `PreconditionFailed` | live probe |
| are superseded bytes retained at all? | **yes, for 7 days** — `soft_delete_policy.retentionDurationSeconds: 604800` | `gcloud`, confirmed by listing |
| is a superseded generation recoverable? | **yes, administratively** — it appears in `list_blobs(soft_deleted=True)` as `<name>#<generation>` | live probe |

**Correcting the first version of this finding.** An earlier probe used
only a pinned read, got `NotFound`, and concluded the superseded bytes
were gone. They are not: soft delete retains them for seven days. The
pinned read 404s because a normal read does not see soft-deleted objects,
not because nothing is there. The conclusion happened to be right about
what a *consumer* can do and wrong about what *exists* — which is the
distinction that matters when the question is "can we diagnose a
superseded artifact after the fact."

Exact-generation reads **are** guaranteed, so the reviewer's fallback —
content-addressed names becoming mandatory — is **not** triggered. The
declaration, revised:

> Artifacts are **immutable by policy**, enforced by preconditions. A
> superseded generation is a **halt-worthy anomaly** detected by a failed
> pinned read or a digest mismatch, never a supported read path. Superseded
> bytes are nonetheless retained for **7 days** by the bucket's soft-delete
> policy and can be listed and restored administratively — so a halt has a
> bounded **forensic window** in which what replaced an artifact can still
> be examined. That window is a diagnostic affordance, not durable
> retention, and no code path depends on it.

**Permissions, deliberately not changed.** The pipeline's service account
cannot read bucket config (`storage.buckets.get` denied, 403). That is the
correct posture rather than a gap: the driver never needs bucket metadata,
and granting a permission to make a one-off audit convenient would widen
the credential that runs unattended on a rented VM. Config questions are
answered with `gcloud` under a human identity, which is where they belong.

### The write-once invariant — enforced, not described

Generation pinning guarantees "the consumer reads what was committed" only
while the pinned generation survives, and with versioning off a
supersession destroys it. So semantic naming plus manifest pinning is
acceptable **only** under a genuinely write-once policy for anything a
consumer pins. Enforced in code:

- **LINEAGE** — everything scientific; anything that is or can be a parent
  in a manifest. **Create-once.** `force=True` on one of these is a named
  refusal, raised *before* `produce` runs, so a refusal never burns the
  GPU time it exists to protect. Regeneration means a **new name** — as
  Phase A did, writing `encoded_train_s1200` rather than overwriting
  `encoded_fit_s1200`.
- **RUN_SCOPED** — reports and timing summaries. Never a parent, never
  pinned.

**Classification is fail-closed.** An undeclared kind is LINEAGE. A new
scientific artifact inherits create-once automatically; falling into the
overwrite-capable class requires an explicit declaration and can never
happen by omission. Proven by test rather than assumed.

**The lineage walk is transitive.** Checking immediate parents leaves the
property one link deep — a run-scoped object three hops up is still a
mutable thing inside a chain of digests claiming to be fixed. The walk
runs on the **consume** side as well as the publish side, and the two are
not redundant: publish binds what this code writes, consume binds what it
is willing to read, including a manifest left by an older commit or by
hand.

**Reports are run-scoped, and no overwrite path survives anywhere.**
`force=True` previously overwrote one fixed report name per rung, so a
resumed run destroyed the record of what the attempt that died had seen —
the same "an unwritten result does not survive" failure this project
already has a lesson about, and reports are pre-test-package material.
Report names now carry a run id, both attempts survive, and the storage
model is uniformly append-only. No kind currently retains an overwrite
path; if one ever must, the plan has to state which, why historical
replacement is acceptable, and that no package claim depends on
recovering earlier versions.

All four negative paths were confirmed by breaking what they watch, each
firing its own test and no other: force against LINEAGE; a second create
against a committed name losing its precondition; a RUN_SCOPED artifact
offered as a parent, on the publish side and the consume side separately;
and the fail-closed default.

These measurements and this deviation are carried into the pre-test
package.

---

## Decision 2 — equivalence at 60,000 is a new prudential extension, not the locked gate

`DESIGN.md:330` is literal and scoped: *"at both the 1,000- and
5,000-image stages, JAX and sklearn must produce (a) max absolute
difference in clipped validation predictions <= 1e-8, and (b) identical
alpha selection."* It names 1,000 and 5,000, it passed at both, and it is
**discharged**. `FINDINGS.md`'s "the ladder's third real-data equivalence
gate" phrasing overclaimed what was frozen and does not survive this
plan.

What runs at 60,000 is a **new, prudential extension** at 12x the largest
verified scale — a genuinely different conditioning regime — **fold-level
only**. That preserves the gate's literal quantities (clipped validation
predictions, identical alpha selection, both fold-level concepts) without
seven full-corpus oracle fits on a metered A100.

**The framing describes what the check IS. It does not soften what a
failure MEANS**, and the halt rule is verbatim:

> Disagreement beyond the frozen tolerance on any fold or condition —
> predictions or alpha selection — **halts everything before Stage 4**,
> full diagnosis required. *"The locked gate still passed"* is **not an
> available reading** of an extension failure at production scale.

Stated in advance so neither the framing nor the verdict can be chosen
after a number exists. The measured oracle-side CPU cost is reported;
whether it overlaps GPU work or runs serially is implementation detail.

---

## Decision 3 — the encoder-on-noisy-inputs gate is discharged, with the reasoning on record

Not re-run at stage 3. Three reasons, recorded so this is a reasoned
discharge rather than an omission:

1. It is a **stage-1 device** by its own docstring and by the design's
   ladder. Stage 2 already deliberately declined to re-run it, recording
   diagnostics instead. No new gates get invented mid-ladder in either
   direction.
2. The quantity it guards — noisy-input convergence — is **independently
   evidenced at 60,000** by Phase A's recorded tail: median and p95 both
   exactly 0.0, max 2.468e-10, four orders below the solver's `rtol=1e-6`.
**Struck, and the correction recorded rather than edited away.** An
earlier draft of this section carried a third reason: *"under the amended
absolute-convergence clause the measured values pass trivially — both
medians are literally zero."* That claimed a gate-pass from data never
collected under the gate's rule. Only the **noisy** median was measured at
60,000; Phase A encodes corrupted images, no clean 60,000 encode exists,
and therefore no clean median exists to form the ratio the gate is defined
on. The discharge stands without it, on (1) and (2) alone.

The standing language, everywhere this is referred to: **"discharged as a
stage-1 device; Phase A's noisy final-Delta distribution is reported as
independent convergence evidence."** Never "passes the amended gate at
60k" — that sentence describes a computation nobody ran.

---

## Decision 4 — the audit runs after Phase B, against Phase B's own artifacts

Separate session, sequenced **after** Phase B, for failure isolation —
bundling would make one session's death cost three results — and for a
sharper reason: the audit's 1,200-step inputs are **Phase B's own
persisted evolved-feature artifacts**, consumed through the validated
path. The audit then compares 150-step features against the actual
production features — the exact objects the confirmatory result will rest
on — rather than against a recomputation that merely ought to match them.

**Requirement this places on this plan, made explicit:** evolved thetas
and features are persisted per graph, with fingerprints. Resumability
requires it anyway; the audit's design makes it load-bearing.

**The 150 side must differ from Phase A in `encoder_steps` ONLY** — same
official indices, same corruption realizations, same folds, gauges,
preprocessing and fingerprint contract. The audit driver **asserts** this
by comparing the two artifacts' recorded fingerprint configs field by
field rather than assuming it; the config already carries `encoder_steps`,
so the artifacts self-describe and the assertion is over recorded data,
not over intent.

**The 150-step encode is a named, authorized addition.**
`encode_stage3_local.py` becomes parameterized by step count — the
fingerprint config already carries `encoder_steps`, so the artifact
self-describes — producing `encoded_train_s150.npz` plus manifest under
the full contract, run locally.

*Correcting my own estimate*: I put this at ~11 minutes by carrying Phase
A's 679s unscaled. The encoder's cost is linear in step count and it runs
two passes per image, so 150 steps is roughly **1/8 of Phase A — order 85
seconds** plus fixed corruption/topology/load overhead. Desktop's figure
was right and mine was wrong by about 8x.

---

## Blocking facts read from the code

**Sizing is not measured, and principle 18 forbids extrapolating it.**
Features are `2 * 505 - 2 = 1008` dims. At n=60,000 float64: one
condition's matrix is 484 MB, six held simultaneously ~2.9 GB, one fold's
SVD input (48,000 x 1008) 387 MB plus workspace. An A100 has 40 GB, so
this is very likely fine — and "very likely fine" is the reasoning
principle 18 exists to forbid. Stage 2A extrapolated data generation
correctly from n=1,000 to 60,000 and dismissed classifier CV as "a few
seconds" at the same n=1,000, where it went on to dominate by 79x.

**Mechanical facts, verified.** `EVOLVE_CHUNK = 250` divides 60,000
exactly (240 chunks); stage 2's divisibility halt stays valid unchanged.
Phase B has **no encode step** — stage 2's `step4_encode` becomes a
validated consume, which changes the step numbering.

---

## Step structure

Mirrors `run_ladder_stage2.py` deliberately: same composition rules, same
resumability, same halt discipline. Differences are marked.

| # | step | notes |
|---|---|---|
| — | bootstrap | pinned-commit clone; closure-keyed pre-flight |
| — | `stage_kmnist` | validated consume; manifests now present |
| 0 | pre-flight | x64 enabled before `jax.numpy` binds; device report |
| 1 | corpus | **changed** — all 60,000 official indices, ascending; Freeze 2 role flags carried |
| 1b | topologies | consume stage 1's `topologies.npz` under `CONTENT_ONLY` |
| 2 | corruption | **changed** — regenerated in-session for 60,000; **bit-exact** cross-check against stage 2's stored rows, matched by official index |
| 2b | **sizing probe** | **new** — one condition, one fold: peak memory and wall-clock reported before the full run proceeds |
| 3 | encoded input | **changed** — validated consume of `encoded_train_s1200.npz`; its manifest identity (payload sha256, fingerprint commit) into the run report; one-image re-encode spot-check at **ULP tolerance**, never exact equality |
| 4 | restrict | unchanged; the artifact is already 505-dim |
| 5 | evolution | batched, four graphs, success-flag gating, 240 chunks of 250, per-graph CPU reference; **thetas persisted per graph with fingerprints** (Decision 4) |
| 6 | features | six conditions, gauge-fixed at reference node 363; **persisted per condition with fingerprints** (Decision 4) |
| 7 | ridge | 5-fold CV, nine alphas, seven refits; fold-level oracle extension (Decision 2); addendum-#1 diagnostics below |
| 8 | CNN | 54k fit / 6k locked validation, locked config, three seeds, clipped-val selection, preservation items |
| 9 | stats smoke | in-sample, non-inferential, as stage 2 |
| 10 | report | named items on their own terms |

Every artifact goes through `ensure_artifact` with `fingerprint=`, so
publication is atomic with the upload and a forced overwrite republishes
rather than orphaning a sidecar.

### The two spot-checks, with tolerances matched to what each measures

- **Corruption regeneration vs stage 2's stored rows: BIT-EXACT.** Pure
  numpy PCG64 keyed on `SHA256(split:index:42)`, architecture-independent.
  Anything less than exact here would be a real defect.
- **One-image re-encode vs the stored Mac-generated array: ULP
  tolerance, never exact equality.** Phase A recorded max 3 ULP
  (4.441e-16) between ARM and x86 encodings of the same images, with
  within-architecture results bit-exact in both directions. An exact
  comparison would fail on a healthy run for a reason already understood.

### Ridge diagnostics

The mechanism was settled in `STAGE3_PLAN.md`'s own analysis: compute
per-fold quantities **for all nine alphas and column-index after**
selection, rather than retaining fits (~180 MB at stage 3). `s_max` and
`s_min` are recorded as absolute magnitudes — a ratio cannot be
un-divided, which is why `fold_cond` alone was insufficient. Recorded per
fold: `kappa_alpha`, scale-aware numerical rank, coefficient norms,
boundary-alpha flags, full validation curves, and centering margins under
the n-scaled tolerance.

**A convention note is required in the artifact**: which alpha column each
per-fold quantity was indexed at, and under which regime. Without it a
reader cannot tell a fixed-alpha row from a reselected one.

### Debt restated so it is not lost twice

`AUDIT_PROTOCOL.md` requires the per-fold mean of the new per-image OOF
MSEs to reproduce **the stored fold aggregates from the stage-1 and
stage-2 run artifacts**. The bit-exact cross-check delivered with
`oof_per_image_mse` discharged the Tier-1 version of this, not the
protocol's. It belongs to the audit driver, not to Phase B.

---

## Verification

- Every new guard confirmed by breaking what it watches. Specifically: the
  corruption cross-check must fail on a perturbed row; the ULP spot-check
  must fail on a genuinely wrong image and pass on the stored one; the
  sizing probe must be able to halt.
- Resumability proven the established way — kill the session mid-run,
  confirm the next run resumes having lost at most one step — now with the
  added property that each resumed step validates what it resumes from.
- Session conventions unchanged: `EXEC_TIMEOUT`, 60s heartbeat, sentinel
  plus exit code, unconditional teardown with `stop` status checked.
- Full suite green before each commit.

## What this plan does not do

Stage 4, in any form: no test-split object is read, written, or named.
The amendment audit and the ARM/x86 stress run are separate sessions
after Phase B. Pre-test item 9 (the frozen Stage 4 command) stays
deferred until Phase B's object paths exist, since it depends on names
this plan proposes rather than fixes.
