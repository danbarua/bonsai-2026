Simplified Technical English version of `experiments/stage2b_denoising/PHASE_B_PLAN.md`.

# Stage 2B ladder stage 3, Phase B — plan of record

**Status: this is a plan, not a frozen protocol.** The frozen documents
— `DESIGN.md`, `AUDIT_PROTOCOL.md`, `COMPANION_PROTOCOLS.md`, and the
five frozen decisions in `STAGE3_PLAN.md` — are untouched by this plan,
and they govern wherever they speak to a topic. This plan governs
everything else. Running this plan — from the sizing probe onward —
waits on Dan's explicit release to proceed.

Claude Desktop reviewed this plan on 2026-08-07. The four decisions that
review settled are recorded below, at the exact points where they apply.

## Why this document exists

`STAGE3_PLAN.md` item 6 is the entire content that was frozen about
Phase B:

> **Phase B** — stage-2 driver structure plus kappa_alpha / rank /
> coefficient norms / CNN preservation.

Everything else in this document is new, and is being reviewed here for
the first time. This is worth stating plainly, because the alternative —
treating a short summary as if it carried the same authority as the full
document it summarizes — is a mistake this stage's own history has now
recorded four times, most recently in the very review round that
produced this plan.

---

## What Phase B is

Phase B is a GPU cloud-session driver. It reads Phase A's 60,000-image
encoded artifact, regenerates the corruption and the clean target values
itself, in-session, and runs evolution, ridge, and CNN training at full
scale. It produces no test-split quantity of any kind, and it reads no
test-split object.

It inherits two rules that did not exist yet when stage 2 ran:

- **`consume_validated` is the only way to read data.** `ensure_artifact`'s
  skip branch is itself a validated read, so every resumed step checks
  what it is resuming from. This is enforced by a check that reads the
  syntax tree of every Stage 2B script.
- **The pre-flight check is based on the driver's own source-file list.**
  A GPU target refuses to run when a file in the driver's own list of
  source files differs from the current commit, and it proceeds normally
  when the rest of the code repository is untidy elsewhere.
  `test_every_repo_fetching_gpu_target_runs_the_closure_check` is
  derived automatically, so this new target inherits this safety check
  automatically, on the very day it is written.

---

## Decision 1 — Freeze 4: preconditions for writing data, and what "content-addressed naming" now means

Freeze 4 was agreed to with two parts that had not yet been implemented:
writing data with a precondition attached, and publishing it without a
race condition. The team resolved this as **parts (a) plus (c)**, and
disclosed this openly to ChatGPT at the same time, rather than letting
the pre-test review package be the first place anyone learned that part
of a frozen decision was still unimplemented.

**(a) Preconditions for writing data — implemented.** Every write that
lands a real artifact carries a server-side compare-and-set check:
`if_generation_match=0` to create a new object, or the already-observed
generation number to replace an existing one. A check performed
separately, before a write, is not a real precondition — the gap in time
between the check and the write is exactly where the race condition
lives. The scope of this rule, stated as a test rather than just a
comment: this guard covers the final, composed destination file and its
manifest sidecar file. The chunked upload route's individual *parts*
deliberately carry no such guard, because a resumed upload rewriting a
damaged part is an overwrite done on purpose, and guarding against it
would break the very resume feature it exists to provide. The race
condition Freeze 4 is meant to prevent is two writers replacing each
other's artifact, and the artifact itself — not an individual chunk — is
the actual destination that matters.

This failure mode is a real, live risk, not just a theoretical one: the
resumability design explicitly allows a second cloud session to resume
work after the first one is only *presumed* dead — and "presumed dead,
but actually still uploading" is exactly the situation the checked exit
status of the `stop` command exists to prevent.

**(c) Content-addressed naming — NOT satisfied by part (a) alone.** An
earlier claim that this was satisfied "in substance" was rejected on
review, and replaced with a stronger requirement instead: **the manifest
file is the true commit point.** This is now implemented exactly that
way.

**Write order** — the payload file is written first, under its own
precondition; the exact generation number that write produces is
recorded; the sidecar manifest file is written second, under its own
precondition, recording that exact generation number.

**Read order** — the manifest file is read first; the producer recorded
in it is checked against the consumer's own expected fingerprint
*before any payload bytes are moved at all*; only then is the payload
fetched, **at that exact recorded generation**, never simply "whatever
has this name right now"; and finally, the checksum is recomputed on the
bytes actually fetched.

**The crash window** — a payload file with no manifest is
**UNCOMMITTED**. This is more than just "failing a validity rule": it
means a session that died between the two writes has left behind bytes
that no consumer is allowed to accept, and the next run must not treat
them as a completed step.

**A competing writer** — loses on its precondition check, and it can
never silently replace an artifact that has already been committed.
Both the payload write and the sidecar write are guarded this way.

Pinning to an exact generation is not redundant with the checksum, and
there is a test proving why: a replacement written with the exact
**same bytes**, at a new generation number, still satisfies the
`payload_sha256` checksum check, but still fails the generation pin. The
checksum answers "are these the right bytes"; the generation number
answers "is this the exact object that was actually committed" — and
only the second question is structural, not just about content.

### How long deleted data is retained, measured rather than assumed

Pinning to a generation number protects against race conditions, but it
does not protect the ability to re-fetch an older version later, unless
the storage bucket actually keeps superseded generations around. The
team measured this against the real bucket,
`bonsai-2026-stage2b-cache`, on 2026-08-07, in two ways: reading the
bucket's own declared configuration through the `gcloud` command-line
tool, and running live tests on scratch objects that were deleted
afterward.

| question | finding | how this was found |
|---|---|---|
| can a specific generation of an object be pinned and read back? | **yes** — the team read back 5,366 bytes at an explicit generation number | live test |
| is object version history turned on for the bucket? | **no** | `gcloud storage buckets describe` returns no `versioning` setting at all, and a pinned read of a superseded generation returns a 404 "not found" error |
| does the storage service refuse a stale precondition? | **yes** — it returns a `PreconditionFailed` error | live test |
| are superseded bytes kept at all? | **yes, for 7 days** — `soft_delete_policy.retentionDurationSeconds: 604800` | confirmed via `gcloud`, and by listing objects |
| can a superseded generation be recovered? | **yes, but only by an administrator** — it appears in `list_blobs(soft_deleted=True)` as `<name>#<generation>` | live test |

**Correcting an earlier version of this finding.** An earlier test used
only a pinned read, got a "not found" error, and concluded that
superseded bytes were simply gone. They are not gone: the bucket's soft-
delete policy retains them for seven days. The pinned read returns "not
found" because a normal read does not see soft-deleted objects at all,
not because nothing exists there. That earlier conclusion happened to be
right about what an ordinary *consumer* can do, but wrong about what
actually *exists* in the bucket — and that distinction matters when the
real question is "can we still diagnose what a superseded artifact
looked like, after the fact."

Reads pinned to an exact generation number **are** guaranteed to work,
so the reviewer's fallback proposal — making content-addressed file
names mandatory instead — is **not** needed. The revised statement of
this policy:

> Artifacts are **immutable by policy**, and this is enforced by write
> preconditions. A superseded generation is a **halt-worthy anomaly**,
> detected by either a failed pinned read or a checksum mismatch — it is
> never a supported way to read data. Superseded bytes are nonetheless
> retained for **7 days** by the bucket's soft-delete policy, and can be
> listed and restored by an administrator — so a halt has a bounded
> **forensic window** in which whatever replaced an artifact can still
> be examined. That window exists only as a diagnostic convenience; it
> is not durable, long-term retention, and no code in this project
> depends on it existing.

**Permissions, deliberately left unchanged.** The pipeline's own service
account cannot read the bucket's configuration
(`storage.buckets.get` is denied, with a 403 error). This is the correct
setup, not a gap: the driver itself never needs the bucket's
configuration, and granting this permission just to make one-off manual
audits more convenient would widen the credentials carried by a program
running unattended on a rented cloud machine. Questions about
configuration are instead answered using `gcloud`, under a human's own
identity — which is where they belong.

### The write-once rule — enforced in code, not just described in a document

Pinning to a generation number only guarantees "the consumer reads what
was actually committed" for as long as that pinned generation survives
— and with version history turned off, a later overwrite destroys it
completely. So combining a fixed name with a pinned manifest is only
acceptable **if** the policy for anything a consumer might pin to is
genuinely write-once. This is enforced directly in code:

- **LINEAGE** artifacts — everything scientific; anything that is, or
  could become, a parent listed in some other artifact's manifest. These
  are **create-once**. Passing `force=True` to one of these raises a
  named refusal error *before* the actual computation runs, so a refusal
  never wastes the GPU time it exists to protect. Regenerating one of
  these means writing under a **new name** — exactly as Phase A did,
  writing `encoded_train_s1200` as a new name, rather than overwriting
  `encoded_fit_s1200`.
- **RUN_SCOPED** artifacts — reports and timing summaries. These are
  never treated as a parent of anything, and never pinned.

**Classification defaults to the safer choice when undeclared.** Any
artifact kind that has not been explicitly declared is treated as
LINEAGE by default. A new scientific artifact automatically inherits
create-once protection; falling into the overwrite-allowed category
requires an explicit declaration, and this can never happen simply by
omission. This behavior is proven by an automatic test, not just
assumed to work correctly.

**The chain of parent artifacts is checked all the way through, not just
one link.** Checking only the immediate parent of an artifact leaves
this property only one link deep — a run-scoped, mutable object sitting
three links up the chain is still a mutable thing hiding inside a chain
of checksums that all claim to be fixed. This chain-walk runs both on
the **read** side and the **publish** side, and these two are not
redundant with each other: publishing binds what this code writes;
reading binds what this code is willing to accept, including a manifest
left behind by an older code commit, or created by hand.

**Reports are run-scoped, and there is no overwrite path left anywhere.**
`force=True` used to overwrite one fixed report name per ladder rung, so
a resumed run destroyed the record of what the earlier, failed attempt
had actually seen — the same "an unwritten result does not survive"
failure mode this project already has a documented lesson about. Report
names now include a run ID, so both attempts survive side by side, and
the whole storage model is now uniformly append-only. No artifact kind
currently keeps an overwrite path; if one ever needs one, this plan
requires stating exactly which kind, why replacing history is acceptable
in that case, and confirming that no package-level claim depends on
being able to recover the earlier versions.

**The reuse half of this problem, found when the amended alpha grid
landed.** The write-once rule stops an artifact from being overwritten,
but it does nothing to stop a re-run from silently CONSUMING an artifact
that was computed under a different configuration. Phase B's
`ridge_cv.json` and `ridge_final.npz` sit in the bucket with valid
manifests; `ensure_json` passes no `expected_fingerprint` argument, so
the thirteen-decade re-run would have hit `ensure_artifact`'s skip
branch, silently accepted the nine-decade results, computed nothing new,
and reported `STAGE3_OK` anyway — satisfying the reviewer's "do not
splice" rule only **by accident**, since nothing is spliced together
when nothing is actually computed. This was fixed by deriving the
artifact's name from the grid itself (`ridge_cv_g13_<hex>`), so any
change to the grid automatically changes the file name, and the older
tables survive as history. Deriving names from content, and pinning
expectations with a fingerprint, are complementary: the first stops the
team from writing over history, the second stops the team from reading
the wrong history, and neither one covers what the other one does.

All four negative-path checks here were confirmed by breaking exactly
what they watch, and each broke exactly one test, not any others:
forcing an overwrite of a LINEAGE artifact; a second create attempt
against an already-committed name, which loses its precondition check; a
RUN_SCOPED artifact offered as if it were a parent, tested on both the
publish side and the read side separately; and the default,
fail-closed classification.

### The orphan payload, and how to recover from it

The three refusal checks above are each individually correct, and
together they close off every way out of one particular bad state. A
cloud session that dies between the payload write and the sidecar write
leaves bytes sitting under a LINEAGE name, with no manifest describing
them. On the next run's resume attempt: `consume_validated` refuses to
read it (there is no recorded provenance for it), the create path's
`if_generation_match=0` check fails against the object that already
exists, and `force=True` raises a `WriteOnceViolation` error. **The run
has no automatic way forward.** The team confirmed this by actually
constructing this exact state against a fake test bucket, before Phase B
was even written — not just reasoned about in the abstract — and it is a
real, live risk at 60,000-image scale, where a single run plausibly
spans more than one cloud session.

The recovery tool is `discard_uncommitted(name)`, and using it does not
weaken the write-once rule. **The write-once rule protects the commit
itself, not the raw act of writing bytes.** A lineage artifact is
immutable because its generation number is pinned — by its own manifest,
and possibly also by another artifact naming it as a parent. An orphaned
object has neither of these things pinning it, so nothing can hold a
valid reference to it, and deleting it removes bytes that no consumer
was ever actually allowed to read in the first place. This function
refuses to touch anything that carries a manifest, and it re-reads that
manifest **at the moment it is called**, rather than trusting an earlier
listing, because the write that commits the payload might land in the
gap between listing and deleting.

**This recovery is not automatic, and that is a deliberate decision,
not an oversight.** The driver halts and simply names the recovery
command; it does not run it automatically. An orphaned object means a
session died in the middle of a step, and that is worth a person seeing
directly. More concretely: the resumability design explicitly allows for
a second session resuming after the first is only *presumed* dead — so
if that first session is actually still alive, just running slowly,
automatically discarding its payload would let it later commit a
manifest that points to a generation number that no longer exists,
turning what should be a clean halt into a corrupted lineage that then
needs to be diagnosed by hand.

**It deletes the exact object name only, never a whole path prefix.**
`manifest_object_name` is simply the payload's own name plus a suffix,
so calling `delete_prefix(payload_name, ...)` would match a *committed*
artifact's own manifest too, stripping away its provenance record while
leaving its actual bytes behind. This particular mistake does not appear
in any docstring or any halt message in the code.

The team confirmed this with five deliberate breaks, each one triggering
the one test that names it: refusing to touch a committed artifact;
reading the manifest at call time rather than from an earlier listing;
refusing to touch anything with a manifest name; using an exact-name
delete instead of a prefix-based one; and `ensure_artifact`'s own write-
once check that runs before producing new data. The first two of these
breaks trigger the same pair of tests — the call-time read is what feeds
directly into the committed-artifact check, so this is really one
mechanism covered by two tests, not two fully independent safety checks,
and it is described here that way rather than simply counted as five.

These measurements, and this one deviation, are carried forward into the
pre-test review package.

---

## Decision 2 — running the equivalence check at 60,000 images is a new, cautious extra check, not the original locked gate

`DESIGN.md:330` is worded exactly, and its scope is explicit: *"at both
the 1,000- and 5,000-image stages, JAX and sklearn must produce (a) max
absolute difference in clipped validation predictions <= 1e-8, and (b)
identical alpha selection."* It names only 1,000 and 5,000 images. It
passed at both sizes already, so this original requirement is
**discharged**, fully satisfied. `FINDINGS.md`'s phrase, "the ladder's
third real-data equivalence gate," overstated what was actually frozen
in the design, and this plan does not carry that phrasing forward.

What actually runs at 60,000 images is a **new, cautious extra check**,
at 12 times the largest scale ever verified before — a genuinely
different, more challenging numerical situation — and it checks
**fold-level results only**. This preserves the same literal quantities
the original gate checks (clipped validation predictions, and identical
alpha selection, both still computed per fold), without requiring seven
full-corpus reference fits on a metered A100 GPU.

**Describing this check as "new and extra" describes what the check IS.
It does not soften what a failure of it MEANS**, and the halt rule is
stated exactly as follows:

> Disagreement beyond the frozen tolerance, on any fold or any
> condition — whether in predictions or in alpha selection — **halts
> everything before Stage 4 can run**, and requires full diagnosis.
> *"The original locked gate still passed"* is **not** an acceptable
> reading of a failure in this extra, production-scale check.

This is all stated in advance, so that neither the framing nor the final
verdict can be chosen only after a number already exists. The measured
CPU cost of running the sklearn oracle is reported; whether it runs at
the same time as the GPU work, or runs separately afterward, is left as
an implementation detail.

---

## Decision 3 — the encoder-on-noisy-inputs gate is discharged, with the reasoning recorded

This gate is not re-run at stage 3. There are three reasons, recorded
here so this is clearly a deliberate, reasoned decision, not an
oversight:

1. This gate is, by its own documentation and by the design's own ladder
   structure, a **stage-1 device**. Stage 2 already deliberately chose
   not to re-run it, recording diagnostic numbers instead. No new gate
   is invented in the middle of the ladder, in either direction.
2. The quantity this gate actually guards against — whether the encoder
   converges properly on noisy input — is **independently confirmed at
   60,000 images** by Phase A's own recorded tail measurement: the
   median and the 95th-percentile value are both exactly 0.0, the
   maximum is 2.468e-10, which is four orders of magnitude below the
   solver's relative tolerance of 1e-6.
   **A third reason was struck out, and the correction is recorded here
   rather than silently edited away.** An earlier draft of this section
   gave a third reason: *"under the amended absolute-convergence rule,
   the measured values pass trivially — both medians are literally
   zero."* That claim described a gate result computed from data that
   was never actually collected under the gate's own rule. Only the
   **noisy** median was ever measured at 60,000 images; Phase A only
   encodes corrupted images, no clean 60,000-image encoding exists, and
   so there is no clean median available to form the ratio this gate is
   defined on. The discharge of this gate stands without this third
   reason, resting on reasons (1) and (2) alone.

The standard wording to use, everywhere this decision is referred to:
**"discharged as a stage-1 device; Phase A's noisy final-Delta
distribution is reported as independent convergence evidence."** Never
"passes the amended gate at 60k" — that phrase describes a computation
that nobody actually ran.

---

## Decision 4 — the amendment audit runs after Phase B, against Phase B's own saved artifacts

This runs in a separate session, sequenced **after** Phase B. This is
for isolating failures — bundling the two together would mean one
session's death costs three results instead of one — and there is also
a sharper reason: the audit's 1,200-step inputs are **Phase B's own
saved evolved-feature artifacts**, read through the validated read path.
This means the audit compares 150-step features against the actual
production features — the exact objects the confirmatory result will
rest on — rather than against a fresh recomputation that only ought to
match them.

**This places one explicit requirement on this plan**: evolved theta
(phase) values and features must be saved per graph, with fingerprints
attached. Resumability already requires this anyway, but the audit's own
design makes it essential, not just convenient.

**The 150-step side must differ from Phase A in `encoder_steps` ONLY**
— using the same official image indices, the same corruption draws, the
same folds, the same gauge, the same preprocessing, and the same
fingerprint contract. The audit driver **checks** this directly, by
comparing the two artifacts' recorded fingerprint configurations field
by field, rather than simply assuming it is true. The configuration
already records `encoder_steps`, so the artifacts describe themselves,
and this check is over actual recorded data, not over an unverified
intention.

**The 150-step encode is a named, deliberately authorized addition.**
`encode_stage3_local.py` becomes parameterized by step count — the
fingerprint configuration already records `encoder_steps`, so the
artifact describes itself — producing `encoded_train_s150.npz`, plus a
manifest, under the full contract, run locally.

*Correcting my own earlier estimate*: I originally estimated this at
about 11 minutes, by simply carrying over Phase A's 679-second runtime
unscaled. But the encoder's cost is linear in step count, and it runs
two passes per image, so 150 steps should take roughly **one-eighth of
Phase A's time — on the order of 85 seconds** — plus fixed overhead for
corruption, topology, and loading. Desktop's earlier figure was right,
and mine was wrong, by about a factor of 8.

---

## Facts read directly from the code, and used to block or unblock decisions

**Sizing has not been measured, and working rule 18 forbids assuming it
scales the same way as an earlier measurement.** Features are
`2 * 505 - 2 = 1008` values wide. At n=60,000, in float64: one
condition's matrix is 484 MB, and one fold's SVD input (48,000 by 1008)
is 387 MB, plus additional workspace memory. An A100 GPU has 40 GB of
memory, so this is very likely fine — and "very likely fine" is exactly
the kind of reasoning working rule 18 exists to forbid. Stage 2A
correctly extrapolated its data-generation cost from n=1,000 all the way
to n=60,000, but dismissed classifier cross-validation fitting cost as
"a few seconds" at that same n=1,000 — and that cost went on to dominate
the total runtime by 79 times at full scale.

**Corrected: there are SEVEN conditions, not six.** `ALL_CONDITIONS` has
five entries (`pre_evolution` plus the four evolved graphs), and
`RAW_CONDITIONS` adds two more. Held in memory at the same time, on the
host machine: the five 1008-value conditions take 2,419 MB, `raw_784`
takes 376 MB, and `raw_505` takes 242 MB — **3.04 GB in total**, plus the
target values `Y` at 242 MB. Seven conditions is also the number that
makes `DESIGN.md`'s "42 SVDs" figure come out correctly (7 conditions
times 5 fold-level SVDs, plus 7 final refits), so an earlier note saying
"six conditions, 2.9 GB" was wrong, in a way that the SVD count already
contradicted.

### The dominant cost is CPU time, and it is not the part the sizing probe was originally going to measure

Stage 2's recorded time for its `8_ridge` step was **305.53 seconds at
n=5,000** — the largest single step after encoding. Broken down, rather
than scaled as a single blended rate: for each condition, this step runs
`cross_validate_alpha` (5 JAX SVDs), *and* `ridge_equivalence_check`,
and this second function calls `sklearn_ridge_predict`, which fits
`Ridge(solver="svd")` **once per alpha value** — nine independent CPU
SVDs per fold. Across all seven conditions, that totals **35 JAX SVDs
and 315 sklearn SVDs**, so this step is actually dominated by CPU work,
running on a metered A100 GPU that sits mostly idle for it, at roughly
0.87 seconds per fit.

The SVD of an `(n by 1008)` matrix scales linearly with `n`, for `n`
much larger than 1008, so at 12 times the scale, this ridge step is
projected to take **about 3,700 seconds — roughly an hour of sklearn
computation, on a GPU that is idle the whole time**. Stage 2's own
recorded projection agrees closely with this: 3,299 seconds, projected
to 54,000 images.

**Two consequences follow from this, and only one of them is actually a
decision.**

**`EXEC_TIMEOUT` is not a decision — it is simply fixed by the numbers.**
The projected total time (evolution 492 seconds, features 562 seconds,
ridge about 3,700 seconds, CNN about 1,194 seconds, plus bootstrap
setup, package installation, and a 229 MB download) comes to about 5,900
seconds of compute before overhead. The default timeout of 3,600 seconds
would time out in the middle of the ridge step **even on a perfectly
healthy run**. The stage-3 target overrides this default to **10,800
seconds**.

**The sklearn oracle runs serially, inside the GPU session, and the
resulting roughly $3-4 of idle A100 time is spent deliberately.**
Decision 2 grants this choice exactly as written — *"whether it overlaps
GPU work or runs serially is implementation detail"* — and the artifacts
Decision 4 requires the team to save would technically let this step run
locally afterward, for free. It stays in-session anyway, because
splitting it out would mean Phase B's own report could no longer carry
the equivalence check's verdict, and a halt rule that lands in a
different session from the run it is meant to gate is exactly the kind
of coordination problem `docs/MULTI_AGENT_PRACTICE.md` warns against.
The scope of this check is unchanged — all seven conditions, all five
folds — because Decision 2's halt rule applies to *any* fold or
condition, and narrowing it down to only the primary condition, purely
to save money, would mean reopening an already-settled, frozen decision.

**Mechanical facts, verified directly.** `EVOLVE_CHUNK = 250` divides
60,000 exactly, giving 240 chunks; stage 2's existing check for exact
divisibility remains valid, unchanged. Phase B has **no encode step** at
all: stage 2's own `step4_encode` step becomes a validated read instead,
which changes how later steps are numbered.

### The chain of what gets read, checked against the real storage bucket

This check is read-only, and was done before anything was actually
provisioned, because a missing manifest discovered at step 1b would
otherwise halt the run *after* the team had already paid for a GPU
session. Of 48 payload files under `stage2b/` in the bucket, **only
five carry manifests**: the four staged KMNIST IDX data files, and
`encoded_train_s1200.npz`. Everything under `stage1/` and `stage2/`
predates the fingerprint contract entirely.

`encoded_train_s1200.npz` — the one input that matters most — does have
a manifest. It records a `payload_generation` value of
1786105203101186, and a `payload_sha256` value of `8a26353d…`, declares
no parent artifacts, and **passes `verify_lineage`**. Step 3's validated
read works exactly as this plan assumes it will.

**Correcting this plan's own earlier statement about step 1b.** An
earlier version of this plan said stage 1's `topologies.npz` file is
read under a `CONTENT_ONLY` policy. That describes a check that cannot
actually run here: `CONTENT_ONLY` is a *fingerprint* policy, and with no
manifest at all, there is no fingerprint to compare, under any policy.
The premise was wrong, not the intent behind it. What actually runs
instead is `require_manifest=False` — the named, pre-contract opt-out
already described above, the same one stage 2 uses for its own cross-
rung corruption read — plus a **payload checksum pinned directly as a
constant in the driver's own code**:

| object | pinned sha256 |
|---|---|
| `stage1/common/topologies.npz` | `f671e63cc00b1612db0da5976c14b8880e4c4f90ae7fb192297721665f1907a4` |
| `stage2/common/corruption.npz` | `cd9ac32357f6aeefbaca006fb97cd53c52273f4acca4960115a72fc2983438cf` |

**A pinned checksum is not the same thing as provenance, and it is not
labeled as one.** It says "these are the exact bytes stage 2 read,"
which is a weaker claim, but a true one. It says nothing at all about
what code produced those bytes. Retrofitting manifests onto already-
completed ladder rungs is refused, for the same reason the manifest
contract already gives elsewhere: it would fabricate provenance that
never actually existed, rather than accurately recording it.

---

## Step structure

This deliberately mirrors `run_ladder_stage2.py`: the same composition
rules, the same resumability behavior, the same rules for when to halt.
Differences from stage 2 are marked below.

| # | step | notes |
|---|---|---|
| — | bootstrap | pinned-commit clone; pre-flight check based on the driver's own source-file list |
| — | `stage_kmnist` | validated read; manifests are now present |
| 0 | pre-flight | 64-bit precision enabled before `jax.numpy` binds; a device report is printed |
| 1 | corpus | **changed** — all 60,000 official indices, in ascending order; Freeze 2's role flags are carried along |
| 1b | topologies | **amended** — reads stage 1's `topologies.npz` under `require_manifest=False`, plus a pinned payload checksum; see above |
| 2 | corruption | **changed** — regenerated in-session, for all 60,000 images; checked to be **bit-exact** against stage 2's stored rows, matched by official index |
| 2b | **sizing probe** | **new, amended** — measures one condition and one fold, with **both legs measured and projected separately**: the JAX SVD leg, and the sklearn oracle leg. Halts against pre-specified budgets before the full run proceeds |
| 3 | encoded input | **changed** — a validated read of `encoded_train_s1200.npz`; its manifest identity (payload checksum, fingerprint commit) is recorded into the run report; one image is re-encoded and spot-checked at **ULP tolerance**, never requiring an exact match |
| 4 | restrict | unchanged; the artifact is already restricted to 505 values |
| 5 | evolution | batched, across all four graphs, with success-flag checks, in 240 chunks of 250 images each, plus a per-graph CPU reference check; **theta values saved per graph, with fingerprints** (Decision 4) |
| 6 | features | six conditions, gauge-fixed at reference node 363; **saved per condition, with fingerprints** (Decision 4) |
| 7 | ridge | 5-fold cross-validation, nine alpha values, seven refits; the fold-level oracle extension (Decision 2); additional diagnostics described below |
| 8 | CNN | 54,000 for fitting plus 6,000 locked validation, the locked configuration, three random seeds, selection by clipped validation loss, plus preservation items |
| 9 | stats smoke test | in-sample, non-inferential, matching stage 2's approach |
| 10 | report | named items reported on their own terms |

Every artifact passes through `ensure_artifact` with a `fingerprint=`
argument, so publishing an artifact and uploading it happen as one
atomic action, and a forced overwrite always republishes a fresh
manifest, rather than leaving an orphaned sidecar file behind.

### The sizing probe's budgets, fixed here before any measurement exists

These are committed both in this document and as constants in the
driver's own code, so the verdict cannot be chosen only after a number
already exists — the same discipline Decision 2 applies to the
equivalence extension. The probe measures one condition and one fold,
and projects **each leg using its own separate multiplier**; a single
blended rate for both legs together is exactly what working rule 18
forbids.

| quantity | what the probe measures | multiplier | budget |
|---|---|---|---|
| JAX SVD wall-clock time | one `svd_ridge_fit` call, at 48,000 by 1008 | **x42** (35 fold-level SVDs plus 7 final refits) | — |
| sklearn oracle wall-clock time | one `Ridge(solver="svd")` fit, at a single alpha value | **x315** (7 conditions times 5 folds times 9 alpha values) | — |
| **projected total ridge time** | the sum of the two rows above | | **<= 7,200 seconds** |
| **projected total run time** | ridge time plus the measured or projected time for other steps | | **<= 9,000 seconds** |
| **peak device memory** | the highest JAX memory usage recorded during the probe fold | | **<= 12 GB** |

Here is the reasoning behind the memory budget, so it can be argued with
directly rather than just obeyed: at 48,000 training rows,
`svd_ridge_fit` holds `X` (387 MB), `U` (387 MB, using
`full_matrices=False`), `Y_tilde` (194 MB), `Vt` (8 MB), `W` at a shape
of `(9, 1008, 505)` (37 MB), and `Z` (4 MB) in memory at once — about
**1.02 GB of named arrays**, plus additional GPU solver workspace memory
this simple model does not attempt to predict. The budget is set at
**8 times** this modeled figure, which is enough to catch a modeling
error of roughly an order of magnitude, while still leaving a healthy
run room for a workspace larger than the simple model predicts. This
budget is set well below the GPU's full 40 GB capacity on purpose: a
threshold set at 30 GB would let almost any wrong model pass, and gate
nothing useful at all.

The wall-clock time budgets carry margin above the 12-times projection
(about 5,900 seconds), rather than being set exactly equal to it,
because that projection is itself the thing being tested. **Exceeding
either budget halts the run before the full run proceeds**, and this
halt is the probe correctly doing its job, not an obstacle to work
around.

### The two spot-checks, each with a tolerance matched to what it actually measures

- **Corruption regeneration versus stage 2's stored rows: BIT-EXACT.**
  This uses pure numpy, with the PCG64 random-number generator, keyed on
  `SHA256(split:index:42)`, which does not depend on chip type. Anything
  less than an exact match here would be a real, genuine defect.
- **One image, re-encoded, versus the stored Mac-generated array: ULP
  tolerance, never an exact match.** Phase A recorded a maximum
  difference of 3 ULP (4.441e-16) between ARM-chip and x86-chip
  encodings of the same images, with results that were bit-exact within
  each chip type in both directions tested. An exact-match comparison
  here would fail even on a perfectly healthy run, for a reason the team
  already understands and has already documented.

### Additional ridge diagnostics

The mechanism for this was already settled in `STAGE3_PLAN.md`'s own
analysis: the team computes per-fold quantities **for all nine alpha
values, and indexes by column afterward, once the winning alpha is
known**, rather than saving every individual fit (which would need about
180 MB at stage 3's scale). `s_max` and `s_min` (the largest and
smallest singular values) are recorded as absolute magnitudes, not just
as a ratio — because a ratio cannot be un-divided back into its two
original parts, which is why `fold_cond` (the ratio alone) was not
enough on its own. The following are recorded per fold: `kappa_alpha`
(a scale-aware condition number), a scale-aware numerical rank, the
norms of the fitted coefficients, flags marking whether the selected
alpha sat at a grid boundary, the full validation curves, and the
centering margins under the tolerance that scales with `n`.

**A note on convention is required inside the artifact itself**: which
alpha column each per-fold quantity was indexed at, and under which
alpha regime (fixed or reselected). Without this note, a reader cannot
tell a fixed-alpha row apart from a reselected-alpha row.

### Debt restated here, so it does not get lost a second time

`AUDIT_PROTOCOL.md` requires that the per-fold mean of the new per-image
out-of-fold MSE values must reproduce **the stored fold-aggregate values
from the stage-1 and stage-2 run artifacts themselves**. The bit-exact
cross-check already built into `oof_per_image_mse` only discharges a
weaker, Tier-1 version of this requirement, not the full protocol
requirement. That full requirement belongs to the audit driver, not to
Phase B.

---

## Disclosed differences from how stage 2 behaved

These are recorded here after the fact, rather than quietly repaired
without comment, which is this project's standard way of handling a
past non-disclosure.

### The stats-smoke-test budget guard was dropped, and dropping it was not recorded at the time

Stage 2 gated its statistics smoke test on a projected cost:
`STATS_SMOKE_BUDGET_S = 60.0`, and the step was skipped, with a reason
written into the run record, whenever a linear projection from stage 1's
measured 10.86 seconds at n=1,000 exceeded this budget. The stated
reason for allowing this skip was that stage 1's own run had already
proven the connection between ridge output and the statistics code.

`run_ladder_stage3.py` has no such guard at all: its `step9_stats_smoke`
step now runs unconditionally. At n=60,000, this step took **776.6
seconds — 18% of the run's 4,297.2-second total** — for a step that is,
by its very design, in-sample, training-side, and not a statistical
inference, and whose own output artifact carries a banner saying exactly
that.

**The change in behavior, and the failure to disclose it, are two
separate issues, and only one of them is the actual defect.** Running
this smoke test at full production scale is a defensible choice on its
own: it exercises the machinery at the actual scale it will really be
used at, which is worth something the stage-1 check alone does not
provide. Its result stands, and no scientific claim depends on it. What
was wrong is that a driver silently stopped doing what the previous
rung had done, and this change was recorded in neither this plan nor
the commit that introduced it — so nobody reviewing either one would
have known to look for it. The team caught this themselves, by reading
the timing numbers at the end of the run, which is later than it should
have been found, and finding it late is not a substitute for having
disclosed it up front.

**Forward rule: whether to run this smoke test, or skip it, is an
explicit decision made per session.** Any later driver — including the
audit session — must state which choice it makes, and why, rather than
silently inheriting either behavior just because it was copied from
whichever earlier rung it started from.

## Verification

- Every new safety check is confirmed by deliberately breaking the thing
  it watches. Specifically: the corruption cross-check must fail on a
  deliberately altered row; the ULP spot-check must fail on a genuinely
  wrong image, and pass on the correctly stored one; and the sizing
  probe must actually be able to halt the run.
- Resumability is proven the way this project always proves it: kill the
  cloud session in the middle of a run, and confirm the next run resumes
  having lost at most one step's worth of work — now with the added
  property that each resumed step also validates what it is resuming
  from.
- Session conventions are unchanged: `EXEC_TIMEOUT`, a 60-second
  heartbeat, the sentinel plus exit code, and unconditional teardown
  with the `stop` command's status always checked.
- The full test suite must be green before each commit.

## What this plan does not do

This plan does not touch stage 4 in any form: no test-split object is
read, written, or even named here. The amendment audit and the ARM/x86
stress-test run are both separate sessions, run after Phase B. Pre-test
item 9 (the frozen Stage 4 command) stays deferred until Phase B's own
object paths actually exist, since it depends on names this plan only
proposes, and has not yet fixed.
