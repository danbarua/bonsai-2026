# Stage 2B: fingerprint mechanism, Phase A regeneration, protocol freezes, Phase B

**Status: committed plan of record for ladder stage 3.** Reviewed by
Claude Desktop and ChatGPT across eight rounds; conditional consensus
reached 2026-08-06, conditional on the five freezes at the end of this
document being committed (they are, here and in `AUDIT_PROTOCOL.md`).
Execution additionally requires Dan's explicit release.

## Context

Stage 3 Phase A completed (54,000 fit-side images encoded locally, 7.9
min, artifact in GCS). A six-round ChatGPT review then blocked Phase B
behind provenance work and blocked Stage 4 behind an expanded pre-test
package. Claude Desktop consolidated that exchange into a briefing
(`inbox/2026-08-06T18-08-49Z.md`) headed **"Review has now CONVERGED"**
with a seven-item programme.

I verified that briefing against the eleven archived review rounds
before planning against it. It is substantially faithful, but the
convergence claim is false and three of its items are superseded.

---

## How a late twelfth message changed the programme (resolved)

`.claude/claude2gpt/archive/2026-08-06T18-10-40Z.md`. When this plan was
first drafted the message had not yet been processed on either side; it
has since been read, archived, and its four points adopted. Recorded
here because it is the reason three of the briefing's seven items were
superseded, and because the chronology was initially reported wrongly by
me.

**Chronology, corrected — my first draft got this backwards and it
matters.** Verified from the timestamps: Desktop→ChatGPT 18:08:02Z,
Desktop→me (the "CONVERGED" briefing) 18:08:49Z, ChatGPT→Desktop
18:10:40Z. The twelfth message arrived **111 seconds AFTER the briefing**,
so it did not exist when the briefing was written. My draft claimed it
predated the briefing, which would have made the omission negligent
rather than merely premature — an unfair characterisation, and wrong.
The error came from an exploration agent's summary that placed 18:10:40Z
"after 18:08:02Z and before 18:08:49Z" (arithmetically impossible); I
repeated it without checking the subtraction. Desktop caught it.

What survives the correction: the convergence claim was premature, and
three of the briefing's seven items are superseded by a message that now
exists. The durable process rule is Desktop's sharper version, not the
one I proposed: **convergence is declared by the reviewer, never the
responder; and both inboxes are checked immediately before any
consolidation send.**

It contradicts the briefing on three of seven items and adds a fourth
requirement:

| ChatGPT 18:10:40Z | Briefing |
|---|---|
| "full training" = **60,000**; choosing 54k now "is a post hoc amendment, not a silent clarification" | item 5 recommends **54k** |
| Runtime `sys.modules` traversal "is not a complete source-identity guarantee"; wants **static import closure + runtime closure**, established before generation **and revalidated after** | item 1 specifies "(Import-graph traversal **at runtime**, filtered to repo paths.)" |
| **Either** alpha regime triggers review: "The alpha mechanism determines interpretation, not whether review occurs" | item 3: fixed-alpha decisive, reselected-alpha a "named review item" |
| **NEW**: the 150-vs-1200 audit must use **out-of-fold predictions under the frozen folds**, paired per-image OOF MSE; in-sample MSE from a full refit is "weak and potentially misleading" | item 3 says only "Ridge MSE at both budgets" |

All four were adopted: 60,000 (Freeze 2), static+runtime closure
(Q4), either-regime triggers (Freeze 1/Q3), and OOF predictions (Q5).

## The 54k-vs-60k question, resolved against the design text — and it goes against my own earlier position

Desktop delegated this to me as a named open question, recommending 54k.
I encoded 54k in Phase A on that reading, and wrote in
`encode_stage3_local.py`'s docstring that encoding the 6,000 "would
produce an artifact nothing reads." **On the evidence I now think that
was wrong**, and two of my exploration agents split on it, so here is the
raw evidence rather than a verdict:

- **`DESIGN.md:479`** defines the exact term: *"at stage 3, `"full
  training"` = 54,000 fit + 6,000 locked validation."* The term is
  quoted and defined as the composite. `DESIGN.md:311` then says *"7
  **full-training** SVDs for the final ridge refits."*
- **`DESIGN.md:492`**, the compute table: *"42 SVDs (35 fold-level + 7
  final refits, **~48-60k x 1008**)"*. 48,000 = 0.8 x 60,000 — exactly a
  5-fold CV training portion of 60k — and 60,000 is the refit. Under 54k
  the cell would read ~43-54k. Both numbers are internally consistent
  only under 60k. **Neither Desktop nor ChatGPT cited this line.**
- Against: `DESIGN.md:544` uses "training side" for 60k and "fit" for
  54k, and a refit is a fit; plus CNN comparability. But that is an
  inference from role labels, against an explicit definition of the
  literal term plus corroborating arithmetic.

**Decided: 60,000.** Desktop withdrew its 54k recommendation as "a
comparability inference I invented against locked text I had not
re-read"; ChatGPT's independent reading agreed. ChatGPT's methodological
point is the sharper one and is recorded as the reason this was decided
explicitly rather than by convenience: resolving it *after* Phase A had
already encoded 54k, in the direction that avoids re-work, would have
been a post-hoc choice either way.

**Consequences, all adopted:**
1. Phase A must be re-scoped to encode the 6,000 validation images too
   (~1 extra minute of encode; the regeneration was happening anyway).
2. The regeneration's acceptance test changes shape — the new artifact
   is not a byte-comparison against the old one, because it covers a
   different population.
3. `encode_stage3_local.py`'s "Scope" docstring is wrong and must be
   corrected, not quietly edited.
4. The deferred 6k / full-60k final-Delta tail counts (briefing item 4,
   "if/when those images are encoded") become mandatory immediately.

## A second internal contradiction in the briefing

Item 3 requires the 150-vs-1200 audit at **"full 60k both budgets"**.
Item 4 states the 6k validation subset **"has NEVER been encoded"**.
Both cannot hold. Traced to origin: Desktop's first message to ChatGPT
(`archive/2026-08-06T12-28-20Z.md`) described Phase A as *"full
60,000-image encoding"* — it was 54,000. ChatGPT built the audit scope on
that error; Desktop corrected the population at 18:08 but not the scope
the error had produced. Resolving 54k-vs-60k resolves this too.

## Dropped in consolidation — restore before freezing anything

- **The audit trigger lost two of its three conditions.** ChatGPT
  specified, and Desktop relayed verbatim in addendum #3, three
  triggers: primary-contrast sign reversal, any graph changing
  improvement→deterioration, material graph-order reversal. The
  briefing's item 3 encodes only pairwise order reversal. ChatGPT
  re-listed all three at 18:10 — live, not stale.
- **A FINDINGS correction that never reached me at all**: ChatGPT
  (17:26) — *"median and p95 ratios do not by themselves prove a
  whole-distribution shift."* Desktop conceded it to ChatGPT 29 seconds
  after sending addendum #2, so it appears in neither.
- Two of four ABS_CONV_EPS justification axes ("the encoder
  implementation", "downstream feature sensitivity"); `s_min`/`s_max` as
  recorded fields; "masks" from the separately-hashed inputs list.

---

## Verified blocking facts (read from the code, not assumed)

**Phase A's artifact carries no provenance whatsoever.** Its
`summary_json` records scientific params and timings only — no commit,
no source hashes, no environment. Confirmed by loading it.

**Phase A ran from a dirty tree.** `encode_stage3_local.py` was
uncommitted when the run launched (committed afterwards in `55a6fea`).
Narrowed per Freeze 5, because the original wording overclaimed: the
file has exactly one commit ever and its working-tree sha256 matches
`55a6fea`'s byte-for-byte, which establishes that the CURRENT bytes match
the later commit — **not** that the bytes which ran were those bytes.
Without a launch-time source hash there is no artifact proving it, and my
own recollection of not having edited the script between launch and
commit is not evidence. What is *not* provable from any
record is which versions of the other participating modules were live.
That gap is precisely what the regeneration closes empirically.

**GCS custom metadata is entirely unused.** `blob.metadata` is never
set, read, or passed anywhere in `stage2b_gcs.py`; the uploaded artifact
returns `metadata: None`. Also `md5_hash: None` — confirming the
composite/chunked path — so the fingerprint must not depend on MD5.

**Every write constructs `bucket.blob(name)` inline and discards it**
(`upload_file:673`, `_compose:925`), so there is currently *nowhere* to
attach metadata without holding those handles. `download_file:1091` is
the only place a handle survives two operations.

**`ensure_artifact`'s trust point is a single line** —
`stage2b_gcs.py:1299`, `if not force and object_exists(...)`. That is
where a consume-time fingerprint check belongs. Two consequences:
`force=True` bypasses it entirely (and `step11_report` uses
`force=True`), and `stage_kmnist` calls `download_file` **directly**,
bypassing `ensure_artifact` altogether — a gate placed only in
`ensure_artifact` does not cover it.

**A commit-keyed fingerprint would refuse legitimate cross-stage
reuse.** `step1b_topologies` deliberately consumes *stage 1's*
`topologies.npz`, and `stage_kmnist` consumes IDX files staged under
stage 1 — both written under different commits and configs. Which
fields participate must be selectable per artifact kind, or every
stage-2/3 run refuses its own inputs.

**`StepResult.summary()` is pinned by exact dict equality**
(`tests/test_stage2b_gcs.py:831`) — adding a field is a mandatory test
edit, not optional.

**The closest existing idiom to copy** is the chunked-upload checkpoint
(`_checkpoint_state:819`, `_load_checkpoint:853`): a versioned format
string, compare every field, any disagreement discards wholesale.

## The audit tolerance is specified at the wrong level — with the numbers

The frozen spec says tolerance = M x (max recorded audit-independent
cross-implementation discrepancy). I computed both readings from the
stored artifacts:

| level | value | source |
|---|---|---|
| per-coordinate prediction (`max_abs_clipped_pred_diff`) | **1.1512e-12** | stage-1 `rewired` |
| per-graph evolved-minus-pre **contrast** | **1.3878e-17** | stage-2 `lattice`; three of four graphs exactly 0.0 |

**~83,000x apart.** The audited quantity is the contrast, so the
contrast-level number is the matched one — and it is directly computable
from `mean_clipped_val_mse_jax` / `_sklearn`, already stored in
`ridge_cv.json`. The prediction-level reading gives a *looser* tolerance,
meaning more reversals dismissed as dust. Both sit far below the ~2-4e-3
signal scale so the trigger works either way, but only one is derived at
the level of the thing being tested. Recommend contrast-level, with the
literal reading recorded as the rejected alternative.

Worth noting the trigger is not hypothetical: stage-2 CV already shows
both signs (`T` -2.30e-3, `lattice` -3.90e-3 improving; `rewired`
+4.02e-3, `curr_random` +2.38e-3 worsening).

## Ridge diagnostics need a real change, not just reporting

`cross_validate_alpha` keeps only `fold_cond` — the **ratio**
`s_max/s_min`. `kappa_alpha` needs `s_max` and `s_min` as absolute
magnitudes, and a ratio cannot be un-divided. The per-fold `fit` object
(carrying `singular_values` and `W`) is rebound each iteration and
discarded at `stage2b_ridge.py:403-411`. Also: the selected alpha is not
known inside the loop (`select_alpha` runs after it), so per-fold
quantities must be computed for all nine alphas and column-indexed after,
or the fits retained (~180 MB at stage 3). Recommend the former.

---

## Proposed sequencing

Ordered by what blocks what. All five contested points are now resolved;
the remaining gate is ChatGPT's review of this plan plus Dan's release.

1. **Fingerprint mechanism** — `stage2b_gcs.py` + tests. Static ∪ runtime
   import closure, established pre-generation and revalidated after.
   Per-kind field selection required (cross-stage reuse). Consume-time
   check placed to cover both the `force=True` bypass and the direct
   `download_file` path.
2. **OOF prediction support in `stage2b_ridge`** — module-first, own
   tests, cross-checked against the stored fold aggregates. Blocks the
   audit driver, not the regeneration, so it runs in parallel with (1).
3. **Phase A regeneration at 60k** under the contract. 54k arrays
   payload-compare against the existing artifact; 6k arrays fingerprinted
   at birth with tails computed; acceptance report states the split.
   Depends on (1).
4. **Protocol-document freezes**, each committed before its results are
   inspected — all three trigger conditions, either alpha regime,
   contrast-level tolerance with both required notes, gauge-fixed
   metrics, deterministic stress set.
5. **FINDINGS corrections** — three labeled rationale categories, the two
   adopted sentences, the whole-distribution concession, final-Delta
   reporting discipline, corrected populations. Independent; can start
   immediately.
6. **Phase B** — stage-2 driver structure plus kappa_alpha / rank /
   coefficient norms / CNN preservation.
7. **Pre-test package** — 10 items + fingerprint evidence + protocol
   results + negative-path table. Item 9 (frozen Stage 4 command) is the
   only genuinely new deliverable; draft early.

**Delegation** (worktrees, matching the ladder-stage-1 plan's pattern):
FINDINGS corrections, the protocol documents, and the negative-path
evidence table are independent and parallelizable. The fingerprint
mechanism, the regeneration, and Phase B stay undelegated.

**Negative-path evidence already exists for four of the reviewer's five
demands** — `test_ladder_missing_sentinel_fails_even_on_a_zero_exit`,
`test_a_corrupted_download_raises_naming_the_object_and_both_digests`,
`test_an_object_with_no_recorded_digest_is_refused_rather_than_trusted`,
`test_a_leak_never_masks_the_scientific_verdict`. Only stale-artifact
refusal is new. The package needs a citation table, not new work.

## Verification

Every new guard confirmed by breaking what it watches and observing the
specific expected failure, per this project's standing corollary — the
fingerprint refusal, the per-kind field selection (must still accept
stage-1 topologies), the `force=True` bypass, and the `download_file`
path that skips `ensure_artifact`. Payload comparison verified by
deliberately perturbing one array. Full suite green before each commit.

## The five questions — RESOLVED (Desktop, 20:23:04Z)

All five answered; ChatGPT's plan review still pending, and nothing
executes until it lands and Dan releases.

**Q1 — 60,000.** `DESIGN.md:479`'s definition plus `:492`'s arithmetic
governs; ChatGPT's independent reading agrees; equal sample counts
across unlike model classes were never a design requirement. Desktop's
54k recommendation withdrawn as "a comparability inference I invented
against locked text I had not re-read."

*Acceptance-test shape-note, folded in*: the regeneration now spans two
populations with different evidentiary status. The **54k arrays**
payload-compare bit-exact against the uploaded artifact. The **6k arrays
are new** — no prior artifact exists to compare against, so they are
fingerprinted at birth, and their final-Delta tail is computed
immediately alongside the full-60k tail. The acceptance report must
state that split explicitly rather than letting one "regeneration
passed" blur two different kinds of evidence.

**Q2 — SUPERSEDED by Freeze 1 (see below). My contrast-level
recommendation was wrong and the reason is instructive**: I read "three
of four graphs at exactly 0.0" as thin support for the maximum. The
correct inference is that exact zeros in a difference of two
independently-computed aggregates are the *symptom of cancellation*, not
evidence of agreement — both implementations' errors correlate on the
same images. Level-matching a cancellation-prone empirical aggregate does
not make it a reliable resolution estimate. Replaced by an analytic bound
propagated through the metric; the empirical 1.3878e-17 survives as a
secondary check only. Full derivation in `AUDIT_PROTOCOL.md`.

**Q3 — all three trigger conditions restored**, under **either** alpha
regime: primary-contrast sign reversal; any graph changing
improvement→deterioration; numerically resolved pairwise order reversal.
The alpha mechanism determines interpretation, not whether review occurs.
The fixed-decisive / reselected-review-item split is superseded.

**Q4 — static import closure ∪ runtime closure**, established before
generation and revalidated after, every exclusion frozen and justified.
Runtime-alone misses conditional, lazy and branch-specific imports —
precisely the omitted-helper failure mode the check exists to close.

**Q5 — yes, OOF per-image predictions under the frozen folds.** This is a
real extension to a verified module, so it gets module-first treatment:
implemented with its own tests before any audit driver calls it. It
carries a strong built-in cross-check — the per-fold mean of the new
per-image OOF MSEs must reproduce the already-stored fold-aggregate
values, pinning new machinery against numbers this project already
trusts. Full-refit in-sample MSE is descriptive only.

All implementation findings above (kappa_alpha approach, per-kind
fingerprint fields, the `force=True` and direct-`download_file` bypasses,
the `StepResult.summary()` test edit) endorsed as written. FINDINGS
restorations confirmed: the whole-distribution concession, both missing
ABS_CONV_EPS justification axes, `s_min`/`s_max` as recorded fields, and
masks in the separately-hashed inputs list.

---

## The five freezes (ChatGPT, conditional consensus 2026-08-06)

Execution through fingerprint implementation, Phase A regeneration, the
amendment audit, and Phase B is **consented once these are committed**.
Mismatch gates halt automatically; successful completion of planned steps
creates **no** additional discretionary review. Stage 4 stays blocked
behind the package review and explicit release.

**Freeze 1 — audit tolerance: analytic, not empirical.** My Q2 verdict
overruled; see above for why the argument is better. `d = 1.151190e-12`
propagated through the metric: `|Delta MSE| <= 2d + d^2` (predictions and
targets both in [0,1], so `|q-y| <= 1`). Contrast `4d + 2d^2`; pair
ordering `4d + 2d^2` if the shared `pre` term cancels algebraically, else
`8d + 4d^2`. Safety factor `M = 100` frozen before any result →
**4.604761e-10** (contrast), **9.209522e-10** (4-term pair). Sign
convention frozen: `Delta_g = MSE_evolved_g - MSE_pre`, negative =
improvement. **Removed**: the prospective "insensitive across 5 orders of
M" claim — stage-2 signal sizes do not establish unseen audit effect
sizes; robustness is reported *after* the audit. Frozen in
`AUDIT_PROTOCOL.md`.

**Freeze 2 — population and index identity.** Roles: official corpus
60,000; CNN weight-fit 54,000; CNN validation 6,000; **ridge CV and
final-fit 60,000**. The 6k are held out from CNN gradient updates only,
not from training-side analysis. Regenerated artifact carries official
KMNIST indices plus CNN fit/validation membership flags. All old-vs-new
comparison is **by official image index, never positional prefix**.

*Gate already verified, before any execution*: Phase A's corruption RNG
consumed **official** dataset indices, not fit-local `0..53,999`.
Confirmed empirically by recomputing `epsilon_for` under both hypotheses
and comparing against the stored encoded array at rows 0 / 27000 / 53999
— official matches all three, fit-local fails rows 27000 and 53999. (Row
0 matches both, since official index 0 equals fit-local 0; testing that
row alone would have been vacuous.) No halt.

**Freeze 3 — fixed-alpha defined.** Fixed := the alpha selected from the
production 1,200-step representation per condition via the frozen 5-fold
procedure, applied identically to both budgets' OOF fits. Reselected :=
independent per budget. Report both, plus the production alpha.
Per-budget fold-fitted scalers retained (production preprocessing), so
fixed-alpha isolates **the effect of alpha reselection** — not raw
representation change completely. Stated honestly in the protocol;
shared-scaler comparison optional secondary.

**Freeze 4 — atomic payload + manifest.** Immutable/content-addressed
naming; manifest carrying payload SHA-256, dtype, shape, ordering/index
hash, producer fingerprint, parent hashes, GCS object generation;
consumer-side SHA-256 recomputation; validity only when payload and
manifest both exist and agree; publication via temp-object-then-promotion
or generation-match preconditions (no overwrite races); and **one central
validated consume path** used by `ensure_artifact`, `force=True`
workflows, report steps, and `stage_kmnist` alike — raw download bypass
mechanically prohibited or detected. Into the module docstring verbatim:
*"An object merely existing is never sufficient evidence that it is
resumable."*

**Freeze 5 — the dirty-tree claim, narrowed.** My "the script that ran is
provably the committed one" overclaimed: without a launch-time source
hash, matching bytes now proves only that CURRENT bytes match the later
commit. My own session memory that I edited no code between launch and
commit is not evidence. State only: (a) the recovered script matches the
later commit; (b) provenance for the other participating modules is
unavailable; (c) payload identity under the clean fingerprinted
regeneration is strong *empirical* evidence of computational
equivalence; (d) **the regenerated artifact, not the old one, is the
authoritative Phase A input going forward.**
