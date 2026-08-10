Simplified Technical English version of `experiments/stage2b_denoising/STAGE3_PLAN.md`.

# Stage 2B: fingerprint mechanism, Phase A regeneration, protocol freezes, Phase B

**Status: this is the committed plan of record for ladder stage 3.**
Claude Desktop and ChatGPT reviewed it across eight rounds. Conditional
agreement was reached on 2026-08-06, on the condition that the five
frozen decisions at the end of this document get committed. They are
committed, both here and in `AUDIT_PROTOCOL.md`. Actually running this
plan additionally requires Dan's explicit release.

## Context

Stage 3 Phase A was already complete: 54,000 fit-side images encoded
locally, in 7.9 minutes, with the artifact saved in GCS. A six-round
ChatGPT review then blocked Phase B behind further provenance work, and
blocked Stage 4 behind an expanded pre-test review package. Claude
Desktop then combined that whole exchange into a briefing document
(`inbox/2026-08-06T18-08-49Z.md`), headed **"Review has now
CONVERGED,"** with a seven-item programme of work.

I checked that briefing against all eleven archived review rounds
before planning against it. It is largely faithful to those rounds, but
its claim of convergence is false, and three of its seven items are
superseded.

---

## How a late twelfth message changed the programme (now resolved)

`.claude/claude2gpt/archive/2026-08-06T18-10-40Z.md`. When this plan was
first drafted, this message had not yet been processed by either side.
It has since been read, archived, and its four points adopted. This is
recorded here because it is the reason three of the briefing's seven
items were superseded, and because I first reported the timing of these
events wrongly.

**Chronology, corrected — my first draft got this backward, and it
matters.** Verified from the timestamps: Desktop sent its message to
ChatGPT at 18:08:02Z; Desktop sent the "CONVERGED" briefing to me at
18:08:49Z; ChatGPT sent its reply to Desktop at 18:10:40Z. The twelfth
message arrived **111 seconds AFTER the briefing**, so it did not exist
yet when the briefing was written. My draft claimed it predated the
briefing, which would have made the briefing's omission of it negligent,
rather than simply premature — an unfair description, and a wrong one.
This error came from an earlier exploration agent's summary, which
placed 18:10:40Z "after 18:08:02Z and before 18:08:49Z" — a claim that is
arithmetically impossible. I repeated that error without checking the
subtraction myself. Desktop caught the mistake.

What survives this correction: the claim of convergence was premature,
and three of the briefing's seven items are superseded by a message that
now exists. The rule this leaves behind for the future is Desktop's
sharper version, not the one I originally proposed: **convergence is
declared by the reviewer, never by the person being reviewed; and both
message inboxes are checked immediately, before sending any summary that
consolidates them.**

This later message contradicts the briefing on three of its seven
items, and adds one new requirement:

| ChatGPT, 18:10:40Z | Briefing |
|---|---|
| "full training" means **60,000** images; choosing 54,000 now "is a post hoc amendment, not a silent clarification" | item 5 of the briefing recommends **54,000** |
| checking `sys.modules` at runtime alone "is not a complete source-identity guarantee"; requires **both a static import list and a runtime import list**, established before generation, **and re-checked after** | item 1 of the briefing specifies only "(Import-graph traversal **at runtime**, filtered to repo paths.)" |
| **Either** alpha regime triggers a review: "The alpha mechanism determines interpretation, not whether review occurs" | item 3 of the briefing treats fixed-alpha as decisive, and reselected-alpha only as a "named review item" |
| **NEW REQUIREMENT**: the 150-vs-1200 audit must use **out-of-fold predictions, under the frozen folds**, with a paired per-image out-of-fold MSE; a full in-sample refit MSE is "weak and potentially misleading" | item 3 of the briefing says only "Ridge MSE at both budgets" |

All four of these points were adopted: 60,000 images (Freeze 2), both a
static and a runtime import list (Question 4), either alpha regime
triggering review (Freeze 1 / Question 3), and out-of-fold predictions
(Question 5).

## The 54,000-versus-60,000 question, resolved against the design's own text — and against my own earlier position

Desktop had delegated this question to me, and recommended 54,000. I
encoded 54,000 images in Phase A on that basis, and I wrote, inside
`encode_stage3_local.py`'s own documentation, that encoding the extra
6,000 images "would produce an artifact nothing reads." **Looking at the
evidence now, I think that was wrong**, and two of my own exploration
agents split on this question, so here is the raw evidence, rather than
just a verdict:

- **`DESIGN.md:479`** defines this exact term: *"at stage 3,
  `"full training"` = 54,000 fit + 6,000 locked validation."* The term
  is directly quoted and defined as this combined total.
  `DESIGN.md:311` then says *"7 **full-training** SVDs for the final
  ridge refits."*
- **`DESIGN.md:492`**, the compute-cost table, says: *"42 SVDs (35
  fold-level + 7 final refits, **~48-60k x 1008**)"*. 48,000 equals
  0.8 times 60,000 — exactly a five-fold cross-validation training
  portion of 60,000 images — and 60,000 is the size of the final refit.
  Under a 54,000-image reading, this cell would instead read
  "~43-54k". Both numbers are only internally consistent under the
  60,000-image reading. **Neither Desktop nor ChatGPT had cited this
  exact line before.**
- Against this reading: `DESIGN.md:544` uses "training side" to mean the
  full 60,000, and "fit" to mean the 54,000, and a refit is technically a
  kind of fit, plus there is a case for CNN comparability. But that
  argument is only an inference drawn from role labels, set against an
  explicit definition of the literal term, plus arithmetic that
  corroborates it.

**Decided: 60,000.** Desktop withdrew its own 54,000 recommendation,
describing it as "a comparability inference I invented against locked
text I had not re-read." ChatGPT's own independent reading agreed with
this. ChatGPT's underlying point is the sharper one, and it is why this
was decided explicitly, rather than by convenience: resolving this
question *after* Phase A had already encoded 54,000 images, in whichever
direction avoids extra rework, would have been a post-hoc choice either
way.

**Consequences, all adopted:**
1. Phase A must be re-scoped to also encode the 6,000 validation images
   (about 1 extra minute of encoding time; the regeneration was already
   happening anyway).
2. The regeneration's acceptance test changes shape: the new artifact is
   no longer a simple byte-comparison against the old one, because it
   now covers a different population of images.
3. `encode_stage3_local.py`'s "Scope" documentation is wrong and must be
   corrected, not quietly edited without comment.
4. The previously deferred final-Delta tail counts for the 6,000 and the
   full 60,000 images (briefing item 4, "if/when those images are
   encoded") become mandatory immediately.

## A second internal contradiction inside the briefing

Item 3 of the briefing requires the 150-vs-1200 audit to run at
**"full 60k both budgets."** Item 4 states the 6,000-image validation
subset **"has NEVER been encoded."** Both statements cannot be true at
the same time. Traced back to its origin: Desktop's first message to
ChatGPT (`archive/2026-08-06T12-28-20Z.md`) had described Phase A as a
*"full 60,000-image encoding"* — it was actually 54,000. ChatGPT built
its audit scope on top of this error; Desktop corrected the population
size at 18:08, but did not correct the audit scope that this earlier
error had produced. Resolving the 54,000-versus-60,000 question above
also resolves this second contradiction.

## Dropped during consolidation — must be restored before anything is frozen

- **The audit trigger lost two of its three conditions.** ChatGPT had
  specified, and Desktop relayed word-for-word in addendum #3, three
  separate trigger conditions: a sign reversal on the primary contrast,
  any graph changing from improvement to deterioration, and a material
  reversal in graph ordering. The briefing's item 3 only encodes the
  pairwise order-reversal condition. ChatGPT re-listed all three
  conditions again at 18:10 — this is a live requirement, not an old,
  stale one.
- **A correction to `FINDINGS.md` that never reached me at all**:
  ChatGPT, at 17:26, stated — *"median and p95 ratios do not by
  themselves prove a whole-distribution shift."* Desktop conceded this
  point to ChatGPT only 29 seconds after sending addendum #2, so it
  never made it into either document.
- Two of the four `ABS_CONV_EPS` justification factors were also
  dropped ("the encoder implementation" and "downstream feature
  sensitivity"); so were `s_min` and `s_max` as recorded fields, and
  "masks" from the list of separately-hashed inputs.

---

## Blocking facts, verified by reading the code directly, not assumed

**Phase A's artifact carries no provenance at all.** Its
`summary_json` file records only scientific parameters and timing
information — no code commit, no source-file checksums, no environment
information. This was confirmed by actually loading the file.

**Phase A ran from an untidy code tree.** `encode_stage3_local.py` had
uncommitted changes when the run started (it was committed afterward, as
commit `55a6fea`). This claim is narrowed to what Freeze 5 states below,
because the original wording overstated it: this file has exactly one
commit in its entire history, and its current working-tree checksum
matches commit `55a6fea`'s bytes exactly — but this only establishes
that the CURRENT bytes match that later commit, **not** that these were
the exact bytes that actually ran at launch time. Without a source
checksum taken at launch time, there is no artifact proving this either
way, and my own memory of not having edited the script between launch
and commit time is not itself evidence. What is *not* provable from any
existing record is which versions of the other participating code
modules were actually running at that time. That exact gap is what the
regeneration closes, empirically.

**GCS custom metadata is completely unused.** `blob.metadata` is never
set, read, or passed anywhere in `stage2b_gcs.py`; the uploaded
artifact's own metadata field returns `None`. The same is true for
`md5_hash: None`, which confirms the composite-and-chunked upload path
was used — so the fingerprint mechanism must not depend on MD5
checksums.

**Every write builds a `bucket.blob(name)` object inline and then
discards it** (`upload_file:673`, `_compose:925`), so there is currently
*nowhere* in the code to attach metadata to, without first holding onto
those object handles somewhere. `download_file:1091` is the only place
in the code where a handle survives across two separate operations.

**`ensure_artifact`'s trust point is a single line of code** —
`stage2b_gcs.py:1299`, reading `if not force and object_exists(...)`.
This is exactly where a fingerprint check, done at read time, belongs.
This has two consequences: `force=True` bypasses this check entirely
(and `step11_report` already uses `force=True`), and `stage_kmnist`
calls `download_file` **directly**, bypassing `ensure_artifact`
altogether — so a check placed only inside `ensure_artifact` would not
cover this second path at all.

**A fingerprint check keyed strictly on the producing commit would break
legitimate reuse across stages.** `step1b_topologies` deliberately reads
*stage 1's* `topologies.npz` file, and `stage_kmnist` reads IDX data
files staged back under stage 1 — both of these were written under
different code commits and different configurations than the stage
reading them now. Which fields participate in the fingerprint check must
be selectable separately, per kind of artifact, or else every stage-2
and stage-3 run would end up refusing its own legitimate inputs.

**`StepResult.summary()` is pinned by an exact dictionary-equality
check** (`tests/test_stage2b_gcs.py:831`) — adding any new field to it
requires a mandatory test edit; this cannot be skipped.

**The closest existing pattern to copy** is the chunked-upload
checkpoint system (`_checkpoint_state:819`, `_load_checkpoint:853`): a
versioned format string, comparing every field, where any disagreement
at all discards the whole checkpoint.

## The audit tolerance was specified at the wrong level — with the actual numbers

The frozen specification says: tolerance equals a safety factor `M`,
multiplied by the largest recorded cross-implementation discrepancy that
does not depend on this audit. I computed both possible readings of
this, directly from the stored artifacts:

| level | value | source |
|---|---|---|
| per-coordinate prediction (`max_abs_clipped_pred_diff`) | **1.1512e-12** | stage-1 `rewired` |
| per-graph evolved-minus-pre-evolution **contrast** | **1.3878e-17** | stage-2 `lattice`; three of four graphs sit at exactly 0.0 |

**These two readings are about 83,000 times apart.** The quantity this
audit actually tests is the contrast, so the contrast-level number is
the one that matches it — and it can be computed directly from
`mean_clipped_val_mse_jax` and `_sklearn`, both already stored in
`ridge_cv.json`. The prediction-level reading gives a *looser*
tolerance, meaning more reversals would be dismissed as noise. Both
readings sit far below the roughly 2-4e-3 scale of the actual signal, so
the trigger condition works correctly either way, but only one of these
two readings is actually derived at the level of the thing being tested.
I recommend the contrast-level reading, with the prediction-level
reading recorded as the alternative that was rejected.

It is worth noting that this trigger condition is not hypothetical:
stage-2 cross-validated contrasts already show both signs at once (`T`
at -2.30e-3, `lattice` at -3.90e-3, both improving; `rewired` at
+4.02e-3, `curr_random` at +2.38e-3, both worsening).

## The ridge diagnostics need a real code change, not just better reporting

`cross_validate_alpha` currently keeps only `fold_cond` — the **ratio**
`s_max/s_min`. `kappa_alpha` needs `s_max` and `s_min` as separate
absolute magnitudes, because a ratio cannot be un-divided back into its
two parts. The per-fold `fit` object, which carries `singular_values`
and `W`, is currently rebound on every loop iteration and then discarded
(`stage2b_ridge.py:403-411`). Also: the selected alpha value is not
actually known yet inside this loop (`select_alpha` runs only after the
loop finishes), so per-fold quantities must either be computed for all
nine alpha values and indexed by column afterward, or every individual
fit must be kept in memory (about 180 MB, at stage-3 scale). I recommend
computing all nine and indexing afterward.

---

## Proposed sequencing

Ordered by what blocks what. All five contested points are now
resolved; the one remaining gate is ChatGPT's review of this whole plan,
plus Dan's release.

1. **The fingerprint mechanism** — `stage2b_gcs.py` plus its tests. This
   needs both a static and a runtime import list, established before
   generation and re-checked afterward. Selecting which fields
   participate, per kind of artifact, is required (for legitimate cross-
   stage reuse). The read-time check must be placed so it covers both
   the `force=True` bypass and the direct `download_file` path.
2. **Out-of-fold prediction support inside `stage2b_ridge`** — built as
   its own module first, with its own tests, cross-checked against the
   stored fold-aggregate values. This blocks the audit driver, but not
   the regeneration, so it can run in parallel with item (1).
3. **Phase A regeneration at 60,000 images**, under the new contract.
   The 54,000 arrays get a payload comparison against the existing
   artifact; the 6,000 new arrays get a fingerprint at the moment they
   are created, with their own tail measurements computed; the
   acceptance report states this split explicitly. This depends on
   item (1).
4. **Freezing the protocol documents**, each one committed before its
   own results are inspected — covering all three trigger conditions,
   either alpha regime, the contrast-level tolerance with both required
   notes, gauge-fixed metrics, and the deterministic stress set.
5. **Corrections to `FINDINGS.md`** — the three labeled rationale
   categories, the two adopted sentences, the whole-distribution
   concession, the final-Delta reporting discipline, and the corrected
   image populations. This item is independent, and can start
   immediately.
6. **Phase B** — the stage-2 driver structure, plus `kappa_alpha`,
   numerical rank, coefficient norms, and CNN preservation.
7. **The pre-test review package** — 10 items, plus fingerprint
   evidence, plus the protocol results, plus the negative-path table.
   Item 9 (the frozen Stage 4 command) is the only genuinely new
   deliverable in this list; drafting it early is worthwhile.

**Delegation** (using separate worktrees, matching the pattern used for
the ladder-stage-1 plan): the `FINDINGS.md` corrections, the protocol
documents, and the negative-path evidence table are all independent of
each other, and can be worked on in parallel. The fingerprint mechanism,
the regeneration, and Phase B must stay undelegated, worked on directly.

**Negative-path evidence already exists for four of the reviewer's five
demands** — see `test_ladder_missing_sentinel_fails_even_on_a_zero_exit`,
`test_a_corrupted_download_raises_naming_the_object_and_both_digests`,
`test_an_object_with_no_recorded_digest_is_refused_rather_than_trusted`,
and `test_a_leak_never_masks_the_scientific_verdict`. Only the stale-
artifact refusal demand is genuinely new. The pre-test package needs a
citation table linking demands to tests, not new work to build.

## Verification

Every new safety check is confirmed by deliberately breaking the exact
thing it watches, and observing the specific expected failure — this
project's standard corollary to its verification rule. This applies to:
the fingerprint refusal, the per-kind field selection (which must still
correctly accept stage-1's topologies), the `force=True` bypass, and the
`download_file` path that skips `ensure_artifact` entirely. Payload
comparison is verified by deliberately corrupting one array on purpose.
The full test suite must be green before each commit.

## The five questions — RESOLVED (Desktop, 20:23:04Z)

All five questions are now answered. ChatGPT's review of this plan is
still pending, and nothing in this plan actually executes until that
review lands, and Dan gives his release.

**Question 1 — 60,000.** `DESIGN.md:479`'s definition, together with
`:492`'s arithmetic, governs this. ChatGPT's own independent reading
agrees. Requiring equal sample counts across model classes that are not
even comparable to begin with was never actually a design requirement.
Desktop's earlier 54,000 recommendation is withdrawn, described as "a
comparability inference I invented against locked text I had not
re-read."

*A note on the acceptance test's shape, folded in here*: the
regeneration now spans two populations of images with different kinds of
evidence behind them. The **54,000 arrays** get a payload comparison,
checked bit-exact against the previously uploaded artifact. The **6,000
arrays are new** — there is no prior artifact to compare them against,
so they instead get a fingerprint at the moment they are created, and
their final-Delta tail is computed immediately, alongside the full
60,000-image tail. The acceptance report must state this split
explicitly, rather than letting a single "regeneration passed" message
blur together two genuinely different kinds of evidence.

**Question 2 — SUPERSEDED by Freeze 1 (see below). My earlier
contrast-level recommendation was wrong, and the reason why is worth
learning from**: I had read "three of four graphs at exactly 0.0" as
weak, but supportive, evidence for using that number as the maximum. The
correct reading is instead that exact zeros in a difference between two
independently-computed values are actually the *symptom of
cancellation*, not evidence of agreement between them — both
implementations' errors are correlated on the same images. Matching the
level of the quantity being audited does not make a cancellation-prone,
measured aggregate into a reliable estimate of numerical resolution.
This was replaced instead with an analytic bound, carried through the
metric's own formula. The measured value of 1.3878e-17 survives only as
a secondary check. The full derivation is in `AUDIT_PROTOCOL.md`.

**Question 3 — all three trigger conditions are restored**, checked
under **either** alpha regime: a sign reversal on the primary contrast;
any graph changing from improvement to deterioration; and a numerically
resolved pairwise order reversal. The alpha mechanism only determines
the interpretation of a trigger, not whether a review actually happens.
The earlier split, where fixed-alpha was decisive and reselected-alpha
was only a named review item, is superseded.

**Question 4 — a static import list, combined with a runtime import
list**, both established before generation, and both re-checked
afterward, with every exclusion frozen and justified. Using the runtime
list alone would miss conditional imports, lazily-loaded imports, and
imports specific to one code branch — precisely the "reimplemented
helper" failure mode this whole check exists to close off.

**Question 5 — yes, out-of-fold per-image predictions, under the frozen
folds.** Since this is a real extension to an already-verified module,
it gets module-first treatment: it is implemented with its own tests,
before any audit driver is allowed to call it. It carries a strong
built-in cross-check: the per-fold mean of the new per-image out-of-fold
MSE values must reproduce the already-stored fold-aggregate values, which
pins this new machinery against numbers the project already trusts. A
full in-sample refit MSE is descriptive information only.

All the implementation findings above (the `kappa_alpha` approach, the
per-kind fingerprint field selection, the `force=True` and direct-
`download_file` bypasses, and the `StepResult.summary()` test-edit
requirement) are endorsed exactly as written. The `FINDINGS.md`
restorations are confirmed: the whole-distribution concession, both of
the missing `ABS_CONV_EPS` justification factors, `s_min` and `s_max` as
recorded fields, and masks included in the list of separately-hashed
inputs.

---

## The five freezes (ChatGPT, conditional agreement 2026-08-06)

Running the fingerprint implementation, the Phase A regeneration, the
amendment audit, and Phase B is **agreed to, once these five decisions
are committed.** Mismatch checks halt automatically; successfully
completing the planned steps creates **no** additional discretionary
review requirement. Stage 4 stays blocked behind the pre-test review
package, and behind an explicit release.

**Freeze 1 — the audit tolerance is analytic, computed from formulas,
not measured empirically.** My earlier Question 2 verdict is overruled;
see above for why the counter-argument is the stronger one. The value
`d = 1.151190e-12` is carried through the metric's formula:
`|Delta MSE| <= 2d + d^2` (since predictions and targets are both in the
[0, 1] range, so `|q-y| <= 1`). The contrast bound is `4d + 2d^2`; the
pair-ordering bound is `4d + 2d^2` if the shared `pre` term cancels out
algebraically, or `8d + 4d^2` if it does not. A safety factor of
`M = 100` is frozen before any result exists, giving thresholds of
**4.604761e-10** for the contrast, and **9.209522e-10** for the 4-term
pair ordering. The sign convention is frozen as:
`Delta_g = MSE_evolved_g - MSE_pre`, where a negative value means
improvement. **Removed**: an earlier, forward-looking claim that the
trigger verdict would be "insensitive across 5 orders of magnitude of
M" — signal sizes measured at stage 2 do not establish what effect sizes
this audit will actually find at full scale; robustness is instead
reported *after* running the audit. This is frozen in
`AUDIT_PROTOCOL.md`.

**Freeze 2 — population and image-index identity.** The roles are:
official corpus 60,000; CNN weight-fit subset 54,000; CNN validation
subset 6,000; **ridge cross-validation and final-fit corpus 60,000**.
The 6,000 validation images are held out only from CNN weight updates,
not from training-side analysis in general. The regenerated artifact
carries official KMNIST indices, plus flags marking CNN fit-versus-
validation membership. Every comparison between old and new data is
**by official image index, never by row position in a file**.

*Already verified, before any execution*: Phase A's corruption random-
number generator consumed **official** dataset indices, not fit-local
indices running `0..53,999`. This was confirmed empirically, by
recomputing `epsilon_for` under both possible readings, and comparing
the result against the stored encoded array at rows 0, 27,000, and
53,999 — the official-index reading matches all three rows; the fit-
local reading fails at rows 27,000 and 53,999. (Row 0 matches under
either reading, since official index 0 equals fit-local index 0; testing
only that one row would have proven nothing.) No halt occurred.

**Freeze 3 — fixed alpha, defined precisely.** Fixed alpha means: the
alpha value selected from the production 1,200-step representation, per
condition, using the frozen 5-fold procedure, then applied identically
to both budgets' out-of-fold fits. Reselected alpha means: an
independent selection, made separately for each budget. The results
report both values, plus the production alpha value. Per-budget, fold-
fitted scalers are kept (matching production preprocessing), so fixed-
alpha isolates only **the effect of reselecting alpha** — not the effect
of a raw representation change, completely on its own. This is stated
honestly in the protocol; a shared-scaler comparison is optional,
secondary work only.

**Freeze 4 — the payload and its manifest are written as one atomic
unit.** This uses immutable, content-addressed naming; a manifest that
carries the payload's SHA-256 checksum, its data type, shape, an
ordering or index checksum, the producer's fingerprint, its parent
checksums, and the GCS object generation number; a consumer-side
recomputation of the SHA-256 checksum; validity only when both the
payload and the manifest exist and agree with each other; publication
either through a temporary object that is later promoted, or through
generation-match preconditions (so there are no overwrite race
conditions); and **one single, central, validated read path**, used
consistently by `ensure_artifact`, by `force=True` workflows, by report-
writing steps, and by `stage_kmnist` alike — with any raw download that
bypasses this path either mechanically prohibited, or at least detected.
The following sentence is recorded word-for-word inside the module's own
documentation: *"An object merely existing is never sufficient evidence
that it is resumable."*

**Freeze 5 — the "dirty tree" claim, narrowed.** My earlier claim that
"the script that ran is provably the one that was committed" overstated
the evidence: without a source checksum taken at the moment of launch,
matching bytes now only proves that the CURRENT bytes match the later
commit. My own memory of not having edited any code between launch time
and commit time is not itself evidence. The correct, narrower statement
is only: (a) the recovered script matches the later commit; (b)
provenance for the other participating code modules is unavailable; (c)
payload identity, under a clean, fingerprinted regeneration, is strong
*empirical* evidence of computational equivalence; and (d) **the
regenerated artifact, not the old one, is now the authoritative Phase A
input going forward.**
