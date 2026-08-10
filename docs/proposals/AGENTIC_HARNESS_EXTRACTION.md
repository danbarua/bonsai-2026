# Agentic-research harness: retrospective and extraction spec

Status: proposal. Nothing here is extracted; nothing here is a
recommendation to build now. Findings are filed as tasks, not as further
documents.

Method: four cold-read agents with no prior project context, given measured
scoping data and forbidden to re-derive it. One held an efficacy ledger, one
an extraction spec, one the vocabulary, one the greenfield question. The
ledger adjudicated the others.

---

## 1. What this reviews, and the base rate

Over roughly 72 hours a team of AI agents built CI, a vacuous-test review, a
binding-clause gate inventory, provenance capture hooks, an agent mail mesh,
and about 80,000 words of process documentation, around a small piece of
science.

Measured from `git log`, non-merge, since 2026-08-08: **190 commits — 121
tooling/tests only, 32 docs only, 20 science + tooling, 17 science only.**
81% touched no science.

**State this once and weigh it throughout: the apparatus is days old.** Most
guards have had almost nothing to guard. "Only latent findings in three days"
and "only latent findings in three years" are different claims, and only the
first is supported here. Every verdict below is a reading of a short window.

## 2. The headline, restated after a definitional check

The ledger's first framing was: *every defect where a result was actually
wrong was caught by a human-shaped act, never by an automated guard.*

That framing is **partly an artifact of the verdict taxonomy**, and the
taxonomy was supplied by the review's own author. `RESULT WAS WRONG` requires
the bad claim to have already entered a durable record. Pre-merge guards fire
before that by construction, so the only verdict they can earn is
`WOULD HAVE BEEN WRONG`. Humans read committed records; guards fire on
uncommitted ones. The two populations cannot land in the same bucket.

**The argument is structural and needs no count**: `RESULT WAS WRONG` requires
a durable record by definition, so a pre-merge guard can only ever earn
`WOULD HAVE BEEN WRONG`. The comparison was rigged by its own definitions.

The commits corroborate it. All six `RESULT WAS WRONG` rows were fixed by
commits touching already-committed files — `f522e41`, `7e46952`, `e6d704c`,
`c6e0312` and `e51050f` all carry deletions; `4173bf7` is pure insertion
because the halt never existed and the wrong `STAGE3_OK` sat in a run
artifact. The single automated catch, `8fd8aa7`, fired pre-commit.

This is a category-J instance — right field, wrong question — inside a review
whose subject is category-J instances. It is recorded rather than quietly
corrected because the catalogue exists for exactly this.

**Two findings survive the correction, and neither depends on the taxonomy:**

1. **The guards largely police each other.** Of 28 mechanical findings, most
   are defects *in guards*, not defects on a path to a science claim.
2. **The two most elaborate components are net-negative in defects.** The
   provenance hooks produced defect #18 and caught nothing; `review_delta.sh`
   produced four and caught nothing. A component that manufactures the class
   of claim it was built to prevent is not invisible deterrence — it is added
   surface. This argument is independent of how verdicts were bucketed.

## 3. The scorecard, split three ways

The original scorecard collapsed "fired, only latent" with "never fired."
Those are different, and for a *preventive* control the difference is
decisive: a spend guard that never fired means nobody tried to overspend,
which is its success condition, not its failure.

| Class | Meaning |
|---|---|
| **UNPAID** | Fired; every finding latent |
| **UNFIRED — hazard absent** | Preventive; the hazard never occurred |
| **UNFIRED — untested** | Never exercised; cannot yet tell which |
| **PAID** | Caught a wrong or would-be-wrong result |

| Component | Caught | Class |
|---|---|---|
| External reviewer reading `gates.toml` | 6 (5 wrong) | PAID |
| Peers using the tools on real work | 6 (3 wrong) | PAID |
| Reading the artifact, no guard | 1 (1 wrong) | PAID |
| Catalogue/doc-citation guards | 1 (1 would-be) | PAID |
| Deliberate breakage tests | 10 | UNPAID |
| Gate inventory | 7 | UNPAID (but see §6.3 — the credit sits with one schema field, not the reconciler) |
| PR #28 vacuous-test review | 9 | UNPAID |
| In-session review | 6 | UNPAID |
| Cloud Build anti-vacuity | 2 | UNPAID |
| CI-image / dependency guards | 2 | UNPAID |
| Cloud Build spend guard | 0 | UNFIRED — hazard absent |
| Transit-integrity check | 0 | UNFIRED — hazard absent |
| `review_delta.sh` / publisher | 0 (source of 4) | UNFIRED — untested |
| Provenance capture hooks | 0 (source of #18) | UNFIRED — untested |

The spend guard and transit-integrity check are **not** candidates for
removal. Replacing a pre-hoc block with a post-hoc billing alert is a
downgrade sold as a saving.

**Ledger integrity.** 42 rows were catalogued; **four (#11, #12, #13, #30)
carry no fixing SHA and no evidence artifact.** The review's own rule was "no
SHA, no row." They are retained as marked-incomplete rather than counted, so
the defensible figure is **38 evidenced rows**, not 42.

## 4. What the ordering says

Every guard built *ahead* of use is UNPAID. The PAID rows are dominated by
peers using tools on real work and outside readers reading records.

An earlier draft concluded: "it is the ordering that paid, not the tooling."
That overclaims — it converts a correlation with implementation order into a
mechanism, and this review has no way to discriminate the two. The defensible
statement:

> In this 72-hour window, observed payoff is concentrated in controls
> introduced or exercised in response to concrete incidents. Prospectively
> built controls have not yet demonstrated comparable payoff.

That is weaker than "build reactively" and it is what the evidence supports.

## 5. Extraction spec

Testability requirement: for each "lifts as-is", name the one constant that
must change. If none can be named, the claim is untested and marked so.

### 5a. Lifts as-is

| Component | The one constant |
|---|---|
| `tools/ci/check_suite_not_vacuous.py` | `DEFAULT_BASELINE` (L57) |
| `tools/ci/assert_no_cloud_credentials.py` | `CREDENTIAL_ENV_VARS[0]` (L31) |
| `tools/ci/check_workflow_parity.sh` | `WORKFLOWS` array (L26) |
| `tools/ci/review_delta.sh` | `IN_SCOPE` (L76) |
| `tools/ci/review_run.sh` | `REVIEW_WORKFLOW` display name (L27) |
| `tools/gates/triage_candidates.py` | `MODEL` (L54) |
| `infra/*.tf` | `var.project_id` (`variables.tf:9`) |
| `src/bonsai/stats/permutation.py` | its file location only |

### 5b. Corrections to the supplied coupling map

The map was written by the review's author and given to the agents as
measured fact. A cold read corrected it in four places — the clearest
evidence the cold-read design worked:

- `check_workflow_parity.sh` — `REF_B` was listed as a coupling point. It is
  `${2:-origin/stage2b}`, a positional default, already overridable.
  **Both readings then missed the larger fact: nothing invokes it.** It is
  named in three documents and in zero executable surfaces — no Makefile
  target, no CI job, no test. Overridable does not help a script no caller
  runs, and a coupling audit that only asks "could this be parameterised?"
  cannot see that. It scores UNFIRED–untested, and its default pair
  (`origin/main` vs `origin/stage2b`) omits `stage2b-ci`, the base branch the
  workflow's own trigger names, so the pair most worth checking is the one a
  bare invocation does not check. Verified 2026-08-10: all three branches
  carry blob `3ffe0837` and both pairs report no drift, so the condition is
  clean — this is about the instrument, not the current state.
- `permutation.py` — rated "Med, coupled to `src/bonsai/`." It imports
  numpy/scipy/itertools/multiprocessing and nothing intra-project. Moving it
  is `git mv`.
- `gate_inventory.py` — rated "highest, `REPO_ROOT` only." The obligation
  vocabulary is hardcoded (`_VALUE_STATUSES`, `_REQUIRED_DIMENSIONS`,
  `_OPTIONAL_BY_KIND`). It splits into **two** assets: a generic clause
  extractor (RFC2119 table, `clause_id = sha256(normalised)[:12]`) and a
  project-specific lifecycle checker whose vocabulary belongs in the TOML.
- Provenance hooks — the scripts are generic; the coupling is the **matcher
  registration** in settings, i.e. a template, not a code change.

### 5c. Lifts with parameterisation

- **Scope regex, one key, three consumers.** `IN_SCOPE`
  (`review_delta.sh:76`), the jq filter `(^|/)tests?/`
  (`publish_review.sh:104,131,158`), and the workflow's `paths:` are three
  independent literals that must agree. Extract one key; generate or assert
  the workflow `paths:` from it.
- **Environment prefix.** `BONSAI_PROVENANCE_ROOT`, `BONSAI_PROBE_LOG` → one
  configurable prefix.
- **`ci_targets.py`** — real work. `CI_INVOCABLE_TARGETS`, `SPEND_MARKERS`,
  a cross-tree `import _makefile`, and `_ALLOWED_RECIPE_REQUIRES` bake in the
  build system. Needs a config table plus a pluggable recipe parser.

### 5d. Bonsai-only — do not extract

`ci_skip_baseline.txt` (measured data; regenerate) · the `github` skill (a map
of this repo) · `gates.toml` rows (89 measured rows; clause ids hash
normalised text, so rewording orphans a mapping — the schema lifts, the
content does not) · the review prompt (names this project's taxonomy) · the
provenance probes (measurements of one harness version; the emitters lift) ·
`infra/variables.tf` defaults.

### 5e. Dependency order

1. Config/env layer — everything reads it.
2. **Anti-vacuity exit-code contract** (exit 2 on zero candidates derived,
   exit 1 on any finding) — must precede any guard, because it fixes what
   "green" means and cannot be retrofitted across guards already written.
   **Reconciles with §6's "build only after a defect proves it":** define the
   generic exit-code *convention* up front; do not build a dedicated
   anti-vacuity *scanner* until the hazard manifests. A convention costs a
   paragraph and cannot be retrofitted; a scanner costs a component and can.
3. Scope regex — before its three consumers, which must agree.
4. Review output JSON schema — before the publisher; the schema is its
   contract.
5. Obligation vocabulary — before `gate_inventory.py`; clause ids hash
   normalised text, so freezing the schema late orphans every mapping.
   **Reconciles with §6's "defer the reconciler":** defer the *tool*, freeze
   the *schema shape* early. They are not the same decision, and a reader who
   defers both orphans every mapping the day the reconciler arrives.
6. Recipe-parser interface — before `ci_targets.py`, which imports it.
7. Provenance root convention — before both hook sets.
8. Terraform last; it creates the triggers the workflows assume.

### 5f. Known defect, recorded rather than frozen

The review workflow re-runs on every push to an open PR into the checkpoint
branch. Three cost points, in the order they were measured: **$1.2769 /
169.7 s / 26 turns / 13,751 output tokens** per run from
`tools/ci/review_run.sh`; then **$5.04** on one PR #29 run (31394098469);
then **~$0.25** for the same surface locally on Haiku. The last two are Dan's
measurements, recorded in `f0438cb`, not reproduced here.

**The frequency defect stands; the cost defect was fixed by narrowing the
model, not the trigger.** `f0438cb` pins the reviewer to Haiku at low effort,
caps turns at 40, and disallows `Agent`/`Task`/`ScheduleWakeup`/web tools.
Task #22's cost half closes; its frequency half and #23's scope question do
not — the workflow still fires on `synchronize`. The workflow's in-line prose
defends the current trigger and must not be carried forward as settled.

**The same run is an efficacy-ledger row, and a rare one.** PR #29 run
31394098469 spawned five background agents, waited 300 s, examined 6 of 12
files, posted a PR comment saying the review was in progress, and **exited
success**. That verdict was durable — a posted comment, not a transcript —
so by §3's taxonomy it scores **RESULT WAS WRONG**. It is recorded here as an
addendum and is deliberately NOT folded into §3 — that ledger was frozen at
`79ecd7f`, this run postdates its cut, and principle 3 says an exploratory
addition to a corrected family is flagged as nominal rather than absorbed.
§3 stands at 6 of 38. Read this row alongside it, not inside it.

**And then, in the same PR, it caught something.** Run 31403231653 — Haiku,
1 m 28 s — examined 2 of 2 changed test files, reported `Not examined: None`,
and found `assert selected_deltas[0] == deltas.max() or True` at
`tests/test_stage2b_arm_x86_propagation.py:71`. The `or True` makes the
assertion unfalsifiable. Verified here that the assertion would **pass**
unaided (`selected_deltas[0]` is exactly `deltas.max()`), so it is a leftover
debug escape, not a disabled check concealing a defect — which is why it
scores **LATENT**, not a wrong result.

Two things about that row matter more than the finding itself. First, the
same workflow appears in this addendum as both the defect and, one commit
later, the catcher — the narrowing worked, on its first real surface. Second,
it does not disturb §2's headline. That headline is about **deterministic
guards** catching **wrong results**; this is an LLM reviewer catching a latent
one. A cheap complete pass beat an expensive partial one, which is a claim
about coverage, not about the class of thing automation can find.

It is also the review workflow appearing in a ledger as a defect rather than
a catcher, which no §3 row does.

Two guards landed against it, and both are narrowings — the shape principle
21 warns about. `1d1e3ab` fails the job when the push's own test set is not
fully examined (whole-PR partial stays report-only, correctly, since older
files go unread on purpose); `2135f8e` carries the unfinished remainder
forward in a sticky comment so the backlog survives the next push. `1d1e3ab`
was break-tested in both directions — incomplete-this-push fails, complete-
this-push-with-older-files-unread passes — which is what principle 21's
corollary asks for and what most of this repo's narrowings did not get.

### 5g. Justified by principle, not by a caught defect

Flagged by the extraction agent, adjudicated against the ledger:
`assert_no_cloud_credentials.py` (UNFIRED — hazard absent; keep) ·
`triage_candidates.py` · `ci_targets.py` (cites principle twice; its "caught"
example is hypothetical) · `gate_inventory.py`'s trichotomy · the 100%
coverage floor and 24 finding kinds. None of these cite an incident.

### 5h. One design choice NOT to carry forward: content-derived identity

Raised by external review as a challenge rather than a refinement, and
accepted.

Sections 5d and 5e both note that clause ids are `sha256` of normalised
clause text, and treat that as a constraint to work within — "freeze the
schema early, because rewording orphans a mapping." That accepts the premise.
The premise is the defect.

**Using mutable prose as identity conflates two jobs.** Identity should answer
"which obligation is this?" and survive an editorial reword. Content hashing
should answer "has this text changed since it was dispositioned?" and fire
loudly when it does. This project's design makes the first job depend on the
second, so improving a sentence silently destroys its disposition — and the
lesson a naive reader takes from that is "don't reword protocol documents,"
which is the wrong adaptation.

**The greenfield form:** give each obligation a stable explicit id, assigned
once and never derived. Store the normalised-text hash *alongside* it as a
separate field. A reword then produces a visible "text changed, disposition
needs re-confirmation" finding instead of an orphan. This is strictly more
information than the current scheme and costs one column.

This supersedes, for a new project, the "freeze the schema shape early"
advice in §5e.5 — that advice remains correct *for this repository*, where
89 rows already carry content-derived ids and changing the scheme now is its
own migration.

### 5i. A gate this document argued for, which the event then did not need

The DAG extraction of `DESIGN.md` found that rung 3 → rung 4 — the edge into
the one-shot confirmatory evaluation, the single most consequential edge in
the design — has **no stated exit gate**. Its only protection is an ACCESS
rule ("no test-side result accessed before stage 4"), which is a permission
rather than an acceptance criterion: it says who may look, not what must hold
before looking is warranted. That was written up as the clearest case of the
gates-on-edges model earning its keep, since a prose reader notices nothing —
no section is missing, the ladder describes rung 4 fully, and the gap appears
only when you ask what *opens* the edge.

**Stage 4 has since run. The edge was traversed ungated and the result was
clean.** Recorded here because it cuts against the argument this document
makes, and a ledger that only accumulates confirmations is the thing §1 warns
about. Scored **UNFIRED — hazard absent**, not "a gate we got away with
missing"; the distinction is the whole point of separating those two verdicts,
and claiming the stronger one here would be unfalsifiable.

What actually protected rung 4 was a different mechanism entirely: the
feasibility ladder. Part 1 returned an honest FAIL at 150 steps and caught an
under-converged encoder *before* the A100 run, which would otherwise have
spent the one-shot evaluation on it. A staged ladder that can fail cheaply and
an exit gate on the final edge are not substitutes — but on the evidence here,
the ladder is the one that paid, and a greenfield harness told to build both
should build the ladder first.

Three findings of identical shape came out of the same extraction and remain
unadjudicated, now also historical: `active_indices` (the 505 restriction) has
no stated producer; the Stage-2 condition-number diagnostic requires a table
but names no threshold that would overturn the float64 choice; CNN
"best-of-3" states a selection rule with no acceptance criterion. An agent
cannot distinguish "the document omits this" from "the document assumes it
obvious", which is why none of the four is asserted as a defect.

## 6. Greenfield: what to define up front

Binding rule applied: nothing enters this list without a ledger defect it
would have caught. The list is short, and **all four items are conventions or
formats, not software.**

1. **A verdict record computed from per-gate evidence rows, never asserted.**
   No run prints OK unless every named gate has a row carrying its evaluated
   value; a gate with no row is a missing row, not a pass. Catches #17a
   (`4173bf7`) directly; blocks #39 (`7e46952`).
2. **A citation convention** — any "it fired" or quantitative claim in a
   durable record carries a SHA, run id, or log line beside it. Catches #39,
   #40 (`f522e41`), and the coverage claim drawn from the bad provenance
   record in #18 (`c6e0312`, `63ac018`). Cost: one line of norm.
3. **A flat, human-readable inventory of claim-bearing gates, with one
   evidence field that must be filled honestly** — but not the reconciler.
   An outside reader of `gates.toml` produced five of the six wrong-result
   findings (#36: `e51050f`, `4c4871b`, `2dc8c9b`, `d54554c`).

   The evidence field is a late amendment to this item, and the file argued
   against my first draft. `gates.toml:1394-1400` records that filling
   `break_demonstrated` honestly is what exposed the ridge equivalence gate
   having no negative-path test at all (142 tests green under a hardcoded
   verdict) and the index-join guard being a source grep that stayed green
   while all three joins were replaced — adding, pointedly, "**neither was
   found by review**." Both remain LATENT (ledger rows 19, 20; `8017225`,
   `bff25eb`), so this does not move the gate inventory to PAID. But it
   relocates the credit: the catching was done by **a schema field demanding
   evidence a human had to supply**, not by the reconciler that checks the
   file, and not by the outside-reader pass in item 4. "The row that would not
   go in is the instrument" is the file's own phrase for it, and it is the
   single most transferable sentence in this repository.

   Build the list and that one field. Defer the reconciler and the per-kind
   contracts — but see §5e: freeze the schema's shape early even while
   deferring the tool, because clause ids hash normalised text.
4. **A standing outside-reader pass over durable records**, by someone who
   did not write them. Cited by #36, #17a, #18, #23 (`e6d704c`), #40 — five
   of six wrong results. The highest-yield item in the ledger, and not code.

### Build only after a defect proves it

| Component | Trigger |
|---|---|
| Anti-vacuity *scanner* (the convention goes in on day 1 — §5e.2) | First time a suite passes with everything skipped *and* a claim is drawn from it |
| Deliberate breakage tests | First time a guard a *result depended on* is found unable to fail |
| Doc-count tests | A second prose-count error reaching a science record. Cheaper: stop putting literal counts in prose |
| Dependency pinning guards | First time a dependency change moves a numeric result |
| A review-delta tool | Only after the platform's own diff view is demonstrably insufficient — it was built first and cost four defects |

### Not measured at all

The **agent mail mesh** and the **80,253 words of process docs** (32
docs-only commits) appear nowhere in the ledger. Absence from a ledger is
weaker evidence than an UNPAID verdict.

Precisely: **the burden of justification has not been met** — which is not the
same as demonstrated waste, and over a three-day window the distinction is
load-bearing. The claim is that apparatus at this cost should carry a
measurement and currently carries none, not that it is known to be worthless.

## 7. Vocabulary

**Load-bearing** — something reads, keys on, or refuses because of the term:
`binding_gate`/`binding_value`/`binding_claim` (distinct required-field
contracts in `gate_inventory.py:327-328`) · `binding clause` · `narrowing`
(promoted to principle 21; changes practice in two tools) · `departed`
(defines a third output mode; nine tests key on it) · `adjacent evidence` ·
`load-bearing scratch` · `presence-shaped`.

**Decoration** — `anti-vacuity` (see the contradiction below) · `checkpoint
branch` (a rename of *integration branch*; the attached rule and the concrete
value carry the weight) · `self-referential pin` (*tautological test*) ·
`break-confirmation`/`break-test` (labels for a rule that is already stated
in `CLAUDE.md`).

**Resolving the one contradiction between agents.** The vocabulary agent
called `anti-vacuity` decoration; the extraction agent placed the anti-vacuity
exit-code contract at dependency-order position 2. Both are right, and
together they are the cleanest vocabulary finding here: **the mechanism earned
its place; the coined noun is inflation.** Nothing keys on the word.

**Undefined but relied upon — these are harness defects, not nomenclature
complaints.** `adjacent evidence` is an **uncontrolled input to a decision
procedure**: zero definitions, five to six uses, all in `gates.toml`; a cold
reader must guess whether "adjacent" means a different clause, module, or
claim, and guessing "different module" fills two deliberately empty rows and
flips readiness from failing to passing. Highest-risk gap. It and the
`break_demonstrated` collision below are filed as tasks #26 and #25 **as
defects**. The second is the more serious on reflection: it undermines the one
evidentiary field §6.3 identifies as the most transferable mechanism here.
Then the status values `pending_consumer`/`pending_package`/`not_applicable`,
defined only in divider comments — `pending_package` fails readiness while
`not_applicable` passes, so a wrong guess silently downgrades an obligation.
Then `superseded_by`/`amendment_locator`/`canonical_clause`, defined only in
Python (these fail loudly, so lower risk), and `session marker`.

**Drift.** `break_demonstrated` is defined **twice, differently** —
`gates.toml:48` (a gate flips red under deliberate local disable) versus
`:614` (a value fails when the production value or its propagation changes).
The reconciler enforces the field's presence, not which semantics were used.
This is a real hole, not cosmetic. Separately: three names for the gate
inventory, one of which (`FINDINGS.md:1099`, "binding-**gate** inventory")
names one of three kinds as the whole; and the taxonomy's file order runs
K, M, L, which is navigational only since nothing parses the letters.

**The glossary asymmetry.** `docs/GLOSSARY.md` carries a plain-English twin,
headed at `:265` and ending at `theta_static` around `:393`. It covers **26 of
26 science entries and 0 of 4 process entries**, stopping immediately above
the process cluster at `:216-263` (`binding clause`, `the (binding-clause)
inventory`, the three kinds, `negative attestation`). (An
earlier "60/40 process-over-science" figure was quoted in the brief with no
stated denominator and is withdrawn.) The implication holds and sharpens: the
vocabulary written for humans to read was translated; the vocabulary the
agents wrote for themselves never was — and it is the half a newcomer must
learn before touching `gates.toml` at all.

**Discharged 2026-08-10.** The twin now carries all four process entries, in
the technical glossary's own order. The finding above stands as the state at
review time; the asymmetry it describes no longer exists. Worth noting what
writing them exposed: three of the four are only explicable through the
INCIDENT that produced them — the inventory's plain-English entry is mostly
the story of a run reporting `STAGE3_OK` for a gate that was never
implemented. A term whose everyday reading has to recount an incident is a
term doing real work, which is the §7 test applied by accident.

**Coping mechanisms.** Those compensating for limitations *intrinsic* to
agents generalise to any harness: convergence declared by the reviewer not the
responder; a summary borrows authority its source never granted; commit before
you need to; the session marker distinguishing "didn't run" from "ran, found
nothing"; point at an artifact, never at who checked.

Those compensating for limitations *of this setup*, with what would remove
each: peer visibility → a shared run-state channel or lease · non-propagating
surface (MCP schemas, hooks) → a version stamp peers can observe · batch
declared by merge → a CI cost model that can afford per-push runs · "break
what the guard watches" → a mutation-testing harness, which `break_demonstrated`
is a hand-typed prose stand-in for · "a citation of a gate is not evidence of
a gate" → executable locators that resolve and run, instead of trusting a
string.

Two entries were mis-filed: **derive sets, don't hand-list** and **know what
fields MEAN** (the `filename` vs `previous_filename` bug) are ordinary
engineering discipline, not coping with agent frailty.

## 8. What would settle the open question

The deterrence hypothesis — that guards prevented defects from ever being
written, so they cannot appear in a ledger of caught defects — is
**unfalsifiable as this repository is instrumented.** Of the last 100 workflow
runs, six failed; all six are the review workflow failing on merge commits
into the checkpoint branch. Tooling failing on tooling. Not one red run blocks
a science claim, and there is no store of rejected-then-corrected events
anywhere.

What would distinguish the readings: **a persisted log of blocked-then-
corrected events, retaining the pre-correction text.** It does not exist here.

An earlier draft ended "every efficacy claim about this apparatus is
untestable in both directions." That is too broad, and it discounts the
ledger this document spent its length building. Several narrow effects are
directly evidenced and stand:

- `review_delta.sh` caused four catalogued defects and caught none.
- Provenance capture produced at least one incorrect provenance record, which
  was then relied upon.
- External review detected specified errors in committed dispositions.
- `break_demonstrated` elicitation exposed two latent failures that review
  missed.
- The spend and transit-integrity guards did not encounter their hazards in
  the observed window.

What is untestable is the **counterfactual net-prevention effect**: how many
defects would have occurred without the apparatus, including any deterred
before they took observable form. That single quantity is unmeasured, and it
is the one on which "was the apparatus worth it" turns.

### 8a. A control arm existed after all

Added after §9 was written, on new evidence rather than reflection.

`experiments/stage2a_dynamics_classification/` is **complete, locked, and has
no `gates.toml`** — the inventory was invented for Stage 2B. So the repository
contains a stage that ran to completion without the apparatus. Three cold-read
agents characterised it (32,860 words across 7 docs). This does not measure
deterrence, but it does answer a question §8 called unanswerable: **what
actually goes wrong in a stage with no inventory?**

**Stage 2A: 15 catalogued defects — 9 wrong results already in durable
records** (11 distinct claims), 2 would-be-wrong, 4 latent. The tooling ledger
in §3, same verdict standard and method, different subject: **6 wrong of 38.**

Different populations and discovery processes, so this is not a clean rate
comparison. The direction is not in doubt. **The tooling produced mostly
latent defects; the science produced mostly wrong claims.** §1's 81% figure
sharpens accordingly: the effort went where the defects were not.

**Two or three of the fifteen would a gate inventory have caught** — all of
them documented HALT/`never` clauses with no executable behind them. The
largest class was something else entirely: **eight of fifteen were prose-claim
errors** — a conclusion drawn backwards, a comparison that could not support
its inference, framing words ("independent corroboration", "equally-sized",
"ruled out") that overstated. Every one caught by a human reading prose, at
`cd25a86` and `5d3a29d`. The apparatus was built for a different class.

**And the same external review pass also surfaced the defects the inventory
would have covered structurally** — so review is a superset, consistent with
§3's finding that it was the only PAID component.

**Correction to §3.** The gate inventory's row should not be read as UNPAID
alone. A cold read established a capability the ledger could not score: a
*code-resident* inventory can only enumerate gates that exist, so a **missing**
gate is invisible — which is why nobody noticed that no Stage 2A driver
branches on its go/no-go report. `gates.toml`'s failing-empty state is designed
against exactly that. The honest verdict is neither UNPAID nor vindicated:
**a narrow instrument doing a narrow job, where the job is a minority of what
goes wrong.**

**The finding that changes what to build.** Stage 2A's lines of enquiry were
checked against their pre-registered closure criteria — possible only because
the stage is closed. **Four of nine closed cleanly.** Two are SILENT (a
pre-registered gauge-sensitivity comparison never mentioned again, whose
function exists and is called by no pipeline; and a named watched-for outcome
adjudicated for three of its four cases). One DRIFTED. Two closed on criteria
never stated. A documented approval gate lapsed unremarked, and storage — a
required report — was never reported at all.

| Aspect | Defects | Tooling built |
|---|---|---|
| Gates | 2–3 of 15 | `gates.toml` + reconciler, ~2,100 lines |
| Lines of enquiry | **5 of 9 lines mis-closed** | **none** |

**The apparatus was built for the aspect with the fewest defects.**

**Correction to how this section justified itself.** An earlier version said
no Stage 2B line has closed, so only Stage 2A could show this. That is false
and the error was mine: `experiments/stage2b_denoising/FINDINGS.md` is 8,328
words of evaluations, and I scoped the review without reading it. Stage 2B's
*confirmatory* line is open; several feasibility lines are closed, one of them
through a full FAIL → diagnosis → disclosed amendment → re-run cycle.

The narrower claim that survives: Stage 2A is the control arm because it has
**no gate inventory** and because its *primary* line closed, so pre-registered
closure criteria can be checked against a completed confirmatory result. That
was always the load-bearing reason. "2B has no evaluations" was not, and was
wrong.

**And Stage 2B's FINDINGS is ahead of this document on three counts I
proposed as novel.** Its Part 3 separates (a) the contemporaneous decision-time
reason, (b) a durable rationale identified retrospectively, and (c) subsequent
evidence narrowing (a)'s premise — explicitly preserving (a) "as the decision
record… not as a claim that still stands unmodified." That is the stacked
criterion history of §5h, applied to a decision rather than a clause, and it
predates the version sketched here. It also performs the invalidation check
§5h says amendments omit: a looser reading of the amended rule was tested,
found to select `S*=300`, and rejected because that value fails its own
same-step re-run — establishing that **the correction did not select the
outcome**. And (c) is a recorded reopen: the exact-zero premise "did not
survive scale", with 79 nonzero final-Deltas at 54,000 images.

So the amend-without-invalidation finding in §7 is narrower than stated. One
gates.toml row records supersession without invalidation. The FINDINGS record
of the same amendment does the invalidation work in prose. The defect is that
the two live in different documents and only one of them is machine-checkable.

### 8b. The first two blocked-then-corrected events, and why they disagree

§8 says the log that would settle this "does not exist here." It still
doesn't. But on 2026-08-10 the repository produced the first two entries it
*would* have contained, and git retains the pre-correction text, so they can
be read now without building anything — which matters, because §8's own
instruction is not to build this speculatively.

Both arose from `cf5ffca`, the Protocol 1/2 closure commit, which was pushed
described as "code, tests, FINDINGS, gates, verification, commit and push"
and was red on two tests at the moment it landed.

| | event A | event B |
|---|---|---|
| guard | `test_stage2b_gcs_makefile.py` | `test_stage2b_gate_corpus.py` |
| defect | `stage2b-protocol2` ran a GCS-touching driver with no `$(GCS_ENV)` | FINDINGS.md derives 60 clause candidates; the exemption still declares 59 |
| pre-correction text | `cd $(STAGE2B_DIR) && uv run python run_abs_conv_eps_sensitivity.py` | exemption count `"FINDINGS.md": 59` |
| also found by | **the Codex review, independently** | **nobody** |
| corrected | yes, in `c702bae` | **no — still red** |

**Event A is the guard's clearest win and simultaneously the weakest possible
evidence for it.** Two detectors fired on one defect: this repository's
AST-walking guard, and a reviewer in another harness reading the target. The
correction cannot be attributed to either alone. This is principle 4 arriving
from the other direction — not choosing the strongest of several controls
after the fact, but being unable to separate two that both fired.

**Event B is the control, and it is the one that did not get fixed.** Only a
deterministic guard reported it; no human or LLM reviewer did; and it survived
the closure commit. The obvious hypothesis is that a defect surfaced by a
reviewer someone is reading gets attention, and one surfaced only by a red
test in a suite nobody ran does not. Two events is not evidence for that, and
it is recorded as a hypothesis with a cheap test attached: **watch whether
event B is still red at the next checkpoint merge.**

What both events do establish, against §8's stronger phrasing: the guards are
capable of catching genuinely new defects on the day they are introduced,
which the ledger to that point had not shown. Neither would have become a
wrong scientific claim — A misroutes artifacts, B stales an exemption — so
both score **WOULD HAVE BEEN WRONG (operational)**, and §2's headline about
deterministic guards and wrong *results* is untouched.

This does not license building a lines-of-enquiry tool. §6's rule still binds
— the first move is the format, not the machinery, and §6.2's citation
convention plus §6.4's outside reader are what caught the prose-claim class
that dominates here.

## 9. What this establishes, and what it does not

Stated narrowly, because the broad version was wrong twice already.

**This document does not establish that an agentic-research harness is
useless.** It establishes four narrower things:

1. The present harness was **massively overbuilt relative to its demonstrated
   three-day value**.
2. Several sophisticated controls **created their own failure surface** — two
   of them net-negative in defects within the observation window.
3. The highest observed returns came from **independent reading, real-world
   use, and forcing claim-makers to produce causal evidence**.
4. The **counterfactual preventive value of the remaining machinery is
   unmeasured**, and nothing here was instrumented to measure it.

The greenfield implication is therefore not "no tooling." It is: **start with
provenance conventions, evidence-bearing claims, explicit obligations, and
independent review; instrument prevented failures; then earn automation
incrementally from observed failure modes.**

That conclusion survives the taxonomy correction in §2, which the original
"humans good, guards bad" framing did not.

---

# The same document, in plain English

Nothing new is claimed here. This is sections 1–9 again, for a reader who has
not worked on this project. Where the version above uses a word this team
invented, this version says what it means.

## What happened

A group of AI agents spent about three days working on a research project.
Almost all of that time went into building tools to check their own work,
rather than into the research.

The numbers: 190 pieces of work were saved in that period. 17 of them touched
the actual science. The other 173 were tools, tests, and documents about how
to work. They also wrote roughly 80,000 words of process documentation, which
is about the length of a novel.

This document asks a simple question about all that tooling: **did any of it
actually catch a mistake that mattered?**

## How we checked

Four AI reviewers were given the project cold. None of them had worked on it,
so none of them had a reason to defend it. One built a list of every mistake
the tools ever caught. One asked which tools would be useful on a different
project. One looked at the private vocabulary the team invented. One asked
what a sensible team would have built on day one.

The one with the list of mistakes got the final say, because it was the only
one working from evidence rather than opinion.

## What we found

**A catalogue of 42 entries, of which 38 count.** The review set itself a
rule: no entry counts unless you can point at the exact change that fixed it.
Four entries fail that rule, so they are left in the catalogue but kept out of
every conclusion drawn from it.

Of the 38 that count: in six, something written down and saved was actually
wrong. In one, a tool caught a wrong thing before it got saved. The other 31
were real problems that never affected any result.

**The most useful thing in this review:** the two most complicated tools the
team built caused more problems than they found. One of them, a tool for
recording how numbers were produced, wrote down that it had recorded
something when it had not. Someone then quoted that false record. The other,
a tool for spotting which test files changed, missed eight files that were
deleted and gave a clean report over them. Neither tool ever caught anything.

That finding is the one to keep. It does not depend on any judgement call
about how mistakes were categorised.

**A mistake I made, and had to correct.** My first summary said no automated
check ever caught a genuinely wrong result, and that people caught all of
them. That was unfair, and the unfairness was built into my own scoring.

I had defined "a wrong result" as something already written down and saved.
But automated checks run *before* things get saved. So they could never
qualify, no matter how well they worked. I was comparing two groups that
could not be scored the same way. It is the same kind of error this project
catalogues, made inside a review of that kind of error.

**A place where the project's own files corrected me.** I had written that a
future team should build the simple list of things-to-check, but skip the
detailed form that goes with it. Then I found a note in the project saying
that one field on that form — the one asking "prove this check can actually
fail" — is what exposed two broken checks that human review had missed. My
advice would have thrown away the one part that worked.

The two problems it found were still harmless, so this does not make the tool
a success. But it moves the credit. What worked was **a form that forced a
person to supply evidence**, not the software that read the form afterwards.

## What actually caught the real mistakes

People did. Specifically:

- someone outside the work reading the records and disagreeing
- someone using a tool for real and noticing it misbehave
- an author going back and re-checking their own claim

Every tool built *before* anyone needed it scored zero.

I first wrote "so it is the timing that mattered, not the tools." That claims
more than we know — it treats an observed pattern as an explanation. The
careful version: in these three days, the things that paid off were the ones
built or used in response to an actual problem. The things built in advance
have not paid off *yet*. That is a description, not a rule.

## An important thing I got wrong the first time

"Never went off" is not the same as "useless."

There is a check that stops the team from spending money by accident. It has
never gone off. That is what success looks like for that kind of check — it
means nobody has tried to overspend. My first draft suggested replacing it
with a billing alert, which would tell you *after* the money is gone. That is
worse, dressed up as a saving.

So the scoring now separates three things: checks that ran and found only
harmless problems; checks that never ran because the bad thing never
happened; and checks nobody has ever exercised, where we honestly cannot
tell.

## A fair objection, which we cannot answer

Someone could reasonably say: "the tools worked *so* well that mistakes never
got made, so of course your list of caught mistakes is empty." That is a real
possibility and this review cannot rule it out.

To settle it you would need a record of every time a tool stopped something,
kept alongside what the person was about to write. This project has no such
record. Of the last 100 automated runs, six failed, and all six were the
checking tools failing on themselves. None of them stopped a piece of science.

I first wrote that this makes *every* claim about the tooling untestable.
That went too far and threw away the evidence this document spent its length
gathering. Plenty is directly known: one tool caused four problems and found
none; another wrote down a false record that someone then quoted; outside
readers found specific errors; one form field exposed two broken checks.

What we genuinely cannot measure is the only question that settles the whole
argument: **how many mistakes would have happened without any of it.** That
number is unknown, and nothing here was built to find out.

## A check we argued for, which turned out not to be needed

Mapping the plan as a diagram turned up what looked like the best find of the
whole exercise. The final experiment can only be run once — look at the
answer, and you have used up your one look. The plan says clearly *who* is
allowed to look and *when*, but never says what has to be true first. That is
a permission, not a standard. Reading the plan as prose you notice nothing
missing, because nothing is missing; the section describing the final run is
complete. The hole only appears when you ask what opens the door.

The final run has since happened, with no such check in place, and it went
fine.

That is recorded here rather than quietly dropped, because it argues against
the case this document makes, and a list that only ever collects wins is the
exact problem described at the start. It is scored as "the bad thing never
happened", not as "we got away with it" — those are different claims and only
the first one is supported.

What did protect the final run was something else: the practice of doing a
small cheap version first. The small version failed honestly, which caught a
problem before the expensive run rather than after. If a new project could
only build one of the two — the cheap rehearsal or the door check — the
evidence here says build the rehearsal.

## The words the team invented

Some of the invented terms do real work. Three labels for different kinds of
commitment genuinely change what a person must write down. A word for "a test
that was narrowed down" changed how two tools were built.

Some are just new names for old ideas. "Checkpoint branch" is an integration
branch. "Self-referential pin" is a test that checks a value equals itself.

Two terms are relied on and defined nowhere at all. The worse one is
"adjacent evidence." It appears five times, always as the reason a row in a
file is left blank, and blank rows fail the project's own readiness check. A
new reader has to guess what it means, and a wrong guess flips the project's
status from "not ready" to "ready."

One term is defined twice, differently, which is worse than not defining it.
`break_demonstrated` means one thing in the first half of a file and another
thing in the second half. The software only checks that the field is filled
in, not which meaning was used.

**And the finding that prompted this section.** The project keeps a glossary
with a plain-English version alongside it. That plain-English version explains
all 26 scientific terms and none of the 4 terms the agents invented for
themselves. The vocabulary written for humans got translated. The vocabulary
the agents wrote for their own use never did — and it is the half you need in
order to touch the project's central file.

This section exists because it would have been absurd to write that finding
and not act on it.

## What a new project should do on day one

Four things, and none of them are software:

1. **Never let a run announce success on its own say-so.** Every check it
   claims to have passed must have a line showing what it actually measured.
   No line means no pass. A run once reported success for a check that had
   never been written.
2. **Any claim with a number in it carries a link to where the number came
   from.** Three of the six real mistakes were claims nobody could trace.
3. **Keep a plain list of what the project has promised to check, with one
   column that forces you to prove each check can fail.** Not the software
   that reads the list — just the list and that one column.
4. **Have someone who did not write the work read it.** This caught five of
   the six real mistakes. It costs nothing and it beat every tool.

## What to build later, if ever

Most of the tooling should wait for a real problem to justify it. The
document gives each one a trigger: build the check for "did all the tests
quietly skip?" the first time all the tests quietly skip *and* somebody
believes the result. Build the tool for comparing changed files only after
the standard one proves insufficient — this project built it first, and it
cost four problems.

Two large things do not appear in the evidence at all: the messaging system
the agents use to talk to each other, and the 80,000 words of process
documentation. Not appearing is weaker evidence than scoring badly.

The precise complaint is that nobody has shown these are worth their cost —
not that they have been shown to be waste. Over three days those are very
different statements, and only the first is supported.

## If you want to reuse any of this elsewhere

Eight of the tools would work in another project by changing exactly one
setting each, and the document names which setting for each one. That naming
is the point: if nobody can name the one thing that would have to change,
nobody has actually checked that the tool is portable.

A few need real work. One has the build system baked into it. One depends on
a private form whose shape must be decided early, because the project
identifies each entry by a fingerprint of its own text — reword an entry
later and the link to it breaks.

**That last part is a mistake worth not repeating.** Naming a thing by its own
wording means you cannot improve the wording without losing track of the
thing. A new project should give each entry a plain label that never changes,
and keep the fingerprint of the text in a separate column. Then rewording
produces a useful warning — "this text changed, check the decision still
holds" — instead of silently breaking the link. The cost is one extra column.
The current design teaches people not to improve their own documents, which
is the wrong lesson to build in.

And one warning is recorded rather than hidden: the automated review re-runs
on every single change. That is a known problem, not a feature. Part of it has
since been fixed — the review now uses a smaller, cheaper model and is
forbidden from spawning helpers of its own, which took one run from about $5
down to roughly 25 cents. But it still runs on every change, and nobody has
yet decided what it should and should not look at. Making each run cheap is
not the same as deciding it should happen.

That episode is worth recording for a second reason. The expensive run had
looked at half the files it was given, written a note saying it was still
working, and then reported success anyway. A person reading the note could
tell it was unfinished; the pass/fail flag that everything automatic reads
could not — which is precisely the failure
this whole review process exists to catch, committed by the review process
itself. Two fixes followed: the job now fails if it leaves any of the current
batch unexamined, and whatever it did not get to is carried forward in writing
so it cannot quietly vanish. The first of those was tested by breaking it on
purpose in both directions, which is the standard this project sets and does
not always meet.
