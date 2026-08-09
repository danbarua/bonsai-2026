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
peers using tools on real work and outside readers reading records. On this
evidence it is the **ordering** that paid, not the tooling: build in response
to an incident, and the same component lands differently.

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
branch, at a measured **$1.2769 / 169.7 s / 26 turns / 13,751 output tokens**
per run (from `tools/ci/review_run.sh`). Tasks #22/#23 concern this. The
workflow's in-line prose defends the current trigger and must not be carried
forward as settled.

### 5g. Justified by principle, not by a caught defect

Flagged by the extraction agent, adjudicated against the ledger:
`assert_no_cloud_credentials.py` (UNFIRED — hazard absent; keep) ·
`triage_candidates.py` · `ci_targets.py` (cites principle twice; its "caught"
example is hypothetical) · `gate_inventory.py`'s trichotomy · the 100%
coverage floor and 24 finding kinds. None of these cite an incident.

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
| Anti-vacuity check | First time a suite passes with everything skipped *and* a claim is drawn from it |
| Deliberate breakage tests | First time a guard a *result depended on* is found unable to fail |
| Doc-count tests | A second prose-count error reaching a science record. Cheaper: stop putting literal counts in prose |
| Dependency pinning guards | First time a dependency change moves a numeric result |
| A review-delta tool | Only after the platform's own diff view is demonstrably insufficient — it was built first and cost four defects |

### Not measured at all

The **agent mail mesh** and the **80,253 words of process docs** (32
docs-only commits) appear nowhere in the ledger. Absence from a ledger is
weaker evidence than an UNPAID verdict — but unmeasured apparatus at that
cost is, by default, unjustified.

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

**Undefined but relied upon.** `adjacent evidence` — zero definitions, five to
six uses, all in `gates.toml`; a cold reader must guess whether "adjacent"
means a different clause, module, or claim, and guessing "different module"
fills two deliberately empty rows and flips readiness green. Highest-risk gap.
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
Until it does, every efficacy claim about this apparatus — including the
negative ones in this document — is untestable in both directions.

---

# The same document, in plain English

Nothing new is claimed here. This is sections 1–8 again, for a reader who has
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

**A catalogue of 42 mistakes.** In six of them, something written down and
saved was actually wrong. In one, a tool caught a wrong thing before it got
saved. The other 35 were real problems that never affected any result.

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

Every tool built *before* anyone needed it scored zero. Every tool that came
*after* a real problem scored something. On this evidence it is the **timing**
that mattered, not the tools.

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

So: **every claim about whether this tooling was worth it — including the
critical ones in this document — is currently untestable.** That is the
honest position.

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
documentation. Not appearing is weaker evidence than scoring badly. But
something that expensive and that unmeasured is, by default, unjustified.

## If you want to reuse any of this elsewhere

Eight of the tools would work in another project by changing exactly one
setting each, and the document names which setting for each one. That naming
is the point: if nobody can name the one thing that would have to change,
nobody has actually checked that the tool is portable.

A few need real work. One has the build system baked into it. One depends on
a private form whose shape must be decided early, because the project
identifies each entry by a fingerprint of its own text — reword an entry
later and the link to it breaks.

And one warning is recorded rather than hidden: the automated review currently
re-runs on every single change, at about $1.28 a time. That is a known
problem, not a feature, and it is already on the list to fix.
