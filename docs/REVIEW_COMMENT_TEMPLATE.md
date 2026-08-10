# The review comment: one comment, maintained

The vacuous-test review keeps **one** comment per pull request and rewrites
it on every run. This file is the structure it maintains.

It lives here rather than in the workflow prompt for a practical reason: the
workflow file must be byte-identical on `main` and `stage2b` or the action
skips and reports success, so every prompt edit costs a hand-sync of two
branches. A template in the repository can be improved by anyone, on any
branch, without that.

## Why one comment rather than a thread

A new comment per run makes the newest review the hardest to find, buries
whether a finding was ever fixed, and re-reports the same thing until a
reader learns to skim — at which point the review has stopped working. It
also grows the context every subsequent run must read, since prior comments
are injected back into the prompt.

The rule that follows: **a finding disappears from the comment only when it
is FIXED, never because it has already been mentioned.** Suppressing a
still-open finding to avoid repetition would convert a defect into a
silence, which is the exact failure this project catalogues.

## The structure

Everything below is maintained in place. Keep the headings, keep the order,
and keep the tables even when empty — an empty table headed "Open findings"
says something a missing section does not.

```markdown
## Vacuous-test review

**Scope.** `tests/*.py` changed in this PR. Dot-directory tooling is out of
scope and is not counted.

**Status:** <one line — e.g. "2 open findings, 1 fixed since last run">
**Last run:** <ISO timestamp> · <N> of <M> changed test files examined

### Open findings

| # | file::test | category | what would have to change for it to fail | first seen |
|---|---|---|---|---|
| 1 | `tests/test_x.py::test_y` | A | nothing — it greps source | this run |
| 2 | `tests/test_z.py::test_w` | F | nothing — narrowing checked by the glob | run 3 |

<!-- `first seen` takes "this run" or a run number YOU RECORDED IN THE RUN
     LOG BELOW. If there is no prior comment, every row says "this run".
     Never "prior review round" — that is a value invented to fill a
     column, and the column is not worth a fabricated history. -->


### Fixed since first reported

| # | file::test | category | fixed in |
|---|---|---|---|
| 2 | `tests/test_z.py::test_w` | F | `abc1234` |

### Examined

<details><summary>Files read, and when</summary>

| file | last examined | verdict |
|---|---|---|
| `tests/test_x.py` | run 4 | 1 open finding |
| `tests/test_z.py` | run 4 | clean |

</details>

### Left scope

<!-- In-scope tests that were DELETED or MOVED OUT of `tests/` in this push.
     The delta reports these under a `DEPARTED` heading; they do not exist in
     the working tree, so do not try to read them. Record where each one went
     and, in one clause, whether the move looks legitimate.

     Keep the section even when empty -- "none" is a fact worth stating on a
     PR that removed tests, and its absence is not. -->

| file | where it went | note |
|---|---|---|
| `tests/test_x.py` | `.claude/tooling/test_x.py` | moved out of the science suite |

### Not examined

<!-- Only in-scope files. If a changed test file was not read, say so and
     why. An absent row here is a claim that everything in scope was read. -->

### Run log

<!-- One line per run. Newest first. Keep it short; this is provenance, not
     narrative. -->

- run 4 · <timestamp> · re-examined 2 changed files · no new findings
- run 3 · <timestamp> · full pass over 7 files · 2 findings
```

## Rules for maintaining it

1. **Read the existing comment first.** It is your own prior state, and it
   is injected into your context along with the rest of the thread.

2. **If `<comments>` says "No comments", YOU HAVE NO PRIOR STATE.** This is
   your first run on this pull request. Every finding is `first seen: this
   run`, the *Fixed* table is empty because nothing has been fixed yet, and
   the run log has exactly one line. Do not write "prior review round", do
   not carry anything forward, and do not infer a history from the fact that
   this document describes one.

   Observed on PR #28: the first run received `<comments>: No comments` and
   an empty `<review_comments>`, and still filed a finding marked *"first
   seen: prior review round"*. The finding was real; its history was
   invented. A required column with no honest value gets one made up — which
   is the same failure `tools/gates/gate_inventory.py` documents for
   `discharged_in`, and the reason that field is required only when a row
   claims to be discharged.

3. **Carry every open finding forward**, with its original `first seen` —
   **only when there is a prior comment to carry it from.**
   Re-verify it against the current code: if the code changed such that the
   finding no longer holds, move it to *Fixed* and name the commit. If it
   still holds, leave it — do not re-litigate it, and do not restate its
   reasoning at length a second time.
3. **A finding you did not re-check is still open.** Never drop a row
   because this run did not look at that file.
4. **Number findings once.** A finding keeps its number for the life of the
   PR, so a comment elsewhere can refer to "finding 2" and still be right
   next week.
5. **Only examine what changed since your last run**, plus anything with an
   open finding, plus anything still listed under *Not examined* (or on an
   unchecked `- [ ]` line). Re-reading an unchanged file you already cleared
   is the cost this structure exists to remove.

   **Carry-forward is mechanical.** `tools/ci/review_delta.sh` re-reads this
   comment on every run and unions those unfinished paths into the next
   review's file list — even on a push that touches no tests. That is what
   keeps a partial pass (PR #29 run 31394098469: 6 of 12 files, $5, "in
   progress") from evaporating on the next synchronize. A path leaves that
   backlog only when a later run actually examines it and drops it from this
   comment. Leaving *Not examined* empty while files remain unread is a
   silent drop.
6. **If a previous run reported clean and you find something, say so
   explicitly** in the run log. Two runs of this review have disagreed about
   the same file on the same PR, and a disagreement is information — the
   later verdict is not automatically the right one.

7. **A test leaving scope is a reportable event, not an absence.** Deleting a
   test file, or moving it out of `tests/`, removes it from every future
   review — so the run in which it happens is the only one that can say so.
   Record it under *Left scope* and carry any open finding it held into
   *Fixed* only if the code it guarded also went; otherwise the finding is
   still open and now unguarded, which is worse, and it stays open.

   This is not hypothetical. `review_delta.sh` reported "no in-scope test
   files changed" for a push that moved eight tests out of `tests/`, because
   the compare API returns a rename's DESTINATION in `.filename` and the
   source only in `.previous_filename`. Fixed 2026-08-08; the delta now emits
   a `DEPARTED` section, and this is where it lands.

## What must never happen

- A finding vanishing without appearing under *Fixed*.
- A test leaving scope with no row under *Left scope* — the one run that
  could report it is the one where it happened.
- *Examined* listing a file the run did not actually read.
- The status line claiming full coverage when *Not examined* is non-empty.
- Findings gating the build. This comment advises; the deterministic checks
  in `cloudbuild.yaml` are what gate, and keeping those apart is deliberate.
