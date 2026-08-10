---
description: Map of this project's GitHub surface — the Actions review workflow, Cloud Build CI, the Terraform that creates it, and the guards under tools/ci/. Use before touching a workflow file, a trigger, a CI guard, or anything that reads the GitHub API.
argument-hint: "[review | ci | infra | api]"
---

# GitHub, Actions and Cloud Build in this repository

This surface has conventions, guards and measured notes that already exist.
Rediscovering them by trial is expensive here in a specific way: **the two
most common failures both report success.** A workflow that skips is a green
job, and a review that examined nothing posts a clean comment.

This file is a map. Each document it names is the authority for its area and
moves independently — read the source rather than trusting a summary of it.

## The load-bearing gotchas

Four facts that are not guessable and that have each cost a real failure:

1. **A Claude workflow file must be byte-identical to the version on the
   default branch, or the action SKIPS — and a skip reports job success.**
   Editing `.github/workflows/claude-code-review.yml` on one branch silently
   disables it until `main` matches. `tools/ci/check_workflow_parity.sh` is
   the guard. This is why the review's prompt is kept short and its
   maintained content lives in `docs/REVIEW_COMMENT_TEMPLATE.md`: a template
   in the repository can be improved on any branch; a prompt cannot.

2. **Supplying `prompt:` on a `pull_request` event selects AGENT mode, which
   injects no GitHub context at all.** `track_progress: true` forces TAG
   mode, which supplies the PR body, comments, review comments and changed
   files, and passes the prompt as `<custom_instructions>`. Without it the
   review runs on the prompt alone and reports on nothing.

3. **Cloud Build triggers are `location = "global"`.** `cloudbuild.yaml`'s
   own `gcloud builds list` is un-regioned and cannot see regional builds.
   2nd-generation repository connections are regional, so adopting one means
   fixing that query first.

4. **Reach `stage2b-ci` by pull request, never a direct push.** A direct
   merge skips the review and produces a green CI that nothing reviewed. The
   PR is what makes the review fire on the diff and CI gate the merge.

## Where things are

### The review

| | |
|---|---|
| `.github/workflows/claude-code-review.yml` | fires on PRs whose BASE is `stage2b-ci`, when `tests/**` changed |
| `docs/GITHUB_ACTIONS_NOTES.md` | what `claude-code-action` actually does, cited to file and line |
| `docs/REVIEW_COMMENT_TEMPLATE.md` | the one sticky comment it maintains, and the rules for it |
| `tools/ci/review_delta.sh` | push delta + DEPARTED + sticky OUTSTANDING carry-forward |
| `tools/ci/vacuous_review_local.sh` | local Haiku preflight; `make vacuous-review PR=N` |
| `tools/ci/publish_review.sh` | publishes the result and fails if the review produced none |
| `tools/ci/review_run.sh` | a run's own telemetry: artifact URL, duration, turns, cost |

Two things about the API that mislead if assumed:

- The action's PR query selects `path additions deletions changeType` and
  **carries no diffs**. A review that needs file contents reads them from the
  checked-out working tree.
- `.files[].filename` from the compare API is the path **at `after`** — for a
  rename, the destination. The source is in `.previous_filename`. Filtering
  only `.filename` cannot see a file that moved OUT of a watched directory.

- **Sticky carry-forward is mechanical.** `review_delta.sh` re-reads the
  PR's vacuous-test sticky and unions *Not examined* / unchecked `- [ ]`
  paths into the next run — even on a docs-only push. Without that, a
  partial pass (PR #29: 6 of 12 files, $5 Sonnet) evaporates on the next
  synchronize.
- **Local preflight before the checkpoint PR:** `make vacuous-review PR=N`
  (Haiku default). Measured on PR #29: local Haiku ~$0.25 / ~1 min / 12 files;
  Actions Sonnet was ~$5 / partial; Actions Haiku re-run ~$0.32 / 12 files.
  Do not pass `--bare` to the CLI here — bare skips OAuth and reports
  "Not logged in" on a logged-in machine. Workflow is Haiku-pinned
  (`--model haiku`, Agent disallowed); keep it byte-identical on `main` and
  `stage2b` or the action SKIPS green (`check_workflow_parity.sh`).

### CI

| | |
|---|---|
| `cloudbuild.yaml` | the build itself |
| `tools/ci/check_suite_not_vacuous.py` + `tools/ci/ci_skip_baseline.txt` | every skip in CI is accounted for; both directions fail the build |
| `tools/ci/assert_no_cloud_credentials.py` | CI is credential-free, and proves it |
| `tools/ci/ci_targets.py` | what CI may invoke — the spend guard over the Makefile's GPU and GCS targets |
| `tools/ci/check_workflow_parity.sh` | the byte-identical check from gotcha 1 |

The skip baseline is a statement about capability, not a way to make a build
green. Adding an entry asserts CI is *expected* to lack something; if the
right answer is that a test does not belong in the science suite, move the
test rather than widening the baseline.

### Infrastructure

| | |
|---|---|
| `infra/` | Terraform for the whole Cloud Build setup; `terraform -chdir=infra plan` |
| `infra/triggers.tf` | three triggers — checkpoint (PRs into `stage2b-ci`), deps, manual |
| `infra/README.md` | the branch-protection table |

**There is deliberately no per-push trigger.** Agents run the fast suite
locally and habitually; a cloud run of the same suite on the same commit adds
Linux/x86, a clean checkout, and a record, and none of those change between
one push and the next. The reasoning and the measurement behind it are in the
header of `infra/triggers.tf`.

Every trigger names an explicit `service_account`. A trigger without one runs
as the legacy default Cloud Build identity, which on this project holds
project-level roles including `storage.objects.delete` — reaching the science
bucket. `tools/ci/assert_no_cloud_credentials.py` cannot see that: a Cloud Build
credential arrives from the metadata server, not an environment variable.

## Working on any of it

- **Prefer `gh` to reconstructing context.** The base, head, file list,
  comments and check results are all in GitHub already. Three range-based
  review designs were built here from git plumbing and all three failed the
  same way. The checkout may also be shallow.
- **A narrowing must be tested with the narrowing, not with the broader
  form.** An explicit list, a filtered target, or a path filter verified by
  running the general case proves the code works and says nothing about the
  narrowing. Derive the set from the filesystem or the AST where possible;
  where a list must stay explicit, assert it equals the derived set in both
  directions.
- **Break-confirm every guard.** Change the thing it watches, watch it fail
  with the specific expected message, then revert. A guard nobody has seen
  fail is not yet a guard — and on this surface, the untested failure mode is
  usually a silent green rather than a red.
- **Announce before changing shared surface.** A workflow file, a trigger, or
  a hook registration alters behaviour inside other sessions. Workflow and
  hook changes in particular do not propagate until a session restarts, so
  announcing is the only channel. See `docs/MULTI_AGENT_PRACTICE.md`.
