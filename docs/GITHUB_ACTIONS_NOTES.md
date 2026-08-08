# What we learned about `claude-code-action`

Everything here was read out of the action's own source at
`anthropics/claude-code-action@v1` (and, where noted,
`github/github-mcp-server` v0.17.1, which it runs as a container). Every
claim carries a file:line so it can be re-checked when the action moves.
Nothing here is inferred from behaviour alone.

It is written because this repository spent a day building three
increasingly elaborate workarounds for a problem that did not exist, and the
cost of each was a green build that meant nothing.

---

## 1. The one thing to know: two modes, and only one has context

The action auto-detects a mode. You never select it directly, and the choice
silently determines whether the model receives any GitHub context at all.

`src/modes/detector.ts:64-77` — on a `pull_request` event:

```ts
if (context.eventAction && supportedActions.includes(context.eventAction)) {
  // If prompt is provided, use agent mode (default for automation)
  if (context.inputs.prompt) {
    return "agent";
  }
}
```

`src/modes/agent/index.ts:86-90` — agent mode's *entire* prompt file:

```ts
const promptContent =
  context.inputs.prompt ||
  `Repository: ${context.repository.owner}/${context.repository.repo}`;
```

**Supplying `prompt:` selects agent mode, and agent mode injects nothing.**
No changed files, no PR body, no comments. `prepareAgentMode` never calls
`fetchGitHubData`.

This is the trap, and it is the wrong way round from what you would guess:
*writing a more specific prompt takes context away.* A workflow that says
"review this PR" and then discovers the model cannot see the PR will
naturally reach for `gh` and `git` to fetch what it is missing — which is
how a review ends up reconstructing, badly, what the platform already knew.

### The fix

```yaml
with:
  track_progress: true
  prompt: |
    ...your instructions...
```

`src/modes/detector.ts:21-31` forces tag mode for PR events even when
`prompt` is set, and `src/create-prompt/index.ts:479-488` then appends your
prompt to the full auto-context:

```ts
if (context.githubContext?.inputs?.prompt) {
  return defaultPrompt + `\n\n<custom_instructions>\n${...}\n</custom_instructions>`;
}
```

Same instructions, all the context. Two consequences to accept: it creates a
tracking comment on the PR (`src/modes/tag/index.ts:46`), and it adds
`mcp__github_comment__update_claude_comment` to the allowed tools.

Valid events and actions are gated at `detector.ts:100-114` —
`opened, synchronize, ready_for_review, reopened, labeled`. On anything else
it **throws**, which is why a `workflow_dispatch` entry point cannot have
this context (see §5).

---

## 2. What tag mode actually injects

From `src/create-prompt/index.ts:664-713`:

| Tag | Contents |
|---|---|
| `<formatted_context>` | title, author, `head -> base`, state, labels, ±line counts, commit count, changed-file **count** |
| `<pr_or_issue_body>` | PR body, sanitized, image URLs rewritten to local paths |
| `<comments>` | issue-level comments |
| `<review_comments>` | review bodies plus inline comments, with `path:line` |
| `<changed_files>` | `- path (CHANGETYPE) +N/-M SHA: <sha>` per file |
| `<event_type>`, `<is_pr>`, `<repository>`, `<pr_number>`, … | scalars |
| `<custom_instructions>` | your `prompt` input |

### Two limits worth knowing before you design around it

**There are no diffs.** `PR_QUERY`
(`src/github/api/queries/github.ts:46-53`) selects only:

```graphql
files(first: 100) { nodes { path additions deletions changeType } }
```

No `patch`, no diff text, anywhere. The default prompt compensates by
*telling the model to run git itself* (`index.ts:719`): "When comparing PR
changes, use `origin/${baseBranch}` as the base reference". If your checkout
is shallow (`fetch-depth: 1`) that ref is absent, so either fetch deeply or
tell the model explicitly not to.

**The SHA is not a blob SHA.** `fetcher.ts:493` computes it locally:

```ts
const sha = execFileSync("git", ["hash-object", file.path], ...)
```

Useful for identity; useless for fetching from the API. Sentinels:
`"deleted"` for deleted files, `"unknown"` on failure.

---

## 3. The silent ceilings — the dangerous part

`fetchGitHubData` issues **one** GraphQL call with no pagination loop
(`fetcher.ts:415`), so everything past the first page is dropped without a
word:

| Field | Cap | Line |
|---|---|---|
| `files` | 100 | `github.ts:46` |
| `comments` | 100 | `github.ts:54` |
| `reviews` | 100 | `github.ts:68` |
| review `comments` | 100 | `github.ts:80` |
| `commits` | 100 | `github.ts:33` |

Worse, `fetcher.ts:427-432`:

```ts
if (pullRequest.files === null) {
  console.warn(`GitHub did not return the file list for PR #${prNumber} (diff likely too large); proceeding without file-level context`);
}
changedFiles = pullRequest.files?.nodes ?? [];
```

**On a very large PR the file list comes back empty and the action
continues.** The model sees `Changed Files: unknown (file list unavailable)`
and `<changed_files>No files changed</changed_files>` — which is
indistinguishable, from inside the model, from a PR that touched nothing.

The failure this produces is the worst kind: a review that **honestly**
reports "no test files changed" over a PR that changed hundreds. It is not a
careless model, so no amount of prompt discipline prevents it, and any check
that trusts the review's own account of its scope will agree with it.

**The only defence is a second, independent source.** Our publisher asks
GitHub directly (`gh pr diff --name-only`, needs `GH_TOKEN`) and fails when
the two disagree. Both directions are pinned in
`tests/test_publish_review.sh`: contradicted fails, confirmed passes —
otherwise the guard makes the honest case unreportable.

---

## 4. Consequences for a review workflow

Once you have tag mode, most of what a hand-rolled review workflow does
becomes unnecessary. What we deleted:

- **All git.** The changed-file list is injected. For a *vacuous-test*
  review no diff is needed at all: vacuity is a property of the test as
  written — a test that cannot fail could not fail before the diff either —
  so the list says which files to read and the checked-out tree supplies
  them.
- **All `gh` from the model.** Same reason. It remains in the *publisher
  step*, which is a different thing: an independent check, not the review.
- **`fetch-depth: 0`.** Only ranges needed deep history.
- **The commit-range input.** See §5.

The allowlist that remains is `Read,Grep,Glob` plus two MCP tools. Note that
in tag mode **the report IS the tracking comment**: without
`mcp__github_comment__update_claude_comment` the review runs, costs money,
and publishes nothing.

---

## 5. `workflow_dispatch` cannot have PR context

Not "is awkward to give context to" — cannot.

1. `src/github/context.ts:222-228` — `workflow_dispatch` returns an
   `AutomationContext`, which by type has **no `entityNumber` and no `isPR`**.
2. `context.ts:298-302` — `isEntityContext()` is true only for the five
   entity events; `workflow_dispatch` is an automation event.
3. `detector.ts:80` — automation events fall through to `return "agent"`.
4. `modes/tag/index.ts:38-40` — tag mode hard-throws "Tag mode requires
   entity context".
5. `detector.ts:83-97` — `track_progress` throws on any non-entity event.

`entityNumber` is assigned in exactly five places (`context.ts:177, 187, 198,
208, 218`), all from `payload.issue.number` or `payload.pull_request.number`.
No input, env var, or `client_payload` writes it. The official
`workflow_dispatch` example (`examples/manual-code-analysis.yml:34-36`)
hand-rolls its context for exactly this reason, and
`docs/custom-automations.md:25` still lists `workflow_dispatch` as
"coming soon".

**So a dispatched run is structurally blind.** We removed ours. To re-run a
review, "Re-run all jobs" replays the original `pull_request` event with its
context intact; a new push re-triggers via `synchronize`.

---

## 6. MCP servers, and when each is installed

`src/mcp/install-mcp-server.ts`. Installation is *conditional on the
allowlist* — naming a tool prefix is what turns the server on.

| Server | Prefix | Installed when | Tools |
|---|---|---|---|
| `github_comment` | `mcp__github_comment__` | always in tag mode; in agent mode only if listed (`:109`) | `update_claude_comment` |
| `github_inline_comment` | `mcp__github_inline_comment__` | PR context **and** listed (`:146-150`) | `create_inline_comment` |
| `github_ci` | `mcp__github_ci__` | PR context, a token, **and** a live `actions: read` probe (`:171-179`) | `get_ci_status`, `get_workflow_run_details`, `download_job_log` |
| `github_file_ops` | `mcp__github_file_ops__` | `use_commit_signing: true` (`:127`) | `commit_files`, `delete_files` |
| `github` | `mcp__github__` | listed (`:208`) | external container, see below |

### If you enable `mcp__github__`

It runs `ghcr.io/github/github-mcp-server` with only
`GITHUB_PERSONAL_ACCESS_TOKEN` and `GITHUB_HOST` set
(`install-mcp-server.ts:219`). That means **read-only mode is OFF and the
default toolsets are live** — `merge_pull_request`, `push_files`,
`delete_file` and friends are all exposed. Set `GITHUB_READ_ONLY=1` as the
server-side belt to the allowlist's braces.

The real tool names for reviewing (from the server's Go source) are
`pull_request_read` — a method-dispatch tool covering `get`, `get_files`
(with per-file `patch`), `get_diff`, `get_reviews`, … — and
`get_file_contents`, whose `ref` accepts `refs/pull/{n}/head`.

Names that **do not exist**, despite being the obvious guesses:
`get_pull_request`, `get_pull_request_files`, `get_pull_request_diff`,
`list_pull_request_files`, `get_pull_request_comments`. A wrong name in an
allowlist never matches and never errors — the tool is simply absent and the
model quietly works around it.

Also note `request_copilot_review` is **mutating** despite the name, and is
in a default toolset.

---

## 7. Inputs that turn out not to exist

`override_prompt` is **not** in `action.yml`. It survives only in
`src/entrypoints/collect-inputs.ts:17` as a telemetry key and in the docs as
deprecated. The same is true of `mode`, `model`, `allowed_tools`,
`disallowed_tools`, `custom_instructions`, `direct_prompt`, and `max_turns` —
all removed, all still referenced in blog posts and older examples.

Current inputs are `prompt`, `claude_args` (raw CLI args), `settings`,
`track_progress`, `display_report`, `use_sticky_comment`, the auth family,
and the actor filters. Outputs are `conclusion`, `execution_file`,
`structured_output`, `branch_name`, `github_token`, `session_id`.

Two traps we hit:

- **`execution_file` is a PATH, not JSON.** `structured_output` is the
  schema-validated object. Passing the first where the second is expected
  costs a red build — and, on the day, a guard that printed "the review
  produced no result" directly beneath the review's rendered result.
- **`--allowed-tools` vs `--allowedTools`.** The examples use the camelCase
  form; we use the hyphenated one because that is the spelling with measured
  evidence behind it here (denials fell from twelve to four). Switching on
  the strength of an example risks silently disabling the allowlist, so our
  test accepts either and fails loudly when neither is present — "no
  allowlist" and "wrong allowlist" need different fixes.

---

## 8. Workflow-file rules that bit us

- **A workflow must be byte-identical to the default branch's copy**, or the
  action **skips and the job reports SUCCESS**. A green run and a real review
  are indistinguishable from outside. `tools/ci/check_workflow_parity.sh`
  exists solely for this, and it needs to compare the *working tree* as well
  as the two published refs — comparing only the two published copies
  reported "no drift" while the branch about to become the default carried a
  different file.
- **A `workflow_dispatch` free-text input is unvalidated.** `origin/staging...HEAD`
  is well-formed, names a branch that does not exist, and produced a green
  59-second run.
- **`HEAD~1..HEAD` is empty after a merge** — the merge commit's tree equals
  its parent's — so it reviews nothing and reports success.
- **`gh pr diff` returns HTTP 406 above 20,000 lines.** On a large PR it
  cannot retrieve the diff at all, and a review that proceeds on what it
  could reach will describe a subset as the whole unless something measures
  coverage.

---

## 9. The general lesson

Every failure above is one shape: **a green that means less than it says**,
and in every case the cause was reconstructing context the platform already
had rather than reading how the platform provides it.

The specific mistake worth naming is that a *more specific prompt removed
context*. That is not guessable from the outside, it has no runtime symptom,
and it makes the workflow appear to work while the model is blind. We found
it by cloning the action and reading `detector.ts` — after building three
workarounds for its consequences.

When an integration behaves oddly, read its source before designing around
it. It is usually a shorter path than the workaround, and it is the only one
that tells you what the workaround would have been hiding.
