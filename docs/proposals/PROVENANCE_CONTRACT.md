# The provenance contract: capture at birth, citation at use

Design for GitHub issue #22 (load-bearing scratch hooks). This is the
document CLAUDE.md principle 24 forward-references as "the provenance
contract (capture-at-birth, citation-at-use)" — that reference currently
has no path, and this file is meant to be it.

**Status: design.** One part is built and measured (the payload probe in
`tools/provenance/`, whose numbers this document quotes). The capture
hooks, the classification skill, and the citation verifier are specified
here and not yet implemented.

**Where this lives, and why here.** `docs/proposals/` rather than `docs/`,
because the infrastructure track edits shared documents only as
patches-in-waiting. Two companion proposals accompany this one:
`CLAUDE_MD_PRINCIPLE_24_AMENDMENT.md` (adding the path principle 24 is
missing) and nothing else — this document does not amend
`PROJECT_MEMORY.md` or `MULTI_AGENT_PRACTICE.md`.

---

## 1. The problem, restated precisely

A number that anchors a decision is either reproducible from committed
code, or it is not a decision anchor (principle 24). The failure mode is
narrow and specific: an **ephemeral script** — a heredoc, a `python -c`, a
tmp file, an inline `mighty-colab exec` — whose **output enters a durable
record** while the script itself dies with the session. The number
survives; its generator does not. Six months later the number is a
cached conclusion whose provenance is a chat transcript nobody can re-read.

Two sides, and the whole design turns on keeping them separate:

- **Capture** is forensic and unconditional. It runs at the moment the
  scratch executes, costs the author nothing, blocks nothing, and judges
  nothing. Its output is evidence, not a citable source.
- **Citation** is a verifier-side check on durable documents and frozen
  constants. It runs later, in a different process, and it fails closed
  only where the cost of a wrong number is highest.

The reason for the split is behavioural. A guard that fires during
exploration teaches route-around, and this project has the receipts on
what route-around costs (`experiments/stage2b_denoising/README.md`,
"Guards you must not route around", and the fake-bucket incident behind
it). Exploration must stay free. The tax is levied at the moment a number
stops being exploratory and enters a document — which is a different
moment, in a different tool call, and is where the checklist belongs.

### 1.1 A contradiction in the source material, and how it is resolved

Issue #22 §2 says empirical claims "must cite a repo path **or a capture
ID**." CLAUDE.md principle 24 says the remedy for captured scratch that
matters is "promotion to committed code — **never citing the capture**."
The infrastructure kickoff brief restates principle 24's side explicitly:
captures are "never cited — promotion to committed code is the only remedy
for scratch that matters."

**This design follows principle 24 and the brief.** A capture ID is not a
citation and the verifier does not accept one. Recording that here rather
than implementing it silently, because the issue text says otherwise and a
reader comparing the two deserves to know which way it went and why.

The reasoning, beyond deference: a capture ID that satisfies a citation
check would make the capture log load-bearing. The log is gitignored,
run-scoped, and machine-written — promoting it to a citable source
recreates `class0_constructions.pkl` exactly, one level up. The log's
value is that it lets you *find* the script you now have to promote, not
that it stands in for having promoted it.

---

## 2. The measured substrate

Everything in §3 depends on what a hook actually receives. That was
measured, not assumed, because the published documentation is ambiguous on
the point (it describes a 10,000-character cap on hook *output* and a
transcript write lag, neither of which is a statement about the inbound
payload).

Generator: `tools/provenance/run_truncation_probe.py`, which drives
`emit_bytes.py` (a position-labelled stream ending in a sentinel carrying
the digest of everything before it) through a headless `claude -p`
subprocess whose `--settings` registers `probe_hook_payload.py`. Rerun with:

```
uv run python tools/provenance/run_truncation_probe.py
```

| Condition | Field carrying output | Inline cap | Persisted file | Shape of loss |
|---|---|---|---|---|
| Success, ≤ 30,000 chars | `tool_response.stdout` | none hit | **absent** | none — complete |
| Success, > 30,000 chars | `tool_response.stdout` | **30,000 chars** | **present, byte-exact** | tail dropped, silently |
| Output on fd 2 (stderr) | `tool_response.stdout` (folded; `stderr` key empty) | same 30,000 | same | same |
| **Failure (non-zero exit)** | `error` — **no `tool_response` at all** | **10,040 chars** | **absent** | **middle elided** |

Specifics worth keeping:

- The 30,000 cap is **constant, not proportional**: 50 KB and 1 MB of
  output both arrive as exactly 30,000 characters.
- The persisted copy is **complete and verified**, not merely large:
  200,095 and 1,000,015 bytes recovered byte-exact, terminal sentinel
  intact, and the SHA-256 of the recovered body matching the digest the
  emitter computed before sending. `persistedOutputSize` equalled the true
  size in both cases.
- Truncation of the inline field is **unmarked**. The text simply stops
  mid-line. A hook that reads `stdout` and does not compare against
  `persistedOutputSize` cannot tell a complete result from a cut one.
- The persisted path lives under
  `~/.claude/projects/<project>/<session-uuid>/tool-results/`. It is
  session-scoped and will be reaped.
- On the failure path, labelled lines 0..296 were present but
  **non-contiguous** — head and tail retained, middle discarded. The loss
  compounds: the failure text appears to be an elision of the
  already-30,000-capped text, not of the original.

### 2.1 What follows for the design

1. **No command rewriting.** The original sketch for "untruncated stdout"
   was a `PreToolUse` hook rewriting the command via `updatedInput` to tee
   output somewhere. `persistedOutputPath` makes that unnecessary. This is
   a large win: rewriting arbitrary Bash has real blast radius (quoting,
   pipelines, `pipefail`, exit-code propagation, `run_in_background`), and
   a capture mechanism that changes what the user's command *does* is a
   worse failure than an incomplete log.
2. **Copy bytes, never store the path.** The persisted file is ephemeral.
   A capture record holding a pointer into a directory that gets reaped is
   `class0_constructions.pkl` at a new grain, inside the very tool built to
   prevent it. Capture copies the content into the run-scoped log
   directory and records its digest.
3. **The Pre/Post pair is justified by measurement, not by caution.** The
   issue proposed it "to survive mid-run death." The failure path measures
   out as the worst case in the table: no persisted file, a 10,040-char
   cap, and middle elision. A scratch script that dies — the case that
   matters most — is precisely the one no post-hoc event can reconstruct.
   `PreToolUse` is where the script text is available unconditionally.
4. **The script, not the stdout, is the primary asset.** Stdout
   completeness is best-effort with a recorded fidelity flag. The script is
   captured completely in every case, because `PreToolUse` sees
   `tool_input` before anything runs. Promotion to committed code needs the
   generator; the output can be regenerated once the generator exists.

---

## 3. Part 1 — Capture

### 3.1 Events and registration

| Event | Matcher | Purpose |
|---|---|---|
| `PreToolUse` | `Bash`, `mcp__mighty-colab__exec`, `mcp__mighty-colab__run` | Write the `open` record: script text, referenced-file snapshots, cwd, git state. Survives death of the call. |
| `PostToolUse` | same | Write the `close` record: output (from `persistedOutputPath` when named, else inline), fidelity flag, duration. |
| `PostToolUseFailure` | same | Write the `close` record with `outcome: failed`, the capped `error` text, and `fidelity: elided`. |

Registration goes in the committed `.claude/settings.json` — which now
**exists**, shipped at `3ec12ad` on `stage2b` by the c2c-mail hooks. An
earlier draft of this document said the project had none; that was true at
the time of branching and is no longer.

The consequence is a rule, not a footnote: **merge into that file, never
create or clobber it.** Two tracks now write it, so the collision is a
named risk any PR here must handle explicitly, and rebasing against current
`stage2b` before building is the standing discipline.

The c2c-mail hooks in `.claude/hooks/c2c-mail/` are the layout convention
this feature follows: a per-feature directory, shared logic in `lib/`,
`test/` holding break-tests that run each hook against synthetic stdin
under throwaway environment overrides, and a README recording design
decisions and the incidents behind them. Capture departs on one point only
— it is written in Python rather than bash, because it needs shell-aware
tokenising, heredoc extraction, and a predicate importable by pytest.

Narrowing uses the `if` field (permission-rule syntax) so the hook process
is not spawned on every tool call:

```json
{"matcher": "Bash", "hooks": [{"type": "command",
  "if": "Bash(*python -c*)", "command": "..."}]}
```

The `if` clauses are an optimisation, not the decision. The hook **also
self-filters** using the predicate in §3.2, for the reason the probe hook
already does: a settings file is read by parallel sessions, and a mistake
in `if` syntax should cost a missed capture, never a log of somebody else's
work.

### 3.2 What counts as scratch

**The governing rule, and it generalises well past capture:** classify on
what CONSUMES the payload, never on how the payload arrives. `-c` flags,
heredocs and pipes describe *delivery*; the program that receives the text
decides whether any code is being run at all.

`stage2b-lead` derived this from two field-reported defects that were the
same mistake seen from opposite sides — `uv run python -c` invisible because
it was not the first token, and `git commit -F - <<EOF` captured because a
heredoc was present at all. Keying on delivery produced both; keying on the
consumer fixed both.

It is not a capture-local note. The c2c mail hooks filter on *addressee*
rather than on transport for the same reason, and the same error is
available anywhere a system decides what something is by how it showed up.

A single committed predicate — `is_scratch(tool_name, tool_input) ->
(bool, reason)` — is the sole authority. Not a regex scattered across
settings, and not a hand-maintained list in prose.

Principle 21 applies awkwardly here and the design should say so plainly:
this set is **not derivable**. There is no filesystem or AST enumeration of
"ways to run an ephemeral script"; it is inherently a pattern list, and a
pattern list is exactly the shape principle 21 warns has under-covered four
times in this project. The mitigation is not derivation but a **committed
corpus test**: `tests/test_provenance_capture.py` holds positive and
negative example invocations, each pinned with the reason it is in the
corpus, and the predicate is tested against all of them. When a new scratch
shape is discovered in the wild, the fix is a corpus entry plus a pattern —
never a pattern alone.

Positive (capture):

- `python -c`, `python3 -c`, `uv run python -c` — inline code
- a heredoc whose body is fed to an interpreter (`<<'EOF'`, `<<EOF`)
- a script path under `/tmp`, `$TMPDIR`, or a `mktemp` result
- `mighty-colab exec -f <path>`, `mighty-colab run <path>` — the local
  file is read and sent to a remote kernel
- `mcp__mighty-colab__exec` with inline code or a local file argument
- piped code: `echo ... | mighty-colab exec`, `cat x.py | ...`

Negative (do not capture):

- any invocation of a committed path (the generator already exists — that
  is the happy path, and `diagnose_encoder_gate_failure.py` is the model)
- `git`, `ls`, `grep`, `pytest`, `make`, and other read/navigate commands
- anything under `tests/`

The negative list matters as much as the positive one. Capturing
everything produces a log nobody reads, which is the same outcome as
capturing nothing but with more disk.

### 3.3 The record

Append-only JSONL at
`.provenance/runs/<session_id>/capture.jsonl` (gitignored). One `open`
record per invocation, one `close` record joined to it by `tool_use_id`.

```jsonc
{
  "capture_id": "20260807T164512Z-3f9a2c",   // timestamp + digest prefix
  "phase": "open",                            // open | close
  "tool_use_id": "toolu_01ABC...",            // joins open to close
  "session_id": "...", "ts_utc": "...",
  "cwd": "...",
  "git": {"commit": "15fa025...", "branch": "infra-tooling", "dirty": true},
  "tool_name": "Bash",
  "trigger_reason": "inline_c",               // which predicate rule fired
  // What executed it, where knowable. A capture recording WHAT ran but not
  // WHICH VERSION ran it inherits the same skew blindness it exists to
  // close -- the c2c track lost a message to exactly that gap (a deployment
  // silently lacking a feature the caller assumed present), and a GPU number
  // from an older mighty-colab or a different CUDA build is a different
  // number. Absent rather than guessed when it cannot be determined.
  "executor": {"tool": "mighty-colab", "version": "0.5.0",
               "endpoint": "colab:gpu1", "resolved_from": "--version"},
  "script": {
    "text": "...",                            // the code, in full
    "source": "inline_c | heredoc | file_snapshot | stdin_pipe",
    "sha256": "..."
  },
  "referenced_files": [                       // see 3.4
    {"path": "/tmp/probe.py", "sha256": "...", "bytes": 812,
     "snapshot": "blobs/ab/cdef...", "existed": true}
  ]
}
```

```jsonc
{
  "capture_id": "...", "phase": "close", "tool_use_id": "toolu_01ABC...",
  "outcome": "ok | failed",
  "duration_ms": 1840,
  "output": {
    "blob": "blobs/12/3456...",               // copied bytes, never a path
    "bytes": 200095,
    "source": "persisted | inline | error_field",
    "fidelity": "complete | capped | elided"
  }
}
```

`fidelity` is decided by measurement, not hope:

- `complete` — read from `persistedOutputPath` and its length equals
  `persistedOutputSize`; or read inline with no persisted path named and
  length < the observed 30,000 cap.
- `capped` — inline text at exactly the cap with no persisted file to
  recover from.
- `elided` — from the `error` field on the failure path.

Recording fidelity rather than asserting completeness is the point. An
auditor reading `fidelity: elided` knows the output is partial; an auditor
reading a log that silently dropped 970,000 characters does not.

### 3.4 The file snapshot — the one thing transcripts provably lose

`mighty-colab exec -s <name> -f script.py` reads a **local** file and sends
its contents to a remote kernel. The session transcript records the command
line — the path — and never the content. When the path is a tmp file, the
content is gone the moment the session ends, and the transcript's record of
"what was run on the GPU" is a filename.

This is the strongest single justification for capture-at-birth, and it is
unaffected by every truncation finding in §2: it is a `PreToolUse` file
read, done before the tool executes.

At `open`, for each local path the predicate identifies as an argument to a
remote-exec command, capture reads the file, stores the bytes as a
content-addressed blob under `.provenance/runs/<session_id>/blobs/`, and
records path, size and digest. Content addressing means re-running the same
script fifty times costs one blob.

### 3.5 Run-scoped conventions: forensic, never a parent, never cited

Three rules, each with a mechanism rather than an exhortation:

1. **Forensic.** The log is evidence for an auditor reconstructing where a
   number came from. It is not an input to any computation.
2. **Never a parent.** No committed artifact may descend from it. Nothing
   reads the log except a human and the audit tooling. *Mechanism:*
   `.provenance/` is gitignored, so no artifact derived from it can be
   committed without the derivation itself being visible in a diff.
3. **Never cited.** A capture ID is not a citation (§1.1). *Mechanism:* the
   verifier in §5 explicitly rejects capture-ID-shaped tokens where a
   citation is required, and has a break-test proving it rejects them.

### 3.5.1 Storage and death model

Stated explicitly, because a forensic log with an unexamined lifecycle is
the failure it was built to prevent, one level up.

- **Where.** `.provenance/runs/<session_id>/` in the working tree, holding
  `capture.jsonl` plus a content-addressed `blobs/` directory. In the
  working tree rather than under `~/.claude/` so that a worktree's captures
  travel with the worktree and die with it, and so `git status` never shows
  them (the path is gitignored).
- **How big.** Blobs are content-addressed, so re-running the same script
  fifty times costs one copy. Output blobs are capped: above a ceiling the
  record keeps head, tail, digest and true size rather than the whole
  payload, and says so in `fidelity`. **That ceiling is a policy choice, not
  a measured threshold** — nothing about it scales with a quantity, so
  principle 22 does not govern it, and it should not be dressed up as
  empirical. `tools/provenance/capture_stats.py` reports what the local logs
  actually contain, and to date nothing has come close to the ceiling; but
  those logs are gitignored and local-only, so that observation is context
  and explicitly not the anchor. Turning this into a measured threshold
  would require committed code regenerating real outputs, not a reading of
  the forensic log.
- **How it dies.** Pruned **by age, never by size.** A size-capped log
  evicts its oldest records first, which are exactly the ones whose
  generator is most likely already gone — the failure mode inverted. Age
  pruning removes records whose window for mattering has closed.
- **What deletes it.** Nothing automatic on session end. A session that
  dies mid-run is the case this exists for, so teardown must not be
  entangled with it. Pruning is a separate, idempotent pass.
- **What must never read it.** Any committed artifact. The log is a leaf
  (§3.5), and `.provenance/` being gitignored is the mechanism: no
  derivation from it can be committed without the derivation being visible
  in a diff.

### 3.6 Non-goals, held literally

- **Never blocks.** The hook exits 0 unconditionally — including on its own
  internal error, including on malformed input, including when the log
  directory is unwritable. It writes an error record if it can. A capture
  path that can break a session is worse than no capture path, and one
  `exit 2` from a forensic hook would train exactly the route-around this
  design exists to avoid. (`probe_hook_payload.py` already follows this
  rule and is the reference implementation of it.)
- **Does not review.** Capture forms no opinion about whether a spike was
  worth running.
- **Does not replace `diagnose_*.py`.** Writing a committed diagnostic
  script is the happy path and always was. Capture exists for the case
  where somebody did not.

---

## 4. Part 2 — Classification at edit time

Mechanical trigger, judgment payload.

**Trigger:** a `PreToolUse` hook on `Edit|Write|NotebookEdit` whose
`tool_input.file_path` matches a durable document — the derived set from
§5.1, not a hand-list.

**Payload:** the hook returns
`hookSpecificOutput.additionalContext` containing the classification
checklist, which is the operative half of principle 24 stated as a
question the author answers at the moment of writing:

> Does this edit introduce a quantitative claim or a frozen constant?
> - **IN scope** — empirical measurement of the system under study whose
>   output enters a durable record or anchors a frozen parameter. It needs
>   a citation to committed code.
> - **OUT of scope** — discarded exploration (must stay cheap), and inline
>   analytic derivation (the derivation is its own provenance).
>
> If IN and the generator was ephemeral: the remedy is to promote it to
> committed code. Not to cite the capture.

**It never blocks.** `additionalContext` only; no `permissionDecision`,
no exit 2. The author is writing a document at that moment, which is
exactly when they have the context to answer, and exactly when a hard stop
would be most expensive.

**Chat and spike work are never touched.** The trigger is a durable-doc
file path. Editing a scratch file, running an experiment, or talking
through an idea produces no prompt at all.

### 4.1 What this hook must not become

An end-of-session sweep that grades the session it ran in. The
reviewer-declares-convergence rule (`docs/MULTI_AGENT_PRACTICE.md`) applies:
a session auditing its own provenance is the responder declaring its own
convergence. Sweeps belong to a separate verifier instance (§5), invoked
deliberately, reading committed state.

---

## 5. Part 3 — The citation verifier (design only)

A pytest module, `tests/test_provenance_citations.py`, generalising the
citation-pinning already proven in
`tests/test_stage2b_negative_path_evidence.py` — which asserts every test
name a document cites still exists, derives the truth set from the AST
rather than a grep, and carries exemptions each with a reason and a test
that the exemption still names something absent. That file is the template;
this is the same shape at repository scale.

### 5.1 Derive both sets — the verifier is itself a narrowing

Principle 21 applies **to** this verifier, not merely through it. Any
hand-listed set of documents-to-scan or constants-to-check will
under-cover, and — the part that keeps biting — verifying it by running
`pytest tests/` would hide the gap exactly as `STAGE2B_TEST_FILES` did.

- **Durable documents:** filesystem walk — `docs/**/*.md`,
  `experiments/**/FINDINGS.md`, `**/DESIGN*.md`,
  `**/*_PLAN.md`, `docs/PROJECT_MEMORY.md`. Never a list.
- **Frozen constants:** AST walk for module-level `UPPER_CASE` assignments
  bound to numeric literals under `src/bonsai/` and `experiments/`. Never a
  list. (`GRAD_NORM_REL`, `MEAN_X_TOL` and the encoder-gate thresholds are
  the instances principle 22 and 23 were written from; the walk finds them
  without being told they exist.)
- Where a list must stay explicit, assert list == derived **in both
  directions**: entries missing, and entries naming things that no longer
  exist.

### 5.2 A syntactic trigger, not a judgment call

"Durable empirical claim" as prose yields a verifier nobody can run
deterministically and whose failures are arguable. The trigger must be
mechanical:

- a numeric literal in scientific notation (`1.78e-14`), **or**
- a decimal with three or more significant figures, **or**
- a `p = ...` / `p < ...` construction,
- appearing on a line that is **not** inside a fenced code block, and not
  inside a table row marked as an appendix of measured values.

A **citation** satisfying it is a backticked repo-relative path that
exists, or a backticked identifier resolvable by AST walk to a definition
in committed code. A capture ID is explicitly **not** a citation (§1.1).

**Attribution is section-scoped, not line-scoped.** A citation satisfies
every triggering line from its own heading down to the next heading of the
same or higher level. Without this rule the check is unusable, and the
proof is this document: §2 states `30,000`, `200,095`, `1,000,015` and
`10,040` in prose under a heading that names
`tools/provenance/run_truncation_probe.py` as their generator once. Under a
line-scoped rule every one of those is an uncited empirical claim in a
fail-closed target, and the first thing anyone implementing the verifier
would do is exempt the document that specifies it. A measurement table
belongs under one attribution, not repeated per cell.

The corollary is that a section heading is a scope boundary the author is
responsible for: putting an unrelated number under a heading whose citation
does not generate it defeats the check. That is a real weakness of
section-scoping and it is accepted deliberately, because the line-scoped
alternative is not adopted — it is exempted into uselessness.

This will over-trigger anyway — version numbers, dates, counts of things.
That is what the exemption table is for, in the exact `NOT_CITATIONS`
shape: each entry carries a reason, and a test asserts each exemption still
refers to something that genuinely is not a claim.

### 5.3 Fail-closed vs flag-only

| Target | Mode | Why |
|---|---|---|
| Frozen constants (AST-derived) | **fail-closed** | A wrong threshold silently changes what an experiment concludes. Principles 22 and 23 are both incidents of exactly this. |
| Quantitative claims in FINDINGS / DESIGN / PROJECT_MEMORY | **fail-closed** | These are the durable record. |
| Gray-zone qualitative statements | **flag only** | A guard that fires on exploration trains route-around. |

The asymmetry is deliberate and is the behavioural core of the whole
design: hard where a wrong number is expensive and the author is already in
"writing it down" mode, soft everywhere else.

### 5.3a Acceptance criteria — binding

Set by `claude-desktop-orchestrator` on 2026-08-08, after the capture hooks
shipped four defects past a green suite. Recorded as requirements rather
than intentions, because the verifier's own glue will be configuration and
taxonomy G therefore applies to it with full force.

1. **It runs against the LIVE repository, with violations seeded in situ.**
   Synthetic corpora may supplement, never substitute. A verifier tested
   only against synthetic inputs has demonstrated nothing about whether it
   runs where it matters.
2. **Its own wiring is proven the `session_open` way** — some marker whose
   absence is DIAGNOSTIC, distinguishing "the verifier ran and found
   nothing" from "the verifier never ran." These are the two readings that
   silently collapsed for the capture hooks, and the failure was invisible
   precisely because a clean result and an absent result look identical.
3. **Its first named test case is the shape this project produced three
   times in one night: component correct, wiring absent.**

### 5.4 Break-tests are part of the definition

A guard you have not seen fail is not yet a guard. Each of these is a test
that seeds a violation and asserts the specific expected failure:

1. Add a module-level `UPPER_CASE` numeric constant with no citation →
   the constant check must fail, naming that constant.
2. Rename a file cited by a durable document → the path check must fail,
   naming the stale citation.
3. Add a new `FINDINGS.md` in a new directory → the **derivation** must
   discover it without any list being edited. (This is the test that would
   have caught `STAGE2B_TEST_FILES`.)
4. Add an exemption naming something that does exist as a real claim →
   the exemption-integrity check must fail.
5. Offer a capture ID where a citation is required → must be **rejected**,
   pinning §1.1 as behaviour rather than prose.

A verifier that cannot be shown to catch a seeded violation is the
fake-bucket lesson again, in process clothing.

---

## 6. Build order, and what is not decided

Built and measured: the payload probe (`tools/provenance/`).

Proposed order, smallest first, each landing with its break-tests:

1. ~~`is_scratch` predicate + corpus test — no hooks, pure function.~~
   **Done** — `.claude/hooks/provenance-capture/scratch_predicate.py`,
   `tests/test_provenance_capture.py`.
2. Capture hook (Pre/Post/Failure), merged into `.claude/settings.json`.
   **Authorized to build.** Conditions attached to that authorization:
   fail-open proven by break-test (missing log directory, unwritable disk
   and malformed input must each pass silently, never block), and the
   storage/death model above stated — which §3.5.1 now does.
3. Classification skill + durable-doc `PreToolUse` hook. **Held**, on
   sequencing rather than merit: it changes behavior inside science-track
   sessions at a moment when those sessions are gated on writing FINDINGS.
   It lands with the verifier round, after the current gates clear.
4. Citation verifier — the largest, and the one most improved by having
   the first three in use first.

Open, deliberately not decided here:

- **Retention period** for `.provenance/runs/`. Needs a week of real
  captures to size honestly; guessing now would be a frozen constant with
  no measurement behind it, which this document is poorly placed to do.
  `tools/provenance/capture_stats.py` is the instrument for answering it
  when the data exists — and it exists to keep that answer from arriving as
  an inline `python -c` whose output lands in a document, which is the
  failure this whole contract describes.
- **Whether the 30,000 cap is stable across harness versions.** It is a
  measured property of one version, not a contract. The capture hook
  therefore compares against `persistedOutputSize` rather than testing
  `len(text) == 30000`, so a changed cap degrades the fidelity flag instead
  of corrupting it. `run_truncation_probe.py` is committed precisely so the
  number can be re-measured rather than trusted.
- **Whether MCP tool inputs need per-server predicates.**
  `mcp__mighty-colab__exec` is the only remote-exec MCP surface today.

---

## Appendix — reproducing the measurements

```
uv run python tools/provenance/run_truncation_probe.py          # full matrix
uv run python tools/provenance/run_truncation_probe.py --case 200000 both 1
```

Six headless sessions, roughly a minute each. Output table columns:
inline stdout length, inline stderr length, `persistedOutputSize`,
recovered length, and whether the recovered body's digest matches the
sentinel the emitter wrote. The `sha_ok` column is the byte-exactness
claim in §2; it is printed by the committed driver rather than read out of
an ad-hoc query, which is the standard this document is arguing for.
