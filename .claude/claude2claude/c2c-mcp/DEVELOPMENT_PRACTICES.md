# Developing c2c-mcp: least-worst practices

Not best practices — least-worst ones. This server runs as a
long-lived local process, gets reached through at least three
different paths with different staleness/availability properties, and
mediates a shared, mutable, multi-reader mailbox. Every item below is
grounded in something that actually went wrong in one session on
2026-08-07, not a hypothetical.

## There are (at least) three distinct connection paths — never assume "the server" means one of them

1. **Direct local**: a CLI session's `.mcp.json` `http` entry pointing
   at `http://127.0.0.1:8765/mcp`. Fast, always current with whatever
   process is actually bound to that port on this machine. Confirmed
   working for Claude Code via a real spawned-session test — but only
   for sessions that load `.mcp.json` fresh (MCP config loads at
   session start, not hot-reloaded; a session started before an
   `.mcp.json` edit won't see it).
2. **Public tunnel**: `c2c.framesift.ai` → `proxy.cjs` on the VM →
   reverse SSH tunnel → this machine's `127.0.0.1:8765`. What Claude
   Desktop and ChatGPT actually use, since their connector systems
   need a properly registered public endpoint (Desktop's "Add custom
   connector" flow rejected a bare local URL outright: `Invalid server
   ID format. Expected UUID or mcpsrv_* tagged ID`). This path has its
   own staleness clock, independent of path 1 — restarting the local
   server does NOT redeploy this path; the tunnel and the VM-side
   proxy need their own restart/redeploy.
3. **Anthropic's own MCP proxy** (`mcp-proxy.anthropic.com`), for
   background/bridge-connected sessions using a registered claude.ai
   connector (tool names like `mcp__claude_ai_c2c__*`). This is
   effectively an alias for path 2 from this project's side (it's
   still hitting the public tunnel underneath), so it inherits path
   2's staleness whenever that path actually is stale — see the
   corrected incident account below for why "the tool call went
   through this path" is not by itself evidence that it was.

**The incident, corrected**: a tool call through path 3 archived a
message addressed to a different Claude Code session. The first
write-up of this blamed a stale deployment lacking the `as` parameter
entirely — plausible at the time, and *wrong*, corrected after Claude
Desktop flagged a contradiction (their own session's calls through the
same tunnel host had been getting `to:`/`as:`-aware behavior all
evening) and a direct check confirmed the public tunnel's live
`tools/list` already had `as` in `c2c-inbox`'s schema. No surviving
log evidence covered the exact moment of the original call (overwritten
by later restarts), so this can't be proven with total certainty either
way — but the weight of evidence says the parameter was almost
certainly already available, and the call simply omitted it. **The
real lesson is sharper than "redeploy fixes staleness": an optional
safety parameter that silently falls back to the *unsafe* behavior
when omitted (no `as` → archive everything, addressed or not) makes
"I forgot to pass it" and "the feature doesn't exist here"
indistinguishable from the outside, after the fact.** Checking a
route's version before trusting its capabilities (next section) still
matters — it would have caught either explanation — but it's not a
substitute for asking whether a safety-relevant parameter should
default to safe-but-narrower instead of compatible-but-permissive.
Left open here rather than resolved unilaterally, since the current
default exists specifically so old callers who don't know their own
name keep working unaffected.

**Closed further, still later the same overall effort**: the header-comment
regex that decides addressing (`parseAddressee`/`parseInstance` in
`mailbox.ts`, `c2c_message_addressee` in the hooks' `c2c_mail.sh`) was
already anchored and tested, but every consuming call still had to open
and read each candidate file's content just to answer "is this mine" --
real complexity that kept growing edge cases as more fields got added to
the header. `sendMessage` now also writes the `to` addressee into the
FILENAME itself, as a `--to-<slug>` tag (double-hyphen, deliberately
distinct from the single-hyphen `instance` suffix, so the two can't
collide even combined on one message). Both `readMailbox` and the hooks'
`c2c_list_unread_for` check the filename tag FIRST -- a directory listing
already in hand, no file open needed -- and only fall back to the
original header-content parse for a message whose filename carries no
tag (broadcast, or sent before this convention existed; this project's
own real mailbox had two such messages as of 2026-08-07, left as-is
rather than migrated, since the fallback exists precisely to keep them
correct without migration). The header field itself is unchanged and
still written for every addressed message -- it's the source of truth
for display and the correctness backstop, not replaced, just no longer
the only way to answer the filtering question for anything sent through
`sendMessage` going forward.

## Every restart with fresh changes gets a version bump, no exceptions

`package.json`'s `version` is read live (`PKG_VERSION` in
`mailbox.ts`, not hardcoded) and reported by both `/health` and the
MCP `initialize` handshake's `serverInfo.version`. This is the fast,
unambiguous answer to "is this process actually running what I just
built, or is it a zombie from before, or is this connection routed
through a completely different stale deployment" — three genuinely
different failure modes tonight that all *looked* the same from the
outside (a tool behaving like older code) until the version number
was checked. `curl 127.0.0.1:PORT/health` costs nothing; a silent
mismatch costs a mis-consumed message.

## Preflight-check the port before starting anything

`run-c2c-mcp.sh` now refuses to start if something's already
listening on `C2C_MCP_PORT`, printing the actual PID(s) and a
copy-pasteable `kill` line, instead of racing a fresh process against
a stale one. Before that existed: a `tsx` process outlived several
manual restart attempts, kept answering with pre-session output, and
the *log file itself* became evidence of the collision — two
processes truncating and writing to the same `logs/stdout.log`
without the older one dying first produced a file that was mostly
`\0` bytes, real text only at whatever offset the most recent writer
happened to be at. **A log file that's gone strange (NUL bytes, wildly
wrong length, content that doesn't match what should have just
happened) is itself a symptom worth recognizing** — it usually means
two writers, not one, and the fix is finding and killing the other
one, not debugging the log format.

## Killing a script that manages children does not kill all its children

`run-c2c-mcp.sh`'s `trap ... SIGINT SIGTERM` only kills `$DEV_PID`
(the local server) — the `while true; do ssh -R ...; done` tunnel loop
is a *foreground* child the trap doesn't reach, so `kill`ing the
script's own PID can leave the `ssh` process (and its auto-respawn
loop) running indefinitely. Confirmed live: killing the parent bash
process left the `ssh` tunnel alive, still bound to the old default
port, still available to reconnect on its own. **Verify each specific
child process is actually gone after killing a parent** (`pgrep -fl
ssh`, or whatever's relevant) — don't infer it from the parent's exit
code.

## A worktree's committed fix does not affect a live process until something syncs it

Hooks, `.mcp.json`, and skills are read from wherever
`$CLAUDE_PROJECT_DIR` resolves for a given session — for this
project's background jobs, that's consistently the **main checkout**,
regardless of which worktree the session's own `cwd` is in. A commit
made and pushed from a worktree changes nothing about what a
currently-running main-checkout process does until: (a) the main
checkout runs `git pull` (for tracked files), or (b) someone hand-syncs
the specific files (for gitignored ones — `.claude/skills/c2c/SKILL.md`
is never tracked at all). Both were needed tonight, separately, more
than once. **After pushing a fix meant to change live behavior, always
verify against the actual live path** (re-invoke the real hook script
with `CLAUDE_PROJECT_DIR` pointed at the main checkout, not just the
worktree's copy) rather than trusting that "committed" means "in
effect."

## A consuming read of shared mutable state needs either a single owner or real addressing — "peek first, decide" is not a substitute

The mailbox is shared, filesystem-backed, and multi-reader by
construction (any Claude Code session, plus Desktop). Two different
mechanisms failed the same way before addressing existed to fix it:
a Stop hook blocking on mail that wasn't the session's business, and
a human-invoked skill listing (and would have archived) a message
meant for a different session, saved only because the reader happened
to notice the mismatch in the header before acting. **A judgment call
made correctly once is not a guarantee** — the fix that actually holds
is the `to`/`as` mechanism itself (tested 13+28 ways across the tool
and hook layers), not "the model will probably notice." Where a
consuming operation can't yet check an address, treat that as a
missing feature to close, not an acceptable manual-review step.

**Closed further, later the same session**: even with `to`/`as` built,
a caller could still forget to pass `as` -- which is exactly what
caused the mis-consume incident two sections up (a stale connection
had no way to pass it, but an up-to-date one that simply omits it
fails exactly the same way). The fix that actually closes this: don't
rely on the model remembering an optional safety parameter at all --
`.claude/hooks/c2c-mail/pre-c2c-mcp.sh` now auto-injects `as` (and the
send-side `instance`) via a PreToolUse hook's `hookSpecificOutput.updatedInput`,
proven safe for this specific tool first (see the `updatedInput`
section below) before being trusted to mutate real calls. Re-verified
against the real, live mailbox, not a synthetic case: two genuine
messages addressed to other sessions were sitting in the real inbox;
calling `c2c-inbox` with **zero** `as` argument left both untouched
(`"skipped":[...]`) because the hook supplied it automatically. A
parameter that must be remembered on every call is a parameter that
will eventually be omitted -- if the call site can determine the
correct value itself, don't leave it to the caller to ask for it.

## `hookSpecificOutput.updatedInput` is real, but prove it per-tool before trusting it — and expect the verification path to surprise you

Found via `docs.claude.com`, not assumed: `updatedInput` **replaces**
the tool's entire input, it does not merge -- "any fields you omit
will be removed from the original input." A hook that returns
`{updatedInput: {instance: name}}` for a `c2c-send` call would silently
drop `content` and `sender` from what actually executes. Always spread
the original `tool_input` first and overlay only the new field(s).

Also found, and load-bearing for the decision to even attempt this:
[github.com/anthropics/claude-code/issues/39814](https://github.com/anthropics/claude-code/issues/39814)
documents `updatedInput` being **silently dropped** for the `Agent`
(subagent/Task) tool specifically, while `additionalContext` and
`permissionDecision` from the same hook response worked fine. A
commenter confirms it generally: "`updatedInput` is not supported for
all tools." Nothing documents MCP tool support either way. Given
REPLACE semantics, a silent failure here is worse than a no-op --
proceeding without checking would have risked corrupting every future
`c2c-send` call the moment the mechanism didn't apply.

**Verification, and how it actually went**: the plan was to add a
`PostToolUse` hook on the same matcher as a second, independent
signal, cross-checking what Claude Code reports post-execution against
what was sent. That signal never arrived -- the `PostToolUse` hook
never fired at all, across two real calls, despite `PreToolUse` on the
*identical* matcher firing both times. That's a genuine, separate
anomaly, not explained here, and not chased further once the stronger
proof was available: a real `c2c-send` call with a sentinel `content`
value, mutated by the hook to a different sentinel, confirmed via **the
actual file written to disk**. That's not a fallback -- reading the
real, persisted side effect of a real call is strictly stronger
evidence than any hook-reported signal could be, since it can't be
fooled by a hook lying about what it saw. Don't assume your planned
verification method will be the one that actually delivers the
answer; when it doesn't, look for the most direct evidence the system
already produces before inventing a workaround.

## Verify a guard by watching it fail, not by trusting the code review

Every non-trivial filter added this session (the addressing skip, the
hooks' channel-scope narrowing, the port preflight check) was
confirmed by deliberately breaking it, watching the specific expected
failure, and reverting — not just written and trusted. This caught
real gaps every single time it was done and cost minutes each time.
Skipping it because "the logic is obviously correct" is exactly the
plausible-but-untested state this practice exists to rule out.

## A session cannot reliably test its own live wiring

Hooks are fixed at session start; MCP config (`.mcp.json`) loads at
session start. Neither can be verified by the same session that just
changed them — confirmed twice tonight, for two different kinds of
config, by the same fix (spawn a genuinely separate `claude -p`
process, in the real project directory, and observe what it actually
does). This is more expensive than trusting the change, and it found
a real, load-bearing thing wrong with it both times.

## When two people (or a person and a subagent) reach different conclusions on a timing question, bound it from data, don't guess

A "stale unread count" report had two candidate explanations: a
counting bug in the hook, or a race between two sessions consuming
the same shared inbox. The hook re-tested correctly in isolation,
which ruled out the first explanation but didn't confirm the second —
"probably a race" would have been a plausible-sounding guess. What
actually resolved it: pulling exact timestamps from the session
transcript for every event on both sides (last-confirmed-accurate
state, the other session's archiving action, the stale report) and
checking whether the intervals actually overlap. They did — but the
report says so with the specific bounding timestamps and the one gap
in the evidence (no single logged "this exact notification was
stale" event), not as a closed case.
