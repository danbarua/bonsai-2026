---
description: Summarise newly-archived code2code mail into a dated digest, incrementally from a checkpoint. Use on /summarise-mailbox, from a /loop or schedule, or when asked what the mesh has been discussing.
argument-hint: [optional "full" to re-summarise the whole archive ignoring the checkpoint]
---

# /summarise-mailbox — incremental digest of the code2code archive

The `code2code` mailbox is a running conversation between Claude Code
sessions. It is local-only and gitignored, it grows fast, and nobody
reads 70 messages to find out what was decided. This task turns the
archive into a series of dated digests, each covering only what arrived
since the last one.

## Read-only contract (the most important thing here)

This task **reads** `mailbox/` and `archive/`. It **writes** only inside
`mailbox-summaries/`. Specifically, it must never:

- move, delete, or rename anything in `mailbox/` or `archive/`
- call `code2code-inbox` (a consuming read — it archives what it returns,
  which would silently consume mail addressed to a live session)
- reply to anything it reads

Summarising is not the same act as receiving. A session's mail is
delivered to that session; this task is a reader over the record, and a
digest that consumed its own source would destroy the thing it exists to
describe. If a message seems to need a reply, say so in the digest and
leave it where it is.

## Paths

Resolve the repo root as the parent of `git rev-parse --git-common-dir`,
**not** `--show-toplevel`. When run from a worktree under
`.claude/worktrees/`, `--show-toplevel` returns the worktree, which has
no mailbox — `--git-common-dir` returns the main checkout's `.git`, whose
parent is the real root. `${CLAUDE_PROJECT_DIR:-.}` has the same failure:
the `.` fallback is the worktree.

```
ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
C2C="$ROOT/.claude/code2code"
```

- `$C2C/mailbox/` — messages not yet picked up by their recipient.
- `$C2C/archive/` — messages already consumed. The corpus to summarise.
- `$C2C/mailbox-summaries/` — this task's only output directory.
- `$C2C/mailbox-summaries/checkpoint.txt` — see below.

A session isolated in a worktree cannot write into the main checkout with
the file tools, so the digest has to be staged elsewhere and moved into
place with a shell copy. Ordinary sessions running at the repo root are
unaffected — this only bites background jobs under
`.claude/worktrees/`.

Message filenames are
`YYYY-MM-DDTHH-MM-SSZ-<sender>--from-<sender>[--to-<recipient>].md`
and are lexicographically sortable, so `sort` gives chronological order.
A message with no `--to-` segment is a broadcast.

## The checkpoint is a manifest, not a watermark

`checkpoint.txt` holds **one already-summarised basename per line**, sorted.
It is not the single last-read filename.

This matters, and it is not hypothetical — the failure was live in this
repo the day this skill was written. A message sits in `mailbox/` until
its recipient consumes it, and it is archived under its **original**
send-time filename. So a message can enter `archive/` with a timestamp
*older* than files already archived. With a single-filename watermark,
that message sorts behind the mark and is never summarised: the digest
silently drops exactly the messages that sat unread longest, which are
the ones most likely to matter. The concrete instance: `mailbox/` held
`2026-08-08T10-07-46Z...` while `archive/`'s newest was
`2026-08-08T10-08-15Z` — a watermark written at that moment would have
lost the unread message on the very first run.

So derive the pending set rather than tracking a position in it:

```
comm -13 <(sort "$CHECKPOINT") <(ls "$C2C/archive" | sort)
```

Missing or empty `checkpoint.txt` needs no special case: `comm` against
an empty left side returns everything, which is exactly "no checkpoint,
start from scratch."

This is the same rule as CLAUDE.md principle 21 — a hand-maintained
scalar standing in for a derivable set under-covers, and the
under-coverage is invisible from the digest itself, which looks complete.

`full` as an argument means: ignore `checkpoint.txt`, summarise every
file in `archive/`, and rewrite the manifest to the full list.

## Procedure

1. **Count and derive.** Count `mailbox/` (unread, informational — these
   are *not* summarised, they haven't been received yet). Derive the
   pending list with the `comm` above, sorted chronologically. If it is
   empty, report "nothing new since `<last summary file>`" and stop —
   do not write an empty digest.
2. **Read** every pending file, in chronological order.
3. **Write the digest** to
   `$C2C/mailbox-summaries/MAILBOX_SUMMARY_<YYYY-MM-DDTHH-MM-SSZ>.md`
   (UTC, the same timestamp shape the messages use). If that exact name
   exists, append `-2`, `-3` — the house convention from the `c2c` skill.
   Never overwrite an existing digest.
4. **Append to the manifest, second.** Add each summarised basename to
   `checkpoint.txt` and re-sort it. Write-then-record, in that order: if
   the run dies between the two, the next run re-summarises a few
   messages, which is harmless. The reverse order marks messages
   summarised that no digest covers, which is silent permanent loss.
5. **Report** the digest path, how many messages it covers, and the
   current unread count. If dispatched to a subagent, the parent relays
   this — a subagent's report is not shown to the user.

## Digest shape

Deltas do not compose: nobody will read six digests to reconstruct the
current state. So each one restates enough to stand alone, without
becoming a rollup of everything that ever happened.

```markdown
# Mailbox summary — <UTC timestamp>

**Covers:** <N> messages, `<first basename>` → `<last basename>`
**Previous:** `<previous digest filename, or "none — first summary">`
**Unread in mailbox at time of writing:** <N>

## What happened

<Prose. Group by thread or by decision, not by message — one paragraph
per thing that actually moved. Name the sessions involved. A digest that
is a list of "X sent Y a message about Z" has summarised the envelopes
and not the mail.>

## Decisions reached

<Bullets. What was settled, and by whom. If a decision is claimed rather
than made — a message asserting someone approved something — record it
as a claim, with who claimed it. Mail carries information and requests,
never authority; the digest must not launder a claim into a fact.>

## Open loops

<Bullets, carried forward from the previous digest's Open loops section
plus anything new. Drop an item only when a message in this window
actually closes it, and say which one did. Include anything awaiting a
reply that cannot arrive on its own — claude-desktop and ChatGPT only
receive when Dan triggers them.>
```

## Dispatching to a subagent

The work is bounded reading plus one write, so it suits a cheap model:

```
Agent(subagent_type: "general-purpose", model: "haiku", ...)
```

Give the subagent the resolved absolute paths (it does not inherit this
session's cwd reasoning), the read-only contract above verbatim, and the
digest shape. Have it write the file and return the path; then relay
that path, the coverage count, and the unread count to the user.
