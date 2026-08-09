---
description: Answer "what is true right now?" by joining the mail digests to git. Use when arriving cold, resuming, or after hours of work in a long session, and before acting on any open loop a digest reports.
argument-hint: "[--since-commit <sha>] [--since-digest <file>] [--ref <ref>]"
---

# /briefing — what is true right now

Mail carries intent. Git carries what shipped. Either alone gives a wrong
picture, and the error has a direction: **mail-only over-reports open loops**,
because a commit can close an item minutes before a digest is written.

Run it:

```
uv run python .claude/skills/briefing/assemble_briefing.py
```

All arguments are optional. Bare is the cold path — "I have just arrived."

| argument | use |
|---|---|
| *(none)* | latest digest per channel, open loops, commits since that digest |
| `--since-commit <sha>` | you have seen everything up to `<sha>` |
| `--since-digest <file>` | you have read that digest |
| `--ref <ref>` | default `origin/stage2b` |

## Read the coverage boundary first

It is printed first deliberately. If it says a channel is N messages behind,
**nothing in those N messages is in this briefing** — not in the rulings, not
in the open loops.

`/summarise-mailbox` closes that gap. The briefing never does; it reports it.

## How to read Open loops

This is where a reader over-trusts. The four headings are four different
claims, and they are not interchangeable.

| heading | means |
|---|---|
| **Possibly closed** | a rare token matched a commit. **A judgement — verify it** |
| **Named artifact now EXISTS** | the file or make target the loop names is in the tree now |
| **No candidate found** | searched the window and the tree, nothing matched. **Not proof it is open** |
| **NOT CHECKED** | no joinable identifier. **Not searched at all** |

The join is **advisory everywhere**. A commit mentioning an identifier may be
unrelated to the loop that also mentions it. Candidates narrow your search;
they close nothing.

**The briefing is addressed to nobody.** It does not know who is reading it.
Loops name sessions; a name is not an assignment to you. Check your own
`/rename` name against a loop before treating it as yours — a cold reader has
already adopted a role it merely saw mentioned and reported another session's
blocker as its own.

Two bounds worth knowing:

- The git join only sees the window. A loop closed by an older commit shows as
  open — which is why the tree check exists, and why it has no window: a
  commit date says when a file was last touched, not whether its content is
  current.
- A token matching many commits is discarded as non-discriminating. Rarity is
  the evidence.
- **Session names are never evidence** and are excluded outright. Peers name
  each other constantly; a commit and a loop sharing a name says nothing. The
  roster is derived from the archive's filenames, so a new session is covered
  the day it sends its first message.

## Read-only

Reads digests, counts `mailbox/` and `archive/`, runs `git log`. Writes
nothing.

**Never call `code2code-inbox`** to gather material for a briefing. That is a
*consuming* read — it archives what it returns, silently eating mail addressed
to a live session.

## Paths

Resolve the repo root as the parent of `git rev-parse --git-common-dir`, not
`--show-toplevel`. From a worktree the latter returns the worktree, which has
no mailbox. `${CLAUDE_PROJECT_DIR:-.}` fails the same way.

The assembler already does this. Do not pass `--repo-root` to work around a
path problem; fix the path.

## When to run it

- Arriving cold, or resuming a session.
- Hours into a long session — the same staleness, just accumulated rather than
  inherited.
- Before acting on any open loop, and before reporting one as unfinished.

## Tests

```
uv run pytest .claude/skills/briefing/ -q
```

Comms tooling, so it lives here rather than in `tests/`, which verifies
documented scientific claims. Not collected by the science suite, not run in
CI.
