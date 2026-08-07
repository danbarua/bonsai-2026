# Claude Desktop — c2c protocol (overview)

The actual behaviour lives in two skill files, loaded into Claude
Desktop directly (not this doc):

- `.claude/skills/c2c-send/SKILL.md`
- `.claude/skills/c2c-inbox/SKILL.md`

This file is a human-readable overview and design record — kept
separate so the skill files themselves stay lean and procedural,
without background/rationale bloating what gets loaded into context.

## How triggering actually works here

Unlike Claude Code's `c2c` skill — which is invoked by an exact
`/c2c send|inbox|cleanup` argument parsed by the CLI's slash-command
loader — Claude Desktop has no such parser. Its skills are matched by
description against the natural language of what the user asks for
("c2c inbox", "check for messages from Claude Code", "send that to
Claude Code" all match `c2c-inbox` / `c2c-send` by *meaning*, not by
exact string). That's why these are two separate, plainly-described
skills rather than one skill with an argument to dispatch on.

Trade-off worth naming: because these are matched by the model rather
than parsed by a fixed grammar, there's more surface for
false-positive triggering (something that merely *resembles* a c2c
request) than Code's exact-string version has. The descriptions are
written to require an explicit ask ("send that to Claude Code" /
"check for messages"), not just adjacent phrasing, but this is a
softer boundary than Code's and worth revisiting if either skill fires
when it shouldn't.

## Confirmed facts (from Dan, not assumed)

- Writing to either mailbox folder is inert — it never notifies or
  wakes the other side. A message sits in the buffer until a human
  explicitly asks that side's Claude to check for it (`c2c-inbox` on
  Desktop, `/c2c inbox` on Code). No cross-context injection happens
  on write, on either end.
- Filesystem operations on Desktop's side require per-action user
  approval and are scoped to this project directory only.

## Known asymmetries vs the Code side

- **Delete → archive.** The Filesystem connector has no delete
  operation (only `move_file`, `write_file`, `edit_file`), so
  `c2c-inbox` moves handled messages to `archive/` instead of
  deleting them. Code's skill already anticipates this — it never
  deletes its own outbox, treating that as "Desktop's job."
- **No autonomous peer-reply.** When `c2c-inbox` finds a message
  addressed to Dan (a question, a decision point), it surfaces that to
  Dan and waits for his actual answer rather than fabricating one —
  Dan is actually present in a Desktop chat in a way Code, running
  unattended, usually isn't.
- **No Desktop-side `cleanup`.** Since Desktop can't delete, it can't
  mirror Code's unconditional wipe of both folders. A full reset
  should be run from the Code side, which already has bash access to
  both directories.

## Message format

Plain markdown. Optional one-line leading comment for sender/timestamp
(`<!-- from: claude-desktop · TIMESTAMP -->` / `<!-- from: claude-code
· TIMESTAMP -->`). No threading IDs, no reply-to chains.
