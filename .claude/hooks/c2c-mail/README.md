# c2c mail-awareness hooks

Makes any Claude Code session in this repo aware of unread mail in the
c2c mailboxes (`.claude/claude2*/inbox/`) automatically, so a human no
longer has to relay "check your inbox" into a running session.

Registered project-wide via `.claude/settings.json` (so every session
inherits them, per this repo's convention of project-level settings
over per-user config) as three hooks, all sharing `lib/c2c_mail.sh`
for the actual "what counts as unread" and notification-formatting
logic:

- **`user-prompt-submit.sh`** (`UserPromptSubmit`) -- on every prompt,
  if mail is waiting, injects a one-line `hookSpecificOutput.additionalContext`
  notification (count + filenames) into that turn. Never blocks.
- **`stop.sh`** (`Stop`) -- when the session tries to end its turn
  with unread mail present, blocks (exit 2, reason on stderr) with
  `unread c2c mail: <filenames> — read and handle it before finishing.`
  So mail doesn't just get silently noticed and then ignored.
- **`session-start.sh`** (`SessionStart`, optional third hook) -- same
  notification as `user-prompt-submit.sh`, so a resumed session opens
  already knowing its backlog instead of waiting for the first prompt.
  Can't block (there's no turn yet to block).

## Design decisions

**Filenames and counts only, never message bodies.** This is the
actual security boundary, not a style preference: prompt-time
injection of mailbox *content* would be a prompt-injection surface,
since anything able to write a file into a watched directory could get
arbitrary text injected into every session's context. The agent reads
mail through the normal tools (`/c2c inbox`, the c2c-mcp `-inbox`
tools) *after* seeing the notification; the hooks only ever point,
never read bodies. Verified by `test/break-tests.sh`'s section (d): a
distinctive marker string placed in a message body is asserted absent
from every hook's output, while the filename is asserted present.

**No idle-session wake -- a considered non-goal.** No fswatch/daemon/
CLI-invoking watcher that would notify or act on mail arriving while
no session is running. That crosses from *notification* into
*autonomous triggering*, and the human-starts-sessions checkpoint is
being kept deliberately: these hooks only ever run as part of a
lifecycle event (`UserPromptSubmit`, `Stop`, `SessionStart`) that a
human already initiated by starting or continuing a session. Recorded
here per the brief that asked for these hooks, which named this
constraint explicitly.

**"Unread" = present in an inbox dir.** The c2c-mcp server's own
`-inbox` tools move handled mail to `archive/` by default, matching
that definition -- confirmed against the actual server implementation,
not assumed from the brief's recollection of it. One real nuance
found while confirming: an `-inbox` call made with `archive: false`
(a deliberate peek) leaves the file in `inbox/`, so these hooks
correctly keep treating it as unread and the Stop hook keeps blocking
after a peek. That's the right behavior -- a peek genuinely hasn't
handled the mail -- but worth knowing if a peek is ever used mid-turn
expecting the Stop hook to then let the turn end.

**Watched dirs are derived, not hand-maintained.** `c2c_watch_dirs` in
`lib/c2c_mail.sh` globs `.claude/claude2*/inbox` under
`$CLAUDE_PROJECT_DIR` by default, rather than hardcoding the channel
names (`claude2claude`, `claude2gpt`) in a list here. A third mailbox
channel then just works, with nothing to drift out of sync with the
real directory layout -- this repo's own `docs/VACUOUS_TESTS.md`
principle 21: a hand-maintained list standing in for a derivable set
will silently under-cover, and the broader tool you verify with is
what hides it. Found live while building this, not anticipated in
advance: a spawned smoke-test session flagged that every break-test
either took the default or overrode it explicitly, so none would have
noticed a third channel going unwatched with the original
hand-maintained version. Set `C2C_MAIL_WATCH_DIRS` (space-separated)
to bypass the glob with an explicit list instead -- used by the tests,
which need a throwaway location, not whatever the glob happens to
match on a given machine.

**Hooks fail open.** Claude Code's own hook contract (confirmed
against current docs before implementing, not assumed): any hook exit
code other than 0 or 2 is treated as a non-blocking error -- shown as
a notice, but the user's work proceeds. Combined with `c2c_watch_dirs`
treating a missing directory as "nothing to report" rather than an
error, a hook failure can never block normal work when the mailbox is
empty or absent, by two independent layers (the script's own logic,
and the runtime's fail-open contract as a backstop).

## Per-session addressing

If multiple Claude Code sessions share the same live checkout (not
worktrees -- `.claude/claude2*/` is gitignored, so a worktree doesn't
see the real mailbox at all unless explicitly pointed at it), a
session's Stop hook only counting mail it never sent for itself was a
real, observed problem: every session's Stop hook blocked on *any*
unread mail, regardless of whether that mail was relevant to what that
particular session was doing, and a session archiving mail just to
unblock its own Stop hook could consume a message meant for a
different session's task, making it invisible to the intended
recipient.

Fixed with an optional `to: <name>` field in the message header
comment (`<!-- from: <sender> · <timestamp> · to: <name> -->`,
matching the exact convention the c2c-mcp server's mailbox.ts
implements for the `-send`/`-inbox` MCP tools' `to`/`as` params).
Absent `to:` means broadcast -- every message sent before this
existed, and every message a peer sends without addressing, stays
visible to every session exactly as before.

All three hooks resolve "who am I" from the hook's own `session_id`
(present in every hook's stdin JSON) by cross-referencing the CLI's
own local session registry (`~/.claude/sessions/*.json`, the same
registry the `code-sessions` MCP tool reads) for the matching
`sessionId`'s `/rename`-set `name`. `c2c_list_unread_for` in
`lib/c2c_mail.sh` then only counts a message as "this session's
unread mail" if it's addressed to that name or unaddressed
(broadcast); a message addressed elsewhere is excluded entirely --
neither notified on nor counted toward the Stop-hook block -- and is
left untouched in the filesystem for its actual addressee, avoiding
the archive-to-unblock conflict entirely.

**Fails open, not closed, when a session's name can't be resolved**
(registry missing, or this session isn't in it for some reason): every
message is then treated as broadcast, i.e. the exact pre-addressing
behavior, rather than silently hiding all mail from a session whose
identity is unknown. Covered by `test/break-tests.sh` section (f),
including the fail-open case and a negative proving the filter isn't
vacuous (mail addressed only to a different session must not block
Stop at all, not just "block less").

## Testing

`test/break-tests.sh` invokes each hook directly with synthetic JSON
on stdin, in its own throwaway `C2C_MAIL_WATCH_DIRS`/`C2C_MAIL_SESSIONS_DIR`/
`CLAUDE_PROJECT_DIR` under a temp directory -- never the real project
mailboxes or the real global session registry. Covers: mail present
(notify / block with the exact required wording), inbox empty (silent
/ stop allowed), a watched dir missing entirely (still silent
success), the `stop_hook_active` loop guard (proven non-vacuously: the
same mail still blocks when `stop_hook_active` is false), the
body-content injection-surface guard, the glob-derived watch-dir
default actually covering a channel that was never hand-listed
anywhere, and per-session addressing (section (f): addressed-elsewhere
mail excluded, broadcast and addressed-to-me mail still counted, the
fail-open case, and a negative proving the filter isn't vacuous).

```bash
bash test/break-tests.sh
```

Every non-trivial check in that script has been confirmed to actually
fail when the behavior it guards is reintroduced as a bug, then
reverted -- not just written and trusted (this repo's principle 21's
second half: a guard you haven't watched fail is not yet a guard).
