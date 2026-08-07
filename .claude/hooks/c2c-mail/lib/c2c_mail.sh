#!/bin/bash
# Shared helpers for the c2c mail-awareness hooks (UserPromptSubmit, Stop,
# SessionStart). Sourced, not executed directly -- keeps "what counts as
# unread mail" and "how a notification is formatted" defined in exactly
# one place, since the three hooks all need the same answer.

# Watched inbox dirs. Default is DERIVED from the actual directory layout
# (glob .claude/claude2*/inbox under $CLAUDE_PROJECT_DIR), not a
# hand-maintained channel list -- a third mailbox channel then just works,
# with nothing here to drift out of sync with reality. This is principle
# 21 from this repo's own docs/VACUOUS_TESTS.md: a hand-maintained list
# standing in for a derivable set will silently under-cover, and the
# broader tool you verify with is what hides it -- found live, not
# anticipated: a spawned smoke-test session flagged that every break-test
# either takes this default or overrides it explicitly, so none would
# have noticed a third channel going unwatched.
#
# Set C2C_MAIL_WATCH_DIRS (space-separated, relative-to-project or
# absolute) to bypass the glob entirely with an explicit list instead --
# used by test/break-tests.sh, which needs a throwaway location, not
# whatever the glob happens to match on this machine.
c2c_watch_dirs() {
  if [ -n "${C2C_MAIL_WATCH_DIRS:-}" ]; then
    printf '%s\n' $C2C_MAIL_WATCH_DIRS
    return
  fi
  local d
  for d in "${CLAUDE_PROJECT_DIR:-.}"/.claude/claude2*/inbox; do
    [ -d "$d" ] && printf '%s\n' "$d"
  done
}

# Prints one path per line for every unread message across the watched
# dirs, oldest-filename-first (the mailbox's own filename convention is
# already a UTC timestamp, so a plain sort is chronological). "Unread" =
# present in an inbox dir -- the c2c-mcp server's own -inbox tools move
# handled mail to archive/ by default, so a file surviving here really is
# unprocessed (nuance: an -inbox call made with archive:false, a
# deliberate peek, leaves the file in place -- the hook keeps blocking
# after a peek, correctly, since the mail genuinely hasn't been handled
# yet). A missing watched dir is not an error: prints nothing for that
# dir and moves on, so a mailbox that's never been used (or was cleaned
# up) is silent success, not a failure.
c2c_list_unread() {
  local dir abs_dir
  while IFS= read -r dir; do
    [ -z "$dir" ] && continue
    case "$dir" in
      /*) abs_dir="$dir" ;;
      *) abs_dir="${CLAUDE_PROJECT_DIR:-.}/$dir" ;;
    esac
    [ -d "$abs_dir" ] || continue
    find "$abs_dir" -maxdepth 1 -name '*.md' -type f 2>/dev/null
  done < <(c2c_watch_dirs) | sort
}

# Reads unread-mail paths on stdin (one per line, as c2c_list_unread
# produces) and, if any are present, prints a hookSpecificOutput JSON
# notification for the named hook event ($1: "UserPromptSubmit" or
# "SessionStart" -- Stop doesn't use this, it blocks instead, see
# stop.sh). Silent (no output, implicit success) when stdin is empty.
#
# Filenames only, NEVER file contents -- this is the actual security
# boundary, not just a style choice: prompt-time injection of mailbox
# *content* would be a prompt-injection surface, since anything able to
# write a file into a watched directory could get arbitrary text injected
# into every session's context. This function only ever sees basenames.
#
# JSON is built with a real encoder (python3 json.dumps), not string
# concatenation -- filenames come from a directory anyone with write
# access to the repo can populate, so they're untrusted input requiring
# proper escaping, unlike stop_hook_active below (emitted by Claude Code
# itself in a fixed format, safe to substring-match).
c2c_emit_notification() {
  local hook_event="$1"
  python3 -c '
import json, sys, os
hook_event = sys.argv[1]
lines = [l.strip() for l in sys.stdin if l.strip()]
if not lines:
    sys.exit(0)
names = [os.path.basename(l) for l in lines]
context = "c2c mail waiting: {} unread ({}). Check with /c2c inbox or the c2c-mcp tools.".format(
    len(names), ", ".join(names)
)
print(json.dumps({"hookSpecificOutput": {"hookEventName": hook_event, "additionalContext": context}}))
' "$hook_event"
}
