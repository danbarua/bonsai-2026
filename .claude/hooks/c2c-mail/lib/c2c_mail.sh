#!/bin/bash
# Shared helpers for the c2c mail-awareness hooks (UserPromptSubmit, Stop,
# SessionStart). Sourced, not executed directly -- keeps "what counts as
# unread mail" and "how a notification is formatted" defined in exactly
# one place, since the three hooks all need the same answer.

# Watched inbox dirs. Default is claude2claude/inbox ONLY -- deliberately
# NOT a glob over every .claude/claude2*/inbox, despite that being this
# file's original design (principle 21 from docs/VACUOUS_TESTS.md: derive
# a set rather than hand-list it, so a new channel isn't silently
# under-covered). That principle still holds for "which channels get
# discovered automatically" in general, but a real incident showed the
# glob was solving the wrong problem here: the intended mail topology is
# ChatGPT -> Claude Desktop -> Claude Code (via claude2claude), NOT
# ChatGPT -> Claude Code directly, so a Claude Code session's Stop hook
# blocking on raw, unaddressed claude2gpt/inbox traffic (content meant
# for Desktop to triage and relay, not for an arbitrary unrelated Code
# session to consume or archive) is a false positive, not under-coverage.
# Confirmed live: a substantive ChatGPT review ruling about an unrelated
# ML pipeline stage landed in claude2gpt/inbox and blocked a Claude Code
# session doing unrelated MCP-server engineering, with no clean way to
# unblock without either mishandling content that wasn't its business or
# leaving the session stuck. The fix is scope, not addressing: Code only
# watches the channel Desktop relays INTO it on, full stop.
#
# Set C2C_MAIL_WATCH_DIRS (space-separated, relative-to-project or
# absolute) to override with an explicit list instead -- used by
# test/break-tests.sh for a throwaway location, and available if a future
# channel genuinely does deliver straight to Code (bypassing Desktop) and
# needs watching too.
c2c_watch_dirs() {
  if [ -n "${C2C_MAIL_WATCH_DIRS:-}" ]; then
    printf '%s\n' $C2C_MAIL_WATCH_DIRS
    return
  fi
  local d="${CLAUDE_PROJECT_DIR:-.}/.claude/claude2claude/inbox"
  [ -d "$d" ] && printf '%s\n' "$d"
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

# ---- Addressing (to: field) -------------------------------------------
#
# Mirrors the same convention the c2c-mcp server's mailbox.ts implements
# (sendMessage/readMailbox's `to`/`asName`): a message's header comment may
# carry an optional "to: <name>" segment, e.g.
# "<!-- from: claude-desktop · 2026-08-07T18:00:00Z · to: c2c-implementation -->".
# Absent to: means broadcast -- every message sent before this convention
# existed, and every message a peer sends without addressing, is broadcast
# and counts as unread for every session, exactly as before. This section
# adds the machinery to resolve "who am I" (from the hook's own session_id)
# and filter c2c_list_unread down to "mail that's actually mine."

# Extracts session_id from a hook's stdin JSON payload (read on stdin,
# printed on stdout). Every hook event's payload documents this field, so
# this is used the same way in all three hooks. Prints nothing on
# malformed/empty input rather than erroring -- callers already treat an
# unresolvable session_id (empty string) as "fail open."
c2c_session_id_from_json() {
  python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
sid = data.get("session_id")
if sid:
    print(sid)
'
}

# The Claude Code CLI's own local session registry, one JSON file per PID
# (see .claude/claude2claude/c2c-mcp/src/mailbox.ts's SESSIONS_DIR for the
# same lookup done from the MCP server side). Overridable, mirroring
# C2C_MAIL_WATCH_DIRS, so tests use a throwaway registry instead of real
# global state.
c2c_sessions_dir() {
  printf '%s\n' "${C2C_MAIL_SESSIONS_DIR:-$HOME/.claude/sessions}"
}

# Resolves a Claude Code session_id (as found in every hook's stdin JSON,
# under session_id) to its human-assigned name (set via /rename), by
# scanning the session registry for a matching sessionId. Prints nothing --
# not an error -- if the registry is missing or no file matches; the caller
# (c2c_list_unread_for) treats that as "can't determine who I am" and fails
# open toward pre-addressing behavior, rather than hiding all mail from a
# session whose name genuinely can't be resolved.
c2c_session_name_for_id() {
  local session_id="$1" dir
  dir="$(c2c_sessions_dir)"
  [ -z "$session_id" ] && return 0
  [ -d "$dir" ] || return 0
  python3 -c '
import json, sys, os, glob
session_id, d = sys.argv[1], sys.argv[2]
for path in glob.glob(os.path.join(d, "*.json")):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        continue
    if data.get("sessionId") == session_id:
        name = data.get("name")
        if name:
            print(name)
        break
' "$session_id" "$dir"
}

# Extracts the optional "to: <name>" addressee from a message file's header
# comment (its first line). Prints nothing if absent (broadcast) -- mirrors
# mailbox.ts's parseAddressee exactly (same tolerance: stops at the first
# "-->" or whitespace after "to:", so it doesn't require byte-for-byte
# agreement with this project's own generated format; same anchoring too --
# "to:" must be preceded by the "·" field delimiter or start-of-line, not a
# bare substring match, so a future field whose VALUE happens to contain
# "to:" -- e.g. "auto-import-photo:staging" -- can't get misparsed as the
# addressee. Confirmed live as a real, not hypothetical, failure mode
# before this anchoring existed -- see mailbox.ts's parseAddressee comment
# for the exact adversarial case).
c2c_message_addressee() {
  local file="$1"
  head -n1 "$file" 2>/dev/null | python3 -c '
import sys, re
line = sys.stdin.readline()
before_close = line.split("-->", 1)[0]
m = re.search(r"(?:^|·)\s*to:\s*(\S+)", before_close, re.IGNORECASE)
if m:
    print(m.group(1))
'
}

# Like c2c_list_unread, but filtered to messages addressed to session_id's
# resolved name, or unaddressed (broadcast). A message addressed to a
# DIFFERENT session is excluded entirely: it isn't returned, so it won't
# block this session's Stop hook or clutter its notifications, even though
# it's still sitting in the filesystem, untouched, for its actual
# addressee to pick up later.
#
# If session_id can't be resolved to a name at all (registry missing, or
# this session isn't in it for some reason), every message is treated as
# broadcast -- fails open to the exact pre-addressing behavior rather than
# silently hiding all mail from a session whose identity is unknown.
c2c_list_unread_for() {
  local session_id="$1" my_name file addressee
  my_name="$(c2c_session_name_for_id "$session_id")"
  while IFS= read -r file; do
    [ -z "$file" ] && continue
    if [ -z "$my_name" ]; then
      printf '%s\n' "$file" # can't resolve who I am -- fail open, show everything
      continue
    fi
    addressee="$(c2c_message_addressee "$file")"
    if [ -z "$addressee" ] || [ "$addressee" = "$my_name" ]; then
      printf '%s\n' "$file"
    fi
  done < <(c2c_list_unread)
}
