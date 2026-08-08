#!/bin/bash
# Shared helpers for the c2c mail-awareness hooks (UserPromptSubmit, Stop,
# SessionStart). Sourced, not executed directly -- keeps "what counts as
# unread mail" and "how a notification is formatted" defined in exactly
# one place, since the three hooks all need the same answer.
#
# This file DUPLICATES filesystem-matching logic that also lives in
# .claude/claude2claude/c2c-mcp/src/mailbox.ts (slugify, the
# --to-/--from- filename tag scheme, the addressing/self-exclusion
# decision) -- deliberately, not an oversight to eventually fix. These
# hooks fire synchronously on every prompt/stop/session-start and must
# answer "is there unread mail" from a plain directory listing, without
# depending on the c2c-mcp server being up or paying an HTTP round-trip
# per hook invocation (unlike the MCP tools themselves, which the model
# calls explicitly and can tolerate that cost for). The tradeoff this
# accepts: every function below that says "mirrors mailbox.ts's X" is a
# second, independent implementation of the same rule, in a different
# language, that must be kept in sync BY HAND when the TS side changes --
# there is no shared source of truth between them. `test/break-tests.sh`
# exists largely to catch exactly this kind of drift (each rule
# proven with its own break-test, not just written and trusted), but a
# change to mailbox.ts's filename/addressing conventions still requires a
# matching, deliberate edit here -- it will not happen automatically.
#
# The actual fix (not done, deliberately deferred rather than half-done
# under time pressure): a server endpoint (e.g. `GET /unread?as=<name>`)
# that answers "what's unread for me" using mailbox.ts's real logic
# directly, with these hooks calling it instead of re-parsing filenames in
# bash. Real cost, not free: these hooks currently have ZERO dependency on
# the c2c-mcp server being up (pure filesystem + bash, fails open
# trivially); an HTTP-based version would need an explicit, tested
# fail-open path for "server unreachable" on every one of the three hook
# entry points, plus break-tests.sh rewritten against a live server
# (closer to how test/addressing.sh and test/code2code.sh already work)
# instead of pure bash fixtures. Considered and explicitly deferred this
# session, not overlooked.

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
# code2code/mailbox/ IS also watched by default, unlike claude2gpt above --
# a different case, not an exception to the reasoning that excluded
# claude2gpt: that exclusion was about topology (Code isn't the intended
# recipient of raw ChatGPT<->Desktop relay traffic), not about "watch
# fewer things." code2code messages are BY Claude Code sessions FOR Claude
# Code sessions -- exactly the mail a Code session's Stop hook exists to
# notice. Addressing (c2c_list_unread_for below) still narrows it to "mine
# or broadcast," same as claude2claude.
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
  # code2code only. The claude2claude channel was removed at 0.8.0 -- every
  # party on it had a better route once Desktop joined the mesh directly.
  local d
  d="${CLAUDE_PROJECT_DIR:-.}/.claude/code2code/mailbox"
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
context = "c2c mail waiting: {} unread ({}). Check with the c2c-inbox/code2code-inbox MCP tools.".format(
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

# Mirrors mailbox.ts's slugify() exactly: lowercase, non-alphanumeric runs
# collapsed to a single hyphen, leading/trailing hyphens stripped. Needed
# because the filename fast path below compares against a SLUGIFIED
# addressee (the `--to-<slug>` filename tag), not the exact header name --
# the two slugifiers must agree, or a name with spaces/mixed case would
# match in the header-parsing fallback but not the filename fast path.
c2c_slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

# Extracts the slugified `to` addressee directly from a FILENAME's
# `--to-<slug>` tag (written by mailbox.ts's sendMessage for any message
# sent through the MCP tools), with no file open at all. Prints nothing if
# the filename carries no such tag -- either a broadcast message, or one
# sent before this convention existed (this project's own real mailbox had
# two such messages as of 2026-08-07; c2c_message_addressee below, not this
# function, is what still gets those right).
c2c_message_to_slug_from_filename() {
  local base="${1##*/}"
  case "$base" in
    *--to-*.md)
      local rest="${base##*--to-}"
      printf '%s\n' "${rest%.md}"
      ;;
  esac
}

# Extracts the slugified `instance` (sender) directly from a FILENAME's
# `--from-<slug>` tag, mirroring c2c_message_to_slug_from_filename exactly
# -- see mailbox.ts's parseFromSlugFromFilename for why this is a separate
# explicit tag rather than parsed out of the bare instance suffix. Unlike
# `--to-`, `--from-` is never the LAST tag when both are present (sendMessage
# always writes `--from-` before `--to-`), so this strips up to the (sole)
# "--from-" first, then trims a trailing "--to-..." tag if one follows,
# then trims ".md".
c2c_message_from_slug_from_filename() {
  local base="${1##*/}"
  case "$base" in
    *--from-*.md)
      local rest="${base##*--from-}"
      rest="${rest%%--to-*}"
      printf '%s\n' "${rest%.md}"
      ;;
  esac
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
# for the exact adversarial case). This is the FALLBACK path now -- see
# c2c_list_unread_for, which only opens a file at all when its filename
# carries no `--to-` tag for the fast path to check instead.
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
# Two-tier check, fast path first: a message whose FILENAME carries a
# `--to-<slug>` tag is decided from the filename alone -- no file open, no
# content regex, just a directory listing already in hand from
# c2c_list_unread. Only a file with no such tag (broadcast, or sent before
# this filename convention existed) falls through to opening it and
# checking the header via c2c_message_addressee, which is unchanged and
# still the correctness backstop for exactly that case.
#
# If session_id can't be resolved to a name at all (registry missing, or
# this session isn't in it for some reason), every message is treated as
# broadcast -- fails open to the exact pre-addressing behavior rather than
# silently hiding all mail from a session whose identity is unknown.
#
# Also excludes a session's OWN unaddressed broadcast from its own unread
# list, via the `--from-<slug>` fast path -- needed for code2code (a
# shared-directory channel; see mailbox.ts's makeSharedChannel), where
# without it, a session broadcasting an announcement would have its own
# Stop hook immediately block on the message it just sent. Inert for
# claude2claude/claude2gpt: Code's own sends never land in the directory
# c2c_list_unread reads from there, so `from_slug` can never match `my_slug`
# on those channels regardless. Only applies when the message is otherwise
# unaddressed (`to_slug` empty) -- an addressed message already gets the
# correct "leave for the real recipient" treatment above regardless of who
# sent it.
c2c_list_unread_for() {
  local session_id="$1" my_name my_slug file to_slug from_slug addressee
  my_name="$(c2c_session_name_for_id "$session_id")"
  [ -n "$my_name" ] && my_slug="$(c2c_slugify "$my_name")"
  while IFS= read -r file; do
    [ -z "$file" ] && continue
    if [ -z "$my_name" ]; then
      printf '%s\n' "$file" # can't resolve who I am -- fail open, show everything
      continue
    fi
    to_slug="$(c2c_message_to_slug_from_filename "$file")"
    if [ -n "$to_slug" ]; then
      # Fast path: filename already says who this is addressed to.
      [ "$to_slug" = "$my_slug" ] && printf '%s\n' "$file"
      continue
    fi
    from_slug="$(c2c_message_from_slug_from_filename "$file")"
    if [ -n "$from_slug" ] && [ "$from_slug" = "$my_slug" ]; then
      continue # my own broadcast -- not unread mail for me
    fi
    # Slow path: no filename tag -- open the file and check its header,
    # exactly as before the filename tag existed.
    # Slugified on both sides, matching mailbox.ts's readMailbox. An exact
    # compare here made case decide whether mail was seen, and the folded
    # form is the one agents actually learn -- it is what the filenames
    # carry, and a directory listing is cheaper to read than a
    # code-sessions call.
    addressee="$(c2c_message_addressee "$file")"
    if [ -z "$addressee" ] || [ "$(c2c_slugify "$addressee")" = "$my_slug" ]; then
      printf '%s\n' "$file"
    fi
  done < <(c2c_list_unread)
}
