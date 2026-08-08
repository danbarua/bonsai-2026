#!/bin/bash
# PostToolUse hook: surfaces c2c/code2code mail MID-TURN, so a long turn
# does not sit on unread mail until it ends.
#
# Why this exists on top of the UserPromptSubmit and Stop hooks. Those two
# cover the turn's edges: one fires when the human speaks, the other when
# the turn ends. Neither helps during a turn, and turns here routinely run
# many minutes across dozens of tool calls. The observed consequence was
# Dan hand-relaying "check your mail" repeatedly in a single session --
# exactly the human-poking-the-agent role the work-and-mail-loop skill says
# it exists to remove. The edges were covered and the middle was not.
#
# Two properties this hook must have that the edge hooks do not need:
#
# 1. IT MUST BE QUIET. PostToolUse fires on every tool call, so announcing
#    the same waiting message each time would bury the turn in duplicates
#    and train the reader to skip it. Each message is therefore announced
#    AT MOST ONCE per session, tracked by basename in an ephemeral state
#    file. A message stays announced whether or not it was acted on --
#    nagging is the Stop hook's job, and it already holds a turn that ends
#    with mail unhandled.
#
# 2. IT MUST BE CHEAP. It runs on every tool call and shares a hook budget
#    with the provenance-capture hooks (measured at ~38ms per invocation,
#    ~77ms per Bash call across Pre+Post). The EMPTY-MAILBOX case -- the
#    common one across a long turn -- is pure bash and starts no
#    interpreter: `find` over the watch dirs, and nothing else.
#
#    Note what that does NOT say. Once any mail exists, the addressing
#    filter and the debounce both need the session id, and parsing it out
#    of the hook's JSON goes through python. So the cost is bimodal, not
#    uniformly small, and the numbers are in test/bench-post-tool-use.sh
#    rather than asserted here. An earlier version of this comment claimed
#    the fast path started no interpreter full stop; that was false --
#    session-id parsing ran first and shelled out on every single call.
#    Caught by measuring rather than by reading.
#
# Fail-open like every hook in this directory: any error path exits 0 and
# announces nothing. A missed notification is a nuisance; a blocked tool
# call is a broken session.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/c2c_mail.sh
source "$SCRIPT_DIR/lib/c2c_mail.sh"

INPUT="$(cat)"

# Cheapest possible bail-out FIRST, before anything parses JSON. c2c_list_unread
# is pure bash + find and needs no session id; only the addressing filter does.
# On a session with an empty mailbox -- the common case across a long turn --
# this exits without starting an interpreter at all.
#
# The ordering is deliberate and was measured: parsing session_id first cost
# ~63ms on every call, because c2c_session_id_from_json shells out to python3.
# Reordering does not reimplement anything; it just declines to pay for a
# value the empty case never uses.
[ -z "$(c2c_list_unread 2>/dev/null)" ] && exit 0

session_id="$(echo "$INPUT" | c2c_session_id_from_json)"
[ -z "$session_id" ] && exit 0

# Ephemeral, per session, outside the repo: this is runtime state, not a
# project artifact, and it must not survive into anyone's working tree.
# Overridable so break-tests get their own, following C2C_MAIL_WATCH_DIRS
# and C2C_MAIL_SESSIONS_DIR. The reason is REPEATABILITY, not safety: state
# surviving between runs makes the debounce assertions order-dependent --
# "first call announces" fails if an earlier run already recorded that
# filename. (It would not corrupt anything real; entries are keyed by
# session id and the suite uses synthetic ones, so the cost of omitting it
# is stray files in the shared temp dir plus a flaky test.)
state_dir="${C2C_MAIL_ANNOUNCED_DIR:-${TMPDIR:-/tmp}/c2c-mail-announced}"
mkdir -p "$state_dir" 2>/dev/null || exit 0
state_file="$state_dir/$(echo "$session_id" | tr -c 'A-Za-z0-9_.-' '_')"
[ -f "$state_file" ] || : > "$state_file"

# Fast path, no interpreter: basenames of what is waiting, minus what this
# session has already been told about.
waiting="$(c2c_list_unread_for "$session_id" 2>/dev/null | while IFS= read -r p; do
  [ -n "$p" ] && basename "$p"
done | sort -u)"
[ -z "$waiting" ] && exit 0

new="$(comm -23 <(printf '%s\n' "$waiting") <(sort -u "$state_file" 2>/dev/null))"
[ -z "$new" ] && exit 0

# Record before announcing. If the announcement fails, the message is still
# marked seen -- the Stop hook is the backstop that will not let a turn end
# on unhandled mail, so a lost mid-turn nudge degrades to the old behaviour
# rather than to a duplicate storm.
printf '%s\n' "$new" >> "$state_file"

printf '%s\n' "$new" | c2c_emit_notification "PostToolUse"
exit 0
