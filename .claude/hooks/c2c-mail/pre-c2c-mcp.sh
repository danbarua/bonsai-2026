#!/bin/bash
# PreToolUse hook for the c2c-mcp/c2gpt-mcp tools (matcher:
# mcp__claude_ai_c2c__*, see .claude/settings.json). Three jobs:
#
# 1. Log the raw call to logs/mcp_calls.log (unchanged behavior from the
#    inline `jq -r . >> ...` command this replaces -- pulled out to a real
#    script so it can share lib/c2c_mail.sh's session-name resolution
#    instead of staying a one-liner forever).
# 2. For c2c-send/c2gpt-send calls with sender="claude-code" that didn't
#    already specify `instance`, AUTO-INJECT this session's own
#    /rename-set name via hookSpecificOutput.updatedInput, so Claude
#    Desktop can tell which Claude Code session actually wrote a given
#    message, without relying on the model remembering to pass it. For
#    c2c-inbox/c2gpt-inbox calls with reader="claude-code" that didn't
#    already specify `as`, AUTO-INJECT it the same way -- this is the
#    more consequential of the two: a consuming read (archive:true, the
#    default) with no `as` doesn't just miss a label, it silently
#    consumes EVERYTHING including mail addressed to a different
#    session, which is exactly what happened in a real incident earlier
#    this session (a stale connection had no way to pass `as` at all).
#    Auto-injection closes that gap going forward for any call that
#    reaches this hook.
# 3. For every other call (or when auto-injection doesn't apply --
#    field already set, name unresolvable, not a send/inbox call), fall
#    back to reminding the model of its own session name via
#    hookSpecificOutput.additionalContext instead.
#
# The raw session_id in the logged JSON is an opaque GUID -- not useful on
# its own for either purpose. Session name is resolved the same way the
# mail-awareness hooks resolve "who am I" for addressing
# (c2c_session_name_for_id). cwd/branch were considered for the
# additionalContext fallback and deliberately dropped: the fallback only
# exists to remind the model of an identity string worth passing to
# instance/as, and cwd/branch aren't valid values for either field --
# session name is the only thing actually worth reminding the model of
# here.
#
# updatedInput IS used here, deliberately, after being proven safe for
# THIS specific tool (c2c-send) -- not assumed from it working elsewhere.
# It's confirmed unreliable for at least one other tool type (silently
# dropped for the Agent/subagent tool, per
# github.com/anthropics/claude-code/issues/39814) and was undocumented for
# MCP tools specifically, with REPLACE (not merge) semantics where a
# silent failure could drop required fields like `content` outright.
# Verified end-to-end, not just pipe-tested: a real c2c-send call with a
# sentinel `content` value, mutated by this hook to a different sentinel,
# was confirmed via the ACTUAL FILE WRITTEN TO DISK to contain the
# mutated value, not the original -- the strongest evidence available,
# stronger than a PostToolUse cross-check (which, separately and
# unexpectedly, never fired at all for this matcher across two real
# calls -- a genuine anomaly, but not one that weakens the direct
# file-based proof). This is why the merge below always spreads the
# ORIGINAL tool_input first and overlays only `instance` -- REPLACE
# semantics means anything not echoed back would be silently dropped from
# the actual call.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/c2c_mail.sh
source "$SCRIPT_DIR/lib/c2c_mail.sh"

INPUT="$(cat)"

mkdir -p "${CLAUDE_PROJECT_DIR:-.}/.claude/claude2claude/c2c-mcp/logs"
echo "$INPUT" | jq -r . >> "${CLAUDE_PROJECT_DIR:-.}/.claude/claude2claude/c2c-mcp/logs/mcp_calls.log"

session_id="$(echo "$INPUT" | c2c_session_id_from_json)"
my_name="$(c2c_session_name_for_id "$session_id")"

tool_name="$(echo "$INPUT" | jq -r '.tool_name // empty')"
sender="$(echo "$INPUT" | jq -r '.tool_input.sender // empty')"
existing_instance="$(echo "$INPUT" | jq -r '.tool_input.instance // empty')"
reader="$(echo "$INPUT" | jq -r '.tool_input.reader // empty')"
existing_as="$(echo "$INPUT" | jq -r '.tool_input.as // empty')"

case "$tool_name" in
  *c2c-send|*c2gpt-send)
    if [ "$sender" = "claude-code" ] && [ -z "$existing_instance" ] && [ -n "$my_name" ]; then
      # Spread the ORIGINAL tool_input first, then overlay instance -- see
      # the top comment on why this can never just be {instance: $name}.
      echo "$INPUT" | jq -c --arg name "$my_name" '{
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          updatedInput: (.tool_input + {instance: $name})
        }
      }'
      exit 0
    fi
    ;;
  *c2c-inbox|*c2gpt-inbox)
    # The read-side mirror of the above, and arguably more important: a
    # consuming read (archive:true, the default) with no `as` doesn't just
    # miss an identity label -- it consumes EVERYTHING indiscriminately,
    # including mail addressed to a different session. That's exactly the
    # mechanism behind a real incident earlier this session (a stale
    # connection's call, with no way to pass `as`, archived a message
    # addressed to a different Claude Code session). Auto-injecting `as`
    # here closes that gap for any future call through THIS hook,
    # regardless of whether the model remembered to pass it.
    if [ "$reader" = "claude-code" ] && [ -z "$existing_as" ] && [ -n "$my_name" ]; then
      echo "$INPUT" | jq -c --arg name "$my_name" '{
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          updatedInput: (.tool_input + {as: $name})
        }
      }'
      exit 0
    fi
    ;;
esac

# Fails open like the addressing code it reuses: if the name can't be
# resolved, say nothing rather than inject a wrong or empty name -- the
# model just won't have identity context to include on this call.
if [ -n "$my_name" ]; then
  python3 -c '
import json, sys
name = sys.argv[1]
context = "This Claude Code instance'\''s session name (set via /rename): \"{}\".".format(name)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": context}}))
' "$my_name"
fi
exit 0
