#!/bin/bash
# PreToolUse hook for the c2c-mcp/c2gpt-mcp tools (matcher:
# mcp__claude_ai_c2c__*, see .claude/settings.json). Two jobs:
#
# 1. Log the raw call to logs/mcp_calls.log (unchanged behavior from the
#    inline `jq -r . >> ...` command this replaces -- pulled out to a real
#    script so it can share lib/c2c_mail.sh's session-name resolution
#    instead of staying a one-liner forever).
# 2. Remind the model of its own identity -- session name, cwd, git
#    branch -- via hookSpecificOutput.additionalContext, so it can include
#    this when calling c2c-send/c2gpt-send with sender="claude-code" and
#    Claude Desktop can tell which Claude Code session (and which worktree
#    /branch it's working on) actually wrote a given message. The raw
#    session_id in the logged JSON is an opaque GUID -- not useful on its
#    own for that purpose. Session name is resolved the same way the
#    mail-awareness hooks resolve "who am I" for addressing
#    (c2c_session_name_for_id); cwd comes straight from the hook's own
#    stdin payload (Claude Code already provides it, no lookup needed);
#    branch is derived from that cwd the same way statusline-command.sh
#    derives it for the status line, for consistency.
#
# ONLY additionalContext, deliberately -- not hookSpecificOutput.updatedInput.
# updatedInput is confirmed unreliable for at least one other tool type
# (silently dropped for the Agent/subagent tool, per
# github.com/anthropics/claude-code/issues/39814) and undocumented for MCP
# tools specifically; a REPLACE (not merge) semantics means a silent
# failure there could drop required fields like `content` outright. Not
# using it here until proven safe for c2c-send/c2gpt-send specifically via
# a real end-to-end PostToolUse check, per this repo's own "verify before
# trusting" discipline.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/c2c_mail.sh
source "$SCRIPT_DIR/lib/c2c_mail.sh"

INPUT="$(cat)"

mkdir -p "${CLAUDE_PROJECT_DIR:-.}/.claude/claude2claude/c2c-mcp/logs"
echo "$INPUT" | jq -r . >> "${CLAUDE_PROJECT_DIR:-.}/.claude/claude2claude/c2c-mcp/logs/mcp_calls.log"

session_id="$(echo "$INPUT" | c2c_session_id_from_json)"
my_name="$(c2c_session_name_for_id "$session_id")"
cwd="$(echo "$INPUT" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("cwd",""))
except Exception:
    pass')"

branch=""
if [ -n "$cwd" ] && git --no-optional-locks -C "$cwd" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  branch="$(git --no-optional-locks -C "$cwd" branch --show-current 2>/dev/null)"
  [ -z "$branch" ] && branch="$(git --no-optional-locks -C "$cwd" rev-parse --short HEAD 2>/dev/null)"
fi

# Fails open like the addressing code it reuses: if the name can't be
# resolved, say nothing rather than inject a wrong or empty name -- the
# model just won't have identity context to include on this call. cwd and
# branch are included whenever available, independent of whether the name
# resolved.
if [ -n "$my_name" ] || [ -n "$cwd" ]; then
  python3 -c '
import json, sys
name, cwd, branch = sys.argv[1], sys.argv[2], sys.argv[3]
parts = []
if name:
    parts.append("session name \"{}\" (set via /rename)".format(name))
if cwd:
    parts.append("working directory {}".format(cwd))
if branch:
    parts.append("git branch \"{}\"".format(branch))
identity = ", ".join(parts)
context = (
    "This Claude Code instance: {}. When calling c2c-send or c2gpt-send "
    "with sender=\"claude-code\", include this identity so Claude Desktop "
    "knows which Claude Code session -- and which worktree/branch -- wrote "
    "the message."
).format(identity)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": context}}))
' "$my_name" "$cwd" "$branch"
fi
exit 0
