#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/c2c_mail.sh"

input="$(cat)"
session_id="$(printf '%s' "$input" | c2c_session_id_from_json)"
my_name="$(c2c_session_name_for_id "$session_id")"
[ -n "$my_name" ] || exit 0

python3 -c '
import json, sys
name = sys.argv[1]
try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)
tool = payload.get("tool_name", "")
tool_input = payload.get("tool_input") or {}
field = None
if tool.endswith("code2code-send") and not tool_input.get("instance"):
    field = "instance"
elif tool.endswith("code2code-inbox") and not tool_input.get("as"):
    field = "as"
if field:
    updated = dict(tool_input)
    updated[field] = name
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": updated
    }}))
else:
    context = "This Codex session thread name is: {}.".format(name)
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": context
    }}))
' "$my_name" <<< "$input"
exit 0
