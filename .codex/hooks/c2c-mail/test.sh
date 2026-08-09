#!/usr/bin/env bash
set -uo pipefail
ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT
export C2C_MAIL_PROJECT_DIR="$ROOT"
export C2C_MAIL_SESSION_INDEX="$ROOT/session_index.jsonl"
export C2C_MAIL_ANNOUNCED_DIR="$ROOT/announced"
mkdir -p "$ROOT/.claude/code2code/mailbox"
printf '%s\n' '{"id":"test-id","thread_name":"my-session"}' > "$C2C_MAIL_SESSION_INDEX"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIL=0
check() {
  if [ "$2" = "$3" ]; then printf 'ok - %s\n' "$1"; else printf 'FAIL - %s (got %s, want %s)\n' "$1" "$2" "$3"; FAIL=$((FAIL+1)); fi
}

payload='{"session_id":"test-id","hook_event_name":"UserPromptSubmit"}'
printf '<!-- from: other -->\nSECRET BODY\n' > "$ROOT/.claude/code2code/mailbox/one--from-other--to-my-session.md"
printf '<!-- from: other -->\nOTHER SECRET\n' > "$ROOT/.claude/code2code/mailbox/two--from-other--to-someone-else.md"

out="$(printf '%s' "$payload" | bash "$HERE/notify.sh" UserPromptSubmit)"
check 'notify includes addressed filename' "$(printf '%s' "$out" | grep -c 'one--from-other--to-my-session.md')" 1
check 'notify excludes other addressee' "$(printf '%s' "$out" | grep -c 'two--from-other--to-someone-else.md')" 0
check 'notify never includes body' "$(printf '%s' "$out" | grep -c 'SECRET BODY')" 0
check 'notify emits Codex event JSON' "$(printf '%s' "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hookSpecificOutput"]["hookEventName"])')" UserPromptSubmit

stop="$(printf '%s' "$payload" | bash "$HERE/stop.sh")"
check 'stop blocks with Codex decision' "$(printf '%s' "$stop" | python3 -c 'import json,sys; print(json.load(sys.stdin)["decision"])')" block

post1="$(printf '%s' "$payload" | bash "$HERE/post-tool-use.sh")"
post2="$(printf '%s' "$payload" | bash "$HERE/post-tool-use.sh")"
check 'post-tool announces first time' "$(printf '%s' "$post1" | grep -c 'one--from-other--to-my-session.md')" 1
check 'post-tool debounces repeat' "$(printf '%s' "$post2" | grep -c 'one--from-other--to-my-session.md')" 0

pre='{"session_id":"test-id","tool_name":"mcp__c2c__code2code-send","tool_input":{"content":"hello"}}'
preout="$(printf '%s' "$pre" | bash "$HERE/pre-c2c-mcp.sh")"
check 'pre-tool injects instance' "$(printf '%s' "$preout" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hookSpecificOutput"]["updatedInput"]["instance"])')" my-session
check 'pre-tool explicitly allows rewrite' "$(printf '%s' "$preout" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hookSpecificOutput"]["permissionDecision"])')" allow

rm "$ROOT/.claude/code2code/mailbox/one--from-other--to-my-session.md"
empty="$(printf '%s' "$payload" | bash "$HERE/stop.sh")"
check 'stop is silent for other-addressed mail only' "${#empty}" 0

exit "$FAIL"
