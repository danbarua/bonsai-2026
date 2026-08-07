#!/usr/bin/env bash
# Regression check for pre-c2c-mcp.sh's auto-injection of `instance` via
# hookSpecificOutput.updatedInput, and its additionalContext fallback.
# Hermetic: throwaway C2C_MAIL_SESSIONS_DIR/CLAUDE_PROJECT_DIR, never the
# real global session registry or the real mailbox logs.
#
# The updatedInput mechanism itself was proven safe for c2c-send/c2gpt-send
# via a real end-to-end call against the live server (a sentinel `content`
# value, mutated by the hook, confirmed via the actual file written to
# disk) BEFORE this auto-injection logic was written -- see the comment
# block at the top of pre-c2c-mcp.sh and DEVELOPMENT_PRACTICES.md for the
# full account, including the anomaly that PostToolUse never fired for the
# same matcher across two real calls. This test locks in the SCRIPT's
# behavior (which fields it decides to mutate, and when); it does not
# re-prove that Claude Code actually honors updatedInput for this tool --
# that was proven live, once, deliberately, and isn't cheaply re-provable
# from a pipe-tested script alone.
#
# Usage: bash test/pre-c2c-mcp.sh   (from c2c-mail/, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
export CLAUDE_PROJECT_DIR="$TMP_ROOT"
export C2C_MAIL_SESSIONS_DIR="$TMP_ROOT/.claude-sessions"
mkdir -p "$C2C_MAIL_SESSIONS_DIR"

FAILURES=0
PASS_COUNT=0

check() {
  local desc="$1" got="$2" want="$3"
  if [[ "$got" == "$want" ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "  ok   $desc"
  else
    FAILURES=$((FAILURES + 1))
    echo "  FAIL $desc -- expected [$want], got [$got]"
  fi
}

cleanup() { rm -rf "$TMP_ROOT"; }
trap cleanup EXIT

cat > "$C2C_MAIL_SESSIONS_DIR/424242.json" <<'EOF'
{"pid":424242,"sessionId":"test-session-id","name":"test-instance-name","status":"idle"}
EOF

run() {
  # run <json-payload>
  echo "$1" | bash "$HOOKS_DIR/pre-c2c-mcp.sh"
}

echo "== 1. c2c-send, sender=claude-code, no instance -> auto-injects via updatedInput, echoing all original fields =="
OUT1="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-send","tool_input":{"sender":"claude-code","content":"hello","to":"someone"}}')"
check "updatedInput present" "$(echo "$OUT1" | grep -c 'updatedInput')" "1"
check "original sender echoed back" "$(echo "$OUT1" | grep -c '"sender":"claude-code"')" "1"
check "original content echoed back (not dropped -- replace semantics)" "$(echo "$OUT1" | grep -c '"content":"hello"')" "1"
check "original to echoed back too (not dropped)" "$(echo "$OUT1" | grep -c '"to":"someone"')" "1"
check "instance added" "$(echo "$OUT1" | grep -c '"instance":"test-instance-name"')" "1"
check "no additionalContext in the same response (auto-injection replaces the reminder, not adds to it)" \
  "$(echo "$OUT1" | grep -c 'additionalContext')" "0"

echo "== 2. c2gpt-send, sender=claude-code, no instance -> ALSO auto-injects (both send tools covered) =="
OUT2="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2gpt-send","tool_input":{"sender":"claude-code","content":"hi"}}')"
check "c2gpt-send also gets updatedInput" "$(echo "$OUT2" | grep -c 'updatedInput')" "1"

echo "== 3. NEGATIVE: instance already provided -> NOT overridden, falls back to additionalContext instead =="
OUT3="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-send","tool_input":{"sender":"claude-code","content":"hello","instance":"someone-elses-choice"}}')"
check "no updatedInput when instance was already explicit (filter has teeth, not just additive)" \
  "$(echo "$OUT3" | grep -c 'updatedInput')" "0"
check "falls back to additionalContext instead" "$(echo "$OUT3" | grep -c 'additionalContext')" "1"

echo "== 4. NEGATIVE: sender=claude-desktop -> no auto-injection (only claude-code is multi-instance here) =="
OUT4="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-send","tool_input":{"sender":"claude-desktop","content":"hello"}}')"
check "no updatedInput for claude-desktop sender" "$(echo "$OUT4" | grep -c 'updatedInput')" "0"

echo "== 5. NEGATIVE: a non-send tool call (code-sessions) -> no auto-injection, normal additionalContext =="
OUT5="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__code-sessions","tool_input":{}}')"
check "no updatedInput for a non-send tool" "$(echo "$OUT5" | grep -c 'updatedInput')" "0"
check "still gets additionalContext (cwd/branch/name reminder)" "$(echo "$OUT5" | grep -c 'additionalContext')" "1"

echo "== 6. fail-open: an unresolvable session_id never auto-injects (would inject an empty/wrong name otherwise) =="
OUT6="$(run '{"session_id":"totally-unknown-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-send","tool_input":{"sender":"claude-code","content":"hello"}}')"
check "no updatedInput when the session name can't be resolved" "$(echo "$OUT6" | grep -c 'updatedInput')" "0"

echo "== 7. c2c-inbox, reader=claude-code, no as -> auto-injects as -- this is the consequential half: without it,"
echo "      a consuming read (archive:true) would silently archive mail addressed to a DIFFERENT session (the"
echo "      exact mechanism behind a real incident earlier tonight, before this auto-injection existed) =="
OUT7="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-inbox","tool_input":{"reader":"claude-code","archive":true}}')"
check "updatedInput present" "$(echo "$OUT7" | grep -c 'updatedInput')" "1"
check "original reader/archive echoed back (not dropped)" "$(echo "$OUT7" | grep -c '"reader":"claude-code"')" "1"
check "as added" "$(echo "$OUT7" | grep -c '"as":"test-instance-name"')" "1"

echo "== 8. c2gpt-inbox, reader=claude-code, no as -> ALSO auto-injects (both inbox tools covered) =="
OUT8="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2gpt-inbox","tool_input":{"reader":"claude-code"}}')"
check "c2gpt-inbox also gets updatedInput" "$(echo "$OUT8" | grep -c 'updatedInput')" "1"

echo "== 9. NEGATIVE: as already provided on -inbox -> NOT overridden =="
OUT9="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-inbox","tool_input":{"reader":"claude-code","as":"explicit-choice"}}')"
check "no updatedInput when as was already explicit" "$(echo "$OUT9" | grep -c 'updatedInput')" "0"

echo "== 10. NEGATIVE: reader=claude-desktop on -inbox -> no auto-injection =="
OUT10="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-inbox","tool_input":{"reader":"claude-desktop"}}')"
check "no updatedInput for claude-desktop reader" "$(echo "$OUT10" | grep -c 'updatedInput')" "0"

echo "== 11. c2c-inbox peek (archive:false) still gets as auto-injected too (harmless -- peeks never consume) =="
OUT11="$(run '{"session_id":"test-session-id","cwd":"/tmp","tool_name":"mcp__claude_ai_c2c__c2c-inbox","tool_input":{"reader":"claude-code","archive":false}}')"
check "peek still gets as injected" "$(echo "$OUT11" | grep -c '"as":"test-instance-name"')" "1"
check "peek's archive:false is preserved, not flipped" "$(echo "$OUT11" | grep -c '"archive":false')" "1"

echo "== 12. logging still happens on every call, auto-inject or not =="
find "$CLAUDE_PROJECT_DIR/.claude/claude2claude/c2c-mcp/logs/mcp_calls.log" -type f > /dev/null 2>&1
check "mcp_calls.log has 11 entries (one per run() call above)" \
  "$(grep -c '"tool_name"' "$CLAUDE_PROJECT_DIR/.claude/claude2claude/c2c-mcp/logs/mcp_calls.log" 2>/dev/null)" "11"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
