#!/usr/bin/env bash
# Break-tests for the c2c mail-awareness hooks, invoking each hook script
# directly with synthetic JSON on stdin -- per this repo's standing
# practice (docs/VACUOUS_TESTS.md, principles 10/20/21): a guard is only
# real once it's been shown to have teeth, not just written and trusted.
#
# Uses its own throwaway C2C_MAIL_WATCH_DIRS under a temp directory, never
# the real .claude/claude2claude/inbox or .claude/claude2gpt/inbox -- this
# repo has already been bitten once this session by a test writing into a
# real, shared mailbox/registry instead of an isolated one.
#
# Usage: bash test/break-tests.sh   (from c2c-mail/, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
export CLAUDE_PROJECT_DIR="$TMP_ROOT"
export C2C_MAIL_WATCH_DIRS=".claude/claude2claude/inbox .claude/claude2gpt/inbox"
# Hermetic session registry too -- without this, c2c_session_name_for_id
# would default to the REAL $HOME/.claude/sessions and read real global
# state. Harmless for sections (a)-(e) (their messages are all broadcast,
# so resolution result doesn't matter), but section (f) below populates
# this directly and every section should stay isolated regardless.
export C2C_MAIL_SESSIONS_DIR="$TMP_ROOT/.claude-sessions"
mkdir -p "$C2C_MAIL_SESSIONS_DIR"
INBOX_C2C="$TMP_ROOT/.claude/claude2claude/inbox"
INBOX_C2GPT="$TMP_ROOT/.claude/claude2gpt/inbox"

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

reset_mailboxes() {
  rm -rf "$INBOX_C2C" "$INBOX_C2GPT"
  mkdir -p "$INBOX_C2C" "$INBOX_C2GPT"
}

mkdir -p "$INBOX_C2C" "$INBOX_C2GPT"

UPS_STDIN='{"session_id":"test","prompt_id":"11111111-1111-1111-1111-111111111111","transcript_path":"/tmp/t.jsonl","cwd":"'"$TMP_ROOT"'","permission_mode":"default","hook_event_name":"UserPromptSubmit","user_prompt":"hello"}'
SESSTART_STDIN='{"session_id":"test","transcript_path":"/tmp/t.jsonl","cwd":"'"$TMP_ROOT"'","permission_mode":"default","hook_event_name":"SessionStart","source":"startup"}'
stop_stdin() { # $1 = stop_hook_active true|false
  echo '{"session_id":"test","prompt_id":"11111111-1111-1111-1111-111111111111","transcript_path":"/tmp/t.jsonl","cwd":"'"$TMP_ROOT"'","permission_mode":"default","hook_event_name":"Stop","last_assistant_message":"done","stop_hook_active":'"$1"'}'
}

# ============================================================
echo "== (a) mail present -> notification emitted / stop blocked with the named reason =="
reset_mailboxes
printf '<!-- from: claude-desktop -->\n\nreal body, should never leak\n' > "$INBOX_C2C/2026-01-01T00-00-00Z.md"

UPS_OUT="$(echo "$UPS_STDIN" | "$HOOKS_DIR/user-prompt-submit.sh")"
UPS_EXIT=$?
check "UserPromptSubmit exits 0 (notify, never block)" "$UPS_EXIT" "0"
check "UserPromptSubmit output mentions the filename" "$(echo "$UPS_OUT" | grep -c '2026-01-01T00-00-00Z.md')" "1"
check "UserPromptSubmit output is valid hookSpecificOutput JSON" \
  "$(echo "$UPS_OUT" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["hookSpecificOutput"]["hookEventName"])' 2>/dev/null)" \
  "UserPromptSubmit"

SS_OUT="$(echo "$SESSTART_STDIN" | "$HOOKS_DIR/session-start.sh")"
check "SessionStart also notifies when mail is present" "$(echo "$SS_OUT" | grep -c '2026-01-01T00-00-00Z.md')" "1"

STOP_OUT="$(stop_stdin false | "$HOOKS_DIR/stop.sh" 2>&1 1>/dev/null)"
STOP_EXIT_CHECK="$(stop_stdin false | "$HOOKS_DIR/stop.sh" >/dev/null 2>/dev/null; echo $?)"
check "Stop exits 2 (blocks) when mail present and not already active" "$STOP_EXIT_CHECK" "2"
check "Stop's stderr reason names the file" "$(echo "$STOP_OUT" | grep -c '2026-01-01T00-00-00Z.md')" "1"
check "Stop's stderr reason uses the required wording" \
  "$(echo "$STOP_OUT" | grep -c 'unread c2c mail:.*read and handle it before finishing')" "1"

# ============================================================
echo "== (b) inbox empty -> silent pass-through / stop allowed =="
reset_mailboxes
mkdir -p "$INBOX_C2C" "$INBOX_C2GPT"

UPS_OUT_EMPTY="$(echo "$UPS_STDIN" | "$HOOKS_DIR/user-prompt-submit.sh")"
UPS_EMPTY_EXIT="$(echo "$UPS_STDIN" | "$HOOKS_DIR/user-prompt-submit.sh" >/dev/null 2>/dev/null; echo $?)"
check "UserPromptSubmit exits 0 on empty inbox" "$UPS_EMPTY_EXIT" "0"
check "UserPromptSubmit produces no output on empty inbox" "$([[ -z "$UPS_OUT_EMPTY" ]] && echo yes)" "yes"

STOP_EMPTY_EXIT="$(stop_stdin false | "$HOOKS_DIR/stop.sh" >/dev/null 2>/dev/null; echo $?)"
check "Stop exits 0 (allows stop) on empty inbox" "$STOP_EMPTY_EXIT" "0"

echo "== (b-continued) watched dir missing entirely (never used / already cleaned up) -> still silent success =="
rm -rf "$TMP_ROOT/.claude"
MISSING_DIR_EXIT="$(echo "$UPS_STDIN" | "$HOOKS_DIR/user-prompt-submit.sh" >/dev/null 2>/dev/null; echo $?)"
check "UserPromptSubmit exits 0 when watched dir doesn't exist at all" "$MISSING_DIR_EXIT" "0"
MISSING_DIR_STOP_EXIT="$(stop_stdin false | "$HOOKS_DIR/stop.sh" >/dev/null 2>/dev/null; echo $?)"
check "Stop exits 0 (allows stop) when watched dir doesn't exist at all" "$MISSING_DIR_STOP_EXIT" "0"
mkdir -p "$INBOX_C2C" "$INBOX_C2GPT"

# ============================================================
echo "== (c) stop_hook_active=true + mail present -> stop ALLOWED (loop guard proven, not assumed) =="
reset_mailboxes
mkdir -p "$INBOX_C2C" "$INBOX_C2GPT"
printf '<!-- from: chatgpt -->\n\nsecond real body\n' > "$INBOX_C2GPT/2026-01-02T00-00-00Z.md"

ACTIVE_EXIT="$(stop_stdin true | "$HOOKS_DIR/stop.sh" >/dev/null 2>/dev/null; echo $?)"
check "Stop exits 0 when stop_hook_active=true, even with mail present" "$ACTIVE_EXIT" "0"
# And prove the negative held it by checking the SAME mail, active=false, still blocks --
# otherwise this "test" could pass vacuously because mail detection itself was broken.
INACTIVE_EXIT="$(stop_stdin false | "$HOOKS_DIR/stop.sh" >/dev/null 2>/dev/null; echo $?)"
check "...and the same mail DOES block when stop_hook_active=false (proves (c) isn't vacuous)" "$INACTIVE_EXIT" "2"

# ============================================================
echo "== (d) notification/block reason contains filenames, NEVER file contents =="
reset_mailboxes
MARKER="SECRET_MARKER_2f9e7c1a_DO_NOT_LEAK"
printf '<!-- from: claude-desktop -->\n\n'"$MARKER"'\n' > "$INBOX_C2C/2026-01-03T00-00-00Z.md"

UPS_LEAK_CHECK="$(echo "$UPS_STDIN" | "$HOOKS_DIR/user-prompt-submit.sh" 2>&1)"
check "UserPromptSubmit output does NOT contain the body marker" "$(echo "$UPS_LEAK_CHECK" | grep -c "$MARKER")" "0"
check "...but DOES still name the file (sanity: hook saw the mail at all)" "$(echo "$UPS_LEAK_CHECK" | grep -c '2026-01-03T00-00-00Z.md')" "1"

SS_LEAK_CHECK="$(echo "$SESSTART_STDIN" | "$HOOKS_DIR/session-start.sh" 2>&1)"
check "SessionStart output does NOT contain the body marker" "$(echo "$SS_LEAK_CHECK" | grep -c "$MARKER")" "0"

STOP_LEAK_CHECK="$(stop_stdin false | "$HOOKS_DIR/stop.sh" 2>&1 1>/dev/null)"
check "Stop's block reason does NOT contain the body marker" "$(echo "$STOP_LEAK_CHECK" | grep -c "$MARKER")" "0"
check "...but DOES still name the file" "$(echo "$STOP_LEAK_CHECK" | grep -c '2026-01-03T00-00-00Z.md')" "1"

# ============================================================
echo "== (e) default watch dirs are DERIVED (glob), not hand-maintained -- a third channel just works =="
# Deliberately unset the override for this section: proving the glob
# default actually covers an arbitrary channel is the entire point, so
# this must NOT be told where to look via C2C_MAIL_WATCH_DIRS.
reset_mailboxes
mkdir -p "$TMP_ROOT/.claude/claude2slack/inbox"
printf '<!-- from: slack-bot -->\n\na third channel nobody hand-added to any list\n' \
  > "$TMP_ROOT/.claude/claude2slack/inbox/2026-01-04T00-00-00Z.md"

GLOB_OUT="$(env -u C2C_MAIL_WATCH_DIRS CLAUDE_PROJECT_DIR="$TMP_ROOT" "$HOOKS_DIR/user-prompt-submit.sh" <<< "$UPS_STDIN")"
check "a third, never-hand-listed channel is picked up via the glob default" \
  "$(echo "$GLOB_OUT" | grep -c '2026-01-04T00-00-00Z.md')" "1"
rm -rf "$TMP_ROOT/.claude/claude2slack"

# ============================================================
echo "== (f) addressing: to: field scopes unread mail to the addressed session (or broadcast) =="
# UPS_STDIN/SESSTART_STDIN/stop_stdin all hardcode session_id "test" -- map
# that exact id to a resolvable name in the (hermetic) session registry, so
# the existing stdin fixtures can be reused instead of inventing new ones.
cat > "$C2C_MAIL_SESSIONS_DIR/424242.json" <<'EOF'
{"pid":424242,"sessionId":"test","name":"me-session","status":"idle"}
EOF

reset_mailboxes
printf '<!-- from: claude-desktop · 2026-01-05T00-00-00Z · to: someone-else -->\n\nnot for me\n' \
  > "$INBOX_C2C/2026-01-05T00-00-00Z.md"
printf '<!-- from: claude-desktop · 2026-01-05T00-01-00Z -->\n\nbroadcast, no to: field\n' \
  > "$INBOX_C2C/2026-01-05T00-01-00Z.md"
printf '<!-- from: claude-desktop · 2026-01-05T00-02-00Z · to: me-session -->\n\naddressed to me\n' \
  > "$INBOX_C2C/2026-01-05T00-02-00Z.md"

UPS_ADDR_OUT="$(echo "$UPS_STDIN" | "$HOOKS_DIR/user-prompt-submit.sh")"
check "addressed-to-someone-else message NOT surfaced" \
  "$(echo "$UPS_ADDR_OUT" | grep -c '2026-01-05T00-00-00Z.md')" "0"
check "broadcast (no to:) message IS surfaced" \
  "$(echo "$UPS_ADDR_OUT" | grep -c '2026-01-05T00-01-00Z.md')" "1"
check "addressed-to-me message IS surfaced" \
  "$(echo "$UPS_ADDR_OUT" | grep -c '2026-01-05T00-02-00Z.md')" "1"

STOP_ADDR_EXIT="$(stop_stdin false | "$HOOKS_DIR/stop.sh" >/dev/null 2>/dev/null; echo $?)"
check "Stop still blocks (broadcast + addressed-to-me mail present)" "$STOP_ADDR_EXIT" "2"

echo "== (f-continued) NEGATIVE: with ONLY someone-else-addressed mail present, Stop must NOT block =="
reset_mailboxes
printf '<!-- from: claude-desktop · 2026-01-05T00-03-00Z · to: someone-else -->\n\nstill not for me\n' \
  > "$INBOX_C2C/2026-01-05T00-03-00Z.md"
STOP_ONLY_OTHER_EXIT="$(stop_stdin false | "$HOOKS_DIR/stop.sh" >/dev/null 2>/dev/null; echo $?)"
check "Stop allowed when ONLY someone-else-addressed mail is present (filter has teeth)" \
  "$STOP_ONLY_OTHER_EXIT" "0"

echo "== (f-continued) fail-open: an unresolvable session_id still sees ALL mail (pre-addressing behavior) =="
UNKNOWN_STDIN='{"session_id":"unresolvable-id-not-in-registry","prompt_id":"22222222-2222-2222-2222-222222222222","transcript_path":"/tmp/t.jsonl","cwd":"'"$TMP_ROOT"'","permission_mode":"default","hook_event_name":"UserPromptSubmit","user_prompt":"hello"}'
UNKNOWN_OUT="$(echo "$UNKNOWN_STDIN" | "$HOOKS_DIR/user-prompt-submit.sh")"
check "unresolvable session still sees mail addressed to someone else (fails open)" \
  "$(echo "$UNKNOWN_OUT" | grep -c '2026-01-05T00-03-00Z.md')" "1"

rm -f "$C2C_MAIL_SESSIONS_DIR/424242.json"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
