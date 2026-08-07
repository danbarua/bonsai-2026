#!/usr/bin/env bash
# Regression check for the code2code channel (code2code-send/code2code-inbox):
# Claude Code sessions messaging each other directly, with no fixed peer
# role (unlike c2c/c2gpt, which always have a non-Code party -- Desktop or
# ChatGPT). Covers what's actually new here, not what addressing.sh/
# instance.sh already prove for the shared mailbox.ts machinery:
#
# - `instance` (send) / `as` (inbox) are REQUIRED, not optional -- a call
#   missing either must fail with a validation error, not silently proceed
#   anonymously (there's no non-Code peer to fall back to broadcast-style
#   sends for on this channel).
# - inbox === outbox is the SAME physical directory (see mailbox.ts's
#   makeSharedChannel) -- a sender must never see their own unaddressed
#   broadcast in their own next -inbox call.
# - The `--from-<slug>` filename tag exists and is combinable with `--to-`
#   in either filename position without one swallowing the other.
#
# Usage: bash test/code2code.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8813
BASE="http://127.0.0.1:$PORT"
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

cleanup() {
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "== build =="
( cd "$PKG_DIR" && npm run build >/dev/null ) || { echo "build failed"; exit 1; }

echo "== start server (BONSAI_PROJECT_ROOT=$TMP_ROOT) =="
BONSAI_PROJECT_ROOT="$TMP_ROOT" C2C_MCP_PORT="$PORT" \
  node "$PKG_DIR/dist/index.js" > "$TMP_ROOT/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "$BASE/health" && break
  sleep 0.1
done
curl -s "$BASE/health" | grep -q '"ok":true' || { echo "server did not start"; cat "$TMP_ROOT/server.log"; exit 1; }

call() {
  curl -s -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"$1\",\"arguments\":$2}}"
}
text_of() {
  node -e "
    const raw = require('fs').readFileSync(0,'utf8');
    const line = raw.split('\n').find(l => l.startsWith('data:'));
    const data = JSON.parse(line.slice(5));
    process.stdout.write(data.result.content[0].text);
  "
}
is_error() {
  node -e "
    const raw = require('fs').readFileSync(0,'utf8');
    const line = raw.split('\n').find(l => l.startsWith('data:'));
    const data = JSON.parse(line.slice(5));
    console.log(data.result.isError ? '1' : '0');
  "
}

echo "== 1. instance is REQUIRED on code2code-send -- omitting it is a validation error, not a silent anonymous send =="
NO_INSTANCE="$(call code2code-send '{"content":"anonymous"}')"
check "call errors" "$(echo "$NO_INSTANCE" | is_error)" "1"
check "no file was written" "$(ls "$TMP_ROOT/.claude/code2code/mailbox" 2>/dev/null | wc -l | tr -d ' ')" "0"

echo "== 2. as is REQUIRED on code2code-inbox -- omitting it is a validation error =="
NO_AS="$(call code2code-inbox '{}')"
check "call errors" "$(echo "$NO_AS" | is_error)" "1"

echo "== 3. session-a sends to session-b -- filename carries both --from- and --to- tags, unambiguously =="
call code2code-send '{"instance":"session-a","content":"hello B","to":"session-b"}' > /dev/null
FILE1="$(ls "$TMP_ROOT/.claude/code2code/mailbox")"
check "exactly one file written" "$(ls "$TMP_ROOT/.claude/code2code/mailbox" | wc -l | tr -d ' ')" "1"
check "filename shape: <ts>-session-a--from-session-a--to-session-b.md" \
  "$(echo "$FILE1" | grep -cE '^[0-9T:Z-]+-session-a--from-session-a--to-session-b\.md$')" "1"

echo "== 4. session-a also sends an unaddressed broadcast -- filename has --from- but no --to- =="
call code2code-send '{"instance":"session-a","content":"restarting the server shortly"}' > /dev/null
FILE2="$(ls "$TMP_ROOT/.claude/code2code/mailbox" | grep -v -- '--to-')"
check "broadcast filename shape: <ts>-session-a--from-session-a.md" \
  "$(echo "$FILE2" | grep -cE '^[0-9T:Z-]+-session-a--from-session-a\.md$')" "1"

echo "== 5. session-a's own -inbox does NOT see either message: one is addressed elsewhere, the other is its own broadcast =="
A_READ="$(call code2code-inbox '{"as":"session-a"}' | text_of)"
check "session-a's inbox is empty" "$(echo "$A_READ" | grep -c 'is empty')" "1"
check "both messages reported as skipped, not lost" "$(echo "$A_READ" | grep -c '2 message(s) left unread')" "1"
check "neither message file was archived (both still in mailbox/)" \
  "$(ls "$TMP_ROOT/.claude/code2code/mailbox" 2>/dev/null | wc -l | tr -d ' ')" "2"

echo "== 6. session-b's -inbox DOES see both (the addressed one, and the broadcast it didn't send) -- both archived =="
B_READ="$(call code2code-inbox '{"as":"session-b"}' | text_of)"
check "session-b sees the addressed message" "$(echo "$B_READ" | grep -c 'hello B')" "1"
check "session-b sees the broadcast too" "$(echo "$B_READ" | grep -c 'restarting the server shortly')" "1"
check "both now archived (mailbox/ empty)" \
  "$(ls "$TMP_ROOT/.claude/code2code/mailbox" 2>/dev/null | wc -l | tr -d ' ')" "0"
check "archive/ has both" \
  "$(ls "$TMP_ROOT/.claude/code2code/archive" 2>/dev/null | wc -l | tr -d ' ')" "2"

echo "== 7. NEGATIVE: session-c never sent anything -- its own -inbox is unaffected by session-a's self-exclusion logic =="
call code2code-send '{"instance":"session-a","content":"another broadcast"}' > /dev/null
C_READ="$(call code2code-inbox '{"as":"session-c"}' | text_of)"
check "session-c (an uninvolved third party) DOES consume a broadcast it didn't send" \
  "$(echo "$C_READ" | grep -c 'another broadcast')" "1"

echo "== 8. peek (archive:false) shows everything regardless of self-exclusion or addressing =="
call code2code-send '{"instance":"session-a","content":"peek target","to":"session-b"}' > /dev/null
A_PEEK="$(call code2code-inbox '{"as":"session-a","archive":false}' | text_of)"
check "session-a's PEEK still shows its own addressed-to-someone-else send" "$(echo "$A_PEEK" | grep -c 'peek target')" "1"

echo "== 9. code2code-archive: the escape hatch for a self-sent broadcast that would otherwise never get archived =="
call code2code-send '{"instance":"session-a","content":"stale announcement, retracting"}' > /dev/null
STALE_FILE="$(ls "$TMP_ROOT/.claude/code2code/mailbox" | grep -v -- '--to-')"
call code2code-inbox '{"as":"session-a"}' > /dev/null
check "session-a's own -inbox still can't archive its own broadcast (confirms the gap this closes)" \
  "$(ls "$TMP_ROOT/.claude/code2code/mailbox" | grep -c "$STALE_FILE")" "1"
ARCHIVE_OUT="$(call code2code-archive "{\"filename\":\"$STALE_FILE\"}" | text_of)"
check "archive reports success" "$(echo "$ARCHIVE_OUT" | grep -c "Archived $STALE_FILE")" "1"
check "file is gone from mailbox/" "$(ls "$TMP_ROOT/.claude/code2code/mailbox" 2>/dev/null | grep -c "$STALE_FILE")" "0"
check "file is now in archive/" "$(ls "$TMP_ROOT/.claude/code2code/archive" 2>/dev/null | grep -c "$STALE_FILE")" "1"

echo "== 10. code2code-archive on a nonexistent filename is a clean no-op, not an error =="
NOOP_OUT="$(call code2code-archive '{"filename":"2020-01-01T00-00-00Z-nobody.md"}' | text_of)"
check "reports not found" "$(echo "$NOOP_OUT" | grep -c 'was not found')" "1"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
