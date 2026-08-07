#!/usr/bin/env bash
# Regression check for optional message addressing (the `to`/`as` params
# added to -send/-inbox and mailbox.ts's sendMessage/readMailbox): a
# message can be addressed to one specific session name, and a consuming
# -inbox read (archive:true) must skip -- not consume -- mail addressed to
# someone else, while a peek (archive:false) always shows everything.
#
# Usage: bash test/addressing.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8799
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

echo "== 1. omitting to: keeps the pre-addressing header format (no to: segment written) =="
call c2c-send '{"sender":"claude-code","content":"broadcast message"}' > /dev/null
RAW_BROADCAST="$(cat "$TMP_ROOT/.claude/claude2claude/outbox"/*.md)"
check "no 'to:' text in an unaddressed message's header" "$(echo "$RAW_BROADCAST" | head -n1 | grep -c 'to:')" "0"

echo "== 2. sending with to: writes it into the header, and -send's response reflects it =="
SEND_OUT="$(call c2c-send '{"sender":"claude-code","content":"for a specific reader","to":"reader-b"}' | text_of)"
check "-send response mentions the addressee" "$(echo "$SEND_OUT" | grep -c 'addressed to reader-b')" "1"
check "header on disk contains to: reader-b" \
  "$(grep -l 'to: reader-b' "$TMP_ROOT/.claude/claude2claude/outbox"/*.md | wc -l | tr -d ' ')" "1"

echo "== 3. peek (archive:false) shows an addressed message regardless of who's asking =="
PEEK_AS_WRONG="$(call c2c-inbox '{"reader":"claude-desktop","archive":false,"as":"reader-a"}' | text_of)"
check "peek as reader-a still shows the message addressed to reader-b" \
  "$(echo "$PEEK_AS_WRONG" | grep -c 'for a specific reader')" "1"
check "peek shows the (to: reader-b) annotation" "$(echo "$PEEK_AS_WRONG" | grep -c '(to: reader-b)')" "1"

echo "== 4. consuming read (archive:true) AS THE WRONG NAME skips the addressed message, doesn't consume it =="
WRONG_READ="$(call c2c-inbox '{"reader":"claude-desktop","as":"reader-a"}' | text_of)"
check "wrong-name consuming read does NOT surface the addressed message" \
  "$(echo "$WRONG_READ" | grep -c 'for a specific reader')" "0"
check "response notes it was left unread for another session" \
  "$(echo "$WRONG_READ" | grep -c 'left unread')" "1"
check "message file is STILL in outbox/ (not archived)" \
  "$(ls "$TMP_ROOT/.claude/claude2claude/outbox" 2>/dev/null | wc -l | tr -d ' ')" "1"
check "broadcast message from step 1 WAS consumed (archived) despite as being set" \
  "$(ls "$TMP_ROOT/.claude/claude2claude/archive" 2>/dev/null | wc -l | tr -d ' ')" "1"

echo "== 5. consuming read AS THE RIGHT NAME does consume the addressed message =="
RIGHT_READ="$(call c2c-inbox '{"reader":"claude-desktop","as":"reader-b"}' | text_of)"
check "right-name consuming read DOES surface the addressed message" \
  "$(echo "$RIGHT_READ" | grep -c 'for a specific reader')" "1"
check "outbox/ now empty -- addressed message was archived" \
  "$(ls "$TMP_ROOT/.claude/claude2claude/outbox" 2>/dev/null | wc -l | tr -d ' ')" "0"
check "archive/ now has both messages" \
  "$(ls "$TMP_ROOT/.claude/claude2claude/archive" 2>/dev/null | wc -l | tr -d ' ')" "2"

echo "== 6. omitting as entirely keeps the pre-addressing behavior: everything visible and consumable =="
call c2c-send '{"sender":"claude-code","content":"another addressed one","to":"reader-c"}' > /dev/null
NO_AS_READ="$(call c2c-inbox '{"reader":"claude-desktop"}' | text_of)"
check "consuming read with NO as consumes an addressed message anyway (back-compat)" \
  "$(echo "$NO_AS_READ" | grep -c 'another addressed one')" "1"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
