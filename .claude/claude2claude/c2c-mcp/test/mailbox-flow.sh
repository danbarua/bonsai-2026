#!/usr/bin/env bash
# End-to-end regression check for src/server.ts + src/mailbox.ts's send/inbox
# direction routing. This is the exact logic that shipped broken once
# already (sendMessage always wrote to outbox/ regardless of `sender`,
# readInbox could only ever read inbox/) -- see git log for
# "send/inbox ignored who was actually asking". Committed per this repo's
# principle 20/21: hand-verified behavior is only a guard once it can fail
# loudly on its own, re-run, not trusted from a session transcript.
#
# Usage: bash test/mailbox-flow.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8797
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

echo "== 0. /health and the MCP initialize handshake both report package.json's real version =="
PKG_VERSION="$(node -p "require('$PKG_DIR/package.json').version")"
HEALTH_VERSION="$(curl -s "$BASE/health" | node -e "console.log(JSON.parse(require('fs').readFileSync(0,'utf8')).version)")"
check "/health version matches package.json (not hardcoded elsewhere)" "$HEALTH_VERSION" "$PKG_VERSION"

INIT_RESP="$(curl -s -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}')"
MCP_VERSION="$(echo "$INIT_RESP" | grep '^data:' | node -e "
  const raw = require('fs').readFileSync(0,'utf8');
  const line = raw.split('\n').find(l => l.startsWith('data:'));
  console.log(JSON.parse(line.slice(5)).result.serverInfo.version);
")"
check "MCP initialize serverInfo.version matches package.json too" "$MCP_VERSION" "$PKG_VERSION"

call() {
  # call <tool> <json-arguments>
  curl -s -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"$1\",\"arguments\":$2}}"
}

text_of() {
  # extracts the single text content block from an SSE-framed tools/call response
  node -e "
    const raw = require('fs').readFileSync(0,'utf8');
    const line = raw.split('\n').find(l => l.startsWith('data:'));
    const data = JSON.parse(line.slice(5));
    process.stdout.write(data.result.content[0].text);
  "
}

echo "== 7. c2gpt channel: chatgpt in the peer role, its own directories =="
call c2gpt-send '{"sender":"chatgpt","content":"from chatgpt"}' > /dev/null
check "c2gpt message landed in claude2gpt/inbox/" \
  "$(ls "$TMP_ROOT/.claude/claude2gpt/inbox" | wc -l | tr -d ' ')" "1"
GPTPEEK="$(call c2gpt-inbox '{"reader":"claude-desktop","archive":false}' | text_of)"
check "c2gpt-inbox as claude-desktop surfaces chatgpt's message" "$(echo "$GPTPEEK" | grep -c 'from chatgpt')" "1"

echo "== 7b. c2gpt is a shared code-side channel: BOTH claude-code and claude-desktop are valid roles =="
CODE_PEEK="$(call c2gpt-inbox '{"reader":"claude-code","archive":false}' | text_of)"
check "c2gpt-inbox as claude-code ALSO surfaces chatgpt's message (not desktop-exclusive)" \
  "$(echo "$CODE_PEEK" | grep -c 'from chatgpt')" "1"
call c2gpt-send '{"sender":"claude-code","content":"from code, direct to gpt"}' > /dev/null
call c2gpt-send '{"sender":"claude-desktop","content":"from desktop, direct to gpt"}' > /dev/null
check "both code-side roles' sends landed in the SAME outbox/ (one shared code-side identity for GPT)" \
  "$(ls "$TMP_ROOT/.claude/claude2gpt/outbox" | wc -l | tr -d ' ')" "2"

echo "== 10. collision suffixing: concurrent same-second sends never overwrite each other =="
node -e "
  process.env.BONSAI_PROJECT_ROOT = process.argv[1];
  import('$PKG_DIR/dist/mailbox.js').then(async ({ sendMessage, CHANNELS }) => {
    const results = await Promise.all([
      sendMessage(CHANNELS.code2code.outbox, 'claude-code', 'collision A'),
      sendMessage(CHANNELS.code2code.outbox, 'claude-code', 'collision B'),
      sendMessage(CHANNELS.code2code.outbox, 'claude-code', 'collision C'),
    ]);
    const names = new Set(results.map(r => r.filename));
    if (names.size !== 3) { console.error('FAIL: expected 3 distinct filenames, got', [...names]); process.exit(1); }
    console.log('ok');
  });
" "$TMP_ROOT/collision-root" > "$TMP_ROOT/collision-out" 2>&1
check "three concurrent sends -> three distinct filenames" "$(cat "$TMP_ROOT/collision-out")" "ok"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
