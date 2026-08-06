#!/usr/bin/env bash
# Regression check for the logErrorRequests middleware in src/index.ts.
# Covers the part that's easy to get subtly wrong: MCP tool-call failures
# come back as HTTP 200 with the error embedded in the JSON-RPC result body
# (result.isError: true), not as an HTTP-level error -- catching only
# res.statusCode >= 400 would silently miss every one of them. Also guards
# the actual chunk-decoding bug hit while writing this: the transport
# writes raw Uint8Array chunks, and naively stringifying one with
# String(chunk) joins byte *values* with commas instead of UTF-8-decoding
# them, silently corrupting the capture with no error thrown anywhere.
#
# Usage: bash test/error-logging.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8796
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
  curl -s -o /dev/null -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":$1}"
}

echo "== 1. valid tools/list produces no log line =="
curl -s -o /dev/null -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

echo "== 2. valid tools/call produces no log line =="
call '{"name":"c2c-send","arguments":{"sender":"claude-code","content":"fine"}}'

echo "== 3. missing sender: HTTP 200 but JSON-RPC isError -- must still log =="
call '{"name":"c2c-send","arguments":{"content":"no sender"}}'

echo "== 4. unknown tool name -- must still log =="
call '{"name":"does-not-exist","arguments":{}}'

echo "== 5. GET /mcp, HTTP 405 -- must log =="
curl -s -o /dev/null "$BASE/mcp"

echo "== 6. unknown route, HTTP 404 -- must log =="
curl -s -o /dev/null "$BASE/does-not-exist"

sleep 0.3
LOG="$(cat "$TMP_ROOT/server.log")"

check "no [c2c-mcp] line for the valid tools/list" \
  "$(echo "$LOG" | grep -c '\[c2c-mcp\].*tools/list')" "0"
check "exactly one [c2c-mcp] error line for the two valid calls combined" \
  "$(echo "$LOG" | grep -c '^\[c2c-mcp\]')" "4"
check "missing-sender error logged with real detail, not just a generic marker" \
  "$(echo "$LOG" | grep -c 'Invalid option: expected one of')" "1"
check "unknown-tool error logged with real detail" \
  "$(echo "$LOG" | grep -c 'does-not-exist not found')" "1"
check "GET /mcp logged as HTTP 405" \
  "$(echo "$LOG" | grep -c 'GET /mcp -> HTTP 405')" "1"
check "unknown route logged as HTTP 404" \
  "$(echo "$LOG" | grep -c 'GET /does-not-exist -> HTTP 404')" "1"
check "no chunk-decoding garbage (comma-joined byte values) leaked into the log" \
  "$(echo "$LOG" | grep -cE '[0-9]+,[0-9]+,[0-9]+,[0-9]+')" "0"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
