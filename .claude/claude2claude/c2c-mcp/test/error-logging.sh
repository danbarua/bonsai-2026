#!/usr/bin/env bash
# Regression check for the request-logging middleware in src/index.ts.
# Covers two things easy to get subtly wrong:
#
# 1. MCP tool-call failures come back as HTTP 200 with the error embedded in
#    the JSON-RPC result body (result.isError: true), not as an HTTP-level
#    error -- catching only res.statusCode >= 400 would silently miss every
#    one of them. Guards the chunk-decoding bug hit while first writing
#    this: the transport writes raw Uint8Array chunks, and naively
#    stringifying one with String(chunk) joins byte *values* with commas
#    instead of UTF-8-decoding them, silently corrupting the capture with no
#    error thrown anywhere.
#
# 2. Successful requests are logged too (to stdout), not just failures --
#    added after "the connector says Connected but shows no actions" turned
#    out to be indistinguishable, from an error-only log, from "the client
#    never actually called anything." Errors and successes must land in
#    separate streams (this test redirects them separately, matching
#    run-c2c-mcp.sh's real >stdout.log 2>err.log, unlike merging both into
#    one file) so a quick glance at err.log alone still tells you if
#    anything is actually broken.
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

echo "== start server (BONSAI_PROJECT_ROOT=$TMP_ROOT, stdout/stderr split like run-c2c-mcp.sh) =="
BONSAI_PROJECT_ROOT="$TMP_ROOT" C2C_MCP_PORT="$PORT" \
  node "$PKG_DIR/dist/index.js" > "$TMP_ROOT/stdout.log" 2> "$TMP_ROOT/err.log" &
SERVER_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "$BASE/health" && break
  sleep 0.1
done
curl -s "$BASE/health" | grep -q '"ok":true' || { echo "server did not start"; cat "$TMP_ROOT/err.log"; exit 1; }

call() {
  curl -s -o /dev/null -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":$1}"
}

echo "== 1. valid tools/list =="
curl -s -o /dev/null -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

echo "== 2. valid tools/call =="
call '{"name":"code2code-send","arguments":{"instance":"session-a","content":"fine"}}'

echo "== 3. missing sender: HTTP 200 but JSON-RPC isError -- must log as an error =="
call '{"name":"code2code-send","arguments":{"content":"no instance"}}'

echo "== 4. unknown tool name -- must log as an error =="
call '{"name":"does-not-exist","arguments":{}}'

echo "== 5. GET /mcp, HTTP 405 -- must log as an error =="
curl -s -o /dev/null "$BASE/mcp"

echo "== 6. unknown route, HTTP 404 -- must log as an error =="
curl -s -o /dev/null "$BASE/does-not-exist"

sleep 0.3
STDOUT_LOG="$(cat "$TMP_ROOT/stdout.log")"
ERR_LOG="$(cat "$TMP_ROOT/err.log")"

echo "== stream separation: successes never land in err.log, errors never land in stdout.log =="
check "valid tools/list logged to stdout, not err.log" \
  "$(echo "$STDOUT_LOG" | grep -c 'POST /mcp -> 200 (method: tools/list)')" "1"
check "...and does NOT appear in err.log" \
  "$(echo "$ERR_LOG" | grep -c 'tools/list')" "0"
check "valid tools/call logged to stdout" \
  "$(echo "$STDOUT_LOG" | grep -c 'POST /mcp -> 200 (method: tools/call)')" "1"

echo "== error content: real detail, not just a generic marker =="
check "missing-field error in err.log names the field, not just a marker" \
  "$(echo "$ERR_LOG" | grep -c 'instance')" "1"
check "...and NOT in stdout.log" \
  "$(echo "$STDOUT_LOG" | grep -c 'isError')" "0"
check "unknown-tool error in err.log with real detail" \
  "$(echo "$ERR_LOG" | grep -c 'does-not-exist not found')" "1"
check "GET /mcp logged to err.log as HTTP 405" \
  "$(echo "$ERR_LOG" | grep -c 'GET /mcp -> HTTP 405')" "1"
check "unknown route logged to err.log as HTTP 404" \
  "$(echo "$ERR_LOG" | grep -c 'GET /does-not-exist -> HTTP 404')" "1"
check "no chunk-decoding garbage (comma-joined byte values) leaked into either log" \
  "$(echo "$STDOUT_LOG$ERR_LOG" | grep -cE '[0-9]+,[0-9]+,[0-9]+,[0-9]+')" "0"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
