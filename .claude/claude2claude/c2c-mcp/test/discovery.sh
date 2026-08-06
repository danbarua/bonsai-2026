#!/usr/bin/env bash
# Regression check for MCP discovery methods (resources/prompts) and the PRM
# well-known alias. Committed per this repo's principle 20/21, guarding the
# exact bug that broke ChatGPT's Developer Mode connector entirely: McpServer
# only wires up resources/list, resources/templates/list, and prompts/list
# handlers lazily, inside registerResource/registerPrompt -- since c2c-mcp
# never calls either, those methods had zero handler and the SDK's generic
# dispatcher returned a hard JSON-RPC "Method not found" (protocol.js's
# ErrorCode.MethodNotFound) instead of a valid empty result. ChatGPT's
# discovery flow calls these unconditionally, so the failure wasn't scoped to
# resources -- it broke visibility into every tool on the server. See
# src/server.ts's createServer() for the fix (declared capabilities + three
# low-level empty-list handlers) and git log for the full diagnosis.
#
# Usage: bash test/discovery.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8795
BASE="http://127.0.0.1:$PORT"
PUBLIC_MCP_URL="$BASE/mcp"
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
# See test/oauth-flow.sh for why C2C_OAUTH_DATA_DIR is set here too.
BONSAI_PROJECT_ROOT="$TMP_ROOT" C2C_MCP_PORT="$PORT" C2C_MCP_PUBLIC_URL="$PUBLIC_MCP_URL" \
  C2C_OAUTH_DATA_DIR="$TMP_ROOT/.oauth-data" \
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
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"$1\",\"params\":{}}"
}

echo "== 1. initialize declares resources and prompts capabilities =="
INIT="$(curl -s -X POST "$BASE/mcp" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"discovery-test","version":"0.0.1"}}}')"
check "capabilities include resources" "$(echo "$INIT" | grep -c '"resources":{}')" "1"
check "capabilities include prompts" "$(echo "$INIT" | grep -c '"prompts":{}')" "1"

echo "== 2. resources/list returns a valid empty result, not an error =="
RESOURCES="$(call resources/list)"
check "no JSON-RPC error" "$(echo "$RESOURCES" | grep -c '"error"')" "0"
check "empty resources array" "$(echo "$RESOURCES" | grep -c '"result":{"resources":\[\]}')" "1"

echo "== 3. resources/templates/list returns a valid empty result =="
TEMPLATES="$(call resources/templates/list)"
check "no JSON-RPC error" "$(echo "$TEMPLATES" | grep -c '"error"')" "0"
check "empty resourceTemplates array" "$(echo "$TEMPLATES" | grep -c '"result":{"resourceTemplates":\[\]}')" "1"

echo "== 4. prompts/list returns a valid empty result =="
PROMPTS="$(call prompts/list)"
check "no JSON-RPC error" "$(echo "$PROMPTS" | grep -c '"error"')" "0"
check "empty prompts array" "$(echo "$PROMPTS" | grep -c '"result":{"prompts":\[\]}')" "1"

echo "== 5. a genuinely-unimplemented method still correctly errors (this isn't a blanket catch-all) =="
UNIMPLEMENTED="$(call completion/complete)"
check "still returns Method not found" "$(echo "$UNIMPLEMENTED" | grep -c 'Method not found')" "1"

echo "== 6. PRM well-known alias serves the same document as the canonical path =="
CANONICAL="$(curl -s "$BASE/.well-known/oauth-protected-resource")"
ALIASED="$(curl -s "$BASE/.well-known/oauth-protected-resource/mcp")"
check "alias matches canonical document" "$([[ "$CANONICAL" == "$ALIASED" ]] && echo yes)" "yes"
check "resource field matches the public URL exactly" "$(echo "$ALIASED" | grep -c "\"resource\":\"$PUBLIC_MCP_URL\"")" "1"

echo "== 7. error log names the failing method, not just that something failed =="
sleep 0.3
check "log line names completion/complete" "$(grep -c 'method: completion/complete' "$TMP_ROOT/server.log")" "1"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
