#!/usr/bin/env bash
# Regression check for DCR client persistence across a server restart.
# Committed per this repo's principle 20/21: caught live mid-session when
# restarting the dev server to deploy an unrelated fix orphaned ChatGPT's
# already-registered client (in-memory registry, wiped on restart), breaking
# reconnection with a generic "Invalid authorization request." -- ChatGPT
# reuses a cached client_id rather than re-registering on every reconnect,
# unlike what the original in-memory design assumed. See src/oauth.ts's
# loadClients/saveClients for the fix.
#
# Uses its own throwaway C2C_OAUTH_DATA_DIR (see test/oauth-flow.sh) so this
# never touches a real deployment's actual signing key or client registry --
# doubly important here since this test's whole point is restarting the
# server against shared state.
#
# Usage: bash test/oauth-client-persistence.sh   (from c2c-mcp/, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
DATA_DIR="$TMP_ROOT/.oauth-data"
PORT=8794
BASE="http://127.0.0.1:$PORT"
PUBLIC_MCP_URL="$BASE/mcp"
REDIRECT_URI="https://chatgpt.com/connector/oauth/test-persistence"
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

start_server() {
  BONSAI_PROJECT_ROOT="$TMP_ROOT" C2C_MCP_PORT="$PORT" C2C_MCP_PUBLIC_URL="$PUBLIC_MCP_URL" \
    C2C_OAUTH_DATA_DIR="$DATA_DIR" \
    node "$PKG_DIR/dist/index.js" > "$TMP_ROOT/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 50); do
    curl -s -o /dev/null "$BASE/health" && return 0
    sleep 0.1
  done
  echo "server did not start"; cat "$TMP_ROOT/server.log"; exit 1
}

echo "== build =="
( cd "$PKG_DIR" && npm run build >/dev/null ) || { echo "build failed"; exit 1; }

echo "== start server #1, register a client =="
start_server
REG="$(curl -s -X POST "$BASE/register" -H 'Content-Type: application/json' \
  -d "{\"redirect_uris\":[\"$REDIRECT_URI\"]}")"
CLIENT_ID="$(node -e "console.log(JSON.parse(require('fs').readFileSync(0,'utf8')).client_id)" <<< "$REG")"
check "client_id issued" "$([[ -n "$CLIENT_ID" ]] && echo yes)" "yes"
check "client persisted to disk" "$(grep -c "$CLIENT_ID" "$DATA_DIR/oauth-clients.json" 2>/dev/null)" "1"

echo "== kill server #1, start server #2 against the SAME data dir (no fresh /register) =="
kill "$SERVER_PID" 2>/dev/null
for _ in $(seq 1 50); do
  curl -s -o /dev/null --max-time 1 "$BASE/health" || break
  sleep 0.1
done
start_server

echo "== /authorize with the client_id from server #1, on server #2, with no re-registration =="
CHALLENGE="$(node -e "console.log(require('crypto').createHash('sha256').update('whatever').digest('base64url'))")"
STATUS="$(curl -s -o /dev/null -w '%{http_code}' \
  "$BASE/authorize?response_type=code&client_id=$CLIENT_ID&redirect_uri=$REDIRECT_URI&code_challenge=$CHALLENGE&code_challenge_method=S256&resource=$PUBLIC_MCP_URL&state=s1")"
check "reused client_id survives the restart (200, not 400)" "$STATUS" "200"

echo "== NEGATIVE: a client_id that was never registered at all still correctly rejected =="
NEVER_REGISTERED="$(curl -s -o /dev/null -w '%{http_code}' \
  "$BASE/authorize?response_type=code&client_id=totally-made-up-client-id&redirect_uri=$REDIRECT_URI&code_challenge=$CHALLENGE&code_challenge_method=S256&resource=$PUBLIC_MCP_URL&state=s2")"
check "unknown client_id still rejected (400)" "$NEVER_REGISTERED" "400"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
