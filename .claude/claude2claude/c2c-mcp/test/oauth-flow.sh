#!/usr/bin/env bash
# End-to-end regression check for src/oauth.ts's DCR+PKCE flow. Runs against
# a throwaway mailbox root and a real server process -- not unit tests of
# individual functions, but the actual sequence Claude's client (and an
# attacker probing the public endpoint) would produce. Per this repo's
# principle 20/21: a manually-verified flow is only a guard once it's
# something that fails loudly on its own, so this is meant to be re-run,
# not a one-off transcript.
#
# Usage: bash test/oauth-flow.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8799
BASE="http://127.0.0.1:$PORT"
PUBLIC_MCP_URL="$BASE/mcp"
REDIRECT_URI="https://claude.ai/api/mcp/auth_callback"
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
  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "== build =="
( cd "$PKG_DIR" && npm run build >/dev/null && npm run build-proxy >/dev/null ) || { echo "build failed"; exit 1; }

echo "== start server (BONSAI_PROJECT_ROOT=$TMP_ROOT) =="
# C2C_OAUTH_DATA_DIR keeps the signing key + persisted DCR client registry
# in this throwaway root too -- without it, this test would read and
# overwrite the real deployment's actual OAuth state (same package
# directory either way, since the path is derived from the running script's
# own location, not from BONSAI_PROJECT_ROOT).
BONSAI_PROJECT_ROOT="$TMP_ROOT" C2C_MCP_PORT="$PORT" C2C_MCP_PUBLIC_URL="$PUBLIC_MCP_URL" \
  C2C_OAUTH_DATA_DIR="$TMP_ROOT/.oauth-data" \
  node "$PKG_DIR/dist/index.js" > "$TMP_ROOT/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "$BASE/health" && break
  sleep 0.1
done
curl -s "$BASE/health" | grep -q '"ok":true' || { echo "server did not start"; cat "$TMP_ROOT/server.log"; exit 1; }

node_json() { node -e "const d=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(d$1)"; }

echo "== 1. protected resource metadata =="
PRM="$(curl -s "$BASE/.well-known/oauth-protected-resource")"
check "resource field matches public URL" "$(echo "$PRM" | node_json ".resource")" "$PUBLIC_MCP_URL"
check "authorization_servers includes issuer" "$(echo "$PRM" | node_json ".authorization_servers[0]")" "$BASE"

echo "== 2. authorization server metadata =="
ASM="$(curl -s "$BASE/.well-known/oauth-authorization-server")"
check "registration_endpoint present" "$(echo "$ASM" | node_json ".registration_endpoint")" "$BASE/register"
check "S256 advertised" "$(echo "$ASM" | node_json ".code_challenge_methods_supported[0]")" "S256"

echo "== 3. dynamic client registration =="
REG="$(curl -s -X POST "$BASE/register" -H 'Content-Type: application/json' \
  -d "{\"redirect_uris\":[\"$REDIRECT_URI\"]}")"
CLIENT_ID="$(echo "$REG" | node_json ".client_id")"
check "client_id issued" "$([[ -n "$CLIENT_ID" ]] && echo yes)" "yes"

echo "== 4. PKCE pair =="
VERIFIER="$(node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))")"
CHALLENGE="$(node -e "console.log(require('crypto').createHash('sha256').update(process.argv[1]).digest('base64url'))" "$VERIFIER")"

run_authorize() {
  # POSTs a consent approval (the GET/render path is covered separately by
  # step 5) and echoes the issued code extracted from the redirect Location.
  local state="$1"
  local loc
  loc="$(curl -s -D - -o /dev/null -X POST "$BASE/authorize" \
    --data-urlencode "response_type=code" --data-urlencode "client_id=$CLIENT_ID" \
    --data-urlencode "redirect_uri=$REDIRECT_URI" --data-urlencode "code_challenge=$CHALLENGE" \
    --data-urlencode "code_challenge_method=S256" --data-urlencode "resource=$PUBLIC_MCP_URL" \
    --data-urlencode "state=$state" | grep -i '^location:' | tr -d '\r')"
  echo "$loc" | sed -E 's/.*[?&]code=([^&]*).*/\1/'
}

echo "== 5. GET /authorize renders consent (200) =="
GET_STATUS="$(curl -s -o /dev/null -w '%{http_code}' \
  "$BASE/authorize?response_type=code&client_id=$CLIENT_ID&redirect_uri=$REDIRECT_URI&code_challenge=$CHALLENGE&code_challenge_method=S256&resource=$PUBLIC_MCP_URL&state=s1")"
check "GET /authorize returns 200" "$GET_STATUS" "200"

echo "== 6. POST /authorize (consent approved) issues a code, redirects =="
CODE="$(run_authorize s2)"
check "code issued (non-empty)" "$([[ -n "$CODE" ]] && echo yes)" "yes"

echo "== 7. token exchange with correct verifier =="
TOKRESP="$(curl -s -X POST "$BASE/token" \
  --data-urlencode "grant_type=authorization_code" --data-urlencode "code=$CODE" \
  --data-urlencode "redirect_uri=$REDIRECT_URI" --data-urlencode "client_id=$CLIENT_ID" \
  --data-urlencode "code_verifier=$VERIFIER")"
ACCESS_TOKEN="$(echo "$TOKRESP" | node_json ".access_token")"
REFRESH_TOKEN="$(echo "$TOKRESP" | node_json ".refresh_token")"
check "access_token issued" "$([[ -n "$ACCESS_TOKEN" ]] && echo yes)" "yes"
check "refresh_token issued" "$([[ -n "$REFRESH_TOKEN" ]] && echo yes)" "yes"

echo "== 8. NEGATIVE: replaying the same code fails =="
REPLAY="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/token" \
  --data-urlencode "grant_type=authorization_code" --data-urlencode "code=$CODE" \
  --data-urlencode "redirect_uri=$REDIRECT_URI" --data-urlencode "client_id=$CLIENT_ID" \
  --data-urlencode "code_verifier=$VERIFIER")"
check "reused code rejected (400)" "$REPLAY" "400"

echo "== 9. NEGATIVE: wrong code_verifier fails =="
CODE2="$(run_authorize s3)"
WRONG="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/token" \
  --data-urlencode "grant_type=authorization_code" --data-urlencode "code=$CODE2" \
  --data-urlencode "redirect_uri=$REDIRECT_URI" --data-urlencode "client_id=$CLIENT_ID" \
  --data-urlencode "code_verifier=wrong-verifier-entirely")"
check "wrong verifier rejected (400)" "$WRONG" "400"

echo "== 10. NEGATIVE: expired code fails (65s wait, code TTL is 60s) =="
CODE3="$(run_authorize s4)"
sleep 65
EXPIRED="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/token" \
  --data-urlencode "grant_type=authorization_code" --data-urlencode "code=$CODE3" \
  --data-urlencode "redirect_uri=$REDIRECT_URI" --data-urlencode "client_id=$CLIENT_ID" \
  --data-urlencode "code_verifier=$VERIFIER")"
check "expired code rejected (400)" "$EXPIRED" "400"

MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"oauth-flow-test","version":"0.0.1"}}}'
# tools/call is the only method that actually touches mailbox files, so
# it's the one that must stay gated regardless of discovery being opened up.
MCP_TOOLS_CALL='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"c2c-send","arguments":{"sender":"claude-code","content":"should be blocked pre-auth"}}}'

echo "== 11. MCP tools/call via proxy marker, no token -> 401 with WWW-Authenticate =="
UNAUTH="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'X-C2C-Via-Proxy: 1' -d "$MCP_TOOLS_CALL")"
check "unauthenticated proxied tools/call rejected (401)" "$UNAUTH" "401"
WWWAUTH="$(curl -s -D - -o /dev/null -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'X-C2C-Via-Proxy: 1' -d "$MCP_TOOLS_CALL" | grep -i 'www-authenticate' | grep -c 'resource_metadata=')"
check "WWW-Authenticate carries resource_metadata" "$WWWAUTH" "1"

echo "== 11b. discovery methods stay anonymous even via the proxy marker, no token needed =="
DISCOVERY_NO_TOKEN="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'X-C2C-Via-Proxy: 1' -d "$MCP_INIT")"
check "unauthenticated proxied initialize succeeds (200)" "$DISCOVERY_NO_TOKEN" "200"
TOOLS_LIST_NO_TOKEN="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'X-C2C-Via-Proxy: 1' -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}')"
check "unauthenticated proxied tools/list succeeds (200)" "$TOOLS_LIST_NO_TOKEN" "200"

echo "== 12. MCP call via proxy marker WITH valid token -> 200 =="
AUTHED="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'X-C2C-Via-Proxy: 1' -H "Authorization: Bearer $ACCESS_TOKEN" -d "$MCP_INIT")"
check "authenticated proxied call succeeds (200)" "$AUTHED" "200"

echo "== 13. MCP call with NO marker header (local/direct) and NO token -> still 200 =="
LOCAL="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d "$MCP_INIT")"
check "unauthenticated direct/local call still succeeds (200)" "$LOCAL" "200"

echo "== 14. refresh grant issues a new access token that also works =="
REFRESHED="$(curl -s -X POST "$BASE/token" \
  --data-urlencode "grant_type=refresh_token" --data-urlencode "refresh_token=$REFRESH_TOKEN")"
NEW_ACCESS="$(echo "$REFRESHED" | node_json ".access_token")"
check "refresh issued a new access_token" "$([[ -n "$NEW_ACCESS" && "$NEW_ACCESS" != "$ACCESS_TOKEN" ]] && echo yes)" "yes"
NEWTOK_STATUS="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'X-C2C-Via-Proxy: 1' -H "Authorization: Bearer $NEW_ACCESS" -d "$MCP_INIT")"
check "refreshed token works against /mcp (200)" "$NEWTOK_STATUS" "200"

echo "== 15. real standalone proxy: client-forged marker header gets overwritten =="
C2C_PROXY_LISTEN_PORT=8798 C2C_PROXY_TARGET_PORT="$PORT" \
  node "$PKG_DIR/dist-proxy/proxy.cjs" > "$TMP_ROOT/proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "http://127.0.0.1:8798/health" && break
  sleep 0.1
done
SPOOFED="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:8798/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'X-C2C-Via-Proxy: 0' -d "$MCP_TOOLS_CALL")"
check "forged local-looking header via real proxy still challenged (401)" "$SPOOFED" "401"
THROUGH_PROXY_AUTHED="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:8798/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H "Authorization: Bearer $ACCESS_TOKEN" -d "$MCP_TOOLS_CALL")"
check "valid token through real proxy succeeds (200)" "$THROUGH_PROXY_AUTHED" "200"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
