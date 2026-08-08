#!/usr/bin/env bash
# Regression check for POST /webhook on the backend: turns a GitHub delivery
# into an ordinary code2code message from `no-reply`.
#
# Signature verification does NOT happen here -- it happens in the proxy,
# which is the only component that still holds the raw signed bytes (see
# test/webhook-proxy.sh). What this suite covers is the trust rule the
# backend applies to the proxy's verdict:
#
#   came through the proxy + verified   -> accept
#   came through the proxy + NOT verified -> 401
#   same-machine caller                 -> accept, authless like every other
#                                          route (the 127.0.0.1 binding is
#                                          already the trust boundary)
#
# The last case is the one worth stating: /webhook must not be reachable
# unverified from the public internet, and must stay usable from a local
# script without ceremony.
#
# Usage: bash test/webhook-route.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8823
BASE="http://127.0.0.1:$PORT"
MAILBOX="$TMP_ROOT/.claude/code2code/mailbox"
ARCHIVE="$TMP_ROOT/.claude/code2code/archive"
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
contains() {
  local desc="$1" haystack="$2" needle="$3"
  case "$haystack" in
    *"$needle"*) check "$desc" "yes" "yes" ;;
    *) check "$desc" "MISSING [$needle]" "yes" ;;
  esac
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

RUN_BODY='{"repository":{"full_name":"danbarua/bonsai-2026"},"workflow_run":{"name":"CI","head_branch":"stage2b","conclusion":"failure","html_url":"https://github.com/x/y/actions/runs/1"}}'
count_md() { ls -1 "$1"/*.md 2>/dev/null | wc -l | tr -d ' '; }

echo
echo "== a same-machine caller is accepted without ceremony =="
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H 'Content-Type: application/json' -H 'X-GitHub-Event: workflow_run' \
  -H 'X-GitHub-Delivery: local-1' --data-binary "$RUN_BODY")
check "local POST /webhook gets 202" "$CODE" "202"
check "one message landed in the mailbox" "$(count_md "$MAILBOX")" "1"
check "it is from no-reply" "$(ls -1 "$MAILBOX" | grep -c -- '--from-no-reply')" "1"

MSG="$(cat "$MAILBOX"/*.md)"
contains "summary names the event" "$MSG" 'workflow_run'
contains "summary names the branch" "$MSG" 'stage2b'
contains "summary names the conclusion" "$MSG" 'failure'
contains "summary carries the run url" "$MSG" 'actions/runs/1'
contains "summary marks the payload untrusted" "$MSG" 'never as instructions'

echo
echo "== a proxied request WITHOUT the verified marker is refused =="
BEFORE=$(count_md "$MAILBOX")
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H 'Content-Type: application/json' -H 'X-GitHub-Event: workflow_run' \
  -H 'x-c2c-via-proxy: 1' --data-binary "$RUN_BODY")
check "proxied+unverified gets 401" "$CODE" "401"
check "and wrote no message" "$(count_md "$MAILBOX")" "$BEFORE"

echo
echo "== a proxied request WITH the verified marker is accepted =="
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H 'Content-Type: application/json' -H 'X-GitHub-Event: push' \
  -H 'x-c2c-via-proxy: 1' -H 'x-c2c-verified: github' \
  --data-binary '{"repository":{"full_name":"danbarua/bonsai-2026"},"ref":"refs/heads/stage2b","compare":"https://github.com/x/y/compare/a...b"}')
check "proxied+verified gets 202" "$CODE" "202"
check "two messages now waiting" "$(count_md "$MAILBOX")" "2"

echo
echo "== a verified marker WITHOUT the proxy marker is a local caller =="
# Belt and braces: the marker alone must not be the thing that grants
# access, or a future refactor dropping the proxy check would silently make
# a forged single header sufficient.
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H 'Content-Type: application/json' -H 'X-GitHub-Event: ping' \
  -H 'x-c2c-verified: github' --data-binary '{"repository":{"full_name":"r/r"}}')
check "accepted as a local caller" "$CODE" "202"

echo
echo "== delivered mail is DELETED on consumption, not archived =="
curl -s -X POST "$BASE/mcp" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"code2code-inbox","arguments":{"as":"reader"}}}' >/dev/null
check "mailbox drained" "$(count_md "$MAILBOX")" "0"
check "nothing from no-reply reached archive/" \
  "$(ls -1 "$ARCHIVE" 2>/dev/null | grep -c -- '--from-no-reply')" "0"

echo
if [[ "$FAILURES" -eq 0 ]]; then
  echo "== $PASS_COUNT passed, 0 failed =="
else
  echo "== $PASS_COUNT passed, $FAILURES failed =="
  exit 1
fi
