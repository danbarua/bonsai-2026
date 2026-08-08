#!/usr/bin/env bash
# Regression check for GitHub webhook verification in the reverse proxy
# (src/proxy.ts). The proxy is the only component that still holds the RAW
# request bytes -- the backend's express app parses JSON before any route
# runs, and a re-serialized body does not reproduce what GitHub signed.
#
# The signature this suite expects is computed with `openssl dgst -sha256
# -hmac`, deliberately a DIFFERENT implementation from the proxy's
# node:crypto. If both sides used node, a wrong-but-self-consistent usage --
# hex vs base64, signing the parsed body, the wrong prefix -- would agree
# with itself and pass while being broken against real GitHub.
#
# The assertion that matters most is the forgery one. filteredHeaders()
# forwards every non-hop-by-hop client header, and `x-c2c-verified` is set
# CONDITIONALLY, so without an unconditional delete a caller could simply
# send the header themselves and the backend would believe the proxy issued
# it. That check is here because the guard has to be seen failing, not
# assumed.
#
# Usage: bash test/webhook-proxy.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
BACKEND_PORT=8821
PROXY_PORT=8822
SECRET="test-webhook-secret-do-not-use-anywhere"
SEEN="$TMP_ROOT/seen.jsonl"
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
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null
  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "== build proxy =="
( cd "$PKG_DIR" && npm run build-proxy >/dev/null ) || { echo "build failed"; exit 1; }

# Stand-in backend: records the headers of every request that reaches it, so
# the tests can assert on what the proxy did and did NOT forward. A rejected
# request must leave no line here at all.
cat > "$TMP_ROOT/backend.cjs" <<'EOF'
const http = require('node:http');
const fs = require('node:fs');
const seen = process.env.SEEN_PATH;
http.createServer((req, res) => {
  let body = '';
  req.on('data', c => { body += c; });
  req.on('end', () => {
    fs.appendFileSync(seen, JSON.stringify({ url: req.url, headers: req.headers, body }) + '\n');
    res.writeHead(200, { 'content-type': 'text/plain' });
    res.end('backend ok');
  });
}).listen(Number(process.env.PORT), '127.0.0.1');
EOF

: > "$SEEN"
SEEN_PATH="$SEEN" PORT="$BACKEND_PORT" node "$TMP_ROOT/backend.cjs" &
BACKEND_PID=$!

echo "== start proxy (secret set, 1KB body cap) =="
C2C_PROXY_LISTEN_HOST=127.0.0.1 C2C_PROXY_LISTEN_PORT="$PROXY_PORT" \
  C2C_PROXY_TARGET_HOST=127.0.0.1 C2C_PROXY_TARGET_PORT="$BACKEND_PORT" \
  C2C_GITHUB_WEBHOOK_SECRET="$SECRET" C2C_WEBHOOK_MAX_BYTES=1024 \
  node "$PKG_DIR/dist-proxy/proxy.cjs" > "$TMP_ROOT/proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "http://127.0.0.1:$PROXY_PORT/health" && break
  sleep 0.1
done
# The readiness probe goes THROUGH the proxy, so the backend has already
# recorded it. Reset the log so the counts below start from a clean zero.
: > "$SEEN"

BASE="http://127.0.0.1:$PROXY_PORT"
BODY='{"action":"completed","workflow_run":{"conclusion":"failure"}}'

# openssl, not node -- an independent implementation of the same spec.
sign() { printf '%s' "$1" | openssl dgst -sha256 -hmac "$SECRET" -r | awk '{print $1}'; }
SIG="sha256=$(sign "$BODY")"

seen_count() { wc -l < "$SEEN" | tr -d ' '; }
last_verified() {
  node -e "
    const fs=require('fs');
    const lines=fs.readFileSync(process.argv[1],'utf8').trim().split('\n').filter(Boolean);
    if(!lines.length){process.stdout.write('NO-REQUEST');process.exit(0)}
    const h=JSON.parse(lines[lines.length-1]).headers;
    process.stdout.write(h['x-c2c-verified'] ?? 'ABSENT');
  " "$SEEN"
}

echo
echo "== a correctly signed delivery is verified and forwarded =="
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H "X-Hub-Signature-256: $SIG" -H 'Content-Type: application/json' \
  --data-binary "$BODY")
check "signed delivery gets 200" "$CODE" "200"
check "backend received it" "$(seen_count)" "1"
check "backend sees x-c2c-verified: github" "$(last_verified)" "github"

echo
echo "== a bad signature is rejected AT THE EDGE =="
BEFORE=$(seen_count)
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H "X-Hub-Signature-256: sha256=$(printf 'f%.0s' $(seq 1 64))" \
  -H 'Content-Type: application/json' --data-binary "$BODY")
check "bad signature gets 401" "$CODE" "401"
check "backend never saw the request" "$(seen_count)" "$BEFORE"

echo
echo "== a signature over DIFFERENT bytes than were sent is rejected =="
BEFORE=$(seen_count)
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H "X-Hub-Signature-256: $SIG" -H 'Content-Type: application/json' \
  --data-binary '{"action":"tampered"}')
check "tampered body gets 401" "$CODE" "401"
check "backend never saw the tampered request" "$(seen_count)" "$BEFORE"

echo
echo "== FORGERY: a client cannot supply x-c2c-verified itself =="
# No signature header at all, so this takes the ordinary streaming path and
# is forwarded -- the point is that the header must be stripped on the way.
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/anything" \
  -H 'x-c2c-verified: github' -H 'Content-Type: application/json' \
  --data-binary '{"forged":true}')
check "unsigned request is still forwarded" "$CODE" "200"
check "client-supplied x-c2c-verified is STRIPPED" "$(last_verified)" "ABSENT"

echo
echo "== forged header PLUS a bad signature is still rejected =="
BEFORE=$(seen_count)
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H 'x-c2c-verified: github' \
  -H "X-Hub-Signature-256: sha256=$(printf 'a%.0s' $(seq 1 64))" \
  -H 'Content-Type: application/json' --data-binary "$BODY")
check "forged header does not bypass verification" "$CODE" "401"
check "backend never saw it" "$(seen_count)" "$BEFORE"

echo
echo "== ordinary traffic is untouched =="
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")
check "unsigned GET is proxied normally" "$CODE" "200"
check "and carries no verified marker" "$(last_verified)" "ABSENT"

echo
echo "== oversized signed body is capped, not buffered =="
BEFORE=$(seen_count)
BIG="$(printf 'x%.0s' $(seq 1 4096))"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H "X-Hub-Signature-256: sha256=$(sign "$BIG")" \
  -H 'Content-Type: application/json' --data-binary "$BIG")
check "body over the cap gets 413" "$CODE" "413"
check "backend never saw the oversized body" "$(seen_count)" "$BEFORE"

echo
echo "== a non-numeric byte cap falls back, it does not switch the cap off =="

# `Number("25MB")` is NaN and `size > NaN` is ALWAYS FALSE, so a plausible
# human value would silently disable the memory-DoS guard it was meant to
# configure. The fallback must hold, and it must say so.
kill "$PROXY_PID" 2>/dev/null; wait "$PROXY_PID" 2>/dev/null
BAD_LOG="$TMP_ROOT/proxy-badcap.log"
C2C_PROXY_LISTEN_HOST=127.0.0.1 C2C_PROXY_LISTEN_PORT="$PROXY_PORT" \
  C2C_PROXY_TARGET_HOST=127.0.0.1 C2C_PROXY_TARGET_PORT="$BACKEND_PORT" \
  C2C_GITHUB_WEBHOOK_SECRET="$SECRET" C2C_WEBHOOK_MAX_BYTES="25MB" \
  node "$PKG_DIR/dist-proxy/proxy.cjs" > "$BAD_LOG" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "$BASE/health" && break
  sleep 0.1
done
: > "$SEEN"

check "the bad value is reported" \
  "$(grep -c 'is not a positive integer number of bytes' "$BAD_LOG")" "1"

# Corroborating, NOT discriminating -- say so rather than let it read as
# proof. 4096 bytes is under the 25MiB default, but it is also under a NaN
# cap, since every comparison against NaN is false. This assertion passes
# whether the fallback works or the cap is switched off entirely. The log
# line above is the one that tells those apart.
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/webhook" \
  -H "X-Hub-Signature-256: sha256=$(sign "$BIG")" \
  -H 'Content-Type: application/json' --data-binary "$BIG")
check "the default cap applies, so a 4KB body is accepted" "$CODE" "200"
check "and it reached the backend" "$(seen_count)" "1"

echo
if [[ "$FAILURES" -eq 0 ]]; then
  echo "== $PASS_COUNT passed, 0 failed =="
else
  echo "== $PASS_COUNT passed, $FAILURES failed =="
  exit 1
fi
