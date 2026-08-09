#!/usr/bin/env bash
# Regression check for session names containing SPACES.
#
# `/rename` accepts anything, and "mailbox digest generation" is a live
# session on this machine. The header parsers used `(\S+)`, which stops at
# the first space, so `to: mailbox digest generation` parsed as "mailbox".
# readMailbox compares slugify(to) against slugify(asName), and "mailbox"
# never matches "mailbox-digest-generation" -- so the message was skipped as
# addressed to someone ELSE, by the only session it was for.
#
# Undeliverable, with no error to sender or reader: the file sits in the
# mailbox looking delivered, counted as unread by everyone and claimable by
# nobody. The FILENAME was always correct (`--to-mailbox-digest-generation`);
# only the header parse truncated, and readMailbox reads the header.
#
# Pre-existing -- any addressed message to a multi-word name was lost. It
# became visible only when webhook fan-out started addressing every live
# session by name.
#
# Usage: bash test/multiword-names.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8825
BASE="http://127.0.0.1:$PORT"
MAILBOX="$TMP_ROOT/.claude/code2code/mailbox"
MULTI="mailbox digest generation"
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
    process.stdout.write(JSON.parse(line.slice(5)).result.content[0].text);
  "
}
count_md() { ls -1 "$1"/*.md 2>/dev/null | wc -l | tr -d ' '; }

echo
echo "== a multi-word addressee actually receives its mail =="

call code2code-send "{\"instance\":\"sender-one\",\"to\":\"$MULTI\",\"content\":\"addressed to a multi-word name\"}" >/dev/null
check "the message was written" "$(count_md "$MAILBOX")" "1"
check "the filename carries the full slug" \
  "$(ls -1 "$MAILBOX" | grep -c -- '--to-mailbox-digest-generation')" "1"

# The bug: a DIFFERENT session must not be able to take it either. If the
# addressee parsed as "mailbox", this read would still skip -- so this
# assertion alone does not prove the fix. It is the pair that does.
OTHER="$(call code2code-inbox '{"as":"someone-else"}' | text_of)"
case "$OTHER" in
  *"addressed to a multi-word name"*) check "another session does NOT receive it" "received" "skipped" ;;
  *) check "another session does NOT receive it" "skipped" "skipped" ;;
esac
check "and left it in place" "$(count_md "$MAILBOX")" "1"

# The discriminating one: before the fix this skipped too, and the message
# was claimable by nobody.
MINE="$(call code2code-inbox "{\"as\":\"$MULTI\"}" | text_of)"
case "$MINE" in
  *"addressed to a multi-word name"*) check "the named session DOES receive it" "received" "received" ;;
  *) check "the named session DOES receive it" "skipped" "received" ;;
esac
check "and consumed it" "$(count_md "$MAILBOX")" "0"

echo
echo "== a multi-word SENDER still recognises its own broadcast =="

# parseInstance had the same truncation, and it feeds excludeSelfSent: a
# session that cannot recognise its own name would consume its own
# unaddressed broadcast before anyone else saw it -- exactly what
# excludeSelfSent exists to prevent.
call code2code-send "{\"instance\":\"$MULTI\",\"content\":\"my own broadcast\"}" >/dev/null
check "the broadcast was written" "$(count_md "$MAILBOX")" "1"

SELF="$(call code2code-inbox "{\"as\":\"$MULTI\"}" | text_of)"
case "$SELF" in
  *"my own broadcast"*) check "the sender does NOT consume its own broadcast" "consumed" "skipped" ;;
  *) check "the sender does NOT consume its own broadcast" "skipped" "skipped" ;;
esac
check "so it is still there for someone else" "$(count_md "$MAILBOX")" "1"

THIRD="$(call code2code-inbox '{"as":"third-party"}' | text_of)"
case "$THIRD" in
  *"my own broadcast"*) check "another session CAN consume it" "consumed" "consumed" ;;
  *) check "another session CAN consume it" "not received" "consumed" ;;
esac

echo
if [[ "$FAILURES" -eq 0 ]]; then
  echo "== $PASS_COUNT passed, 0 failed =="
else
  echo "== $PASS_COUNT passed, $FAILURES failed =="
  exit 1
fi
