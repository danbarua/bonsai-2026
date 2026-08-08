#!/usr/bin/env bash
# Regression check for the `no-reply` sender: machine-generated notifications
# (webhook deliveries and the like) are DELETED when consumed, not archived.
#
# Why it matters, and why deletion rather than a shorter retention: archive/
# is the corpus the mailbox digests are derived from. The digest checkpoint is
# a manifest diffed against archive/ (`comm -13`), so anything landing there
# is counted as undigested until a digest covers it. Archiving webhook traffic
# would pad every digest with CI notifications nobody reads a week later, and
# inflate the unsummarised counter on the statusline permanently.
#
# Both retirement paths must behave the same way, because they are the same
# function (mailbox.ts retireMessage) called twice:
#
# - the consuming read     (code2code-inbox  -> readMailbox)
# - the explicit retire    (code2code-archive -> archiveMessageByFilename)
#
# A control case runs beside each, because "the file is not in archive/" is
# also true when nothing was ever written -- the assertion has to distinguish
# deleted-because-no-reply from never-arrived.
#
# Usage: bash test/no-reply.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8817
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
count_md() { ls -1 "$1"/*.md 2>/dev/null | wc -l | tr -d ' '; }

echo
echo "== consuming read: no-reply is deleted, a normal sender is archived =="

call code2code-send '{"instance":"no-reply","content":"build failed on stage2b"}' >/dev/null
call code2code-send '{"instance":"ci-watcher","content":"a real session speaking"}' >/dev/null
check "both messages are waiting in mailbox/" "$(count_md "$MAILBOX")" "2"

INBOX_OUT="$(call code2code-inbox '{"as":"reader"}' | text_of)"

# Delivered: the reader must actually SEE the no-reply message. Deleting it
# without delivering it would also leave archive/ empty, and would be a far
# worse bug than the one this file is guarding.
case "$INBOX_OUT" in
  *"build failed on stage2b"*) check "no-reply message IS delivered to the reader" "yes" "yes" ;;
  *) check "no-reply message IS delivered to the reader" "no" "yes" ;;
esac

check "mailbox/ is drained by the consuming read" "$(count_md "$MAILBOX")" "0"
check "exactly one message reached archive/" "$(count_md "$ARCHIVE")" "1"
check "the archived one is the real session, not no-reply" \
  "$(ls -1 "$ARCHIVE" | grep -c -- '--from-ci-watcher')" "1"
check "no no-reply message survives anywhere in archive/" \
  "$(ls -1 "$ARCHIVE" | grep -c -- '--from-no-reply')" "0"

echo
echo "== explicit retire (code2code-archive) takes the same path =="

# archiveMessageByFilename never opens the file -- it is handed a name -- so
# this is the call site that would silently keep archiving no-reply if the
# rule had been implemented on message CONTENT instead of the filename.
# A DIFFERENT sender name from the first phase, deliberately. sendMessage's
# collision loop (flag: "wx") only checks the mailbox, never archive/, so a
# same-second re-send from the same sender after the first was archived
# produces an identical filename and fs.rename silently overwrites the
# archived copy. That is a real pre-existing hazard, unrelated to no-reply,
# and reusing `ci-watcher` here made this suite fail for that reason rather
# than for anything it is testing.
call code2code-send '{"instance":"no-reply","content":"second delivery"}' >/dev/null
call code2code-send '{"instance":"other-session","content":"second real message"}' >/dev/null
NOREPLY_FILE="$(ls -1 "$MAILBOX" | grep -- '--from-no-reply' | head -1)"
REAL_FILE="$(ls -1 "$MAILBOX" | grep -- '--from-other-session' | head -1)"

call code2code-archive "{\"filename\":\"$NOREPLY_FILE\"}" >/dev/null
call code2code-archive "{\"filename\":\"$REAL_FILE\"}" >/dev/null

check "explicit retire drains mailbox/" "$(count_md "$MAILBOX")" "0"
check "explicit retire archived the real message" \
  "$(ls -1 "$ARCHIVE" | grep -c -- '--from-other-session')" "1"
check "explicit retire DELETED the no-reply message" \
  "$(ls -1 "$ARCHIVE" | grep -c -- '--from-no-reply')" "0"

echo
echo "== a name merely containing 'no-reply' is not the reserved sender =="

# The predicate is an exact match on the parsed from-slug, not a substring.
# `no-reply-bot` is a different sender and its mail belongs in the archive.
call code2code-send '{"instance":"no-reply-bot","content":"not the reserved name"}' >/dev/null
call code2code-inbox '{"as":"reader"}' >/dev/null
check "no-reply-bot is archived like any other sender" \
  "$(ls -1 "$ARCHIVE" | grep -c -- '--from-no-reply-bot')" "1"

echo
echo "== a burst in one second still deletes every message =="

# sendMessage resolves same-second collisions with a `-2`, `-3` suffix at the
# END of the filename. With no `--to-` segment that counter lands inside the
# from-slug, so `--from-no-reply-2.md` parses as the sender "no-reply-2" and
# an exact match would archive it. Found this way: three webhook deliveries
# in one second archived two of themselves. CI sends bursts, so this is the
# common case rather than an edge.
call code2code-send '{"instance":"no-reply","content":"burst one"}' >/dev/null
call code2code-send '{"instance":"no-reply","content":"burst two"}' >/dev/null
call code2code-send '{"instance":"no-reply","content":"burst three"}' >/dev/null
check "three collided filenames are waiting" "$(count_md "$MAILBOX")" "3"
BEFORE_ARCHIVE="$(count_md "$ARCHIVE")"
call code2code-inbox '{"as":"reader"}' >/dev/null
check "burst drains the mailbox" "$(count_md "$MAILBOX")" "0"
check "and none of the burst reached archive/" "$(count_md "$ARCHIVE")" "$BEFORE_ARCHIVE"

echo
if [[ "$FAILURES" -eq 0 ]]; then
  echo "== $PASS_COUNT passed, 0 failed =="
else
  echo "== $PASS_COUNT passed, $FAILURES failed =="
  exit 1
fi
