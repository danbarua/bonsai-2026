#!/usr/bin/env bash
# Regression check for the `instance` field (which Claude Code SESSION
# sent a message, distinct from `sender`'s ROLE): header comment, filename
# slugging, structuredContent, and back-compat when omitted.
#
# Usage: bash test/instance.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
PORT=8802
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
structured_of() {
  node -e "
    const raw = require('fs').readFileSync(0,'utf8');
    const line = raw.split('\n').find(l => l.startsWith('data:'));
    const data = JSON.parse(line.slice(5));
    console.log(JSON.stringify(data.result.structuredContent));
  "
}

echo "== 1. sending with a plain (already-slug-safe) instance name =="
SEND_OUT="$(call c2c-send '{"sender":"claude-code","content":"plain instance test","instance":"c2c-implementation"}' | text_of)"
check "-send response mentions the instance" "$(echo "$SEND_OUT" | grep -c 'from instance c2c-implementation')" "1"
FILE1="$(ls "$TMP_ROOT/.claude/claude2claude/outbox" | head -1)"
check "filename has the instance slug appended after the timestamp, plus the --from- tag" \
  "$(echo "$FILE1" | grep -cE '^[0-9T:Z-]+-c2c-implementation--from-c2c-implementation\.md$')" "1"
check "header has the exact (unslugged, same in this case) instance name" \
  "$(grep -c 'instance: c2c-implementation' "$TMP_ROOT/.claude/claude2claude/outbox/$FILE1")" "1"

echo "== 2. a session name with spaces/mixed case slugifies for the filename but stays exact in the header =="
call c2c-send '{"sender":"claude-code","content":"spacey instance test","instance":"Mail Session Introspection"}' > /dev/null
FILE2="$(ls "$TMP_ROOT/.claude/claude2claude/outbox" | grep -v c2c-implementation)"
check "filename got the slugified form (lowercase, hyphens), plus the --from- tag" \
  "$(echo "$FILE2" | grep -cE '^[0-9T:Z-]+-mail-session-introspection--from-mail-session-introspection\.md$')" "1"
check "header kept the EXACT original name, not slugified" \
  "$(grep -c 'instance: Mail Session Introspection' "$TMP_ROOT/.claude/claude2claude/outbox/$FILE2")" "1"

echo "== 3. -inbox surfaces instance in the annotation and structuredContent =="
PEEK="$(call c2c-inbox '{"reader":"claude-desktop","archive":false}' | text_of)"
check "peek annotates c2c-implementation's message with its instance" \
  "$(echo "$PEEK" | grep -c '(instance: c2c-implementation)')" "1"
STRUCT="$(call c2c-inbox '{"reader":"claude-desktop","archive":false}' | structured_of)"
check "structuredContent includes instance for at least one message" \
  "$(echo "$STRUCT" | grep -c '"instance":"c2c-implementation"')" "1"

echo "== 4. omitting instance: no slug in filename, no instance: in header (back-compat) =="
call c2c-send '{"sender":"claude-code","content":"no instance provided"}' > /dev/null
FILE3="$(ls "$TMP_ROOT/.claude/claude2claude/outbox" | grep -v -e c2c-implementation -e mail-session)"
check "filename has NO extra slug component (just the bare timestamp)" \
  "$(echo "$FILE3" | grep -cE '^[0-9]{4}(-[0-9]{2}){2}T([0-9]{2}-){2}[0-9]{2}Z\.md$')" "1"
check "header has no instance: field at all" \
  "$(grep -c 'instance:' "$TMP_ROOT/.claude/claude2claude/outbox/$FILE3")" "0"

echo "== 5. NEGATIVE: an instance whose ENTIRE name is non-alphanumeric produces no filename slug (would be an empty/degenerate token) =="
call c2c-send '{"sender":"claude-code","content":"symbols only instance","instance":"!!!"}' > /dev/null
# Excludes FILE3 by EXACT name (-x -F), not by shape -- case 4 (no instance
# at all) and this case (a symbols-only instance, which slugifies to "")
# both legitimately produce an IDENTICALLY-shaped bare-timestamp filename,
# so a shape-based exclusion here can't tell them apart and would leave
# FILE4 empty (silently vacuous -- both real candidates would get excluded
# by their own shape). Found while updating this suite for the --from- tag:
# the checks below never actually referenced $FILE4 before this fix, so
# the emptiness was never caught.
# NOTE: case 4 and this case both slugify to an empty instance component, so
# they can collide on the exact same bare-timestamp base if run in the same
# wall-clock second -- sendMessage's own same-second collision handling then
# gives THIS case's file a "-2" suffix. Exclude case 4's file (FILE3) by
# EXACT name only, not by any "-2.md$" pattern -- that pattern would exclude
# exactly the collision-suffixed file this case is likely to produce, which
# is what happened when this suite was first updated for the --from- tag.
FILE4="$(ls "$TMP_ROOT/.claude/claude2claude/outbox" | grep -v -e c2c-implementation -e mail-session | grep -v -x -F -- "$FILE3")"
check "exactly one new file for this case (FILE4 isn't empty/ambiguous)" "$(echo -n "$FILE4" | grep -c .)" "1"
# The symbols-only send should fall back to the bare-timestamp filename shape
# (like case 4), NOT crash and NOT produce a filename with a trailing bare
# hyphen. Scoped to FILE4 specifically, not the whole directory -- other
# files in it (from cases 1/2 above) now LEGITIMATELY contain "--" as part
# of the --from- tag, so a directory-wide "--" check would false-positive
# on those instead of testing what this case actually cares about.
check "the symbols-only send's OWN filename has no dangling/degenerate hyphen" \
  "$(echo "$FILE4" | grep -cE -- '-\.md$|--')" "0"
check "the symbols-only send's filename is bare-timestamp shaped, with or without a collision suffix" \
  "$(echo "$FILE4" | grep -cE '^[0-9]{4}(-[0-9]{2}){2}T([0-9]{2}-){2}[0-9]{2}Z(-[0-9]+)?\.md$')" "1"
check "the symbols-only send still has its header instance: field (raw, unslugged)" \
  "$(grep -rl 'instance: !!!' "$TMP_ROOT/.claude/claude2claude/outbox" | wc -l | tr -d ' ')" "1"

echo "== 6. chronological (oldest-first) ordering is preserved across mixed instance/no-instance filenames =="
ORDER="$(call c2c-inbox '{"reader":"claude-desktop","archive":false}' | text_of | grep -oE '### [0-9TZ:-]+(-[a-z0-9-]+)?\.md' | sed 's/### //')"
SORTED="$(echo "$ORDER" | sort)"
check "readMailbox's returned order already matches a plain lexicographic sort" "$ORDER" "$SORTED"

echo "== 7. parseAddressee/parseInstance are ANCHORED to the · delimiter, not a bare substring search =="
# Direct unit-style check against the compiled parser functions, not an
# HTTP round trip -- this tests a header no current sendMessage call can
# produce (no field besides instance/to exists yet), but the parser must
# not depend on that staying true. Found live: an unanchored /to:\s*(\S+)/
# matched "staging" out of ".../branch: auto-import-photo:staging · to:
# real-target -->" -- the substring "photo:staging" contains a spurious
# "to:" that isn't the real field at all. Two adversarial fields here:
# "branch: auto-import-photo:staging" (embeds a spurious "to:" via
# "photo:") and "other: reinstance:old" (embeds a spurious "instance:" via
# "reinstance:") -- neither is preceded by "·", so neither should match.
ANCHOR_CHECK="$(node -e "
const { parseAddressee, parseInstance } = require('$PKG_DIR/dist/mailbox.js');
const adversarial = '<!-- from: claude-code · 2026-08-07T21:50:00Z · branch: auto-import-photo:staging · other: reinstance:old · instance: real-instance-name · to: real-target -->';
const to = parseAddressee(adversarial);
const inst = parseInstance(adversarial);
console.log(to === 'real-target' && inst === 'real-instance-name' ? 'PASS' : 'FAIL to=' + to + ' instance=' + inst);
")"
check "adversarial substrings in earlier fields don't hijack to:/instance: parsing" "$ANCHOR_CHECK" "PASS"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
