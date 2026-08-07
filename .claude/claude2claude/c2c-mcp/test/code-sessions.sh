#!/usr/bin/env bash
# Regression check for src/mailbox.ts's listCodeSessions + the code-sessions
# tool in src/server.ts. Uses a throwaway CLAUDE_SESSIONS_DIR, never the
# real ~/.claude/sessions/ -- this project has already been bitten once this
# session by a test writing into a real, shared registry (the OAuth client
# data), so this follows the same isolation pattern established there.
#
# Usage: bash test/code-sessions.sh   (from the c2c-mcp/ directory, or anywhere)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_ROOT="$(mktemp -d)"
REPO_ROOT="$TMP_ROOT/repo"
SESSIONS_DIR="$TMP_ROOT/fake-sessions"
PORT=8798
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

mkdir -p "$REPO_ROOT" "$REPO_ROOT/.claude/worktrees/some-feature" "$SESSIONS_DIR" "$TMP_ROOT/other-repo"

echo "== start server (BONSAI_PROJECT_ROOT=$REPO_ROOT, CLAUDE_SESSIONS_DIR=$SESSIONS_DIR) =="
BONSAI_PROJECT_ROOT="$REPO_ROOT" CLAUDE_SESSIONS_DIR="$SESSIONS_DIR" C2C_MCP_PORT="$PORT" \
  node "$PKG_DIR/dist/index.js" > "$TMP_ROOT/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "$BASE/health" && break
  sleep 0.1
done
curl -s "$BASE/health" | grep -q '"ok":true' || { echo "server did not start"; cat "$TMP_ROOT/server.log"; exit 1; }

call() {
  # call <tool> <json-arguments>
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

echo "== 0. no session files at all -> empty, not an error =="
EMPTY="$(call code-sessions '{}' | text_of)"
check "reports no sessions found" "$(echo "$EMPTY" | grep -c 'No Claude Code sessions found')" "1"

echo "== seed fake session files =="
# In-repo, alive: use this test script's OWN server process as a guaranteed-
# live PID, cwd = the repo root itself.
cat > "$SESSIONS_DIR/$SERVER_PID.json" <<EOF
{"pid":$SERVER_PID,"sessionId":"11111111-1111-1111-1111-111111111111","cwd":"$REPO_ROOT","name":"root-session","status":"idle","jobId":"aaaa1111","updatedAt":1786120000000}
EOF

# In-repo (a worktree), NOT alive: PID astronomically unlikely to exist.
cat > "$SESSIONS_DIR/99999901.json" <<EOF
{"pid":99999901,"sessionId":"22222222-2222-2222-2222-222222222222","cwd":"$REPO_ROOT/.claude/worktrees/some-feature","name":"worktree-session","status":"idle","jobId":"bbbb2222","updatedAt":1786120000000}
EOF

# Outside the repo entirely -- must be filtered out.
cat > "$SESSIONS_DIR/99999902.json" <<EOF
{"pid":99999902,"sessionId":"33333333-3333-3333-3333-333333333333","cwd":"$TMP_ROOT/other-repo","name":"unrelated-session","status":"idle","jobId":"cccc3333","updatedAt":1786120000000}
EOF

# Malformed JSON -- must be skipped, not crash the whole call.
echo 'not valid json at all {' > "$SESSIONS_DIR/99999903.json"

echo "== 1. in-repo alive session appears, correctly marked alive =="
OUT="$(call code-sessions '{}' | text_of)"
check "2 sessions counted (in-repo only; malformed skipped, out-of-repo excluded)" \
  "$(echo "$OUT" | grep -oE '^[0-9]+ session' | grep -oE '^[0-9]+')" "2"
check "root-session present" "$(echo "$OUT" | grep -c 'root-session')" "1"
check "root-session marked alive" "$(echo "$OUT" | grep 'root-session' | grep -c 'alive')" "1"
check "root-session NOT marked not-running" "$(echo "$OUT" | grep 'root-session' | grep -c 'not running')" "0"

echo "== 2. in-repo dead-pid session appears, correctly marked not running =="
check "worktree-session present" "$(echo "$OUT" | grep -c 'worktree-session')" "1"
check "worktree-session marked not running" "$(echo "$OUT" | grep 'worktree-session' | grep -c 'not running')" "1"

echo "== 3. NEGATIVE: out-of-repo session is filtered out entirely (proves the filter has teeth) =="
check "unrelated-session does NOT appear" "$(echo "$OUT" | grep -c 'unrelated-session')" "0"

echo "== 4. malformed session file does not error out the whole call =="
check "call still succeeded (no error text)" "$(echo "$OUT" | grep -ci 'error')" "0"

echo
echo "== $PASS_COUNT passed, $FAILURES failed =="
exit $((FAILURES > 0))
