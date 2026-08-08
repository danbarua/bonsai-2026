#!/usr/bin/env bash
# Measure post-tool-use.sh's per-call cost, both paths.
#
# Committed rather than run as a heredoc because the number it prints is the
# disclosure other sessions are owed before this hook lands in their
# settings -- and a number that anchors a decision has to be reproducible
# from committed code (CLAUDE.md principle 24). The precedent is INFRA
# measuring their capture hook's ~38ms and disclosing it unprompted.
#
# Hermetic: its own state dir and its own mailbox under a temp root, so it
# neither reads real mail nor leaves entries behind.
#
# Usage: bash test/bench-post-tool-use.sh [iterations]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$(dirname "$SCRIPT_DIR")"
N="${1:-30}"

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT
export CLAUDE_PROJECT_DIR="$TMP_ROOT"
export C2C_MAIL_ANNOUNCED_DIR="$TMP_ROOT/.announced"
export C2C_MAIL_SESSIONS_DIR="$TMP_ROOT/.sessions"
mkdir -p "$C2C_MAIL_SESSIONS_DIR" "$TMP_ROOT/.claude/claude2claude/inbox"

stdin_for() { echo '{"session_id":"'"$1"'","hook_event_name":"PostToolUse","tool_name":"Bash"}'; }

bench() { # $1 = label, $2 = session-id prefix, $3 = reuse-session (yes/no)
  local label="$1" prefix="$2" reuse="$3" i sid start end
  start=$(date +%s%N)
  for ((i = 0; i < N; i++)); do
    if [ "$reuse" = yes ]; then sid="$prefix"; else sid="$prefix$i"; fi
    stdin_for "$sid" | "$HOOKS_DIR/post-tool-use.sh" >/dev/null 2>&1
  done
  end=$(date +%s%N)
  echo "$label: $(( (end - start) / N / 1000000 )) ms/call over $N calls"
}

echo "== empty mailbox: the path that runs on most tool calls =="
bench "  no mail at all          " "empty" yes

printf '<!-- from: claude-desktop -->\n\nbody\n' \
  > "$TMP_ROOT/.claude/claude2claude/inbox/2026-01-01T00-00-00Z.md"

# Announce once, then measure the repeat path -- mail present, nothing new.
stdin_for seen | "$HOOKS_DIR/post-tool-use.sh" >/dev/null 2>&1
echo "== mail present but already announced: the steady state during a long turn =="
bench "  mail waiting, debounced " "seen" yes

echo "== announcing: rare by construction, once per message per session =="
bench "  fresh session, formats  " "fresh" no
