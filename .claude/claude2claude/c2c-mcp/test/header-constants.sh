#!/usr/bin/env bash
# Drift check for the two cross-file header constants.
#
# `x-c2c-via-proxy` and `x-c2c-verified` are each DEFINED IN MORE THAN ONE
# FILE as a bare string literal, deliberately: src/proxy.ts must keep its
# zero-dependency standalone build, so it cannot import from oauth.ts or
# index.ts (that would pull express's types onto a VM that has nothing but
# node:http). The duplication buys the standalone build and costs a drift
# risk, and this file is the payment.
#
# Renaming one copy and not the other does not fail loudly. It fails CLOSED
# and silently: the backend looks for a header the proxy no longer sets, so
# every verified webhook 401s and looks like a signing problem. Safe, but a
# long evening.
#
# The third assertion is the one that keeps this honest over time: no source
# file may contain a bare `x-c2c-*` literal outside these constant
# definitions. Without it, someone inlining the header at a new call site
# would pass every other check here while creating exactly the drift this
# file exists to catch (CLAUDE.md principle 21 -- derive the set, don't trust
# a list of the places you remembered to look).
#
# Usage: bash test/header-constants.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$PKG_DIR/src"
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

# Every value assigned to a constant of this name, anywhere in src/.
values_for() {
  grep -rhoE "const $1 = \"[^\"]+\"" "$SRC" | grep -oE '"[^"]+"' | tr -d '"' | sort -u
}
definitions_of() {
  grep -rlE "const $1 = \"[^\"]+\"" "$SRC" | sort -u
}

echo "== the two duplicated header constants agree across files =="

for NAME in PROXY_MARKER_HEADER VERIFIED_HEADER; do
  VALUES="$(values_for "$NAME")"
  DISTINCT="$(printf '%s\n' "$VALUES" | grep -c .)"
  FILES="$(definitions_of "$NAME")"
  NFILES="$(printf '%s\n' "$FILES" | grep -c .)"

  check "$NAME resolves to exactly one value" "$DISTINCT" "1"
  # Guards against the check passing vacuously because a copy was DELETED
  # rather than kept in sync -- one definition would agree with itself.
  if [[ "$NFILES" -ge 2 ]]; then
    check "$NAME is still duplicated (>=2 files), so this check is live" "yes" "yes"
  else
    check "$NAME is still duplicated (>=2 files), so this check is live" \
      "only $NFILES file(s): $(printf '%s ' $FILES)" "yes"
  fi
done

echo
echo "== no bare x-c2c-* literal escapes the constants =="

# Any `x-c2c-...` string in src/ that is NOT the right-hand side of one of
# those constant definitions. Such a literal is an inlined header: it will
# not move when the constant is renamed.
#
# Classification is done by re-reading each hit's SOURCE LINE, never by
# filtering the grep output on the value. An earlier version of this check
# excluded hits matching `"x-c2c-(via-proxy|verified)"` -- which is every
# hit, making the whole assertion vacuous. It passed against a deliberately
# inlined literal, and only break-testing showed it. Principle 10, in a
# guard written to enforce principle 21.
HITS="$(mktemp)"
grep -rnoE '"x-c2c-[a-z0-9-]*"' "$SRC" > "$HITS" 2>/dev/null || true

STRAY_LINES=""
while IFS= read -r hit; do
  [[ -z "$hit" ]] && continue
  file="${hit%%:*}"
  rest="${hit#*:}"
  line="${rest%%:*}"
  text="$(sed -n "${line}p" "$file")"
  case "$text" in
    *"const PROXY_MARKER_HEADER = "*|*"const VERIFIED_HEADER = "*) ;;
    *) STRAY_LINES="${STRAY_LINES}${file##*/}:${line} " ;;
  esac
done < "$HITS"
rm -f "$HITS"

check "no inlined x-c2c-* literals in src/" "${STRAY_LINES:-none}" "none"

echo
if [[ "$FAILURES" -eq 0 ]]; then
  echo "== $PASS_COUNT passed, 0 failed =="
else
  echo "== $PASS_COUNT passed, $FAILURES failed =="
  exit 1
fi
