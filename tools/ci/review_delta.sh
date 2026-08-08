#!/bin/bash
# Which in-scope test files changed since the last review ran.
#
# On a `synchronize` event GitHub hands us the two commits bracketing the
# push -- `github.event.before` and `github.event.after` -- and the compare
# API returns the files between them. That is the incremental set: what a
# reviewer needs to look at now, as opposed to everything the pull request
# has ever touched, which is what `<changed_files>` carries.
#
# The distinction is the point. A review that re-reads the whole branch on
# every push produces a fresh full report each time, and a fresh full report
# is not an update -- it buries whether anything got fixed, and it pays to
# re-read files nobody has touched in twenty commits.
#
# GitHub-native on purpose. The context is already in GitHub, and this
# repository has already built three broken designs from reconstructing it
# with git plumbing. There is no `git` call here and the checkout may be
# shallow.
#
# FAIL-OPEN, LOUDLY. A force-push makes `before` unreachable and the compare
# 404s; a first run has no previous review at all. Both fall back to
# reviewing everything and SAY so, because a silent fallback to "review
# nothing" is the failure this whole workflow exists to avoid.
#
# Usage: review_delta.sh <before-sha> <after-sha> <owner/repo>
#   Writes `files` and `mode` to $GITHUB_OUTPUT when set, else to stdout.

set -u

BEFORE="${1:-}"
AFTER="${2:-}"
REPO="${3:-${GITHUB_REPOSITORY:-}}"
OUT="${GITHUB_OUTPUT:-/dev/stdout}"

# `tests/*.py` only -- the same narrowing the prompt applies. Dot-directory
# shell tooling is out of scope and is not counted anywhere.
IN_SCOPE='^tests/[^/]*\.py$'

emit() {  # emit <mode> <files...>
  local mode="$1"; shift
  {
    echo "mode=$mode"
    echo "files<<REVIEW_DELTA_EOF"
    printf '%s\n' "$@" | grep . || true
    echo "REVIEW_DELTA_EOF"
  } >> "$OUT"
}

# The zero SHA is what GitHub sends for a branch that did not previously
# exist. Treat it as "no previous state" rather than as a commit.
ZERO="0000000000000000000000000000000000000000"

if [ -z "$BEFORE" ] || [ "$BEFORE" = "$ZERO" ] || [ -z "$AFTER" ]; then
  emit full
  echo "[delta] no usable previous commit; reviewing everything in scope" >&2
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  emit full
  echo "[delta] gh unavailable; reviewing everything in scope" >&2
  exit 0
fi

if ! compare=$(gh api "repos/${REPO}/compare/${BEFORE}...${AFTER}" \
                 --jq '.files[].filename' 2>/dev/null); then
  # Most often a force-push: `before` is no longer reachable. Not an error,
  # and not a reason to review nothing.
  emit full
  echo "[delta] compare ${BEFORE}...${AFTER} failed (force-push?); " \
       "reviewing everything in scope" >&2
  exit 0
fi

changed=$(printf '%s\n' "$compare" | grep -E "$IN_SCOPE" | sort -u)

if [ -z "$changed" ]; then
  # A real, well-formed answer: this push touched no in-scope test file.
  # Distinct from the failures above, and the prompt treats it differently --
  # there is nothing new to read, only open findings to re-verify.
  emit none
  echo "[delta] no in-scope test files changed in this push" >&2
  exit 0
fi

emit incremental $changed
echo "[delta] $(printf '%s\n' "$changed" | grep -c .) in-scope test file(s) changed since ${BEFORE:0:7}" >&2
