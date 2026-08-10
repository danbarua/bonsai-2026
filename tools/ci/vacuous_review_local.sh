#!/bin/bash
# Run the vacuous-test review locally — default model: Haiku.
#
# Why this exists: PR #29 run 31394098469 cost $5.04 on Sonnet, examined 6 of
# 12 files, and left the rest on an "in progress" sticky. The same job is
# runnable on a laptop with claude -p --model haiku for a fraction of that,
# before a checkpoint PR is even opened. Actions remains the durable record;
# this is the preflight.
#
# What it does NOT do: post sticky/inline comments by default (use --publish
# once you have verified the model is logged in and you want GitHub side
# effects). It always prints the delta and the model transcript to stdout.
#
# Usage:
#   tools/ci/vacuous_review_local.sh --pr 29
#   tools/ci/vacuous_review_local.sh --pr 29 --model sonnet --publish
#   tools/ci/vacuous_review_local.sh --base origin/stage2b-ci --head HEAD
#   tools/ci/vacuous_review_local.sh --base origin/stage2b-ci --head HEAD --delta-only
#
# Env:
#   REVIEW_MODEL   default haiku (overridden by --model)
#   REVIEW_PR      default from --pr
#   REVIEW_REPO    owner/repo (default: gh's current repository)
#   REVIEW_MAX_TURNS  default 25
#   REVIEW_EFFORT     default low
#
# Requires: gh, jq, claude (logged in: claude /login), and a checkout of the
# PR head (or --head pointing at the tree to read).

set -euo pipefail

MODEL="${REVIEW_MODEL:-haiku}"
PR="${REVIEW_PR:-}"
REPO="${REVIEW_REPO:-}"
BASE_REF=""
HEAD_REF=""
PUBLISH=0
MAX_TURNS="${REVIEW_MAX_TURNS:-25}"
EFFORT="${REVIEW_EFFORT:-low}"
DRY_DELTA=0

die() { echo "vacuous_review_local: $*" >&2; exit 1; }

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pr) PR="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --base) BASE_REF="${2:-}"; shift 2 ;;
    --head) HEAD_REF="${2:-}"; shift 2 ;;
    --publish) PUBLISH=1; shift ;;
    --max-turns) MAX_TURNS="${2:-}"; shift 2 ;;
    --effort) EFFORT="${2:-}"; shift 2 ;;
    --delta-only) DRY_DELTA=1; shift ;;
    -h|--help) usage ;;
    *) die "unknown arg: $1" ;;
  esac
done

command -v gh >/dev/null 2>&1 || die "gh is not on PATH"
command -v jq >/dev/null 2>&1 || die "jq is not on PATH"

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) \
  || die "not inside a git checkout"
cd "$ROOT"

DELTA_SH="$ROOT/tools/ci/review_delta.sh"
[ -x "$DELTA_SH" ] || die "missing $DELTA_SH"

if [ -z "$REPO" ]; then
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null) \
    || die "could not resolve REVIEW_REPO; pass --pr inside a gh-linked checkout"
fi

BEFORE=""
AFTER=""
PR_URL="(no PR)"

if [ -n "$PR" ]; then
  pr_json=$(gh pr view "$PR" --repo "$REPO" --json baseRefOid,headRefOid,number,url,baseRefName,headRefName) \
    || die "could not read PR $PR in $REPO"
  BEFORE=$(printf '%s' "$pr_json" | jq -r .baseRefOid)
  AFTER=$(printf '%s' "$pr_json" | jq -r .headRefOid)
  PR_URL=$(printf '%s' "$pr_json" | jq -r .url)
elif [ -n "$BASE_REF" ] && [ -n "$HEAD_REF" ]; then
  BEFORE=$(git rev-parse "$BASE_REF")
  AFTER=$(git rev-parse "$HEAD_REF")
else
  die "pass --pr N, or both --base and --head"
fi

echo "[local-review] repo=$REPO pr=${PR:-none} model=$MODEL" >&2
echo "[local-review] before=${BEFORE:0:12} after=${AFTER:0:12}" >&2

# Sticky carry-forward uses REVIEW_PR when set (same as CI via event path).
export REVIEW_PR="${PR:-}"
export GITHUB_REPOSITORY="$REPO"

delta_out=$(mktemp)
delta_err=$(mktemp)
cleanup() { rm -f "$delta_out" "$delta_err"; }
trap cleanup EXIT

if ! GITHUB_OUTPUT="$delta_out" bash "$DELTA_SH" "$BEFORE" "$AFTER" "$REPO" 2>"$delta_err"; then
  cat "$delta_err" >&2
  die "review_delta.sh failed"
fi
cat "$delta_err" >&2 || true

mode=$(awk -F= '/^mode=/{print $2; exit}' "$delta_out")
files=$(awk '
  /^files<</ {flag=1; next}
  flag && $0 == "REVIEW_DELTA_EOF" {flag=0; next}
  flag {print}
' "$delta_out")

echo "[local-review] mode=$mode" >&2
if [ -n "$files" ]; then
  echo "[local-review] delta body:" >&2
  printf '%s\n' "$files" | sed 's/^/  /' >&2
fi

if [ "$DRY_DELTA" -eq 1 ]; then
  printf 'mode=%s\n' "$mode"
  printf '%s\n' "$files"
  exit 0
fi

command -v claude >/dev/null 2>&1 || die "claude is not on PATH (install Claude Code CLI)"

# No Agent, no free-form Bash, no git. --publish adds gh pr comment only.
ALLOWED="Read,Grep,Glob"
if [ "$PUBLISH" -eq 1 ]; then
  ALLOWED="${ALLOWED},Bash(gh pr comment:*)"
fi

SCHEMA='{"type":"object","properties":{"no_tests_changed":{"type":"boolean"},"files_examined":{"type":"array","items":{"type":"string"}},"findings":{"type":"array","items":{"type":"object","properties":{"file":{"type":"string"},"test":{"type":"string"},"category":{"type":"string"},"what_would_have_to_change":{"type":"string"}},"required":["file","test","category","what_would_have_to_change"]}},"summary":{"type":"string"}},"required":["no_tests_changed","files_examined","findings","summary"]}'

if [ -n "$files" ]; then
  delta_body="$files"
else
  delta_body="(none listed - if mode is full, examine every in-scope test changed on the PR; if none, re-verify open sticky findings only)"
fi

# Write the prompt to a temp file so shell quoting cannot corrupt it.
prompt_file=$(mktemp)
cleanup() { rm -f "$delta_out" "$delta_err" "$prompt_file"; }
trap cleanup EXIT

{
  printf '%s\n' \
    "Review the changed TESTS for vacuity. You are running LOCALLY as a preflight" \
    "of the checkpoint review (PR ${PR:-none}, ${PR_URL})." \
    "" \
    "## Spec (read these first, once)" \
    "" \
    "- docs/VACUOUS_TESTS.md -- taxonomy and working rules. Categories and counts" \
    "  come from the document; do not invent new letters." \
    "- docs/REVIEW_COMMENT_TEMPLATE.md -- inventory structure (for your summary)." \
    "" \
    "## Scope" \
    "" \
    "IN SCOPE: tests/*.py only. Dot-directory tooling is out of scope." \
    "" \
    "CHANGED / OUTSTANDING (mode ${mode}):"
  # '%s\n' treats a leading "-" as data, never as a printf option.
  printf '%s\n' "$delta_body"
  printf '%s\n' \
    "" \
    "incremental -- examine EXAMINE + OUTSTANDING paths. none -- nothing new;" \
    "stop after confirming sticky open findings. full -- review everything in" \
    "scope on this PR." \
    "" \
    "Read files from the working tree with Read/Grep/Glob only. Do NOT run the" \
    "test suite. Do NOT spawn subagents. Do NOT use Bash except gh pr comment" \
    "when publishing is enabled." \
    "" \
    "## Output" \
    "" \
    "1. Structured JSON (schema enforced) with every test file you actually read" \
    "   in files_examined, and each finding taxonomy letter." \
    "2. A short markdown inventory matching REVIEW_COMMENT_TEMPLATE.md headings" \
    "   (Open findings / Examined / Not examined). Put paths you could not finish" \
    "   under Not examined so the next run carries them forward." \
    "3. If publishing: one short gh pr comment supplementary note only -- do not" \
    "   fight the Actions sticky." \
    "" \
    "Findings advise; they do not gate."
} >"$prompt_file"

echo "[local-review] invoking claude --model $MODEL --effort $EFFORT --max-turns $MAX_TURNS" >&2

# Do NOT pass --bare: bare mode skips OAuth/keychain, so a logged-in local
# CLI still reports "Not logged in". Project CLAUDE.md is extra context but
# auth has to work; the allowed-tools list is what keeps the review narrow.
set +e
claude -p \
  --model "$MODEL" \
  --effort "$EFFORT" \
  --max-turns "$MAX_TURNS" \
  --allowed-tools "$ALLOWED" \
  --json-schema "$SCHEMA" \
  --output-format json \
  -- "$(cat "$prompt_file")"
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  die "claude exited $rc (are you logged in? try: claude /login)"
fi
