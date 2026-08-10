#!/bin/bash
# Run the vacuous-test review locally — default model: Haiku.
#
# Why this exists: PR #29 run 31394098469 cost $5.04 on Sonnet, examined 6 of
# 12 files, and left the rest on an "in progress" sticky. The same job is
# runnable on a laptop with `claude -p --model haiku` for a fraction of that,
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
#
# Env:
#   REVIEW_MODEL   default haiku (overridden by --model)
#   REVIEW_PR      default from --pr
#   REVIEW_REPO    owner/repo (default: gh's current repository)
#
# Requires: gh, jq, claude (logged in: `claude /login`), and a checkout of the
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

if [ -n "$PR" ]; then
  pr_json=$(gh pr view "$PR" --repo "$REPO" --json baseRefOid,headRefOid,number,url,baseRefName,headRefName) \
    || die "could not read PR $PR in $REPO"
  BASE_SHA=$(printf '%s' "$pr_json" | jq -r .baseRefOid)
  HEAD_SHA=$(printf '%s' "$pr_json" | jq -r .headRefOid)
  PR_URL=$(printf '%s' "$pr_json" | jq -r .url)
  # Whole-PR first pass: before=base, after=head. review_delta fail-opens to
  # full when before is missing; here both are real commits.
  BEFORE="$BASE_SHA"
  AFTER="$HEAD_SHA"
else
  [ -n "$BASE_REF" ] && [ -n "$HEAD_REF" ] \
    || die "pass --pr N, or both --base and --head"
  BEFORE=$(git rev-parse "$BASE_REF")
  AFTER=$(git rev-parse "$HEAD_REF")
  PR_URL="(no PR)"
fi

echo "[local-review] repo=$REPO pr=${PR:-none} model=$MODEL" >&2
echo "[local-review] before=${BEFORE:0:12} after=${AFTER:0:12}" >&2

# Fetch sticky carry-forward the same way CI will once GITHUB_EVENT_PATH is set.
export REVIEW_PR="${PR:-}"
export GITHUB_REPOSITORY="$REPO"

delta_out=$(mktemp)
trap 'rm -f "$delta_out"' EXIT
if ! GITHUB_OUTPUT="$delta_out" bash "$DELTA_SH" "$BEFORE" "$AFTER" "$REPO" 2>"${delta_out}.err"; then
  cat "${delta_out}.err" >&2
  die "review_delta.sh failed"
fi
cat "${delta_out}.err" >&2 || true

mode=$(awk -F= '/^mode=/{print $2; exit}' "$delta_out")
# multiline files<<EOF ... EOF
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

# Allowed tools: mirror CI's intent. No Agent, no free-form Bash, no git.
# --publish adds gh pr comment for a supplementary note only.
ALLOWED='Read,Grep,Glob'
if [ "$PUBLISH" -eq 1 ]; then
  ALLOWED="${ALLOWED},Bash(gh pr comment:*)"
fi

SCHEMA='{"type":"object","properties":{"no_tests_changed":{"type":"boolean"},"files_examined":{"type":"array","items":{"type":"string"}},"findings":{"type":"array","items":{"type":"object","properties":{"file":{"type":"string"},"test":{"type":"string"},"category":{"type":"string"},"what_would_have_to_change":{"type":"string"}},"required":["file","test","category","what_would_have_to_change"]}},"summary":{"type":"string"}},"required":["no_tests_changed","files_examined","findings","summary"]}'

PROMPT=$(cat <<EOF
Review the changed TESTS for vacuity. You are running LOCALLY as a preflight
of the checkpoint review (PR ${PR:-none}, ${PR_URL}).

## Spec (read these first, once)

- \`docs/VACUOUS_TESTS.md\` — taxonomy and working rules. Categories and counts
  come from the document; do not invent new letters.
- \`docs/REVIEW_COMMENT_TEMPLATE.md\` — inventory structure (for your summary).

## Scope

IN SCOPE: \`tests/*.py\` only. Dot-directory tooling is out of scope.

CHANGED / OUTSTANDING (mode \`${mode}\`):
${files:-(none listed — if mode is full, examine every in-scope test changed on the PR; if none, re-verify open sticky findings only)}

\`incremental\` — examine EXAMINE + OUTSTANDING paths. \`none\` — nothing new;
stop after confirming sticky open findings. \`full\` — review everything in
scope on this PR.

Read files from the working tree with Read/Grep/Glob only. Do NOT run the
test suite. Do NOT spawn subagents. Do NOT use Bash except \`gh pr comment\`
when publishing is enabled.

## Output

1. Structured JSON (schema enforced) with every test file you actually read
   in \`files_examined\`, and each finding's taxonomy letter.
2. A short markdown inventory matching REVIEW_COMMENT_TEMPLATE.md headings
   (Open findings / Examined / Not examined). Put paths you could not finish
   under **Not examined** so the next run carries them forward.
3. If publishing: one short \`gh pr comment\` supplementary note only — do not
   fight the Actions sticky.

Findings advise; they do not gate.
EOF
)

echo "[local-review] invoking claude --model $MODEL --effort $EFFORT --max-turns $MAX_TURNS" >&2

# --bare: skip project hooks/CLAUDE.md inflation; this review has its own spec.
set +e
claude -p \
  --bare \
  --model "$MODEL" \
  --effort "$EFFORT" \
  --max-turns "$MAX_TURNS" \
  --allowed-tools "$ALLOWED" \
  --json-schema "$SCHEMA" \
  --output-format json \
  "$PROMPT"
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  die "claude exited $rc (are you logged in? try: claude /login)"
fi
