#!/bin/bash
# What did the vacuous-test review actually do, and what did it cost?
#
# Everything here is available from `gh`. Reading it off the Actions web page
# and pasting the interesting lines into a chat window is a human relaying
# what an API already serves -- and a paraphrase of a run is a summary
# borrowing authority its source never granted.
#
# The run's own telemetry is the part the web page buries. `claude-code-action`
# uploads its full execution transcript as an artifact, and the last entry is
# a `result` record carrying duration, turn count and cost. That is the
# authoritative number for "what did this review cost", as opposed to an
# estimate reconstructed from token prices.
#
# Usage:
#   review_run.sh                 latest review run
#   review_run.sh <run-id>        a specific run
#   review_run.sh --json          machine-readable, for a script or a test
#
# Env:
#   REVIEW_REPO       owner/repo    (default: gh's current repository)
#   REVIEW_WORKFLOW   display name  (default: Claude Vacuous-Test Review)
#   REVIEW_ARTIFACT   artifact name (default: claude-review-execution)

set -euo pipefail

WORKFLOW="${REVIEW_WORKFLOW:-Claude Vacuous-Test Review}"
ARTIFACT="${REVIEW_ARTIFACT:-claude-review-execution}"

AS_JSON=0
RUN_ID=""
for arg in "$@"; do
  case "$arg" in
    --json) AS_JSON=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) RUN_ID="$arg" ;;
  esac
done

die() { echo "review_run: $*" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || die "gh is not on PATH"
command -v jq >/dev/null 2>&1 || die "jq is not on PATH"

if [ -n "${REVIEW_REPO:-}" ]; then
  REPO="$REVIEW_REPO"
else
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner) \
    || die "not in a GitHub repository and REVIEW_REPO is unset"
fi

# --- locate the run ------------------------------------------------------

if [ -z "$RUN_ID" ]; then
  RUN_ID=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" --limit 1 \
             --json databaseId -q '.[0].databaseId // empty') \
    || die "could not list runs for workflow '$WORKFLOW'"
  [ -n "$RUN_ID" ] || die "no runs found for workflow '$WORKFLOW' in $REPO"
fi

run_json=$(gh run view "$RUN_ID" --repo "$REPO" \
             --json databaseId,displayTitle,headBranch,event,status,conclusion,createdAt,updatedAt,url) \
  || die "run $RUN_ID not found in $REPO"

# --- locate the artifact -------------------------------------------------
#
# Artifacts EXPIRE (90 days by default). An expired artifact is the ordinary
# end state of an old run, not a fault -- but it must be said, because a
# summary printed with every telemetry field empty looks like a run that did
# nothing rather than a record that aged out.

artifacts_json=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}/artifacts" \
                   --jq "[.artifacts[] | select(.name == \"${ARTIFACT}\")]") \
  || die "could not list artifacts for run $RUN_ID"

artifact_json=$(printf '%s' "$artifacts_json" | jq '.[0] // null')

telemetry='null'
artifact_state='present'

if [ "$artifact_json" = "null" ]; then
  artifact_state='missing'
elif [ "$(printf '%s' "$artifact_json" | jq -r '.expired')" = "true" ]; then
  artifact_state='expired'
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  if gh run download "$RUN_ID" --repo "$REPO" --name "$ARTIFACT" --dir "$tmp" >/dev/null 2>&1; then
    exec_file=$(find "$tmp" -name '*.json' -type f | head -1)
    if [ -n "$exec_file" ]; then
      # The transcript is an array of entries; exactly one is the `result`.
      telemetry=$(jq '[.[] | select(.type == "result")] | .[0] // null' "$exec_file")
    else
      artifact_state='unreadable'
    fi
  else
    artifact_state='download-failed'
  fi
fi

summary=$(jq -n \
  --argjson run "$run_json" \
  --argjson artifact "$artifact_json" \
  --argjson telemetry "$telemetry" \
  --arg state "$artifact_state" \
  --arg repo "$REPO" '
  {
    repo: $repo,
    run: {
      id: $run.databaseId, title: $run.displayTitle, branch: $run.headBranch,
      event: $run.event, status: $run.status, conclusion: $run.conclusion,
      created_at: $run.createdAt, url: $run.url
    },
    artifact: (
      if $artifact == null then {state: $state}
      else {
        state: $state, id: $artifact.id, size_bytes: $artifact.size_in_bytes,
        expires_at: $artifact.expires_at, download_url: $artifact.archive_download_url
      } end
    ),
    telemetry: (
      if $telemetry == null then null
      else {
        outcome: $telemetry.subtype, is_error: $telemetry.is_error,
        duration_ms: $telemetry.duration_ms, num_turns: $telemetry.num_turns,
        total_cost_usd: $telemetry.total_cost_usd,
        output_tokens: ($telemetry.usage.output_tokens // null)
      } end
    )
  }')

if [ "$AS_JSON" -eq 1 ]; then
  printf '%s\n' "$summary"
  exit 0
fi

printf '%s' "$summary" | jq -r '
  "run     \(.run.id)  \(.run.conclusion // .run.status)  \(.run.branch)  \(.run.created_at)",
  "title   \(.run.title)",
  "url     \(.run.url)",
  "",
  (if .artifact.state == "present" then
     "artifact  \(.artifact.id)  \(.artifact.size_bytes) bytes  expires \(.artifact.expires_at)",
     "download  \(.artifact.download_url)"
   else
     "artifact  NOT AVAILABLE (\(.artifact.state)) -- telemetry below is empty for that reason, not because the run did nothing"
   end),
  "",
  (if .telemetry == null then
     "telemetry unavailable"
   else
     "outcome   \(.telemetry.outcome)\(if .telemetry.is_error then "  (ERROR)" else "" end)",
     "duration  \(.telemetry.duration_ms / 1000 | . * 10 | round / 10)s",
     "turns     \(.telemetry.num_turns)",
     "cost      $\(.telemetry.total_cost_usd)",
     "out tok   \(.telemetry.output_tokens)"
   end)
'
