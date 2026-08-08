#!/bin/bash
# Publish the vacuous-test review, and FAIL if it produced no result.
#
# Reads the action's `structured_output` -- a schema-validated JSON object,
# produced by the action rather than by the model's goodwill. That
# distinction is the whole reason this script was rewritten: the first
# version asked the prompt to write a markdown file, which produces nothing
# at all if the model simply does not comply.
#
# What it gates on, precisely:
#
#   ABSENCE fails.   No structured output, unparseable output, or a report
#                    claiming neither "no tests changed" nor any file
#                    examined, means the review did not happen. That is a
#                    fact about the run, not a judgement about the code.
#   FINDINGS do not. However many vacuous tests it reports, the build stays
#                    green. An LLM verdict that can turn a build red is the
#                    failure this repository has spent the most effort
#                    avoiding: deterministic checks gate, reviews advise.
#
# The failure it exists for was measured twice on 2026-08-08. A skipped
# action reported job success; then a successful action reported success
# having published nothing. Green and reviewed were indistinguishable from
# outside both times.
#
# Usage: publish_review.sh '<structured_output_json>' [summary-file]

set -u

RAW="${1:-}"
SUMMARY="${2:-${GITHUB_STEP_SUMMARY:-/dev/stdout}}"

fail() {
  # Careful with the claim. This step observes ONE thing: whether what it
  # was handed is a usable result. It cannot see whether the review ran, so
  # it does not say so.
  #
  # The distinction was earned. On its first firing this step printed "the
  # review produced no result" directly beneath the review's actual result,
  # rendered on the same page -- because the workflow had passed it a file
  # PATH instead of the JSON. The guard was right that its input was
  # unusable and wrong about what that implied, which is precisely the
  # overclaim it exists to prevent elsewhere.
  {
    echo "## This step could not read a review result"
    echo
    echo "$1"
    echo
    echo "**What this does and does not tell you.** It means the value handed"
    echo "to this step was not a usable result. It does NOT establish that"
    echo "the review failed to run -- check the rendered report above and the"
    echo "uploaded execution artifact before concluding anything. A wiring"
    echo "mistake here looks identical to a missing review, and on"
    echo "2026-08-08 it was a wiring mistake: \`execution_file\` (a path) was"
    echo "passed where \`structured_output\` (the JSON) was expected."
    echo
    echo "It fails rather than passing quietly because the opposite error is"
    echo "worse: a skipped action reports job success, and so does a"
    echo "successful one that published nothing. Both were observed the same"
    echo "day, and a green badge distinguished neither."
  } >> "$SUMMARY"
  exit 1
}

if [ -z "${RAW// }" ]; then
  fail "The action returned no \`structured_output\` at all."
fi

if ! command -v jq >/dev/null 2>&1; then
  # Do not silently accept unverified output because a tool is missing.
  fail "\`jq\` is unavailable, so the review's output could not be verified."
fi

if ! printf '%s' "$RAW" | jq empty >/dev/null 2>&1; then
  fail "The \`structured_output\` was not valid JSON."
fi

no_tests=$(printf '%s' "$RAW" | jq -r '.no_tests_changed // false')
n_examined=$(printf '%s' "$RAW" | jq -r '(.files_examined // []) | length')
n_findings=$(printf '%s' "$RAW" | jq -r '(.findings // []) | length')
summary=$(printf '%s' "$RAW" | jq -r '.summary // ""')

if [ "$no_tests" != "true" ] && [ "$n_examined" -eq 0 ]; then
  fail "The review examined 0 test files and did not report that the diff
contained none. Those are different claims, and neither was made."
fi

{
  echo "## Vacuous-test review"
  echo
  if [ "$no_tests" = "true" ]; then
    echo "No test files changed; nothing to review."
  else
    echo "Examined **$n_examined** test file(s). Findings: **$n_findings**."
    echo
    echo "<details><summary>Files examined</summary>"
    echo
    printf '%s' "$RAW" | jq -r '(.files_examined // [])[] | "- `" + . + "`"'
    echo
    echo "</details>"
  fi
  if [ -n "$summary" ]; then
    echo
    echo "$summary"
  fi
  if [ "$n_findings" -gt 0 ]; then
    echo
    echo "### Findings"
    echo
    printf '%s' "$RAW" | jq -r '(.findings // [])[] |
      "- **" + .file + "::" + .test + "** [" + .category + "] — " +
      .what_would_have_to_change'
    echo
    echo "_Advisory. These do not fail the build; deterministic checks gate._"
  fi
} >> "$SUMMARY"
exit 0
