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

# Count TEST files, not entries. The field is documented as "every test file
# actually read", and a review legitimately reads other things -- the
# workflow, this script, and docs/VACUOUS_TESTS.md, which the prompt orders
# it to read FIRST. Those land in the same array.
#
# Measured on run 31259604880: an invalid ref was dispatched, the review
# correctly reported that the range did not exist and examined no tests, and
# `files_examined` came back as exactly `["docs/VACUOUS_TESTS.md"]`. One
# entry, so the old count-based guard passed, and a review that examined
# nothing reported job SUCCESS in 59 seconds. The guard was reading a field
# whose semantics it had assumed -- entries, where the claim was test files.
n_test_files=$(printf '%s' "$RAW" | jq -r '
  [ (.files_examined // [])[] | select(test("(^|/)tests?/")) ] | length')

if [ "$no_tests" != "true" ] && [ "${n_test_files:-0}" -eq 0 ]; then
  fail "The review examined 0 TEST files and did not report that the diff
contained none. Those are different claims, and neither was made. It listed
$n_examined path(s), none under a test directory -- which is what a review
that could not reach the diff at all looks like. Check that the dispatched
ref exists and is non-empty before concluding anything about the code."
fi

# --- coverage: how much of the diff did it actually look at? ---------------
#
# "Examined N files" and "reviewed the diff" are different claims, and the
# gap between them is invisible from the report. Measured on run
# 31259263566: a pull_request review examined 10 of 32 changed test files
# and summarised as "no vacuous tests found in this diff" -- describing a
# subset as the whole.
#
# The cause is structural rather than a lazy model, which is why it needs a
# mechanism. `gh pr diff` returns HTTP 406 above 20,000 lines and that is an
# allowlisted tool the review reaches for; on a large PR it simply cannot
# retrieve the diff and proceeds on what it could reach. Nothing in the
# report distinguishes that from a thorough pass.
#
# The changed-file list comes from GitHub, via `gh pr diff --name-only`, not
# from git. The context is already in GitHub: a PR has a base, a head and a
# file list, and reaching for git plumbing inside a GitHub Action to
# reconstruct what the platform already knows is how the range-based design
# this replaced went wrong three separate times.
#
# It REPORTS and does not fail. Partial coverage on a large diff is often
# legitimate, and failing it would train route-around -- the one outcome
# worse than not measuring. Absence still fails, above; this quantifies.
coverage_note=""
if [ "$no_tests" != "true" ] && [ -n "${PR_NUMBER:-}" ] \
   && command -v gh >/dev/null 2>&1; then
  if changed_all=$(gh pr diff "$PR_NUMBER" --name-only 2>/dev/null); then
    changed=$(printf '%s\n' "$changed_all" | grep -E '(^|/)tests?/' | sort -u)
    n_changed=$(printf '%s' "$changed" | grep -c . || true)
    # A zero here means the computation failed or the path filter missed,
    # NOT that the diff is clean -- `no_tests_changed` is the claim for
    # that, and it is false in this branch. Say so rather than reporting
    # flawless coverage of nothing, which is this file's own subject.
    if [ "${n_changed:-0}" -gt 0 ]; then
      examined=$(printf '%s' "$RAW" | jq -r '(.files_examined // [])[]' \
                 | sort -u)
      missed=$(comm -23 <(printf '%s\n' "$changed") \
                       <(printf '%s\n' "$examined") 2>/dev/null | grep . || true)
      n_missed=$(printf '%s' "$missed" | grep -c . || true)
      if [ "${n_missed:-0}" -gt 0 ]; then
        # Count the CHANGED test files that were examined, not every entry
        # in files_examined -- the review legitimately reads the catalogue
        # and the workflow too, and counting those here would overstate
        # coverage in the one place that exists to measure it honestly.
        n_covered=$((n_changed - n_missed))
        coverage_note="PARTIAL: $n_covered of $n_changed changed test files examined; ${n_missed} not looked at."
      else
        coverage_note="Complete: every one of the $n_changed changed test files was examined."
      fi
    else
      coverage_note="Coverage could not be computed (no changed test paths resolved), so the scope of this review is unverified."
    fi
  fi
fi

{
  echo "## Vacuous-test review"
  echo
  if [ "$no_tests" = "true" ]; then
    echo "No test files changed; nothing to review."
  else
    echo "Examined **$n_examined** test file(s). Findings: **$n_findings**."
    if [ -n "$coverage_note" ]; then
      echo
      echo "**Coverage — $coverage_note**"
      if [ "${n_missed:-0}" -gt 0 ]; then
        echo
        echo "A clean result covers what was examined, not what changed."
        echo "\`gh pr diff\` returns HTTP 406 above 20,000 lines, so on a large"
        echo "diff the review proceeds on what it could reach."
        echo
        echo "<details><summary>Changed but not examined</summary>"
        echo
        printf '%s\n' "$missed" | sed 's/^/- `/; s/$/`/'
        echo
        echo "</details>"
      fi
    fi
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
