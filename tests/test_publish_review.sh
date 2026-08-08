#!/bin/bash
# Break-tests for tools/ci/publish_review.sh.
#
# This step is the only thing standing between a green run and a green run
# mistaken for a review. It has failed silently twice in one morning -- once
# because a skipped action reports job success, once because a successful
# action published nothing -- so its behaviour is tested rather than trusted.
#
# The two properties that matter pull in opposite directions and are both
# asserted: ABSENCE must fail the build, and FINDINGS must not. A script that
# failed on findings would make an LLM verdict gate a build; one that passed
# on absence is the bug being guarded.
#
# Shell rather than pytest, matching the c2c-mail hooks: the thing under test
# is a shell script, and a Python wrapper would put a layer between the test
# and the mechanism. tests/test_publish_review_wrapper.py forwards this into
# the pytest suite so it is not a test nobody runs.
#
# Usage: bash tests/test_publish_review.sh

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/../tools/ci/publish_review.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fails=0
n=0
run() {  # run <json> -> sets RC and OUT
  n=$((n + 1))
  OUT="$TMP/summary_$n"
  : > "$OUT"
  bash "$SCRIPT" "$1" "$OUT"
  RC=$?
}
check() {
  if grep -qF -- "$2" "$OUT"; then echo "  ok    $1"
  else
    echo "  FAIL  $1 -- expected: $2"
    echo "        got: $(head -c 400 "$OUT")"
    fails=$((fails + 1))
  fi
}
expect_rc() {
  if [ "$RC" -eq "$2" ]; then echo "  ok    $1 (exit $RC)"
  else echo "  FAIL  $1 -- expected exit $2, got $RC"; fails=$((fails + 1)); fi
}

echo "publish_review.sh"

if ! command -v jq >/dev/null 2>&1; then
  echo "  SKIP  jq unavailable -- these checks require it"
  exit 0
fi

REAL='{"no_tests_changed":false,"files_examined":["tests/test_a.py"],
       "findings":[{"file":"tests/test_a.py","test":"test_x","category":"A",
       "what_would_have_to_change":"nothing"}],"summary":"One vacuous test."}'
CLEAN='{"no_tests_changed":false,"files_examined":["tests/test_a.py"],
        "findings":[],"summary":"All discriminating."}'
NOTESTS='{"no_tests_changed":true,"files_examined":[],"findings":[],
          "summary":"No tests changed."}'
EMPTY_CLAIM='{"no_tests_changed":false,"files_examined":[],"findings":[],
              "summary":"Looks fine."}'

# --- absence fails ---------------------------------------------------------

run ""
expect_rc "no structured output fails" 1
check "says the action returned nothing" "no \`structured_output\` at all"
check "warns green would have lied" "would NOT have meant the review"

run "not json at all {{{"
expect_rc "unparseable output fails" 1
check "names the parse failure" "not valid JSON"

# THE case: a report claiming nothing examined and not claiming the diff had
# no tests. Those are different assertions and this makes neither.
run "$EMPTY_CLAIM"
expect_rc "examined-nothing fails" 1
check "distinguishes the two claims" "examined 0 test files"

# --- findings do NOT fail --------------------------------------------------

run "$REAL"
expect_rc "findings do not fail the build" 0
check "reports the finding" "test_x"
check "names the taxonomy category" "[A]"
check "marks findings advisory" "do not fail the build"
check "states what it examined" "tests/test_a.py"

# --- non-vacuity: a clean review is not reported as broken -----------------

run "$CLEAN"
expect_rc "a clean review passes" 0
check "reports zero findings" "Findings: **0**"
if grep -qF "produced no result" "$OUT"; then
  echo "  FAIL  a clean review was reported as no result"
  fails=$((fails + 1))
else
  echo "  ok    a clean review is not reported as no result"
fi

run "$NOTESTS"
expect_rc "no-tests-changed passes" 0
check "says nothing to review" "nothing to review"

# --- the real payload, captured from the first successful run --------------
#
# Verbatim `structured_output` from run 31255640813, downloaded from that
# run's artifact. A fixture I invented cannot show that the schema and the
# reader agree about what the ACTION actually emits -- only a captured one
# can, and the wiring bug that run exposed was exactly a disagreement about
# which output carried what.

# Read from a file, not inlined: the captured summary contains both quote
# characters, and hand-quoting it into shell corrupted it on the first
# attempt -- the fixture then failed for a reason that had nothing to do
# with the script under test, which is its own small lesson about fixtures
# that are transcribed rather than stored.
FIXTURE="$HERE/fixtures/review_structured_output_31255640813.json"
if [ ! -f "$FIXTURE" ]; then
  echo "  FAIL  captured fixture missing: $FIXTURE"
  fails=$((fails + 1))
fi
REAL_RUN="$(cat "$FIXTURE")"

run "$REAL_RUN"
expect_rc "the real captured payload passes" 0
check "reports nothing to review"     "nothing to review" "$OUT"
check "carries the summary through"   "5b8bac6" "$OUT"
if grep -qF "produced no result" "$OUT"; then
  echo "  FAIL  a real successful review was reported as no result"
  fails=$((fails + 1))
else
  echo "  ok    a real successful review is not reported as no result"
fi

# --- writes to GITHUB_STEP_SUMMARY when no file argument is given ----------

OUT="$TMP/summary_env"
: > "$OUT"
GITHUB_STEP_SUMMARY="$OUT" bash "$SCRIPT" "$REAL"
check "honours GITHUB_STEP_SUMMARY" "tests/test_a.py"

echo
if [ "$fails" -eq 0 ]; then echo "all checks passed"; exit 0; fi
echo "$fails check(s) FAILED"
exit 1
