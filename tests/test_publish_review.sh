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
check "says what it could not read" "could not read a review result"
check "does not overclaim that the review failed" "does NOT establish that"

run "not json at all {{{"
expect_rc "unparseable output fails" 1
check "names the parse failure" "not valid JSON"

# THE case: a report claiming nothing examined and not claiming the diff had
# no tests. Those are different assertions and this makes neither.
run "$EMPTY_CLAIM"
expect_rc "examined-nothing fails" 1
check "distinguishes the two claims" "examined 0 TEST files"

# The invalid-ref case, verbatim from run 31259604880. The review behaved
# correctly -- it reported that origin/staging does not exist and refused to
# substitute a range -- but `files_examined` carried the CATALOGUE it had
# been told to read first. One entry, non-zero, so a count-based guard
# passed and a review that examined no tests went green in 59 seconds.
CATALOGUE_ONLY='{"no_tests_changed":false,
  "files_examined":["docs/VACUOUS_TESTS.md"],"findings":[],
  "summary":"Cannot review: the requested range does not exist."}'
run "$CATALOGUE_ONLY"
expect_rc "a non-test path does not count as a test file" 1
check "says none were under a test directory" "none under a test directory"
check "points at the ref rather than the code" "ref exists"

# Non-vacuity for the filter: it must not have become "fail unless the array
# is all test files". A review that reads the workflow AND a test is normal.
MIXED='{"no_tests_changed":false,
  "files_examined":["docs/VACUOUS_TESTS.md",".github/workflows/x.yml",
                    "tests/test_a.py"],"findings":[],"summary":"Fine."}'
run "$MIXED"
expect_rc "context files alongside a real test still pass" 0

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
if grep -qF "could not read a review result" "$OUT"; then
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
if grep -qF "could not read a review result" "$OUT"; then
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

# --- coverage: examined vs changed -----------------------------------------
#
# The gap this measures was real and invisible. On run 31259263566 a
# pull_request review examined 10 of 32 changed test files and summarised as
# "no vacuous tests found in this diff" -- a subset described as the whole.
# Structural, not lazy: `gh pr diff` returns HTTP 406 above 20,000 lines.
#
# Exercised against a REAL git repository rather than a stubbed `git`,
# because the check's whole substance is which refs and paths git resolves,
# and a stub would assert that the stub works.

echo
echo "publish_review.sh -- coverage"

REPO="$TMP/repo"
mkdir -p "$REPO/tests"
(
  cd "$REPO" || exit 1
  git init -q -b main .
  git config user.email t@t; git config user.name t
  echo x > README; git add -A; git commit -qm base
  # a fake "origin/main" so GITHUB_BASE_REF resolves the way CI's would
  git update-ref refs/remotes/origin/main HEAD
  echo a > tests/test_a.py; echo b > tests/test_b.py
  git add -A; git commit -qm "two test files"
) || { echo "  FAIL  could not build the fixture repo"; fails=$((fails + 1)); }

run_cov() {  # run_cov <json> -> OUT/RC, executed inside the fixture repo
  n=$((n + 1))
  OUT="$TMP/cov_$n"
  : > "$OUT"
  ( cd "$REPO" && GITHUB_BASE_REF=main bash "$SCRIPT" "$1" "$OUT" )
  RC=$?
}

PARTIAL='{"no_tests_changed":false,"files_examined":["tests/test_a.py"],
          "findings":[],"summary":"Nothing vacuous in this diff."}'
FULL='{"no_tests_changed":false,
       "files_examined":["tests/test_a.py","tests/test_b.py"],
       "findings":[],"summary":"All good."}'

run_cov "$PARTIAL"
expect_rc "partial coverage does NOT fail the build" 0
check "reports the shortfall"        "PARTIAL: 1 of 2"
check "names what was not examined"  "tests/test_b.py"
check "warns a clean result is scoped" "covers what was examined"

# The guard seen passing, not only failing -- otherwise it could be a
# routine that always cries partial.
run_cov "$FULL"
expect_rc "complete coverage passes" 0
check "reports completeness" "Complete: every one of the 2"
if grep -qF "PARTIAL" "$OUT"; then
  echo "  FAIL  a fully-covered review was reported as partial"
  fails=$((fails + 1))
else
  echo "  ok    a fully-covered review is not reported as partial"
fi

# Outside a pull request there is no base ref, so no coverage claim is made.
# Silence is correct here; a fabricated "complete" would not be.
run "$CLEAN"
if grep -qE "PARTIAL|Complete: every one" "$OUT"; then
  echo "  FAIL  coverage was claimed with no base ref to compute it from"
  fails=$((fails + 1))
else
  echo "  ok    no coverage claim without a base ref"
fi

echo
if [ "$fails" -eq 0 ]; then echo "all checks passed"; exit 0; fi
echo "$fails check(s) FAILED"
exit 1
