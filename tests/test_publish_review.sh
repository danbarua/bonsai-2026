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
check "points at the PR rather than the code" "PR is
non-empty"

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
# The changed-file list comes from `gh pr diff --name-only`, so `gh` is
# stubbed on PATH here. Worth being explicit about what that does and does
# not establish: it tests the SET COMPARISON -- which changed test files went
# unexamined -- and says nothing about whether the gh invocation is right.
# The earlier git-based version was tested against a real repository for the
# opposite reason: there the substance WAS ref resolution. The substance
# moved, so the fixture moved with it.

echo
echo "publish_review.sh -- coverage"

STUB="$TMP/bin"
mkdir -p "$STUB"
cat > "$STUB/gh" <<'STUBEOF'
#!/bin/bash
# Stands in for `gh pr diff <n> --name-only`. Emits a fixed changed-file
# list, including a non-test path, because the real thing does too and the
# filter has to drop it.
printf 'tests/test_a.py\ntests/test_b.py\nsrc/thing.py\n'
STUBEOF
chmod +x "$STUB/gh"

run_cov() {  # run_cov <json> -> OUT/RC, with the gh stub first on PATH
  n=$((n + 1))
  OUT="$TMP/cov_$n"
  : > "$OUT"
  PATH="$STUB:$PATH" PR_NUMBER=99 bash "$SCRIPT" "$1" "$OUT"
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
# src/thing.py changed too, and is not a test. Counting it would inflate the
# denominator and understate coverage -- the mirror of the bug that let a
# catalogue entry count as a test file.
if grep -qF "of 3 changed test files" "$OUT"; then
  echo "  FAIL  a non-test path was counted as a changed test file"
  fails=$((fails + 1))
else
  echo "  ok    non-test changed paths are not counted"
fi

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

# --- the contradiction: an honest "no tests changed" over a large PR -------
#
# The most dangerous case, and the one no amount of care inside the review
# can catch. In tag mode the action injects the changed-file list from an
# unpaginated `files(first: 100)` query, and GitHub returns NULL for that
# list when a PR's diff is too large -- which the action renders as "No files
# changed" (src/github/data/fetcher.ts:427). From inside the model that is
# indistinguishable from a PR that touched nothing, so `no_tests_changed:
# true` is an HONEST report of what it was shown, and every check that trusts
# the review's own account of its scope agrees with it.
#
# Only a second, independent source breaks the tie. The stub names two
# changed test files while the payload claims none.

NO_TESTS_CLAIM='{"no_tests_changed":true,"files_examined":[],"findings":[],
  "summary":"No test files changed in this PR."}'
run_cov "$NO_TESTS_CLAIM"
expect_rc "no-tests-changed contradicted by GitHub fails" 1
check "gives the count GitHub reported"  "GitHub lists"
check "names the real cause"             "too large"
check "forbids reading it as clean"      "Do not read this as a clean result"

# Non-vacuity: when GitHub agrees there are no test files, the same claim
# must PASS. Otherwise the guard is "always fail on no_tests_changed", which
# would make the honest case unreportable.
STUB_EMPTY="$TMP/bin_empty"
mkdir -p "$STUB_EMPTY"
cat > "$STUB_EMPTY/gh" <<'STUBEOF'
#!/bin/bash
printf 'src/thing.py\nREADME.md\n'
STUBEOF
chmod +x "$STUB_EMPTY/gh"
n=$((n + 1)); OUT="$TMP/cov_$n"; : > "$OUT"
PATH="$STUB_EMPTY:$PATH" PR_NUMBER=99 bash "$SCRIPT" "$NO_TESTS_CLAIM" "$OUT"
RC=$?
expect_rc "no-tests-changed CONFIRMED by GitHub passes" 0
check "says there was nothing to review" "nothing to review"

# With no PR number there is nothing to ask GitHub about, so no coverage
# claim is made. Silence is correct here; a fabricated "complete" would not
# be, and that is the direction this whole file guards.
run "$CLEAN"
if grep -qE "PARTIAL|Complete: every one" "$OUT"; then
  echo "  FAIL  coverage was claimed with no PR to compute it from"
  fails=$((fails + 1))
else
  echo "  ok    no coverage claim without a PR number"
fi

echo
if [ "$fails" -eq 0 ]; then echo "all checks passed"; exit 0; fi
echo "$fails check(s) FAILED"
exit 1
