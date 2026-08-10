#!/bin/bash
# Which in-scope test files changed since the last review ran, which LEFT the
# reviewed surface entirely, and which the prior sticky comment still lists as
# unexamined.
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
# TWO ANSWERS, NOT ONE. A path can change in two ways that a single list
# cannot tell apart:
#
#   EXAMINE   in scope, and still there at `after` -- read it.
#   DEPARTED  in scope at `before`, GONE at `after` -- removed, or renamed
#             out of `tests/`. There is nothing to read, and saying nothing
#             is wrong.
#
# Collapsing those is what this script did until 2026-08-08. `.files[]`
# carries `filename` = the path AT `after`; for a rename that is the
# DESTINATION, and `previous_filename` holds the source. Filtering
# `.filename` against an in-scope pattern therefore cannot see a test that
# moved OUT of `tests/` -- which is precisely the move that removes it from
# review. Verified against `7a5dfaf...b1d1018`, where
# `tests/test_archive_fidelity.py` -> `.claude/.../test_transit_integrity.py`
# came back `status: renamed` and this script answered "no in-scope test
# files changed" for a push that took eight tests off the reviewed surface.
#
# NO STATUS LIST ANYWHERE, and that is deliberate. Enumerating
# `renamed|removed|copied|...` by hand is this project's most-repeated bug
# (principle 21) reintroduced one layer down while fixing a field-semantics
# one. `copied` is the case a status list gets wrong: it carries
# `previous_filename` like a rename, but the source still exists, so it is
# not a departure. Instead: take EVERY path the compare mentions, from both
# fields, and split it on whether it exists in the tree at `after`. That
# derivation is correct for every status, including ones GitHub adds later.
#
# OUTSTANDING CARRY-FORWARD. Measured on PR #29 run 31394098469: a $5.04
# Sonnet pass examined 6 of 12 changed test files, posted an "in progress"
# sticky listing the other six, and exited success. The next synchronize
# only feeds this script the push delta -- so those six evaporate unless
# something re-queues them. The sticky comment is that something: paths the
# prior run admitted it did not finish are merged into EXAMINE whenever they
# still exist at `after`. No workflow edit (byte-parity trap); the PR number
# comes from GITHUB_EVENT_PATH or REVIEW_PR.
#
# FAIL-OPEN, LOUDLY. A force-push makes `before` unreachable and the compare
# 404s; a first run has no previous review at all; the tree API truncates on
# a very large repository. All fall back to reviewing everything and SAY so,
# because a silent fallback to "review nothing" is the failure this whole
# workflow exists to avoid. A missing sticky, by contrast, is ordinary --
# first run, or a comment the parser cannot read -- and is not fail-open:
# there is simply nothing to carry forward.
#
# Usage: review_delta.sh <before-sha> <after-sha> <owner/repo>
#   Writes `files` and `mode` to $GITHUB_OUTPUT when set, else to stdout.
#
#   mode=full         could not compute; review everything in scope
#   mode=none         nothing in scope changed, nothing departed, nothing
#                     outstanding on the sticky
#   mode=incremental  something to act on; `files` says what and how
#
# Env:
#   REVIEW_PR           pull request number (local runs; Actions uses the event)
#   GITHUB_EVENT_PATH   Actions event JSON; .pull_request.number is read when set
#   REVIEW_STICKY_FILE  path to a sticky-comment body (tests); skips the API
#
# The mode vocabulary is deliberately unchanged. A fourth value would have
# to be explained in the workflow prompt, and the prompt has to stay
# byte-identical on two branches or the action skips and reports success --
# so a new value costs a hand-sync. `files` is free text injected into the
# prompt, so the outstanding section carries its own instructions instead.

set -u

BEFORE="${1:-}"
AFTER="${2:-}"
REPO="${3:-${GITHUB_REPOSITORY:-}}"
OUT="${GITHUB_OUTPUT:-/dev/stdout}"

# `tests/*.py` only -- the same narrowing the prompt declares. Dot-directory
# shell tooling is out of scope and is not counted anywhere.
IN_SCOPE='^tests/[^/]*\.py$'

emit() {  # emit <mode> [body...]
  local mode="$1"; shift
  {
    echo "mode=$mode"
    echo "files<<REVIEW_DELTA_EOF"
    if [ "$#" -gt 0 ]; then
      printf '%s\n' "$@" | grep . || true
    fi
    echo "REVIEW_DELTA_EOF"
  } >> "$OUT"
}

fail_open() {  # fail_open <reason>
  emit full
  echo "[delta] $1; reviewing everything in scope" >&2
  exit 0
}

# The zero SHA is what GitHub sends for a branch that did not previously
# exist. Treat it as "no previous state" rather than as a commit.
ZERO="0000000000000000000000000000000000000000"

if [ -z "$BEFORE" ] || [ "$BEFORE" = "$ZERO" ] || [ -z "$AFTER" ]; then
  fail_open "no usable previous commit"
fi

command -v gh >/dev/null 2>&1 || fail_open "gh unavailable"
command -v jq >/dev/null 2>&1 || fail_open "jq unavailable"

# Raw JSON, filtered locally. Pushing the filter into `gh --jq` would put the
# part that was WRONG outside the reach of any test -- the fixtures in
# tests/fixtures/ replay these two responses precisely so the classification
# below is the thing under test.
if ! compare_json=$(gh api "repos/${REPO}/compare/${BEFORE}...${AFTER}" 2>/dev/null); then
  # Most often a force-push: `before` is no longer reachable.
  fail_open "compare ${BEFORE}...${AFTER} failed (force-push?)"
fi

# What actually exists at `after`. This is the authority for "did it depart",
# and one call answers it for every candidate at once.
if ! tree_json=$(gh api "repos/${REPO}/git/trees/${AFTER}?recursive=1" 2>/dev/null); then
  fail_open "tree at ${AFTER} unavailable"
fi

if [ "$(printf '%s' "$tree_json" | jq -r '.truncated // false')" = "true" ]; then
  # A truncated tree cannot answer "does this path exist" -- absence would be
  # indistinguishable from being cut off, and that reads as a false departure.
  fail_open "tree at ${AFTER} is truncated"
fi

classify() {  # classify <section>
  printf '%s' "$compare_json" | jq -r \
    --arg re "$IN_SCOPE" \
    --arg section "$1" \
    --argjson tree "$(printf '%s' "$tree_json" | jq '[.tree[] | select(.type == "blob") | .path]')" '
    # Every path this push mentions, from BOTH fields. `filename` is the path
    # at `after`; `previous_filename` is where a moved path came from. Taking
    # the union means no status is special-cased.
    ([ .files[] | .filename, (.previous_filename // empty) ]
       | unique
       | map(select(test($re)))) as $candidates

    # Renames, so a departure can say where it went rather than just vanish.
    | ([ .files[] | select(.previous_filename != null)
                  | {from: .previous_filename, to: .filename} ]) as $moves

    | if $section == "examine" then
        [ $candidates[] | select(. as $p | $tree | index($p) != null) ] | .[]
      else
        [ $candidates[] | select(. as $p | $tree | index($p) == null)
          | . as $p
          | ($moves | map(select(.from == $p)) | first) as $m
          | if $m then "\($p) -> \($m.to)" else "\($p) (gone)" end
        ] | .[]
      end
  '
}

if ! examine=$(classify examine); then
  fail_open "jq failed classifying the examine set"
fi
if ! departed=$(classify departed); then
  fail_open "jq failed classifying the departed set"
fi

n_examine=$(printf '%s' "$examine" | grep -c . || true)
n_departed=$(printf '%s' "$departed" | grep -c . || true)

# --------------------------------------------------------------------------
# Outstanding paths still listed on the sticky comment
# --------------------------------------------------------------------------
#
# The prior review's sticky is the only durable record of "we did not finish".
# Push-local compare cannot see that. Reading it here -- not in the prompt --
# means a model that ignores "re-read open findings" still gets the paths in
# its list, and a docs-only push still resumes an unfinished pass.

pr_number="${REVIEW_PR:-}"
if [ -z "$pr_number" ] && [ -n "${GITHUB_EVENT_PATH:-}" ] && [ -r "${GITHUB_EVENT_PATH:-}" ]; then
  pr_number=$(jq -r '.pull_request.number // empty' "$GITHUB_EVENT_PATH" 2>/dev/null || true)
fi

sticky_body=""
if [ -n "${REVIEW_STICKY_FILE:-}" ] && [ -r "${REVIEW_STICKY_FILE}" ]; then
  sticky_body=$(cat "${REVIEW_STICKY_FILE}")
elif [ -n "$pr_number" ] && [ -n "$REPO" ]; then
  # Prefer the comment the action maintains (user login claude[bot] or body
  # marker). Take the newest match. Failure is quiet: no sticky means nothing
  # to carry, not "review everything".
  sticky_body=$(gh api "repos/${REPO}/issues/${pr_number}/comments" --paginate \
    --jq '[.[] | select(.user.login == "claude[bot]" or (.body | test("Vacuous-test review"; "i"))) | .body] | last // empty' \
    2>/dev/null || true)
fi

outstanding=""
if [ -n "$sticky_body" ]; then
  # Extract in-scope test paths the prior run has not cleared:
  #   1. ### Not examined ... (template section; primary)
  #   2. unchecked markdown tasks: - [ ] ... `path` ...
  # Bare `test_*.py` mentions on those lines become tests/test_*.py.
  # Never re-queue paths on checked (- [x]) lines.
  outstanding=$(printf '%s\n' "$sticky_body" | jq -nrR --arg re "$IN_SCOPE" '
    def norm:
      if test("^tests/") then .
      elif test("^test_.*\\.py$") then "tests/\(.)"
      else empty end;
    def paths_in:
      [match("`([^`]+)`"; "g") | .captures[0].string | norm | select(test($re))];
    def until_next_heading:
      if length == 0 then []
      elif (.[0] | test("^### ")) then []
      else [.[0]] + (.[1:] | until_next_heading)
      end;

    [inputs] as $lines
    | ($lines | to_entries
        | map(select(.value | test("^### Not examined")) | .key)
        | first) as $not_i
    | (
        if $not_i == null then []
        else
          ($lines[($not_i+1):] | until_next_heading
            | map(paths_in)
            | add // [])
        end
      ) as $from_section
    | (
        $lines
        | map(select(test("^- \\[ \\]")) | paths_in)
        | add // []
      ) as $from_unchecked
    | (
        $lines
        | map(select(test("^- \\[[xX]\\]")) | paths_in)
        | add // []
      ) as $from_checked
    | ((($from_section + $from_unchecked) | unique) - ($from_checked | unique))
    | .[]
  ' 2>/dev/null || true)
fi

# Only paths that still exist at `after` can be examined. A sticky can lag a
# deletion by one run; the departed section already covers real removals.
if [ -n "$outstanding" ]; then
  tree_paths=$(printf '%s' "$tree_json" | jq -r '[.tree[] | select(.type == "blob") | .path] | .[]')
  outstanding=$(comm -12 \
    <(printf '%s\n' "$outstanding" | grep . | sort -u) \
    <(printf '%s\n' "$tree_paths" | grep . | sort -u) || true)
fi

# Drop anything already in this push's examine set -- no duplicate rows.
if [ -n "$outstanding" ] && [ -n "$examine" ]; then
  outstanding=$(comm -23 \
    <(printf '%s\n' "$outstanding" | grep . | sort -u) \
    <(printf '%s\n' "$examine" | grep . | sort -u) || true)
fi

n_outstanding=$(printf '%s' "$outstanding" | grep -c . || true)

if [ "$n_examine" -eq 0 ] && [ "$n_departed" -eq 0 ] && [ "${n_outstanding:-0}" -eq 0 ]; then
  # A real, well-formed answer: this push touched no in-scope test file, took
  # none away, and the sticky carries no unfinished work. Distinct from the
  # fail-open cases above, and the prompt treats it differently -- nothing new
  # to read, only open findings to re-verify.
  emit none
  echo "[delta] no in-scope test files changed or departed in this push" >&2
  exit 0
fi

body=""
if [ "$n_examine" -gt 0 ]; then
  body="EXAMINE -- in-scope test files changed in this push, present in the working tree:
${examine}"
fi

if [ "${n_outstanding:-0}" -gt 0 ]; then
  [ -n "$body" ] && body="${body}
"
  body="${body}OUTSTANDING -- in-scope tests the prior sticky comment still lists as
unexamined (or unchecked). Carry-forward from a partial review: READ THESE
even if this push did not touch them. Clear a path from the sticky only by
examining it; do not drop it because the push delta is empty:
${outstanding}"
fi

if [ "$n_departed" -gt 0 ]; then
  [ -n "$body" ] && body="${body}
"
  # This block is the whole reason the departed set is reported separately.
  # These paths do not exist at `after`, so an instruction to examine them
  # produces an ENOENT and a confused reviewer.
  body="${body}DEPARTED -- in-scope tests that LEFT the reviewed surface in this push,
by deletion or by being moved out of \`tests/\`. THESE FILES NO LONGER EXIST:
do not try to read them. Report that they left scope and where they went, so
their removal is visible in the review rather than silent:
${departed}"
fi

emit incremental "$body"
echo "[delta] ${n_examine} changed, ${n_outstanding:-0} outstanding, ${n_departed} departed, since ${BEFORE:0:7}" >&2
