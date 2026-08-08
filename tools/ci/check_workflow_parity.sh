#!/bin/bash
# Are the Claude workflow files byte-identical across the branches that need
# them to be?
#
# The action refuses to run unless a workflow file matches the version on the
# default branch -- and it refuses by SKIPPING, which reports job success.
# So drift between `main` and a working branch produces a green run that
# reviewed nothing, twice observed on 2026-08-08. There is no loud failure to
# wait for.
#
# That makes hand-editing the same file on two branches a trap: the copies
# must match to the byte, whitespace included, and nothing tells you when
# they stop matching. This compares git blob hashes -- content identity, not
# a diff of what a human can see -- so an invisible trailing space is caught
# the same as a rewritten prompt.
#
# Usage:
#   bash tools/ci/check_workflow_parity.sh [ref-a] [ref-b]
# Defaults to origin/main and origin/stage2b. Fetches nothing: run
# `git fetch origin` first, or it compares whatever your refs currently say.

set -u

REF_A="${1:-origin/main}"
REF_B="${2:-origin/stage2b}"
WORKFLOWS=(
  ".github/workflows/claude-code-review.yml"
  ".github/workflows/claude.yml"
)
# Files the review READS at runtime rather than files that gate it. A missing
# catalogue does not skip the action; it makes the review worse in a way
# nothing reports, which is worth knowing about separately.
READ_AT_RUNTIME=(
  "docs/VACUOUS_TESTS.md"
  "tools/ci/publish_review.sh"
)

blob() {  # blob <ref> <path> -> hash, or "-" when absent from the ref
  # `git cat-file -e` for existence, deliberately, because `git rev-parse`
  # cannot be used for this. For a path that exists ON DISK but not in the
  # ref, rev-parse prints a fatal to stderr AND echoes its argument to
  # stdout with exit status 0 -- so `rev-parse ... || echo -` captures the
  # string "origin/main:tools/x.sh" as though it were a hash, and two
  # absent-but-different paths then look like two present-and-differing
  # ones. That mislabelled a missing file as a benign difference on this
  # script's first run, which is the whole reason the note is this long: a
  # checker that misreports is worse than no checker, because its output is
  # believed.
  if git cat-file -e "$1:$2" 2>/dev/null; then
    git rev-parse "$1:$2"
  else
    echo "-"
  fi
}

fails=0
echo "workflow parity: $REF_A vs $REF_B"

for path in "${WORKFLOWS[@]}"; do
  a=$(blob "$REF_A" "$path")
  b=$(blob "$REF_B" "$path")
  if [ "$a" = "-" ] && [ "$b" = "-" ]; then
    echo "  --    $path absent on both"
  elif [ "$a" = "$b" ]; then
    echo "  ok    $path identical (${a:0:8})"
  else
    echo "  DRIFT $path"
    echo "        $REF_A: $a"
    echo "        $REF_B: $b"
    echo "        The action will SKIP and report success. A green run will"
    echo "        not mean the review ran."
    fails=$((fails + 1))
  fi
done

for path in "${READ_AT_RUNTIME[@]}"; do
  a=$(blob "$REF_A" "$path")
  b=$(blob "$REF_B" "$path")
  if [ "$a" = "-" ] || [ "$b" = "-" ]; then
    missing_on="$REF_A"; [ "$a" != "-" ] && missing_on="$REF_B"
    echo "  WARN  $path missing on $missing_on -- a review dispatched"
    echo "        against that ref reads it from the checkout and will not"
    echo "        find it. Not a skip; a quietly worse review."
  elif [ "$a" = "$b" ]; then
    echo "  ok    $path identical (${a:0:8})"
  else
    echo "  note  $path differs -- expected while branches diverge, and"
    echo "        harmless: each review reads the copy in its own checkout."
  fi
done

# The working tree, against both refs.
#
# Added because the check reported "no workflow drift" while the branch that
# was about to become main carried a different workflow. Both refs agreed
# with each other and neither agreed with the file just edited -- a true
# statement about the pair, read as a statement about the state of things.
# Comparing exactly the two refs it was asked about is what a narrowing does;
# the reassuring summary line is what made it misleading.
for path in "${WORKFLOWS[@]}"; do
  [ -f "$path" ] || continue
  local_blob=$(git hash-object "$path")
  a=$(blob "$REF_A" "$path")
  b=$(blob "$REF_B" "$path")
  if [ "$local_blob" != "$a" ] || [ "$local_blob" != "$b" ]; then
    echo "  AHEAD $path here differs from the published copies"
    echo "        working tree: $local_blob"
    echo "        $REF_A: $a"
    echo "        $REF_B: $b"
    echo "        Both refs need this file before a dispatch or PR check"
    echo "        reflects it. Until then the run uses the OLD workflow and"
    echo "        still reports success."
    fails=$((fails + 1))
  fi
done

echo
if [ "$fails" -eq 0 ]; then
  echo "no workflow drift"
  exit 0
fi
echo "$fails workflow file(s) have drifted -- the review will skip silently"
exit 1
