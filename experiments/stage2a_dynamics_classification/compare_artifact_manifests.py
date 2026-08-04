"""
Compares two ARTIFACT_MANIFEST.json files field-by-field and exits
nonzero on any load-bearing mismatch. Used by `make stage2a-verify` to
turn manifest regeneration into an actual verification gate rather than
a visual `git diff --stat` inspection aid (external review: the
previous `stage2a-verify` regenerated the committed manifest in place
and always exited 0, since `git diff --stat` succeeds whether or not it
prints anything).

Compares every top-level key (`artifacts`, `graphs`, `selected_C`,
`dimensions`, `image_ordering`, `frozen_primary_effect`) except
`environment` -- that key records the generating machine's dependency
versions/platform/git SHA, which legitimately differ run to run without
indicating a reproduction failure.

Usage: python3 compare_artifact_manifests.py <committed.json> <candidate.json>
Exit 0: no load-bearing differences. Exit 1: at least one mismatch (printed).
"""
import json
import sys

IGNORED_KEYS = {"environment"}


def _diff(committed, candidate, path=""):
    diffs = []
    keys = (set(committed) | set(candidate)) - IGNORED_KEYS if path == "" else set(committed) | set(candidate)
    for key in sorted(keys):
        key_path = f"{path}.{key}" if path else key
        if key not in committed:
            diffs.append(f"  + {key_path} only in candidate: {candidate[key]!r}")
            continue
        if key not in candidate:
            diffs.append(f"  - {key_path} only in committed: {committed[key]!r}")
            continue
        c_val, cand_val = committed[key], candidate[key]
        if isinstance(c_val, dict) and isinstance(cand_val, dict):
            diffs.extend(_diff(c_val, cand_val, key_path))
        elif c_val != cand_val:
            diffs.append(f"  ~ {key_path}: committed={c_val!r} candidate={cand_val!r}")
    return diffs


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <committed.json> <candidate.json>", file=sys.stderr)
        sys.exit(2)
    committed_path, candidate_path = sys.argv[1], sys.argv[2]

    with open(committed_path) as f:
        committed = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)

    diffs = _diff(committed, candidate)
    if diffs:
        print(f"MISMATCH: {candidate_path} differs from {committed_path} "
              f"on {len(diffs)} load-bearing field(s) (environment.* excluded):")
        for d in diffs:
            print(d)
        sys.exit(1)

    print(f"OK: {candidate_path} matches {committed_path} on all load-bearing fields "
          f"(environment.* excluded).")


if __name__ == "__main__":
    main()
