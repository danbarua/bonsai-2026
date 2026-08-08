#!/usr/bin/env python3
"""Summarise what the capture logs actually contain.

Exists because the alternative is an inline `python -c` whose output ends up
in a design document — load-bearing scratch, the exact thing the capture
hooks were built to catch (CLAUDE.md principle 24). Any figure this project
quotes about capture volume, output sizes or fidelity should come from
running this, and be re-derivable by running it again.

It reads the run-scoped logs under `.provenance/runs/`, which are forensic
and gitignored. **That does not make its output citable.** The logs are a
leaf: nothing committed may descend from them. What this script produces is
a description of the log's own shape — how big records get, how often
fidelity degrades — used to size parameters like `OUTPUT_BLOB_MAX` and to
answer "is the retention policy right yet". It is not a route for turning a
captured result into a citation, and a number about the SUBJECT of a capture
still has to come from committed code that regenerates it.

Usage:
    uv run python tools/provenance/capture_stats.py
    uv run python tools/provenance/capture_stats.py --root /path/to/repo
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load(root: Path):
    """Every record across every run log under `root`."""
    records = []
    for log in sorted((root / ".provenance" / "runs").rglob("capture.jsonl")):
        session = log.parent.name
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial final line from a killed process
            record["_session"] = session
            records.append(record)
    return records


def _quantile(values, q):
    """Nearest-rank quantile. Explicit rather than numpy, so this stays
    runnable with no dependencies wherever a log happens to live."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def report(records) -> None:
    sessions = {r["_session"] for r in records}
    markers = [r for r in records if r.get("phase") == "session_open"]
    opens = [r for r in records if r.get("phase") == "open"]
    closes = [r for r in records if r.get("phase") == "close"]

    print(f"\nsessions with a log : {len(sessions)}")
    print(f"session markers     : {len(markers)}")
    print(f"captures (open)     : {len(opens)}")
    print(f"captures (close)    : {len(closes)}")

    # A session whose log has no marker was capturing without a commit
    # point -- the ambiguity the marker exists to remove. Worth surfacing,
    # since it means records from that session cannot be trusted as complete.
    unmarked = sessions - {r["_session"] for r in markers}
    if unmarked:
        print(f"\nWARNING: {len(unmarked)} session(s) wrote records with no "
              f"session_open marker: {sorted(unmarked)}")

    if opens:
        print("\ntrigger reasons:")
        for reason, n in Counter(r.get("trigger_reason") for r in opens).most_common():
            print(f"  {n:5d}  {reason}")

    sizes = [r["output"]["bytes"] for r in closes
             if (r.get("output") or {}).get("bytes") is not None]
    if sizes:
        print(f"\noutput bytes over {len(sizes)} close records:")
        print(f"  min    {min(sizes):>10,}")
        print(f"  median {int(statistics.median(sizes)):>10,}")
        print(f"  p90    {_quantile(sizes, 0.90):>10,}")
        print(f"  p99    {_quantile(sizes, 0.99):>10,}")
        print(f"  max    {max(sizes):>10,}")

    if closes:
        print("\nfidelity:")
        for value, n in Counter(
                (r.get("output") or {}).get("fidelity") for r in closes).most_common():
            print(f"  {n:5d}  {value}")
        print("output source:")
        for value, n in Counter(
                (r.get("output") or {}).get("source") for r in closes).most_common():
            print(f"  {n:5d}  {value}")

    # Whether the self-truncation ceiling has ever actually bound. A ceiling
    # nothing has approached is a policy choice, not a measured threshold,
    # and should be described as one.
    truncated = [r for r in closes
                 if (r.get("output") or {}).get("fidelity") == "truncated_by_capture"]
    print(f"\nrecords truncated by capture: {len(truncated)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repository root holding .provenance/runs/")
    args = parser.parse_args(argv)
    records = load(args.root)
    if not records:
        print(f"no capture records under {args.root}/.provenance/runs/")
        return 0
    report(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
