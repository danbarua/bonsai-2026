#!/usr/bin/env python3
"""How much latency does the capture hook add to an ordinary tool call?

The hook is registered on `Bash|mcp__mighty-colab__.*`, so it runs on EVERY
matching tool call in every session in this repository -- not only the ones
it captures. The uncaptured case is overwhelmingly the common one, and it is
pure tax: a process starts, reads stdin, decides "not scratch", and exits.

Worth measuring rather than assuming, for a reason specific to this design.
The capture hooks were accepted on the argument that they never block and
never surprise. A hook that noticeably slows every command in someone else's
session breaks that bargain in a way no test catches, and the natural
response to a tool that makes your work feel sluggish is to disable it --
which is the route-around this whole feature is built to avoid. Latency is
therefore a correctness property here, not a nicety.

Measures three cases separately, because they cost different things:

  not-scratch  -- the common path. Predicate says no, hook exits.
  scratch-pre  -- an `open` record: git state (three subprocesses) and blob
                  writes, so the expensive one.
  scratch-post -- a `close` record: reads output, writes a blob.

Usage:
    uv run python tools/provenance/bench_hook_overhead.py
    uv run python tools/provenance/bench_hook_overhead.py --runs 50
"""
import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SH = REPO_ROOT / ".claude" / "hooks" / "provenance-capture" / "capture.sh"

NOT_SCRATCH = {
    "hook_event_name": "PreToolUse", "tool_name": "Bash",
    "tool_input": {"command": "git status --short"},
    "tool_use_id": "bench", "session_id": "bench-session",
}
SCRATCH_PRE = {
    "hook_event_name": "PreToolUse", "tool_name": "Bash",
    "tool_input": {"command": 'uv run python -c "print(1)"'},
    "tool_use_id": "bench", "session_id": "bench-session",
}
SCRATCH_POST = {
    "hook_event_name": "PostToolUse", "tool_name": "Bash",
    "tool_input": {"command": 'uv run python -c "print(1)"'},
    "tool_use_id": "bench", "session_id": "bench-session",
    "tool_response": {"stdout": "1\n", "stderr": ""},
}


def time_one(payload: dict, root: Path) -> float:
    """Wall-clock seconds for one hook invocation, as the harness sees it."""
    env = dict(os.environ)
    env["BONSAI_PROVENANCE_ROOT"] = str(root)
    start = time.perf_counter()
    subprocess.run([str(CAPTURE_SH)], input=json.dumps(payload), text=True,
                   capture_output=True, env=env, timeout=60)
    return time.perf_counter() - start


def bench(name: str, payload: dict, runs: int, root: Path) -> list[float]:
    # One warm-up, discarded: the first run pays for filesystem caching that
    # a real session has already paid long before.
    time_one(payload, root)
    samples = [time_one(payload, root) for _ in range(runs)]
    ms = sorted(s * 1000 for s in samples)
    print(f"  {name:<14} median {statistics.median(ms):7.1f} ms   "
          f"min {ms[0]:6.1f}   max {ms[-1]:6.1f}")
    return ms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args(argv)

    print(f"capture.sh overhead over {args.runs} runs each "
          f"(warm-up discarded):\n")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        common = bench("not-scratch", NOT_SCRATCH, args.runs, root)
        bench("scratch-pre", SCRATCH_PRE, args.runs, root)
        bench("scratch-post", SCRATCH_POST, args.runs, root)

    median = statistics.median(common)
    print(f"\nThe not-scratch case is the tax on every matching tool call in "
          f"every session.\nMedian {median:.1f} ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
