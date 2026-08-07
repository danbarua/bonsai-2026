#!/usr/bin/env python3
"""Emit a deterministic, position-labelled byte stream on stdout.

The measurement instrument for one question the provenance-capture design
turns on: **does a `PostToolUse` hook receive the tool's full stdout, or a
truncated copy?** The answer decides whether capture can read stdout from
the hook payload at all, or must intercept it at execution time.

Why position-labelled rather than `yes | head -c N`: a truncated stream and
a complete one are indistinguishable if every line is identical. Each line
here carries its own index and the total it belongs to, and the stream ends
with a sentinel carrying the digest of everything before it. From whatever
survives, a reader can tell (a) how much arrived, (b) whether it was taken
from the head, the tail, or both with the middle elided, and (c) whether it
is byte-identical to what was sent.

This file exists because a number produced by an ephemeral script is not a
decision anchor (CLAUDE.md principle 24). The truncation figures quoted in
`docs/proposals/PROVENANCE_CONTRACT.md` come from running this, and can be
re-derived by running it again.

Usage:
    uv run python tools/provenance/emit_bytes.py --bytes 200000
"""
import argparse
import hashlib
import sys

# Fixed line width keeps `--bytes` predictable: byte count is a whole
# number of lines, so "how much survived" reads directly as a line count.
LINE_WIDTH = 100
PREFIX = "LINE"
SENTINEL = "EMIT_END"

# A repeating, non-uniform filler. Uniform filler (all 'x') would make a
# line that lost its middle look intact.
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _line(index: int, total: int) -> str:
    """One labelled line of exactly LINE_WIDTH characters (excl. newline).

    The label is at the START of the line so that a truncation which keeps
    line prefixes still reveals position.
    """
    head = f"{PREFIX}{index:08d}/{total:08d} "
    filler_len = LINE_WIDTH - len(head)
    if filler_len < 0:
        raise ValueError(f"LINE_WIDTH={LINE_WIDTH} too small for label {head!r}")
    # Offset the alphabet by the line index so no two lines are identical.
    filler = "".join(
        _ALPHABET[(index + i) % len(_ALPHABET)] for i in range(filler_len)
    )
    return head + filler


def build_stream(n_bytes: int) -> str:
    """The exact string written to stdout, sentinel included.

    Returned rather than printed so a test can build it without spawning a
    process and compare against a captured copy.
    """
    if n_bytes < 0:
        raise ValueError(f"--bytes must be non-negative, got {n_bytes}")
    per_line = LINE_WIDTH + 1  # + newline
    n_lines = n_bytes // per_line
    body = "".join(_line(i, n_lines) + "\n" for i in range(n_lines))
    digest = hashlib.sha256(body.encode()).hexdigest()
    sentinel = (
        f"{SENTINEL} lines={n_lines} body_bytes={len(body.encode())} "
        f"body_sha256={digest}\n"
    )
    return body + sentinel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bytes", type=int, required=True,
        help="approximate size of the body in bytes (rounded down to whole "
             "lines; the sentinel line is emitted in addition)")
    parser.add_argument(
        "--stream", choices=("stdout", "stderr", "both"), default="stdout",
        help="which stream to emit on. `stderr` matters because a scratch "
             "script that dies leaves its traceback there, and because "
             "`mighty-colab run` splits its own chatter onto stderr and the "
             "script's output onto stdout")
    parser.add_argument(
        "--exit-code", type=int, default=0,
        help="exit with this code after emitting. Non-zero routes the tool "
             "call to PostToolUseFailure instead of PostToolUse -- a "
             "different hook event, whose payload is not assumed to match")
    args = parser.parse_args(argv)
    stream = build_stream(args.bytes)
    if args.stream in ("stdout", "both"):
        sys.stdout.write(stream)
    if args.stream in ("stderr", "both"):
        sys.stderr.write(stream)
    sys.stdout.flush()
    sys.stderr.flush()
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
