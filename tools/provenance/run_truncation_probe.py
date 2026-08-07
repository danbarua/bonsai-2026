#!/usr/bin/env python3
"""Measure what a `PostToolUse` hook receives when a tool emits a lot of stdout.

Runs the probe inside a HEADLESS Claude Code subprocess with its own
`--settings` file, rather than asking a human to hand-edit
`.claude/settings.local.json`. Two reasons, and the second is the point of
this whole directory:

  1. `.claude/settings.local.json` is shared with whatever other session has
     this repo open. A probe should not reach into it.
  2. A number reachable only by "edit your settings, then type these four
     commands" is load-bearing scratch -- its generator is a chat log
     (CLAUDE.md principle 24). Running this file reproduces the measurement.

What it measures, for each stdout size: how many bytes the hook actually
received, whether the labelled lines that arrived are contiguous from the
head, and whether the terminal sentinel survived. Together those separate
"untruncated", "head kept, tail dropped", and "middle elided".

Usage:
    uv run python tools/provenance/run_truncation_probe.py
    uv run python tools/provenance/run_truncation_probe.py --sizes 1000 200000
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO_ROOT / ".provenance"
LOG_PATH = PROBE_DIR / "probe_payload.jsonl"
SETTINGS_PATH = PROBE_DIR / "probe_settings.json"
HOOK = REPO_ROOT / "tools" / "provenance" / "probe_hook_payload.py"
EMITTER = "tools/provenance/emit_bytes.py"

DEFAULT_SIZES = (1_000, 50_000, 200_000, 1_000_000)

# Sentinel line width, from emit_bytes.LINE_WIDTH + newline. Used only to
# turn a byte count back into an expected line count for the report.
PER_LINE = 101


def write_settings() -> Path:
    """Generate the probe's settings file from this script's own location.

    Generated rather than committed with a hardcoded path: the hook command
    needs an absolute path, and an absolute path baked into a committed file
    is wrong on every machine but one.
    """
    settings = {
        "permissions": {
            "allow": [f"Bash(uv run python {EMITTER}:*)"],
            "deny": [],
            "defaultMode": "acceptEdits",
        },
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "if": f"Bash(*{EMITTER}*)",
                            "command": f"uv run python {HOOK}",
                        }
                    ],
                }
            ]
        },
    }
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return SETTINGS_PATH


def run_probe(size: int, settings: Path, timeout: int) -> subprocess.CompletedProcess:
    """One headless session that runs the emitter once at `size`."""
    prompt = (
        f"Run this exact command with the Bash tool, once, and then reply "
        f"with just the word DONE. Do not summarise the output, do not pipe "
        f"it anywhere, do not run anything else.\n\n"
        f"uv run python {EMITTER} --bytes {size}"
    )
    env = dict(os.environ)
    # The hook writes here; keeping it explicit means a stray default cannot
    # send records somewhere this script will not look.
    env["BONSAI_PROBE_LOG"] = str(LOG_PATH)
    return subprocess.run(
        [
            "claude", "-p", prompt,
            "--settings", str(settings),
            "--allowedTools", "Bash",
            "--model", "claude-haiku-4-5-20251001",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


def read_records():
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]


def report(records) -> None:
    """Print the table the design document quotes."""
    print(f"\n{'sent':>10} {'hook saw':>10} {'payload':>10} {'lines':>7} "
          f"{'first':>7} {'last':>7} {'contig':>7} {'sentinel':>9}  verdict")
    print("-" * 96)
    for rec in records:
        if "probe_error" in rec:
            print(f"  PROBE ERROR: {rec['probe_error']}")
            continue
        command = rec.get("command") or ""
        sent = None
        if "--bytes" in command:
            try:
                sent = int(command.split("--bytes")[1].split()[0])
            except (IndexError, ValueError):
                sent = None
        expected_lines = (sent // PER_LINE) if sent is not None else None
        saw = rec["text_len"]
        complete = rec["sentinel_present"] and rec["first_line_index"] == 0 \
            and rec["line_index_contiguous"]
        if complete and expected_lines is not None \
                and rec["n_labelled_lines"] == expected_lines:
            verdict = "UNTRUNCATED"
        elif rec["sentinel_present"] and not rec["line_index_contiguous"]:
            verdict = "MIDDLE ELIDED"
        elif not rec["sentinel_present"]:
            verdict = "TAIL DROPPED"
        else:
            verdict = "PARTIAL"
        print(f"{sent!s:>10} {saw:>10} {rec['payload_json_bytes']:>10} "
              f"{rec['n_labelled_lines']:>7} {rec['first_line_index']!s:>7} "
              f"{rec['last_line_index']!s:>7} "
              f"{str(rec['line_index_contiguous']):>7} "
              f"{str(rec['sentinel_present']):>9}  {verdict}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--keep-log", action="store_true",
                        help="append to any existing log instead of starting clean")
    args = parser.parse_args(argv)

    if shutil.which("claude") is None:
        print("claude CLI not on PATH -- cannot run the probe", file=sys.stderr)
        return 2

    if LOG_PATH.exists() and not args.keep_log:
        LOG_PATH.unlink()

    settings = write_settings()
    print(f"settings: {settings}")
    print(f"log:      {LOG_PATH}")

    for size in args.sizes:
        print(f"\n=== probing --bytes {size} ===")
        try:
            proc = run_probe(size, settings, args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {args.timeout}s")
            continue
        print(f"  exit={proc.returncode} reply={proc.stdout.strip()[:120]!r}")
        if proc.returncode != 0:
            print(f"  stderr: {proc.stderr.strip()[:500]}")

    report(read_records())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
