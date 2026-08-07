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

# Each case answers one question the capture design turns on.
DEFAULT_CASES = (
    # Is anything persisted BELOW the cap? Decides whether the inline-field
    # fallback is the common path or dead code.
    {"bytes": 1_000, "stream": "stdout", "exit_code": 0},
    {"bytes": 25_000, "stream": "stdout", "exit_code": 0},
    # Where is the cap, and is it constant rather than proportional?
    {"bytes": 200_000, "stream": "stdout", "exit_code": 0},
    {"bytes": 1_000_000, "stream": "stdout", "exit_code": 0},
    # Is stderr capped the same way, and does the persisted file contain it?
    {"bytes": 200_000, "stream": "stderr", "exit_code": 0},
    # Does a FAILING call deliver a payload at all, on PostToolUseFailure?
    {"bytes": 200_000, "stream": "both", "exit_code": 1},
)

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
            ],
            # A separate event, fired instead of PostToolUse when the tool
            # call fails. Registered because "survive mid-run death" is the
            # case the capture design exists for -- a scratch script that
            # dies is exactly the one whose record matters most, and it
            # never reaches PostToolUse at all.
            "PostToolUseFailure": [
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
            ],
        },
    }
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return SETTINGS_PATH


def run_probe(case: dict, settings: Path, timeout: int) -> subprocess.CompletedProcess:
    """One headless session that runs the emitter once, per `case`."""
    command = (f"uv run python {EMITTER} --bytes {case['bytes']} "
               f"--stream {case['stream']} --exit-code {case['exit_code']}")
    prompt = (
        f"Run this exact command with the Bash tool, once, and then reply "
        f"with just the word DONE. It is EXPECTED to fail if it exits "
        f"non-zero -- do not retry it, do not investigate, do not summarise "
        f"the output, do not pipe it anywhere, do not run anything else.\n\n"
        f"{command}"
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


def _flag(value) -> str:
    return {True: "yes", False: "no", None: "-"}[value]


def _arg(command: str, name: str, cast=str):
    """Pull a flag's value back out of the recorded command line."""
    if name not in command:
        return None
    try:
        return cast(command.split(name)[1].split()[0])
    except (IndexError, ValueError):
        return None


def report(records) -> None:
    """Print the table the design document quotes.

    Includes the persisted-file columns, not only the cap: the claim that
    the persisted copy is byte-exact is the one the design leans on hardest,
    so a rerun of this driver has to reproduce it without anyone reaching
    for an ad-hoc query.
    """
    header = (f"\n{'sent':>9} {'stream':>7} {'exit':>5} {'event':>20} "
              f"{'inline_out':>10} {'inline_err':>10} {'persistSize':>11} "
              f"{'fullLen':>9} {'sha_ok':>7}  verdict")
    print(header)
    print("-" * len(header.strip()))
    for rec in records:
        if "probe_error" in rec:
            print(f"  PROBE ERROR: {rec['probe_error']}")
            continue
        command = rec.get("command") or ""
        sent = _arg(command, "--bytes", int)
        stream = _arg(command, "--stream") or "stdout"
        exit_code = _arg(command, "--exit-code", int)
        out, err = rec.get("stdout", {}), rec.get("stderr", {})
        expected_lines = (sent // PER_LINE) if sent is not None else None

        primary = out if stream in ("stdout", "both") else err
        if primary.get("body_matches_sentinel") and \
                primary.get("n_labelled_lines") == expected_lines:
            verdict = "inline COMPLETE"
        elif rec.get("persisted_body_matches_sentinel"):
            verdict = "inline capped, persisted COMPLETE"
        elif primary.get("len", 0) == 0:
            verdict = "NO TEXT IN PAYLOAD"
        else:
            verdict = "inline capped, NO COMPLETE COPY"

        print(f"{sent!s:>9} {stream:>7} {exit_code!s:>5} "
              f"{str(rec.get('hook_event_name')):>20} "
              f"{out.get('len', 0):>10} {err.get('len', 0):>10} "
              f"{rec.get('persisted_output_size')!s:>11} "
              f"{rec.get('persisted_len')!s:>9} "
              f"{_flag(rec.get('persisted_body_matches_sentinel')):>7}  {verdict}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="stdout-only cases at these sizes, instead of "
                             "the full default case list")
    parser.add_argument("--case", nargs=3, metavar=("BYTES", "STREAM", "EXIT"),
                        default=None,
                        help="run a single ad-hoc case, e.g. "
                             "--case 200000 both 1. Exists so that "
                             "re-interrogating one condition is a flag on "
                             "committed code rather than a throwaway script")
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

    if args.case:
        cases = [{"bytes": int(args.case[0]), "stream": args.case[1],
                  "exit_code": int(args.case[2])}]
    elif args.sizes:
        cases = [{"bytes": s, "stream": "stdout", "exit_code": 0}
                 for s in args.sizes]
    else:
        cases = list(DEFAULT_CASES)

    for case in cases:
        print(f"\n=== probing bytes={case['bytes']} stream={case['stream']} "
              f"exit={case['exit_code']} ===")
        try:
            proc = run_probe(case, settings, args.timeout)
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
