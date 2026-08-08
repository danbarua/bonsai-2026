"""Does the REGISTRATION actually make the capture hook fire?

Every other test in this feature drives `capture.py` with synthetic stdin.
That proves the hook works and says nothing about whether
`.claude/settings.json` causes it to run — the narrow thing verified
without the wiring it depends on, which is CLAUDE.md principle 21's shape
and the same gap as vacuous-test incident #15 one layer out.

It is not a hypothetical gap. The first live firing found that hook
registrations are read at SESSION START, so a session already running when
the hooks landed captures nothing. Nothing errors, and an absent record is
indistinguishable from "correctly classified as not scratch" — so the
failure is silent in the direction that matters.

Closing it needs a real session by construction, so these tests spawn a
headless `claude -p` subprocess and inspect what the hooks wrote. That
makes them slow and dependent on the CLI, hence `@pytest.mark.slow` and a
skip when `claude` is absent — excluded from the default suite, run
deliberately after any change to the registration or the hook scripts.

`BONSAI_PROVENANCE_ROOT` redirects the spawned session's captures into a
temp directory, so running this never writes to the project's real log.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = REPO_ROOT / ".claude" / "settings.json"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("claude") is None,
                       reason="claude CLI not on PATH; live wiring unverifiable"),
    pytest.mark.skipif(not SETTINGS.exists(),
                       reason="no project settings.json to register hooks"),
]

# A cheap, side-effect-free scratch shape. Deliberately `python -c` rather
# than a heredoc: it is the shape the corpus treats as canonical and the
# one most likely to regress if `_strip_wrappers` changes.
SCRATCH = "uv run python -c \"print('live registration check')\""
NOT_SCRATCH = "git status --short"

# The over-capture stage2b-lead hit in the field: their commit message for
# `7879a4c` landed in a provenance blob because the predicate keyed on
# "heredoc present" without checking what consumed it. Their real command
# was `git commit -q -F - <<'EOF'`; this uses a READ-ONLY git command so the
# live test cannot create anything, while keeping the shape that mattered --
# a heredoc handed to a non-interpreter.
HEREDOC_TO_NON_INTERPRETER = "git log --oneline -1 <<'EOF'\nunused stdin\nEOF"


def run_session(command: str, root: Path, timeout: int = 240):
    """One headless session that runs `command` once and stops."""
    prompt = (
        f"Run this exact command with the Bash tool, once, then reply with "
        f"just DONE. Do not summarise the output or run anything else.\n\n"
        f"{command}")
    env = dict(os.environ)
    env["BONSAI_PROVENANCE_ROOT"] = str(root)
    return subprocess.run(
        ["claude", "-p", prompt, "--allowedTools", "Bash",
         "--model", "claude-haiku-4-5-20251001"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def all_records(root: Path) -> list[dict]:
    out = []
    for log in (root / ".provenance" / "runs").rglob("capture.jsonl"):
        out.extend(json.loads(line) for line in log.read_text().splitlines()
                   if line.strip())
    return out


def test_a_registered_hook_fires_in_a_real_session(tmp_path):
    """The wiring test. Registration -> hook runs -> record on disk.

    Reports what it found rather than merely asserting, per principle 20:
    for a slow test run deliberately and rarely, the transcript is most of
    its value.
    """
    proc = run_session(SCRATCH, tmp_path)
    print(f"\n[live] session exit={proc.returncode} reply={proc.stdout.strip()[:60]!r}")
    records = all_records(tmp_path)
    for record in records:
        print(f"[live] {record['phase']:6s} {record.get('trigger_reason') or ''} "
              f"{record.get('output') or ''}")
    assert records, (
        "no capture record written by a real session -- the hook is not "
        "registered, not executable, or the settings file is not being read. "
        "This is the failure the synthetic-stdin tests cannot see.")

    opens = [r for r in records if r["phase"] == "open"]
    assert opens, f"records written but none is an `open`: {records}"
    assert opens[0]["trigger_reason"] == "inline_c"
    assert opens[0]["script"]["text"] == "print('live registration check')", (
        "the script text did not survive the round trip through the harness")


def test_a_real_session_writes_its_commit_point(tmp_path):
    """The marker that makes absence diagnostic, verified where it counts.

    A `session_open` record proves capture was live for that session, so
    every later absence becomes an inference about the predicate rather
    than an unanswerable question. Verified live rather than only against
    synthetic stdin, because the whole point of the marker is to witness a
    property of the REGISTRATION -- the thing synthetic tests cannot see.
    """
    run_session(SCRATCH, tmp_path)
    markers = [r for r in all_records(tmp_path) if r["phase"] == "session_open"]
    for marker in markers:
        print(f"\n[live] marker source={marker['source']} "
              f"v{marker['hook_version']} git={marker['git']['commit'][:8]} "
              f"dirty={marker['git']['dirty']}")
    assert markers, (
        "a real session wrote no session_open marker -- either SessionStart "
        "is not registered, or the marker is broken. Without it an empty "
        "capture log is ambiguous, which is the defect it exists to remove.")
    assert markers[0]["hook_version"]
    assert markers[0]["git"]["commit"]


def test_both_phases_are_written_and_can_be_joined(tmp_path):
    """`open` and `close` come from two separate hook invocations in two
    separate processes. If they cannot be joined the log is unreadable."""
    run_session(SCRATCH, tmp_path)
    # The session marker is deliberately excluded: it belongs to the session,
    # not to any tool call, and carries no tool_use_id to join on.
    records = [r for r in all_records(tmp_path) if r["phase"] != "session_open"]
    phases = {r["phase"] for r in records}
    assert phases == {"open", "close"}, f"expected both phases, got {phases}"
    ids = {r["tool_use_id"] for r in records}
    assert len(ids) == 1, f"records cannot be joined: {ids}"
    close = next(r for r in records if r["phase"] == "close")
    print(f"\n[live] joined on {ids.pop()}: {close['output']}")
    assert close["output"]["fidelity"] == "complete"


def test_ordinary_commands_produce_no_record_in_a_real_session(tmp_path):
    """The exclusion, verified live rather than only in the corpus.

    The corpus proves the PREDICATE declines `git status`. This proves the
    registered hook declines it too -- a matcher that captured everything
    would pass every corpus test while logging the whole session.
    """
    proc = run_session(NOT_SCRATCH, tmp_path)
    records = all_records(tmp_path)
    markers = [r for r in records if r["phase"] == "session_open"]
    captures = [r for r in records if r["phase"] != "session_open"]
    print(f"\n[live] ordinary command exit={proc.returncode}, "
          f"{len(markers)} markers, {len(captures)} captures (expected 0)")
    for record in captures:
        print(f"[live] UNEXPECTED: {record.get('command')}")

    # The marker is what turns this from a weak test into a strong one.
    # Without it, "no capture records" would also be satisfied by a session
    # in which the hooks never loaded -- the exact ambiguity the marker was
    # added to remove. Asserting it here means this test now distinguishes
    # "hooks were live and the predicate declined" from "hooks were absent".
    assert markers, (
        "no session marker -- hooks were not live, so this test proves "
        "nothing about whether ordinary commands are excluded")
    assert not captures, (
        "an ordinary command was captured -- the matcher or predicate is "
        "over-capturing, which would put unrelated work into a forensic log")


def test_a_heredoc_to_a_non_interpreter_is_not_captured_live(tmp_path):
    """Field-reported over-capture, pinned where the corpus could not see it.

    `test_ordinary_commands_produce_no_record_in_a_real_session` uses
    `git status`, which carries no heredoc -- so it proved git is declined
    only when nothing else about the command looks interesting, and the
    defect shipped underneath it. This is the same command class WITH the
    feature that triggered the bug.
    """
    run_session(HEREDOC_TO_NON_INTERPRETER, tmp_path)
    records = all_records(tmp_path)
    markers = [r for r in records if r["phase"] == "session_open"]
    captures = [r for r in records if r["phase"] != "session_open"]
    print(f"\n[live] heredoc-to-git: {len(markers)} markers, "
          f"{len(captures)} captures (expected 0)")
    for record in captures:
        print(f"[live] UNEXPECTED: {record.get('trigger_reason')} "
              f"{(record.get('script') or {}).get('text', '')[:60]!r}")
    assert markers, "hooks were not live; this test proves nothing"
    assert not captures, (
        "a heredoc fed to a non-interpreter was captured -- prose is "
        "landing in a store meant for code")


def test_the_live_check_is_not_vacuous(tmp_path):
    """All three tests above would pass if `run_session` silently failed to
    start a session at all -- two by finding nothing, one by finding
    nothing. So prove a session really ran and really executed the command.
    """
    proc = run_session(SCRATCH, tmp_path)
    assert proc.returncode == 0, f"session did not run: {proc.stderr[:400]}"
    assert "DONE" in proc.stdout.upper(), (
        f"session ran but did not execute the command: {proc.stdout[:200]!r}")
