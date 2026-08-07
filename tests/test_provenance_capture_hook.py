"""The capture hook, exercised as a real subprocess.

Invoked the way the harness invokes it -- JSON on stdin, exit code and
filesystem effects observed -- rather than by calling `handle()` directly.
The properties that matter here are process-level (does it exit 0 when the
disk is unwritable?) and a direct function call cannot show them.

**The fail-open cases are the point of this file.** Capture was authorized
on the condition that a broken capture hook can never block normal work,
and that condition is discharged by demonstration, not assertion: each of
the three failure modes below is induced for real and the exit code
observed. A hook that has never been seen to survive its own failure is
not yet fail-open (CLAUDE.md principle 21's corollary).
"""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_DIR = REPO_ROOT / ".claude" / "hooks" / "provenance-capture"
CAPTURE_SH = HOOK_DIR / "capture.sh"
CAPTURE_PY = HOOK_DIR / "capture.py"

SCRATCH_CMD = 'uv run python -c "print(6*7)"'


def run_hook(payload: dict, root: Path, script: Path = CAPTURE_SH,
             env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["BONSAI_PROVENANCE_ROOT"] = str(root)
    env.update(env_extra or {})
    argv = ([str(script)] if script.suffix == ".sh"
            else [sys.executable, str(script)])
    return subprocess.run(argv, input=json.dumps(payload), text=True,
                          capture_output=True, env=env, timeout=30)


def records(root: Path, session: str = "s1") -> list[dict]:
    log = root / ".provenance" / "runs" / session / "capture.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def pre_payload(command: str = SCRATCH_CMD, session: str = "s1") -> dict:
    return {
        "hook_event_name": "PreToolUse", "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_use_id": "toolu_1", "session_id": session, "cwd": str(REPO_ROOT),
    }


# --- fail-open: the authorization condition --------------------------------

@pytest.mark.parametrize("script", [CAPTURE_SH, CAPTURE_PY],
                         ids=["via-wrapper", "python-directly"])
def test_fail_open_on_malformed_input(tmp_path, script):
    """Garbage on stdin must not become a broken tool call.

    **Both layers, separately, and that is not belt-and-braces theatre.**
    `capture.sh` ends in an unconditional `exit 0`, so it masks ANY exit
    code from `capture.py` -- meaning a wrapper-only test would pass even
    if the Python layer blocked outright. Confirmed by breaking it:
    returning 2 from `capture.py`'s exception handler left the
    wrapper-routed case green, and only the direct case caught it.

    That is CLAUDE.md principle 21's second half at hook scale -- verifying
    a narrow thing through the broader thing that wraps it proves the
    broader thing works and says nothing about the narrow one.
    """
    argv = ([str(script)] if script.suffix == ".sh"
            else [sys.executable, str(script)])
    proc = subprocess.run(
        argv, input="this is not json{{{", text=True,
        capture_output=True, timeout=30,
        env={**os.environ, "BONSAI_PROVENANCE_ROOT": str(tmp_path)})
    print(f"\n[fail-open] malformed stdin via {script.name} "
          f"-> exit {proc.returncode}")
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("script", [CAPTURE_SH, CAPTURE_PY],
                         ids=["via-wrapper", "python-directly"])
def test_fail_open_when_the_log_directory_cannot_be_created(tmp_path, script):
    """An unwritable tree is an environment problem, not the session's."""
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        proc = run_hook(pre_payload(), readonly, script=script)
        print(f"[fail-open] unwritable root -> exit {proc.returncode}")
        assert proc.returncode == 0, proc.stderr
        assert records(readonly) == []
    finally:
        readonly.chmod(stat.S_IRWXU)


def test_fail_open_when_the_interpreter_is_missing(tmp_path):
    """The wrapper's own reason to exist.

    capture.py cannot catch a failure that stops it running at all, so the
    shell layer has to. Simulated by emptying PATH, which is what a hook
    spawned in a stripped environment actually sees.
    """
    proc = subprocess.run(
        [str(CAPTURE_SH)], input=json.dumps(pre_payload()), text=True,
        capture_output=True, timeout=30,
        env={"PATH": "/nonexistent", "BONSAI_PROVENANCE_ROOT": str(tmp_path)})
    print(f"[fail-open] no python3 on PATH -> exit {proc.returncode}")
    assert proc.returncode == 0, proc.stderr


def test_fail_open_is_not_vacuous(tmp_path):
    """The three cases above would pass on a hook that does nothing at all.

    So prove the same invocation genuinely works when nothing is broken --
    otherwise 'it exited 0' is evidence of absence, not of resilience.
    """
    proc = run_hook(pre_payload(), tmp_path)
    assert proc.returncode == 0
    assert len(records(tmp_path)) == 1, "healthy path wrote no record"


# --- what gets captured ----------------------------------------------------

def test_scratch_writes_an_open_record_with_the_script(tmp_path):
    run_hook(pre_payload(), tmp_path)
    (record,) = records(tmp_path)
    assert record["phase"] == "open"
    assert record["trigger_reason"] == "inline_c"
    assert record["script"]["text"] == "print(6*7)"
    assert record["script"]["sha256"]
    print(f"\n[capture] open record: {record['capture_id']} "
          f"reason={record['trigger_reason']}")


def test_ordinary_commands_write_nothing(tmp_path):
    """The tax on exploration must be zero, and that means zero records."""
    run_hook(pre_payload("git status --short"), tmp_path)
    assert records(tmp_path) == []


def test_git_state_records_dirtiness(tmp_path):
    """A number produced against a dirty tree is not reproducible from the
    commit alone, and the record must say so rather than imply otherwise."""
    run_hook(pre_payload(), tmp_path)
    (record,) = records(tmp_path)
    assert set(record["git"]) == {"commit", "branch", "dirty"}
    assert isinstance(record["git"]["dirty"], (bool, type(None)))


def test_close_record_reads_the_persisted_file_when_named(tmp_path):
    """The measured escape hatch: full output without rewriting the command."""
    payload = pre_payload()
    big = tmp_path / "persisted.txt"
    big.write_text("x" * 50_000)
    payload.update(hook_event_name="PostToolUse", duration_ms=1234,
                   tool_response={"stdout": "x" * 30_000,
                                  "persistedOutputPath": str(big),
                                  "persistedOutputSize": 50_000})
    run_hook(payload, tmp_path)
    (record,) = records(tmp_path)
    assert record["phase"] == "close" and record["outcome"] == "ok"
    assert record["output"]["source"] == "persisted"
    assert record["output"]["fidelity"] == "complete"
    assert record["output"]["bytes"] == 50_000
    print(f"[capture] close record: {record['output']}")


def test_close_record_marks_the_failure_path_elided(tmp_path):
    """The failure payload is capped AND middle-elided with nothing
    persisted, so honesty about fidelity is the only thing on offer."""
    payload = pre_payload()
    payload.update(hook_event_name="PostToolUseFailure",
                   tool_response=None, error="boom" * 100)
    run_hook(payload, tmp_path)
    (record,) = records(tmp_path)
    assert record["outcome"] == "failed"
    assert record["output"]["fidelity"] == "elided"
    assert record["output"]["source"] == "error_field"


def test_output_blobs_are_content_addressed(tmp_path):
    """Re-running the same script fifty times must cost one copy."""
    for i in range(3):
        payload = pre_payload()
        payload.update(hook_event_name="PostToolUse", tool_use_id=f"t{i}",
                       tool_response={"stdout": "identical output"})
        run_hook(payload, tmp_path)
    blobs = list((tmp_path / ".provenance" / "runs" / "s1" / "blobs").rglob("*"))
    assert len([b for b in blobs if b.is_file()]) == 1
    assert len(records(tmp_path)) == 3


def test_remote_exec_snapshots_the_local_file(tmp_path):
    """The one case session transcripts provably lose."""
    script = tmp_path / "spike.py"
    script.write_text("print('gpu')")
    run_hook(pre_payload(f"mighty-colab exec -s gpu1 -f {script}"), tmp_path)
    (record,) = records(tmp_path)
    (ref,) = record["referenced_files"]
    assert ref["existed"] and ref["bytes"] == 12
    blob = tmp_path / ".provenance" / "runs" / "s1" / ref["blob"]
    assert blob.read_text() == "print('gpu')", "snapshot did not keep the bytes"
    print(f"[capture] snapshotted {ref['path']} -> {ref['blob']}")


def test_a_missing_referenced_file_is_recorded_not_fatal(tmp_path):
    """A path that already vanished is exactly the situation of interest."""
    run_hook(pre_payload("mighty-colab exec -s g -f /tmp/gone-12345.py"), tmp_path)
    (record,) = records(tmp_path)
    (ref,) = record["referenced_files"]
    assert ref["existed"] is False and "error" in ref


def test_the_log_is_append_only_across_invocations(tmp_path):
    """Each hook call is a separate process; the log must accumulate."""
    run_hook(pre_payload(), tmp_path)
    payload = pre_payload()
    payload.update(hook_event_name="PostToolUse",
                   tool_response={"stdout": "42\n"})
    run_hook(payload, tmp_path)
    got = records(tmp_path)
    assert [r["phase"] for r in got] == ["open", "close"]
    assert got[0]["tool_use_id"] == got[1]["tool_use_id"], "records cannot be joined"


def test_sessions_do_not_share_a_log(tmp_path):
    """Run-scoped means run-scoped."""
    run_hook(pre_payload(session="alpha"), tmp_path)
    run_hook(pre_payload(session="beta"), tmp_path)
    assert len(records(tmp_path, "alpha")) == 1
    assert len(records(tmp_path, "beta")) == 1


def test_the_hook_prints_only_suppress_output(tmp_path):
    """A forensic hook must not narrate itself into the transcript."""
    proc = run_hook(pre_payload(), tmp_path)
    assert json.loads(proc.stdout.strip()) == {"suppressOutput": True}
