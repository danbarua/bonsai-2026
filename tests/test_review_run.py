"""`review_run.sh` must not print an empty summary that looks like a quiet run.

The script exists so nobody has to read the Actions web page and paste the
interesting lines into a chat window. That only helps if its failure modes are
loud: an artifact that has EXPIRED (the ordinary end state of a run older than
90 days) yields no telemetry at all, and a summary with every field blank
reads as "this review did nothing" rather than "this record aged out".

These run the real script with a stub `gh` on PATH. The stub emulates `gh`'s
own `--jq`/`-q` filtering for the specific calls the script makes, because
that filtering happens inside `gh` and a stub that ignored it would be testing
a different program.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "ci" / "review_run.sh"

RUN_ID = 31271970194

RUN_VIEW = {
    "databaseId": RUN_ID,
    "displayTitle": "Merge `stage2b` onto `stage2b-ci`",
    "headBranch": "stage2b",
    "event": "pull_request",
    "status": "completed",
    "conclusion": "success",
    "createdAt": "2026-08-08T18:27:25Z",
    "updatedAt": "2026-08-08T18:31:00Z",
    "url": f"https://github.com/danbarua/bonsai-2026/actions/runs/{RUN_ID}",
}

# Shaped as claude-code-action's uploaded transcript is: an array of entries,
# exactly one of which is the `result` record.
EXECUTION = [
    {"type": "system", "subtype": "init"},
    {"type": "assistant"},
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 169686,
        "num_turns": 26,
        "total_cost_usd": 1.2769137,
        "usage": {"input_tokens": 44, "output_tokens": 13751},
    },
]


def _artifact(*, expired: bool = False) -> dict:
    return {
        "id": 9025916552,
        "name": "claude-review-execution",
        "size_in_bytes": 110859,
        "expired": expired,
        "expires_at": "2026-11-06T18:27:25Z",
        "archive_download_url": (
            "https://api.github.com/repos/danbarua/bonsai-2026/actions/artifacts/9025916552/zip"
        ),
    }


def _write_stub(bin_dir: Path, *, artifacts: list[dict], execution: list | None) -> None:
    """A `gh` that answers exactly the four calls this script makes."""
    (bin_dir / "artifacts.json").write_text(json.dumps(artifacts))
    (bin_dir / "run_view.json").write_text(json.dumps(RUN_VIEW))
    if execution is not None:
        (bin_dir / "execution.json").write_text(json.dumps(execution))

    stub = bin_dir / "gh"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            # `gh run list ... -q '.[0].databaseId'` -> the bare id
            if [ "$1" = "run" ] && [ "$2" = "list" ]; then
              echo {RUN_ID}; exit 0
            fi
            if [ "$1" = "run" ] && [ "$2" = "view" ]; then
              cat {bin_dir / "run_view.json"}; exit 0
            fi
            if [ "$1" = "run" ] && [ "$2" = "download" ]; then
              if [ -f {bin_dir / "execution.json"} ]; then
                for a in "$@"; do
                  if [ "$prev" = "--dir" ]; then
                    cp {bin_dir / "execution.json"} "$a/claude-execution-output.json"
                    exit 0
                  fi
                  prev="$a"
                done
              fi
              exit 1
            fi
            if [ "$1" = "api" ]; then
              # The script filters by artifact name via --jq; the stub file is
              # already the filtered list.
              cat {bin_dir / "artifacts.json"}; exit 0
            fi
            echo "stub gh: unexpected call: $*" >&2
            exit 1
            """
        )
    )
    stub.chmod(0o755)


def _run(
    tmp_path: Path,
    *,
    artifacts: list[dict],
    execution: list | None,
    args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_stub(bin_dir, artifacts=artifacts, execution=execution)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["REVIEW_REPO"] = "danbarua/bonsai-2026"

    return subprocess.run(
        ["bash", str(SCRIPT), *(args or [])],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )


def test_the_telemetry_comes_from_the_runs_own_result_record(tmp_path: Path) -> None:
    """Cost and duration are read, not estimated from token prices."""
    proc = _run(tmp_path, artifacts=[_artifact()], execution=EXECUTION, args=["--json"])
    assert proc.returncode == 0, proc.stderr

    summary = json.loads(proc.stdout)
    assert summary["run"]["id"] == RUN_ID
    assert summary["artifact"]["state"] == "present"

    telemetry = summary["telemetry"]
    assert telemetry["outcome"] == "success"
    assert telemetry["is_error"] is False
    assert telemetry["duration_ms"] == 169686
    assert telemetry["num_turns"] == 26
    assert telemetry["total_cost_usd"] == pytest.approx(1.2769137)
    assert telemetry["output_tokens"] == 13751


def test_the_artifact_download_url_is_derived_not_pasted(tmp_path: Path) -> None:
    """The whole point: the link comes from the API, not from a human."""
    proc = _run(tmp_path, artifacts=[_artifact()], execution=EXECUTION, args=["--json"])
    artifact = json.loads(proc.stdout)["artifact"]

    assert artifact["id"] == 9025916552
    assert artifact["download_url"].endswith("/artifacts/9025916552/zip")
    assert artifact["expires_at"] == "2026-11-06T18:27:25Z"


@pytest.mark.parametrize(
    "artifacts, execution, expected_state",
    [
        ([_artifact(expired=True)], EXECUTION, "expired"),
        ([], None, "missing"),
        ([_artifact()], None, "download-failed"),
    ],
)
def test_an_unavailable_artifact_says_so_instead_of_printing_blanks(
    tmp_path: Path, artifacts: list[dict], execution: list | None, expected_state: str
) -> None:
    """Absent telemetry must be distinguishable from a run that did nothing.

    This is the same distinction the delta script had to learn: two states
    that render identically, where only one is reassuring.
    """
    proc = _run(tmp_path, artifacts=artifacts, execution=execution, args=["--json"])
    assert proc.returncode == 0, proc.stderr

    summary = json.loads(proc.stdout)
    assert summary["artifact"]["state"] == expected_state
    assert summary["telemetry"] is None

    human = _run(tmp_path, artifacts=artifacts, execution=execution)
    assert "NOT AVAILABLE" in human.stdout
    assert expected_state in human.stdout
    assert "not because the run did nothing" in human.stdout


def test_a_failed_review_run_is_reported_as_an_error(tmp_path: Path) -> None:
    """A red run must not read as a green one."""
    failed = [
        entry
        if entry["type"] != "result"
        else {**entry, "subtype": "error_during_execution", "is_error": True}
        for entry in EXECUTION
    ]
    proc = _run(tmp_path, artifacts=[_artifact()], execution=failed, args=["--json"])

    telemetry = json.loads(proc.stdout)["telemetry"]
    assert telemetry["is_error"] is True
    assert telemetry["outcome"] == "error_during_execution"

    human = _run(tmp_path, artifacts=[_artifact()], execution=failed)
    assert "(ERROR)" in human.stdout


def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK), "review_run.sh must be executable"
