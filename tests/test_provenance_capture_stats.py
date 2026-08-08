"""`capture_stats.py` reads the forensic log. Keep its extraction honest.

The script summarises capture logs so that figures about them come from
committed code rather than an inline `python -c`. That makes its own
extraction load-bearing in a specific way: **an extraction that silently
returns nothing produces a clean, plausible report of nothing**, and every
figure in it would be vacuously correct. Same shape as the vacuity guard in
`test_stage2b_negative_path_evidence.py`, and the reason that guard exists.

Fixtures are synthetic and written to `tmp_path`. Nothing here reads the
real `.provenance/` logs — they are machine-local and gitignored, so a test
depending on them would pass or fail according to what this machine
happened to have run, which is the opposite of a test.
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "provenance" / "capture_stats.py"

spec = importlib.util.spec_from_file_location("_capture_stats", SCRIPT)
capture_stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture_stats
spec.loader.exec_module(capture_stats)


def write_log(root: Path, session: str, records: list[dict]) -> None:
    log = root / ".provenance" / "runs" / session / "capture.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in records))


def marker(**kw):
    return {"phase": "session_open", "source": "startup",
            "hook_version": "1.1.0",
            "git": {"commit": "abc123", "branch": "b", "dirty": False}, **kw}


def opened(reason="inline_c", **kw):
    return {"phase": "open", "trigger_reason": reason,
            "tool_use_id": "t1", **kw}


def closed(nbytes=100, fidelity="complete", **kw):
    return {"phase": "close", "tool_use_id": "t1",
            "output": {"bytes": nbytes, "fidelity": fidelity,
                       "source": "inline", "blob": "blobs/aa/bb"}, **kw}


def test_load_finds_records_across_sessions(tmp_path):
    """The vacuity guard. Every figure the report prints is over this set;
    a `load` returning nothing would make all of them true and empty."""
    write_log(tmp_path, "s1", [marker(), opened(), closed()])
    write_log(tmp_path, "s2", [marker(), opened(), closed()])
    records = capture_stats.load(tmp_path)
    assert len(records) == 6, f"expected 6 records, got {len(records)}"
    assert {r["_session"] for r in records} == {"s1", "s2"}


def test_every_record_is_tagged_with_its_session(tmp_path):
    """The unmarked-session check joins on this tag; an untagged record
    would make that check silently inert."""
    write_log(tmp_path, "s1", [marker(), opened()])
    assert all(r.get("_session") for r in capture_stats.load(tmp_path))


def test_a_partial_final_line_does_not_lose_the_whole_log(tmp_path):
    """A session killed mid-write leaves a truncated last line. That is the
    case this tool most needs to survive, since a dead session is exactly
    what capture exists for."""
    log = tmp_path / ".provenance" / "runs" / "s1" / "capture.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps(marker()) + "\n" + '{"phase": "open", "trig')
    records = capture_stats.load(tmp_path)
    assert len(records) == 1, "a partial line discarded the intact records"


def test_missing_log_directory_is_not_an_error(tmp_path):
    assert capture_stats.load(tmp_path) == []


def test_session_filter_narrows_by_prefix(tmp_path):
    """Answering "is capture live in THAT session" needs a prefix match --
    session ids are UUIDs and nobody retypes one correctly."""
    write_log(tmp_path, "abc12345-aaaa", [marker(), opened()])
    write_log(tmp_path, "def67890-bbbb", [marker(), opened()])
    assert len(capture_stats.load(tmp_path, "abc")) == 2
    assert {r["_session"] for r in capture_stats.load(tmp_path, "abc")} \
        == {"abc12345-aaaa"}


def test_session_filter_matching_nothing_returns_empty_not_everything(tmp_path):
    """The dangerous failure: a filter that silently falls back to
    unfiltered would report another session's records as the one asked
    about -- worse than reporting nothing, because it looks like an answer."""
    write_log(tmp_path, "abc12345", [marker(), opened()])
    assert capture_stats.load(tmp_path, "zzz") == []


def test_unmarked_sessions_are_reported(tmp_path, capsys):
    """A session with records but no marker was capturing without a commit
    point -- the ambiguity the marker removes. It must be visible, not
    inferred by a reader counting directories."""
    write_log(tmp_path, "marked", [marker(), opened(), closed()])
    write_log(tmp_path, "unmarked", [opened(), closed()])
    capture_stats.report(capture_stats.load(tmp_path))
    out = capsys.readouterr().out
    assert "WARNING" in out and "unmarked" in out
    assert "marked" in out


def test_no_warning_when_every_session_is_marked(tmp_path, capsys):
    """The non-vacuity half: a warning that always fires reports nothing."""
    write_log(tmp_path, "s1", [marker(), opened(), closed()])
    capture_stats.report(capture_stats.load(tmp_path))
    assert "WARNING" not in capsys.readouterr().out


def test_size_statistics_describe_the_close_records(tmp_path, capsys):
    write_log(tmp_path, "s1", [marker()]
              + [closed(nbytes=n) for n in (10, 20, 30, 40, 1000)])
    capture_stats.report(capture_stats.load(tmp_path))
    out = capsys.readouterr().out
    assert "1,000" in out, "max not reported"
    assert "min" in out and "median" in out


def test_quantile_is_order_statistic_not_interpolated(tmp_path):
    """Reported quantiles must be values that actually occurred -- an
    interpolated p99 is a number no capture ever produced."""
    values = [1, 2, 3, 100]
    assert capture_stats._quantile(values, 1.0) == 100
    assert capture_stats._quantile(values, 0.0) == 1
    assert capture_stats._quantile(values, 0.5) in values
    assert capture_stats._quantile([], 0.5) is None


def test_truncation_count_is_reported_even_when_zero(tmp_path, capsys):
    """Whether the ceiling has ever bound is the question that decides if it
    is a policy choice or a measured threshold. Zero is the informative
    answer, so it must be printed rather than omitted."""
    write_log(tmp_path, "s1", [marker(), closed()])
    capture_stats.report(capture_stats.load(tmp_path))
    assert "records truncated by capture: 0" in capsys.readouterr().out
