"""The provenance probe's instruments, pinned.

`tools/provenance/` measures what a `PostToolUse` hook receives, and
`docs/PROVENANCE_CONTRACT.md` §2 quotes those measurements as
the substrate the whole capture design sits on. The instruments therefore
need the same treatment as any other load-bearing code here: if
`emit_bytes.build_stream` stops producing a self-describing stream, or
`probe_hook_payload.analyse` stops distinguishing a complete result from a
truncated one, every number in §2 silently becomes unreproducible and
nothing says so.

`build_stream`'s docstring already promised this test ("returned rather
than printed so a test can build it without spawning a process") -- a
promise that shipped in 6e4adf0 with nothing behind it, which is the same
dangling-citation defect the contract exists to catch, one layer down.

Tier 1 only, per the convention in CLAUDE.md: self-contained structural
tests on synthetic data, no subprocess, no `claude` on PATH, no network.
The Tier 2 counterpart -- driving a real headless session -- is
`run_truncation_probe.py` itself, run deliberately.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = REPO_ROOT / "tools" / "provenance"


def _load(name: str):
    """Import a `tools/provenance/` module by path.

    They are deliberately not a package: `probe_hook_payload.py` has to run
    as a bare hook command under whatever interpreter the harness spawns,
    with no path setup and no dependency on this repo being importable.
    """
    path = PROVENANCE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_provenance_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


emit_bytes = _load("emit_bytes")
probe_hook = _load("probe_hook_payload")


# --- the emitter -----------------------------------------------------------

@pytest.mark.parametrize("n_bytes", [0, 500, 25_000, 200_000])
def test_stream_is_whole_lines_plus_a_sentinel(n_bytes):
    """Byte count is predictable, so "how much survived" reads as a line count."""
    stream = emit_bytes.build_stream(n_bytes)
    per_line = emit_bytes.LINE_WIDTH + 1
    expected_lines = n_bytes // per_line
    body_len = expected_lines * per_line
    assert stream[:body_len].count("\n") == expected_lines
    assert stream[body_len:].startswith(emit_bytes.SENTINEL)
    assert stream.endswith("\n")


def test_sentinel_digest_describes_the_body_that_precedes_it():
    """The self-description is the whole instrument.

    Every completeness verdict downstream -- in the hook, in the driver's
    report, and in the contract's byte-exactness claim -- reduces to this
    digest matching. If the sentinel described anything other than the body
    before it, `sha_ok` would be measuring nothing.
    """
    stream = emit_bytes.build_stream(50_000)
    index = stream.rindex(emit_bytes.SENTINEL)
    body, sentinel = stream[:index], stream[index:]
    claimed = sentinel.split("body_sha256=")[1].strip()
    assert hashlib.sha256(body.encode()).hexdigest() == claimed
    assert f"body_bytes={len(body.encode())}" in sentinel


def test_no_two_lines_are_identical():
    """Uniform filler would make a line that lost its middle look intact."""
    body = emit_bytes.build_stream(20_000).splitlines()[:-1]
    assert len(set(body)) == len(body)
    assert body[0].startswith(f"{emit_bytes.PREFIX}00000000/")


# --- the analyser ----------------------------------------------------------

def test_analyse_calls_a_complete_stream_complete():
    result = probe_hook.analyse(emit_bytes.build_stream(20_000))
    assert result["sentinel_present"] is True
    assert result["body_matches_sentinel"] is True
    assert result["contiguous"] is True
    assert result["first_line_index"] == 0


def test_analyse_detects_a_dropped_tail():
    """The success-path cap: head kept, tail gone, and nothing says so.

    Distinguishing this from a complete stream is the reason the capture
    hook compares against `persistedOutputSize` instead of trusting the
    inline field.
    """
    result = probe_hook.analyse(emit_bytes.build_stream(200_000)[:30_000])
    assert result["sentinel_present"] is False
    assert result["body_matches_sentinel"] is None
    # Still contiguous -- a truncated head is NOT detectable from contiguity
    # alone, which is exactly why the sentinel exists.
    assert result["contiguous"] is True
    assert result["first_line_index"] == 0


def test_analyse_detects_an_elided_middle():
    """The failure-path shape, which is a different loss from a dropped tail.

    Contiguity is the only signal separating them, so it gets its own test:
    a scheme that reported `contiguous` True here would describe the
    failure path as if it were the success path.
    """
    lines = emit_bytes.build_stream(200_000).splitlines(keepends=True)
    spliced = "".join(lines[:50] + lines[-50:])
    result = probe_hook.analyse(spliced)
    assert result["contiguous"] is False
    assert result["first_line_index"] == 0
    assert result["last_line_index"] > 50


def test_analyse_survives_an_empty_stream():
    """The failure path delivered no text at all; the analyser must not raise."""
    result = probe_hook.analyse("")
    assert result["len"] == 0
    assert result["n_labelled_lines"] == 0
    assert result["first_line_index"] is None
    assert result["contiguous"] is False


# --- payload shape ---------------------------------------------------------

def test_extract_text_reports_which_field_it_read():
    """An observation that does not name its source field is not evidence.

    The measured payload puts fd-2 output in `stdout` and leaves `stderr`
    empty; a reader can only check that claim if the record says which key
    the text came out of.
    """
    assert probe_hook.extract_text({"stdout": "abc", "stderr": ""}) == ("stdout", "abc")
    assert probe_hook.extract_text({"stdout": "", "stderr": "boom"}) == ("stderr", "boom")
    assert probe_hook.extract_text("raw") == ("<str>", "raw")
    assert probe_hook.extract_text(None) == (None, "")


def test_observe_handles_a_failure_payload_with_no_tool_response():
    """`PostToolUseFailure` carries `tool_response: null` and an `error` key.

    Pinned because it is the asymmetry the Pre/Post pair exists for, and a
    schema change here would quietly turn the failure path into a blank
    record rather than an error.
    """
    record = probe_hook.observe({
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "uv run python x.py"},
        "tool_response": None,
        "error": emit_bytes.build_stream(2_000),
    })
    assert record["tool_response_type"] == "NoneType"
    assert record["text_len"] == 0
    assert record["error_len"] > 0
    assert record["error"]["sentinel_present"] is True
    assert record["persisted_exists"] is False


def test_the_probe_hook_self_filters_on_its_own_marker():
    """It gets registered in a settings file a parallel session reads.

    A broken `if` clause must cost a missing measurement, never a log of
    somebody else's commands -- so the marker has to match the emitter's
    real path rather than drifting from it.
    """
    assert probe_hook.PROBE_MARKER in "uv run python tools/provenance/emit_bytes.py --bytes 5"
    assert (PROVENANCE_DIR / "emit_bytes.py").as_posix().endswith(probe_hook.PROBE_MARKER)
