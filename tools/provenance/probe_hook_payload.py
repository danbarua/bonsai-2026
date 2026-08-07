#!/usr/bin/env python3
"""A `PostToolUse` hook that measures its own input instead of acting on it.

Answers, empirically: what does a hook actually RECEIVE when the tool it
fires on produced a large stdout? The provenance-capture design needs to
know whether `tool_response` carries the tool's complete output or a
truncated copy, because that determines whether stdout can be captured from
the hook payload at all.

Reads the hook JSON on stdin, writes one observation record per invocation
to a JSONL log, and exits 0 unconditionally. It never blocks, never edits
the tool result, and prints `suppressOutput` so a measurement run does not
narrate itself into the transcript it is measuring.

Pair with `emit_bytes.py`, which produces a stream whose surviving fragments
reveal how much arrived and from where. Register with an `if` clause that
matches ONLY the probe command, so registering it cannot affect any other
session's tool calls:

    {"matcher": "Bash", "hooks": [{"type": "command",
      "if": "Bash(* tools/provenance/emit_bytes.py *)",
      "command": "<abs path to this file>"}]}

Log location: `$BONSAI_PROBE_LOG`, else `<repo>/.provenance/probe_payload.jsonl`.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO_ROOT / ".provenance" / "probe_payload.jsonl"

# Mirrors emit_bytes.py's format. Duplicated deliberately rather than
# imported: this file must stay runnable as a bare hook command under
# whatever interpreter the harness spawns, with no path setup and no
# dependency on the repo being importable.
_LINE_RE = re.compile(r"^LINE(\d{8})/(\d{8}) ", re.MULTILINE)
_SENTINEL_RE = re.compile(
    r"EMIT_END lines=(\d+) body_bytes=(\d+) body_sha256=([0-9a-f]{64})")

# Fields a tool result might carry its text in. Recorded by name so the
# observation says WHICH field was read, not merely what was in it.
_TEXT_FIELDS = ("stdout", "text", "content", "output", "stderr")

# Belt as well as braces. The `if` clause in the registration is supposed to
# keep this off every command but the probe's, but this file gets registered
# in a settings file a PARALLEL session also reads -- so it self-filters too,
# and a mistake in the `if` syntax costs a missing measurement rather than a
# log of somebody else's commands.
PROBE_MARKER = "tools/provenance/emit_bytes.py"


def extract_text(tool_response):
    """Return (field_name, text) for the text-bearing part of a result.

    Shape varies by tool and by harness version, so this reports what it
    found rather than assuming a schema.
    """
    if isinstance(tool_response, str):
        return "<str>", tool_response
    if isinstance(tool_response, dict):
        for field in _TEXT_FIELDS:
            value = tool_response.get(field)
            if isinstance(value, str) and value:
                return field, value
    return None, ""


def analyse(text: str) -> dict:
    """Position and completeness evidence for one stream's text.

    Kept separate from `observe` so stdout and stderr get the identical
    treatment -- the question "is stderr capped the same way as stdout" is
    only answerable if both are measured the same way.
    """
    indices = [int(m.group(1)) for m in _LINE_RE.finditer(text)]
    sentinel = _SENTINEL_RE.search(text)
    complete = None
    if sentinel:
        body = text[:sentinel.start()]
        complete = hashlib.sha256(body.encode()).hexdigest() == sentinel.group(3)
    return {
        "len": len(text),
        "n_labelled_lines": len(indices),
        "first_line_index": indices[0] if indices else None,
        "last_line_index": indices[-1] if indices else None,
        "contiguous": (bool(indices)
                       and indices == list(range(indices[0], indices[-1] + 1))),
        "sentinel_present": bool(sentinel),
        "body_matches_sentinel": complete,
    }


def observe(payload: dict) -> dict:
    """Everything about this invocation worth keeping, as a flat record."""
    tool_response = payload.get("tool_response")
    field, text = extract_text(tool_response)

    indices = [int(m.group(1)) for m in _LINE_RE.finditer(text)]
    sentinel = _SENTINEL_RE.search(text)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "hook_event_name": payload.get("hook_event_name"),
        "tool_name": payload.get("tool_name"),
        "tool_use_id": payload.get("tool_use_id"),
        "session_id": payload.get("session_id"),
        "command": (payload.get("tool_input") or {}).get("command"),
        # How big the whole hook payload was, so a cap on the PAYLOAD is
        # distinguishable from a cap on the text field.
        "payload_json_bytes": len(json.dumps(payload)),
        # Top-level shape, recorded because the failure event turned out to
        # carry no `tool_response` at all -- so where its content lives is a
        # question the payload itself has to answer.
        "payload_keys": sorted(payload),
        "payload_field_sizes": {
            k: len(json.dumps(v)) for k, v in sorted(payload.items())},
        "tool_response_type": type(tool_response).__name__,
        "tool_response_keys": (sorted(tool_response)
                               if isinstance(tool_response, dict) else None),
        "text_field": field,
        "text_len": len(text),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_head": text[:160],
        "text_tail": text[-160:],
        # Position evidence: which labelled lines survived, and whether the
        # run is contiguous (head kept) or has a gap (middle elided).
        "n_labelled_lines": len(indices),
        "first_line_index": indices[0] if indices else None,
        "last_line_index": indices[-1] if indices else None,
        "line_index_contiguous": (
            bool(indices) and indices == list(range(indices[0], indices[-1] + 1))),
        "sentinel_present": bool(sentinel),
        "sentinel_claims_lines": int(sentinel.group(1)) if sentinel else None,
        "sentinel_claims_body_bytes": int(sentinel.group(2)) if sentinel else None,
        "sentinel_body_sha256": sentinel.group(3) if sentinel else None,
    }

    # stdout and stderr measured independently and identically. A cap that
    # applies to one is not evidence about the other, and for a scratch
    # script that dies the traceback on stderr IS the record worth keeping.
    if isinstance(tool_response, dict):
        record["stdout"] = analyse(tool_response.get("stdout") or "")
        record["stderr"] = analyse(tool_response.get("stderr") or "")

    # PostToolUseFailure carries no `tool_response` at all; the output of the
    # call that died arrives under `error` instead, and nothing is persisted
    # to disk. Measured separately because that asymmetry is the whole reason
    # capture cannot rely on a single post-hoc event.
    err_field = payload.get("error")
    record["error_len"] = len(err_field) if isinstance(err_field, str) else None
    record["error"] = analyse(err_field) if isinstance(err_field, str) else None

    # The escape hatch. When stdout exceeds the inline cap the harness
    # persists the full output to a file and names it here, so a hook can
    # recover what the inline field lost -- WITHOUT rewriting the command.
    # Measured, not assumed: read the file back and check it against the
    # stream the emitter says it sent.
    persisted = None
    if isinstance(tool_response, dict):
        persisted = tool_response.get("persistedOutputPath")
    record["persisted_output_path"] = persisted
    record["persisted_output_size"] = (
        tool_response.get("persistedOutputSize")
        if isinstance(tool_response, dict) else None)
    record["persisted_exists"] = False
    record["persisted_len"] = None
    record["persisted_sentinel_present"] = None
    record["persisted_body_matches_sentinel"] = None
    if persisted:
        try:
            full = Path(persisted).read_text()
            record["persisted_exists"] = True
            record["persisted_len"] = len(full)
            psent = _SENTINEL_RE.search(full)
            record["persisted_sentinel_present"] = bool(psent)
            if psent:
                # The sentinel carries the digest of the body that preceded
                # it. Recomputing it here is what makes "complete" a checked
                # claim rather than a length that looked about right.
                body = full[:psent.start()]
                record["persisted_body_matches_sentinel"] = (
                    hashlib.sha256(body.encode()).hexdigest() == psent.group(3))
        except Exception as exc:  # noqa: BLE001
            record["persisted_read_error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> int:
    log_path = Path(os.environ.get("BONSAI_PROBE_LOG", DEFAULT_LOG))
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
        command = (payload.get("tool_input") or {}).get("command") or ""
        if PROBE_MARKER in command:
            record = observe(payload)
            record["raw_stdin_bytes"] = len(raw)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 -- see below
        # A measurement instrument that can break the session it measures is
        # worse than no measurement. Record the failure if possible and exit
        # 0 regardless; this is the same rule the real capture hook follows.
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as fh:
                fh.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "probe_error": f"{type(exc).__name__}: {exc}",
                }) + "\n")
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps({"suppressOutput": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
