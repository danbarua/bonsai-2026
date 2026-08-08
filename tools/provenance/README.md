# provenance tools

Instruments for the load-bearing-scratch capture work. The hooks themselves
live in `.claude/hooks/provenance-capture/`; this directory holds the things
that **measure** them, plus the one tool that reads their output.

Nothing here runs automatically. Two of these files are one-off measurement
instruments rather than live infrastructure — the distinction matters,
because mistaking a probe for a running component is how somebody concludes
the system does something it does not.

## Ongoing tools

**`capture_stats.py`** — summarise capture logs: record counts, trigger
reasons, output sizes, fidelity, and sessions that wrote records without a
`session_open` marker. `--session <prefix>` narrows to one run.

It exists so figures about the logs come from committed code rather than an
inline `python -c` whose output lands in a document. It carries a boundary
worth repeating: reading the run-scoped logs does **not** make its output
citable. The logs are a leaf and nothing committed may descend from them.
What it legitimately describes is the log's own shape — for sizing
parameters and answering "is retention right yet". A number about the
*subject* of a capture still has to come from committed code that
regenerates it.

**`bench_hook_overhead.py`** — how much latency the hooks add to a tool
call. Run it when the answer might have changed; the current figure and the
reasoning about whether it is acceptable are in the hooks' own README.

## One-off measurement instruments

These answered specific questions and are kept so the answers can be
re-derived rather than trusted. They are not part of any running path.

**`emit_bytes.py`** — emits a position-labelled stream ending in a sentinel
carrying the digest of everything before it. From whatever survives, a
reader can tell how much arrived, from which end, and whether it is
byte-identical to what was sent. A uniform stream cannot distinguish a
truncated result from a complete one; this can.

**`probe_hook_payload.py`** — a `PostToolUse` hook that records its own
input instead of acting on it. Answers what a hook actually *receives*.

**`probe_hook_registration.json`** — a documentation fixture showing how to
register that probe. **Not live configuration.** The real registrations are
in `.claude/settings.json`.

**`run_truncation_probe.py`** — drives the pair through a headless
`claude -p` subprocess with its own `--settings`, so the measurement is
reproducible from committed code instead of from "edit your settings, then
type these four commands". Its findings — a 30,000-character inline cap
with a byte-exact persisted copy above it, stderr folded into stdout, and a
failure path carrying no `tool_response` at all — are what the capture
design is built on. They are properties of one harness version, not a
contract: re-run rather than trust.

## Why the probes are committed at all

A number that anchors a decision is either reproducible from committed
code, or it is not a decision anchor (CLAUDE.md principle 24). Every
truncation figure the design quotes came from running these. Deleting them
once the answer was known would have turned those figures into exactly the
load-bearing scratch this whole feature exists to prevent — measurements
whose generator was a chat transcript.

## Tests

`tests/test_provenance_probe.py` pins the instruments themselves;
`tests/test_provenance_capture_stats.py` pins the log summariser, including
a vacuity guard, because an extraction that silently returns nothing
produces a clean and entirely false report.
