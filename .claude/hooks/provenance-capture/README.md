# provenance-capture hooks

Records *load-bearing scratch* — an ephemeral script (heredoc, `python -c`,
tmp file, inline remote exec) whose output can enter a durable record while
the script itself dies with the session. CLAUDE.md principle 24; full design
in `docs/proposals/PROVENANCE_CONTRACT.md`.

Registered project-wide in `.claude/settings.json` on three events, all
matching `Bash|mcp__mighty-colab__.*`:

- **`PreToolUse`** — writes the `open` record: the script text itself, git
  state, and content-addressed snapshots of any local file about to be
  shipped to a remote kernel. Written *before* the call runs.
- **`PostToolUse`** — writes the `close` record: output, fidelity, duration.
- **`PostToolUseFailure`** — `close` record with `outcome: failed`.

All three run `capture.sh`, which wraps `capture.py`. `scratch_predicate.py`
is the sole authority on what counts as scratch; the hook adds no patterns
of its own.

## Design decisions

**The Pre/Post split is measurement, not caution.** A *failing* Bash call
delivers no `tool_response` to a hook at all — its output arrives under
`error`, capped at 10,040 characters, with the middle elided and nothing
persisted to disk. A *succeeding* call over 30,000 characters is capped
inline but carries a byte-exact `persistedOutputPath`. So the one case this
tool exists for — a scratch script that dies mid-run — is precisely the one
no post-hoc event can reconstruct. `PreToolUse` is where the script is
available unconditionally, and the script is the asset that supports
promotion to committed code. Measured by
`tools/provenance/run_truncation_probe.py`; rerun it rather than trusting
these numbers, which are properties of one harness version.

**No command rewriting.** An early draft intercepted at `PreToolUse` and
rewrote the command to tee its output. `persistedOutputPath` made that
unnecessary, which is a large win: rewriting arbitrary Bash has real blast
radius (quoting, pipelines, `pipefail`, exit-code propagation, background
runs), and a capture mechanism that changes what the user's command *does*
is a worse failure than an incomplete log.

**Bytes are copied, never referenced.** The harness's persisted output lives
under `~/.claude/projects/<project>/<session>/tool-results/` and is reaped.
A record holding a pointer into an ephemeral directory would be a cached
conclusion whose source disappears — the exact failure this tool exists to
prevent, recreated inside it.

**Fidelity is recorded, never assumed.** `complete` / `capped` / `elided` /
`truncated_by_capture` / `lost`. Inline truncation is *unmarked* by the
harness — the text simply stops mid-line — so a hook that reads `stdout`
without comparing against `persistedOutputSize` cannot tell a whole result
from a cut one. An auditor reading `elided` knows the record is partial; one
reading a silently-truncated log does not.

**It never blocks, and that is enforced twice.** `capture.py` exits 0 on
every path including its own bugs; `capture.sh` ends in an unconditional
`exit 0` so the guarantee survives failures that stop Python running at all
(no `python3` on PATH, unreadable script, import error). Same
belt-and-braces reasoning as the c2c-mail hooks next door. A single `exit 2`
from a forensic hook would teach the route-around this whole design exists
to avoid.

**Ordinary work is not captured, and the negative cases are load-bearing.**
Committed scripts, `pytest`, `make`, `git`, and read/navigate commands write
nothing. Capturing everything produces a log nobody reads — the same outcome
as capturing nothing, but with more disk — and would put an unrelated
session's commands into a forensic log, which is a privacy question rather
than a tidiness one.

**The predicate is a pattern list and says so.** Principle 21 warns that a
hand-maintained list standing in for a derivable set will silently
under-cover. No derivation exists here: nothing enumerates "ways to run an
ephemeral script". The mitigation is the corpus in
`tests/test_provenance_capture.py`, where every entry carries the reason it
is there. A new shape found in the wild gets a corpus entry *and* a rule,
never a rule alone.

## Testing

```bash
uv run pytest tests/test_provenance_capture.py tests/test_provenance_capture_hook.py
```

`test_provenance_capture.py` is the predicate corpus. `..._hook.py` drives
the hook as a real subprocess — JSON on stdin, exit code and filesystem
effects observed — because the properties that matter are process-level and
a direct function call cannot show them.

Every guard here has been watched to fail. Two are worth recording, because
both found a real defect rather than confirming an expectation:

**The corpus completeness check failed on its first run**, catching two MCP
rules that had ad-hoc tests but no worked example. A guard that fails on
real content unprompted is better evidence than a staged break.

**The fail-open tests were initially vacuous, and the break-test is what
exposed it.** Returning 2 from `capture.py`'s exception handler left every
fail-open test *green*, because `capture.sh`'s unconditional `exit 0` masks
any exit code beneath it. The tests were only ever exercising the wrapper.
Fixed by parameterising each fail-open case over both layers — via the
wrapper *and* invoking `capture.py` directly — after which the same break
fails exactly the `python-directly` variants and correctly leaves the
`via-wrapper` ones passing. This is principle 21's second half at hook
scale: verifying a narrow thing through the broader thing that wraps it
proves the broader thing works and says nothing about the narrow one.
