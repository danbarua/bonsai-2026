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

## Hook registrations are loaded once per session

**A session that has neither been restarted nor had its config refreshed
since these hooks landed captures nothing, silently.** Confirmed live:
after the merge to `stage2b` at `07a33a0`, a scratch command in an
already-open session produced no record, while the same command in a
freshly-started session produced a correct `open`/`close` pair.

A **config refresh is enough** — a full restart is not required. Verified
in the field by `stage2b-lead`, whose session predates `07a33a0` and
captures correctly after an in-session refresh. Worth stating precisely,
because "restart" is the more expensive instruction and people skip
expensive instructions.

This matters more than an ordinary gotcha, because the failure is quiet in
the worst direction. Nothing errors. The absence of a record is
indistinguishable from "that command was correctly classified as not
scratch", so a long-running session can look like it is being captured when
it never was — and the natural time to have a long-running session is
during exactly the sustained work most worth capturing.

To confirm capture is live in a session, look for the **`session_open`
marker** rather than for records of your own commands:

```bash
grep -c session_open .provenance/runs/<session_id>/capture.jsonl
```

Check the marker, not a canary command. An earlier version of this note
suggested running a throwaway `python -c` and treating an empty directory
as proof the hooks were dead. That was unsafe advice: at the time
`uv run python -c` was not captured at all — this project's canonical
invocation — so the check reported "not live" against perfectly live hooks.
The predicate bug is fixed, but the lesson outlives it: **a canary tests
the predicate and the wiring at once, and cannot tell you which one
failed.** The marker tests only the wiring, which is the question being
asked.

`session_id` appears in every record, so an empty or missing directory
after a known-scratch command means the hooks are not loaded in that
session rather than that the predicate declined.

The same check is mechanical in `tests/test_provenance_live_registration.py`,
which spawns a real headless session and inspects what the hooks wrote —
the only way to test the wiring rather than the hook, since a registration
needs a session by construction. Slow and CLI-dependent, so it is
`@pytest.mark.slow` and excluded from the default suite. **Run it after any
change to `.claude/settings.json` or to the scripts it names**, which is
exactly when the wiring can break without a single unit test noticing.

## Cost: about 77 ms per Bash tool call, everywhere

The hook is registered on `Bash|mcp__mighty-colab__.*`, so it runs on every
matching tool call in every session in this repository — not only the ones
it captures. Measured by `tools/provenance/bench_hook_overhead.py`: ~38 ms
for the uncaptured case, and since both `PreToolUse` and `PostToolUse` fire,
**roughly 77 ms is added to each Bash call**. Almost all of it is Python
interpreter startup; the predicate itself is negligible.

This is recorded rather than buried because latency is a correctness
property for this design, not a nicety. The hooks were accepted on the
argument that they never block and never surprise, and the natural response
to tooling that makes your own work feel sluggish is to switch it off — the
route-around this feature exists to prevent. A cost imposed on other
people's sessions should be visible to them.

**Why it is not optimised away.** The obvious fix is a fast substring check
in `capture.sh` — bash string matching costs ~1 ms — exiting before Python
starts. That would put a second copy of the predicate in a second language,
and when the two disagree the result is silent under-capture: exactly the
defect this feature has already shipped twice, and the one hardest to
notice because a missing record looks like correct classification.

Narrowing the settings matcher with `if` clauses has the same problem in a
third place. The broad matcher plus an in-process self-filter was chosen
deliberately so that the predicate has exactly one definition.

~77 ms against a tool call that typically costs hundreds of milliseconds to
seconds is single-digit percent. If that ever stops being true, the number
is here to be re-measured rather than re-guessed, and the trade is worth
revisiting — but the fix must derive the fast path from the predicate, not
restate it.

## Known blind spot: remote execs launched through `make`

**A GPU run started by `make stage2b-ladder-stage3` is not captured**, and
that is a known limitation rather than an oversight.

Every GPU target in this repository launches through a Makefile recipe, so
the tool call the hook sees is `Bash("make <target>")` and the
`mighty-colab exec -f` runs in a subprocess make spawns. The predicate
handles `exec -f` correctly — verified directly — but nothing ever hands it
that string. `make` is correctly classified as ordinary work, and the exec
behind it is invisible.

**What still covers the load-bearing claim.** The Stage 2B drivers verify
`BONSAI_DRIVER_SHA256` against the commit's copy on the remote, and the
closure check refuses to launch when any file in the driver's import closure
differs from HEAD. So "which revision reached the GPU" is answerable from the
run report without these hooks. What capture would have added is an
*independent* witness — the script text as sent, and the `git.dirty` flag,
recorded by something other than the driver vouching for itself. Defence in
depth, not the only evidence.

**Why the obvious fix is not taken.** Expanding make targets in the
predicate would mean either reimplementing make's variable expansion — the
reimplement-instead-of-call failure CLAUDE.md principle 16 exists for — or
shelling out to `make -n`. The second is worse than it looks: `$(shell ...)`
expansions execute at parse time even under `-n`, so a hook classifying a
command would run arbitrary shell as a side effect. In a hook that must
never block and must never surprise, that is disqualifying.

Bypassing make to invoke the exec directly also loses the target's closure
check, its refusal to run a commit absent from a remote, and its
unconditional teardown — strictly worse than the gap.

The right eventual fix is capture at the `mighty-colab` layer, which knows
it is shipping a local file regardless of who invoked it.

Pinned by `test_make_wrapped_remote_exec_is_a_known_blind_spot`, so the
limitation stays visible in the suite rather than becoming folklore.

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
