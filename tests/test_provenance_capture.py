"""The scratch-detection corpus.

`is_scratch` decides what the provenance capture hook records. Its rule set
is a pattern list, and CLAUDE.md principle 21 is explicit that a
hand-maintained list standing in for a derivable set will silently
under-cover -- four times in this project already. There is no derivation
available here: nothing enumerates "ways to run an ephemeral script".

So the list is paid for with this corpus instead. Every entry is an
invocation shape this project has actually produced, carrying the reason it
is in the corpus. The rule for extending it: a new scratch shape found in
the wild gets a corpus entry AND a rule, never a rule alone -- a rule
without an example leaves the next reader unable to tell what it was for,
which is how a pattern list rots into folklore.

The negative corpus matters as much as the positive one. Capturing
everything produces a log nobody reads, which is the same outcome as
capturing nothing but with more disk -- and it would put unrelated sessions'
commands into a forensic log, which is a privacy question, not a tidiness
one.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_DIR = REPO_ROOT / ".claude" / "hooks" / "provenance-capture"


def _load(name: str):
    path = HOOK_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_prov_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


predicate = _load("scratch_predicate")
is_scratch = predicate.is_scratch


def bash(command: str):
    return is_scratch("Bash", {"command": command})


# --- positive corpus -------------------------------------------------------
# (command, expected rule, why this shape is in the corpus)
SCRATCH = [
    (
        'python -c "import numpy; print(numpy.std(x))"',
        "inline_c",
        "the canonical shape principle 24 names first",
    ),
    (
        'uv run python -c "print(1)"',
        "inline_c",
        "how this repo actually launches python -- a bare `python -c` rule "
        "would miss every real invocation, since MEMORY.md prefers uv run",
    ),
    (
        'uv run python -c "print(1)" > out.txt',
        "inline_c",
        "redirection must not hide the -c; the output going to a file is "
        "if anything a stronger signal the number will be reused",
    ),
    (
        "cat <<'EOF' | uv run python -\nprint(2)\nEOF",
        "heredoc",
        "the heredoc form this session itself reached for twice while "
        "building the tool meant to prevent it",
    ),
    (
        "uv run python /tmp/quick_check.py",
        "ephemeral_path",
        "a tmp script: committed-looking invocation, vanishing generator",
    ),
    (
        "mighty-colab exec -s gpu1 -f /tmp/spike.py",
        "remote_exec_local_file",
        "THE case session transcripts provably lose -- the transcript keeps "
        "the path, the kernel gets the bytes, the tmp file dies",
    ),
    (
        "mighty-colab exec -s gpu1 -f experiments/stage2b_denoising/probe.py",
        "remote_exec_local_file",
        "a COMMITTED path still captured: which revision was shipped is not "
        "recoverable from the transcript on a dirty tree",
    ),
    (
        "mighty-colab run --gpu T4 spike.py",
        "remote_exec_local_file",
        "`run` takes the script positionally, not behind -f",
    ),
    (
        'echo "print(1)" | mighty-colab exec -s gpu1',
        "piped_into_remote_exec",
        "piped code never touches disk at all",
    ),
    (
        'BEFORE=$(wc -l < log); uv run python -c "print(1)"; echo done',
        "inline_c",
        "FIELD REPORT from stage2b-lead: scratch is rarely the first token "
        "of a Bash call. Inspecting only tokens[0] missed this entirely "
        "while the corpus's single-command entries all passed",
    ),
    (
        'cd experiments && uv run python -c "print(1)"',
        "inline_c",
        "the same miss behind a `cd &&`, which is how half the commands in "
        "this repo are written",
    ),
]


@pytest.mark.parametrize(
    "command,rule,why", SCRATCH,
    ids=[c[:44] for c, _, _ in SCRATCH])
def test_scratch_shapes_are_captured(command, rule, why):
    verdict = bash(command)
    assert verdict.capture, f"MISSED scratch ({why}): {command}"
    assert verdict.reason == rule, (
        f"captured for the wrong reason: got {verdict.reason!r}, "
        f"expected {rule!r} -- {why}")


# The MCP surface carries code in named fields rather than a command line,
# so it needs its own corpus rather than a Bash entry that happens to
# mention it. Kept separate and merged for the completeness check below --
# the check found these two rules had ad-hoc tests but no worked example the
# first time it ran, which is the gap it exists for.
# (tool_name, tool_input, expected rule, why)
MCP_SCRATCH = [
    (
        "mcp__mighty-colab__exec", {"code": "print(1)"},
        "mcp_remote_exec_inline",
        "inline code sent straight to a remote kernel, never on local disk",
    ),
    (
        "mcp__mighty-colab__exec", {"file": "/tmp/spike.py"},
        "mcp_remote_exec_file",
        "the MCP twin of `exec -f`: local file read, bytes shipped, path "
        "is all the transcript keeps",
    ),
]


@pytest.mark.parametrize(
    "tool_name,tool_input,rule,why", MCP_SCRATCH,
    ids=[f"{t}-{r}" for t, _, r, _ in MCP_SCRATCH])
def test_mcp_scratch_shapes_are_captured(tool_name, tool_input, rule, why):
    verdict = is_scratch(tool_name, tool_input)
    assert verdict.capture, f"MISSED scratch ({why}): {tool_name} {tool_input}"
    assert verdict.reason == rule, (
        f"captured for the wrong reason: got {verdict.reason!r}, "
        f"expected {rule!r} -- {why}")


# --- negative corpus -------------------------------------------------------
NOT_SCRATCH = [
    (
        "uv run python tools/provenance/emit_bytes.py --bytes 200000",
        "a committed script IS the happy path -- capturing it would tax the "
        "behaviour the whole design is trying to encourage",
    ),
    (
        "uv run pytest tests/ -m 'not slow'",
        "running the suite is not producing a number for a document",
    ),
    (
        "git log --oneline -5",
        "read/navigate commands are the bulk of all tool calls",
    ),
    (
        "grep -c EOF docs/PROJECT_MEMORY.md",
        "`-c` here means count, not code -- the interpreter allowlist is "
        "what stops this being a false positive",
    ),
    (
        "make stage2b-test",
        "a Makefile target is committed by construction",
    ),
    (
        "ls /tmp",
        "an ephemeral DIRECTORY is not an ephemeral SCRIPT",
    ),
    (
        "git commit -q -F - <<'EOF'\nFreeze the alpha-grid extension\n\nBody.\nEOF",
        "FIELD REPORT from stage2b-lead: their commit message for 7879a4c "
        "landed in a provenance blob. A heredoc says how text ARRIVES; what "
        "consumes it decides whether it is scratch, and git is not an "
        "interpreter. Already in git, so capturing it is pure noise",
    ),
    (
        "gh pr create -F - <<'EOF'\nPR body prose\nEOF",
        "the same defect's next victim -- any heredoc-fed tool (gh, jq -f -, "
        "mail) would land prose in a store meant for code",
    ),
    (
        "cat <<'EOF' > notes.md\njust some prose\nEOF",
        "a heredoc redirected to a FILE runs no code at all",
    ),
    (
        'grep -aE "closure|commit |colab|REFUS|Error" "$LOG" | head -6',
        "FIELD REPORT bug 4, and the nastiest so far: splitting on `|` "
        "without respecting quotes cut this regex into pieces, one of "
        "which was the bare word `colab` -- which matched the remote-exec "
        "binary list. A grep pattern was read as a pipe target",
    ),
    (
        "ps aux | grep -c '[m]ake x'; uv run mighty-colab sessions | tail -1",
        "a remote-exec BINARY in a pipeline is not a remote exec: "
        "`sessions` reports status and consumes no stdin. Only exec/repl/"
        "console take piped code",
    ),
]

# A DOCUMENTED BLIND SPOT, pinned so it stays known rather than becoming an
# accident. Every GPU target in this repo launches through a Makefile, so the
# tool call is `make <target>` and the `mighty-colab exec -f` runs in a
# subprocess make spawns. The predicate never sees the exec, and `make` is
# correctly classified as ordinary work.
#
# Found by stage2b-lead testing before a paid run rather than after. Not
# fixed here, deliberately -- see the README's "Known blind spot" section for
# what covers it and why the obvious fix is worse than the gap. If someone
# later makes these captured, this test failing is the intended signal to
# read that section, not a regression to paper over.
MAKE_WRAPPED_BLIND_SPOT = [
    (
        "make stage2b-ladder-stage3",
        "the exec is real but happens inside a subprocess make spawns",
    ),
    (
        "make stage2a-evolve-train-gpu",
        "same shape on the Stage 2A targets",
    ),
]


@pytest.mark.parametrize(
    "command,why", MAKE_WRAPPED_BLIND_SPOT,
    ids=[c[:44] for c, _ in MAKE_WRAPPED_BLIND_SPOT])
def test_make_wrapped_remote_exec_is_a_known_blind_spot(command, why):
    """Pins current behaviour AND its reason, so the gap stays visible.

    An undocumented gap and a documented one look identical in a passing
    suite. This is the difference: the limitation is asserted, named, and
    pointed at the section explaining what covers it.
    """
    verdict = bash(command)
    assert not verdict.capture, (
        f"{command} is now captured. If that was deliberate, delete this "
        f"case and the README's blind-spot section together; if not, the "
        f"predicate has started matching make targets by accident.")


@pytest.mark.parametrize(
    "command,why", NOT_SCRATCH, ids=[c[:44] for c, _ in NOT_SCRATCH])
def test_ordinary_work_is_not_captured(command, why):
    verdict = bash(command)
    assert not verdict.capture, (
        f"OVER-captured ({why}): {command} -> {verdict.reason}")


# --- extraction ------------------------------------------------------------

def test_inline_code_is_extracted_not_merely_detected():
    """Detection without extraction captures the fact that a script ran and
    loses the script, which is the only part that supports promotion."""
    verdict = bash('uv run python -c "print(42)"')
    assert verdict.script_text == "print(42)"
    assert verdict.script_source == "inline_c"


def test_heredoc_body_is_extracted_without_its_terminator():
    verdict = bash("uv run python - <<'PY'\nimport math\nprint(math.pi)\nPY")
    assert verdict.script_text == "import math\nprint(math.pi)"
    assert "PY" not in verdict.script_text


def test_remote_exec_records_the_path_for_snapshotting():
    verdict = bash("mighty-colab exec -s gpu1 -f /tmp/spike.py")
    assert verdict.referenced_files == ["/tmp/spike.py"]
    assert verdict.script_source == "file_reference"


def test_mcp_remote_exec_is_classified_from_tool_input():
    """The MCP surface carries code in a field, not a command line."""
    verdict = is_scratch("mcp__mighty-colab__exec", {"code": "print(1)"})
    assert verdict.capture and verdict.script_text == "print(1)"


# --- robustness ------------------------------------------------------------

def test_the_predicate_never_raises():
    """It runs inside a hook. Raising would break the session.

    Fails toward NOT capturing: over-capturing on a parse bug would quietly
    log unrelated work, which is worse than a missed record.
    """
    for pathological in ['python -c "unbalanced', "", "   ", "\x00", "|||",
                         "<<EOF", "mighty-colab exec -f"]:
        verdict = bash(pathological)
        assert isinstance(verdict.capture, bool), pathological


def test_every_corpus_entry_carries_a_reason():
    """An entry with no reason is folklore; principle 21's exemption rule."""
    for command, _, why in SCRATCH:
        assert why and len(why) > 20, f"thin reason for {command!r}"
    for tool_name, _, _, why in MCP_SCRATCH:
        assert why and len(why) > 20, f"thin reason for {tool_name!r}"
    for command, why in NOT_SCRATCH:
        assert why and len(why) > 20, f"thin reason for {command!r}"


def test_every_rule_the_predicate_can_return_appears_in_the_corpus():
    """The list-equals-derived check, in the one direction available.

    The rule NAMES are derivable -- they are string literals in the
    predicate's source -- even though the pattern set is not. So assert
    every capturing rule the predicate can emit has at least one worked
    example, which is what stops a rule being added with no example.
    """
    import ast
    source = (HOOK_DIR / "scratch_predicate.py").read_text()
    emitted = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "Verdict"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value is True
                and isinstance(node.args[1], ast.Constant)):
            emitted.add(node.args[1].value)
    covered = ({rule for _, rule, _ in SCRATCH}
               | {rule for _, _, rule, _ in MCP_SCRATCH})
    missing = emitted - covered
    print(f"\n[corpus] capturing rules: {sorted(emitted)}")
    assert not missing, (
        f"rules with no worked example in the corpus: {sorted(missing)} -- "
        f"add one, or the rule is untestable folklore")
