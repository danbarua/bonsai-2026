#!/usr/bin/env python3
"""Decide whether a tool call is *load-bearing scratch* worth capturing.

The sole authority for that question. Everything else in this directory --
the capture hook, its registration, its log -- consumes this and adds no
patterns of its own, so "what counts as scratch" has exactly one definition
and one test corpus.

Scope, from CLAUDE.md principle 24: an ephemeral script whose output can
enter a durable record while the script itself dies. IN -- inline code,
heredocs, tmp files, and local files shipped to a remote kernel. OUT --
anything already committed (that generator exists; it is the happy path),
and ordinary read/navigate commands.

**This set is not derivable, and that is stated rather than papered over.**
Principle 21 says a hand-maintained list standing in for a derivable set
will silently under-cover, and this project has paid for that four times.
There is no filesystem or AST enumeration of "ways to run an ephemeral
script", so no derivation is available. The mitigation is a committed
corpus (`tests/test_provenance_capture.py`) holding every invocation shape
this project has actually seen, each with the reason it is there. A new
scratch shape found in the wild is fixed by adding a corpus entry AND a
rule -- never a rule alone, which would leave the next reader unable to
tell what the rule was for.

Written in Python rather than bash, departing from the c2c-mail hooks next
door: this needs shell-aware tokenising (`shlex`), heredoc body extraction,
and a predicate importable by pytest. The directory layout, the
`$CLAUDE_PROJECT_DIR` registration convention and the break-test discipline
follow that neighbour exactly.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

# Interpreters whose `-c` argument IS the program. Kept explicit rather
# than "anything with -c": `grep -c` counts lines and is not scratch.
_INTERPRETERS = ("python", "python3", "python3.11", "python3.12", "python3.13",
                 "python3.14", "node", "ruby", "perl")

# Remote-execution commands that read a LOCAL file and ship its contents
# elsewhere. This is the case session transcripts provably lose: the
# transcript records the path, the kernel receives the bytes, and when the
# path is a tmp file the bytes are gone the moment the session ends.
_REMOTE_EXEC = ("mighty-colab", "colab")

# Subcommands of a remote-exec tool that treat piped stdin as a PROGRAM.
# `sessions`, `status`, `stop`, `ls` and friends consume no stdin, so a
# pipeline ending in one ships nothing and is not scratch.
_STDIN_CODE_SUBCOMMANDS = ("exec", "repl", "console", "run")

# Directories whose contents are expected to vanish.
_EPHEMERAL_DIRS = ("/tmp/", "/var/folders/", "/private/tmp/")

_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


@dataclass
class Verdict:
    """Why a call was or was not classified as scratch.

    `reason` names the rule that fired in both directions. A capture record
    that says only "captured" cannot be audited for over- or under-capture;
    one that says `inline_c` can.
    """
    capture: bool
    reason: str
    script_text: str | None = None
    script_source: str | None = None  # inline_c | heredoc | stdin_pipe | file_reference
    referenced_files: list[str] = field(default_factory=list)


def _tokens(command: str) -> list[str]:
    """Best-effort shell tokenisation.

    Falls back to a whitespace split: an unbalanced quote must not make the
    predicate raise inside a hook, and a coarse answer beats no answer.
    """
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _strip_wrappers(tokens: list[str]) -> list[str]:
    """Drop `uv run`, `env FOO=1`, `nohup`, `time` and friends.

    Without this, `uv run python -c ...` reads as the program being `uv`,
    and every inline script launched the way this repo actually launches
    them would be missed.
    """
    out = list(tokens)
    while out:
        head = Path(out[0]).name
        if head in ("uv", "uvx") and len(out) > 1 and out[1] in ("run", "tool"):
            out = out[2:]
            # `uv run --with x python ...`: skip flags and their values.
            while len(out) > 1 and out[0].startswith("-"):
                out = out[2:] if "=" not in out[0] else out[1:]
            continue
        if head in ("nohup", "time", "command", "exec"):
            out = out[1:]
            continue
        if head == "env":
            out = out[1:]
            while out and "=" in out[0] and not out[0].startswith("-"):
                out = out[1:]
            continue
        break
    return out


def _command_portion(command: str) -> str:
    """The command text, with any heredoc BODY removed.

    Segmentation splits on `;`, `&&` and newlines, and a heredoc body is
    arbitrary text full of both. Removing it first stops a line of Python
    inside a heredoc from being mistaken for another shell command.
    """
    match = _HEREDOC_RE.search(command)
    if not match:
        return command
    line_end = command.find("\n", match.end())
    return command if line_end == -1 else command[:line_end]


# Characters shlex treats as punctuation, and the subset of them that end
# one command and begin another. `(` `)` `<` `>` group and redirect; they do
# not separate.
_PUNCTUATION = set("();<>|&")
_SEPARATOR_CHARS = set(";|&")


def _newlines_to_separators(command: str) -> str:
    """Make an unquoted newline an explicit `;`, leaving quoted text alone.

    A newline separates commands in shell, but `shlex` treats it as ordinary
    whitespace, which would merge every line of a multi-line block into one
    segment and hide a `python -c` on line three. Replacing newlines
    wholesale would instead corrupt a multi-line quoted `-c` program, which
    is a common shape here. So the replacement is quote-aware.
    """
    out: list[str] = []
    quote: str | None = None
    for char in command:
        if quote:
            if char == quote:
                quote = None
            out.append(char)
        elif char in "'\"":
            quote = char
            out.append(char)
        else:
            out.append(";" if char == "\n" else char)
    return "".join(out)


def _shell_tokens(command: str) -> list[str]:
    """Tokenise with shell operators as their own tokens, quotes respected.

    `punctuation_chars=True` is what makes `|` a token rather than part of a
    word, WITHOUT treating a pipe inside a quoted string as a separator.
    That distinction is not academic: splitting the raw string on `|` cut
    `grep -aE "closure|commit |colab|REFUS|Error"` into fragments, one of
    which was the bare word `colab`, which then matched the remote-exec
    binary list. A grep pattern was classified as a pipe into a GPU kernel.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        # Unbalanced quotes. Fall back coarsely rather than raise -- a hook
        # depends on this -- and accept that odd input classifies poorly.
        return command.split()


def _segments(command: str) -> list[list[str]]:
    """Token lists for each command within a compound shell invocation."""
    segments: list[list[str]] = []
    current: list[str] = []
    text = _newlines_to_separators(_command_portion(command))
    for token in _shell_tokens(text):
        # shlex groups runs of punctuation into one token, so a `);` arrives
        # whole rather than as `)` and `;`. Testing membership against a set
        # of exact operators would miss it -- which is how a segment boundary
        # gets silently dropped and two commands merge into one.
        if token and set(token) <= _PUNCTUATION:
            if set(token) & _SEPARATOR_CHARS and current:
                segments.append(current)
                current = []
            continue  # grouping/redirect punctuation carries no program
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _resolved_program(segment: list[str]) -> str | None:
    """What this segment actually EXECUTES, after stripping wrappers.

    The distinction the whole predicate turns on. `-c` and heredocs describe
    how code ARRIVES; the program that consumes it decides whether any code
    is being run at all. Keying on delivery produced both field-reported
    bugs at once -- `uv run python -c` invisible because it was not the
    first token, and `git commit -F - <<EOF` captured because a heredoc was
    present at all.
    """
    tokens = _strip_wrappers(segment)
    return Path(tokens[0]).name if tokens else None


def _consumes_code(command: str) -> bool:
    """Does any part of this command feed text to an interpreter?"""
    return any(_resolved_program(s) in _INTERPRETERS for s in _segments(command))


def _inline_code(command: str) -> str | None:
    """The argument to an interpreter's `-c`, anywhere in the command.

    Every segment is checked, not just the first: scratch is rarely the
    leading token of a real Bash call. It sits behind a `cd ... &&`, after a
    variable assignment, or at the end of a `;`-chain.
    """
    for segment in _segments(command):
        tokens = _strip_wrappers(segment)
        if not tokens or Path(tokens[0]).name not in _INTERPRETERS:
            continue
        for i, token in enumerate(tokens[1:], start=1):
            if token == "-c" and i + 1 < len(tokens):
                return tokens[i + 1]
    return None


def _heredoc_body(command: str) -> str | None:
    """The body of a heredoc, if one is present and terminated."""
    match = _HEREDOC_RE.search(command)
    if not match:
        return None
    marker = match.group(2)
    after = command[match.end():]
    lines = after.splitlines()
    if not lines:
        return None
    body: list[str] = []
    for line in lines[1:]:
        if line.strip() == marker:
            return "\n".join(body)
        body.append(line)
    # Unterminated: still scratch, and the partial body is worth keeping.
    return "\n".join(body) if body else None


def _remote_exec_files(tokens: list[str]) -> list[str]:
    """Local paths a remote-exec command will read and ship elsewhere."""
    tokens = _strip_wrappers(tokens)
    if not tokens or Path(tokens[0]).name not in _REMOTE_EXEC:
        return []
    files: list[str] = []
    subcommand = None
    for i, token in enumerate(tokens[1:], start=1):
        if subcommand is None and not token.startswith("-"):
            subcommand = token
            continue
        if token in ("-f", "--file") and i + 1 < len(tokens):
            files.append(tokens[i + 1])
        # `mighty-colab run script.py` takes the script positionally.
        elif (subcommand == "run" and not token.startswith("-")
              and token.endswith((".py", ".ipynb"))):
            files.append(token)
    return files


def _ephemeral_paths(tokens: list[str]) -> list[str]:
    return [t for t in tokens
            if any(t.startswith(d) for d in _EPHEMERAL_DIRS)
            and t.endswith((".py", ".sh", ".ipynb", ".R", ".jl"))]


def is_scratch(tool_name: str, tool_input: dict,
               repo_root: Path | None = None) -> Verdict:
    """Classify one tool call. Never raises -- a hook depends on it."""
    try:
        verdict = _classify(tool_name, tool_input, repo_root)
    except Exception as exc:  # noqa: BLE001
        # Fail OPEN, in the direction of not capturing. A predicate that
        # throws would break the hook; one that over-captures on a parse
        # bug would quietly log unrelated work.
        return Verdict(False, f"predicate_error: {type(exc).__name__}: {exc}")

    # **A capture must attest to something.** A verdict carrying neither
    # script text nor a referenced file would write a record whose
    # `trigger_reason` asserts that code was captured while the record holds
    # no code -- a positive artifact stating something untrue, in a store
    # whose only value is being trustworthy about what ran.
    #
    # That is a worse failure than silent under-capture, and it happened: a
    # grep pattern was classified as a pipe into a GPU kernel, and the
    # resulting record was read by a human as evidence of coverage it did
    # not have. Under-capture leaves an absence you can be suspicious of;
    # this leaves a confident lie. Blocked structurally rather than by
    # fixing each classifier that could produce it.
    if verdict.capture and not verdict.script_text \
            and not verdict.referenced_files:
        return Verdict(False, f"{verdict.reason}_but_attested_nothing")
    return verdict


def _classify(tool_name: str, tool_input: dict,
              repo_root: Path | None) -> Verdict:
    if tool_name.startswith("mcp__mighty-colab__") or tool_name.endswith("__exec"):
        code = tool_input.get("code") or tool_input.get("script") or ""
        path = tool_input.get("file") or tool_input.get("f")
        if path:
            return Verdict(True, "mcp_remote_exec_file",
                           script_source="file_reference",
                           referenced_files=[str(path)])
        if code:
            return Verdict(True, "mcp_remote_exec_inline", script_text=code,
                           script_source="inline_c")
        return Verdict(False, "mcp_remote_exec_no_code")

    if tool_name != "Bash":
        return Verdict(False, "not_a_command_tool")

    command = tool_input.get("command") or ""
    if not command.strip():
        return Verdict(False, "empty_command")

    tokens = _tokens(command)

    # Remote exec is checked FIRST and unconditionally, before the
    # committed-path exemption. `mighty-colab exec -f experiments/x.py` ships
    # a committed file, but WHICH REVISION it shipped is not recoverable from
    # the transcript on a dirty tree -- so the snapshot is worth taking even
    # when the path is tracked.
    for segment in re.split(r"\|\||&&|\||;", command):
        remote_files = _remote_exec_files(_tokens(segment))
        if remote_files:
            return Verdict(True, "remote_exec_local_file",
                           script_source="file_reference",
                           referenced_files=remote_files)

    inline = _inline_code(command)
    if inline is not None:
        return Verdict(True, "inline_c", script_text=inline,
                       script_source="inline_c")

    body = _heredoc_body(command)
    if body is not None:
        # A heredoc is only scratch if something EXECUTES it. `git commit
        # -F -`, `gh pr create -F -` and `cat <<EOF > notes.md` all take
        # heredocs and run no code; capturing them puts prose into a store
        # meant for code, and in git's case duplicates what git already has.
        if _consumes_code(command):
            return Verdict(True, "heredoc", script_text=body,
                           script_source="heredoc")
        return Verdict(False, "heredoc_not_fed_to_an_interpreter")

    ephemeral = _ephemeral_paths(tokens)
    if ephemeral:
        return Verdict(True, "ephemeral_path", script_source="file_reference",
                       referenced_files=ephemeral)

    # Code piped into a remote kernel: the pipeline's left side is the
    # program, and it never touches disk.
    segments = _segments(command)
    for i, segment in enumerate(segments[1:], start=1):
        if _resolved_program(segment) not in _REMOTE_EXEC:
            continue
        # Being the mighty-colab BINARY is not enough. `sessions`, `status`
        # and `stop` report or manage; they consume no stdin and running one
        # in a pipeline ships nothing. Only these subcommands treat piped
        # text as a program.
        subcommand = next((t for t in _strip_wrappers(segment)[1:]
                           if not t.startswith("-")), None)
        if subcommand not in _STDIN_CODE_SUBCOMMANDS:
            continue
        return Verdict(True, "piped_into_remote_exec",
                       script_text=" ".join(segments[i - 1]),
                       script_source="stdin_pipe")

    return Verdict(False, "not_scratch")
