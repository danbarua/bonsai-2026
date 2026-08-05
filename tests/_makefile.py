"""One Makefile parser, shared by the tests that assert against it.

There were briefly two: `test_mighty_colab_contract.py` grew one to check
GPU recipes, and `test_stage2b_gcs_makefile.py` grew another to check the
bucket. They drifted immediately -- the second learned to join backslash
continuations, after a `GCS_ENV` spanning two lines was read as half its
value and made a passing assertion mean the opposite of what it said, and
the first never learned it.

Two hand-maintained copies of one derivable thing, one corrected and the
other silently not, is the shape CLAUDE.md principle 21 is about. Writing
that principle and leaving the duplication in place would have been
funny, so this module exists.

Pure text parsing: no `make` invocation, no shell, no network.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = REPO_ROOT / "Makefile"


def _text():
    return MAKEFILE.read_text()


def recipes():
    """Every recipe, as {target_name: recipe_text}.

    A recipe is the run of tab-indented lines following a `target:` line.
    Continuations are left in place rather than joined: callers assert
    with substring checks, and joining would only obscure which line
    matched. `make_var` joins them, because a variable's value is the
    joined form.
    """
    out, current, body = {}, None, []
    for line in _text().splitlines():
        if line.startswith("\t"):
            if current:
                body.append(line)
            continue
        if current:
            out[current] = "\n".join(body)
            current, body = None, []
        match = re.match(r"^([A-Za-z0-9_-]+):(?!=)", line)
        if match:
            current, body = match.group(1), []
    if current:
        out[current] = "\n".join(body)
    return out


def make_var(name):
    """The value of a `NAME ?= value` (or `:=`, or `=`) assignment.

    Backslash continuations are joined first: a pattern anchored to a
    single line silently returns the first fragment of a multi-line value,
    which reads as "the variable does not mention X" when it does.

    Returns None when the variable is not declared at all, so a caller can
    distinguish "wrong value" from "missing entirely". The last assignment
    wins, matching how `make` itself resolves a repeated one.
    """
    joined = re.sub(r"\\\n\s*", " ", _text())
    pattern = rf"^{re.escape(name)}\s*[?:]?=\s*(.*?)\s*$"
    matches = re.findall(pattern, joined, re.MULTILINE)
    return matches[-1] if matches else None
