"""The `tools/` directories follow one convention, and it is load-bearing.

Measured across `tools/ci`, `tools/gates` and `tools/provenance`, the
convention is already uniform:

    *.sh   executable, shebang, header comment
    *.py   NOT executable, shebang, module docstring

The half that looks like an oversight is the important half. Every Python
tool carries `#!/usr/bin/env python3` and none is executable, so none can be
run as `./tools/ci/whatever.py`. That is deliberate protection rather than a
missing `chmod`: these tools import `numpy`, `yaml` and the `bonsai` package,
which live in the uv-managed `.venv`. `env python3` resolves to the SYSTEM
interpreter, so making one executable would let it start, import nothing it
needs, and fail somewhere less obvious than the command line — or worse,
succeed against different versions than CI uses.

The supported entry point is `uv run python tools/...`, and a permission bit
is what enforces it.

These are tests rather than a README because the convention has to hold on
the day someone adds a tool, and a README is read on the day someone goes
looking. Nothing here needs a new document: `tools/provenance/README.md`
earns its place by drawing the probe-versus-live-component distinction, and
`.claude/skills/github/SKILL.md` maps `tools/ci`.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"


def _scripts(suffix: str) -> list[Path]:
    return sorted(
        p
        for p in TOOLS.rglob(f"*{suffix}")
        if "__pycache__" not in p.parts and p.is_file()
    )


def test_the_scan_finds_tools_in_every_directory():
    """Anti-vacuity: an empty scan would pass every assertion below.

    Enumerated from the filesystem, so a new `tools/` subdirectory is covered
    the day it appears rather than when someone remembers this list.
    """
    subdirs = {p.parent.name for p in _scripts(".py") + _scripts(".sh")}
    assert len(subdirs) >= 3, (
        f"expected tool scripts in at least three subdirectories, found "
        f"{sorted(subdirs)} -- the scan is broken or the layout changed")
    assert len(_scripts(".py")) >= 5 and len(_scripts(".sh")) >= 3


def test_python_tools_are_not_executable_so_they_go_through_uv():
    """A `chmod +x` here is a bug, not a convenience.

    `#!/usr/bin/env python3` resolves to the system interpreter, which does
    not have this project's dependencies or the editable-installed `bonsai`
    package. The permission bit is the only thing making
    `./tools/ci/thing.py` fail immediately rather than misleadingly.
    """
    executable = [
        str(p.relative_to(REPO_ROOT)) for p in _scripts(".py") if os.access(p, os.X_OK)
    ]
    assert not executable, (
        f"Python tools marked executable: {executable}. Their shebang names "
        f"the SYSTEM python, which lacks this project's dependencies -- run "
        f"them as `uv run python <path>` instead, and drop the exec bit")


def test_shell_tools_are_executable_and_say_what_they_are_for():
    """`.sh` files here ARE invoked directly, by CI and by hand."""
    problems = {}
    for path in _scripts(".sh"):
        text = path.read_text()
        name = str(path.relative_to(REPO_ROOT))
        if not os.access(path, os.X_OK):
            problems[name] = "not executable"
        elif not text.startswith("#!"):
            problems[name] = "no shebang"
        else:
            body = text.splitlines()[1:]
            if not body or not body[0].startswith("#"):
                problems[name] = "no header comment saying what it is for"
    assert not problems, f"shell tools breaking the convention: {problems}"


def test_every_python_tool_says_what_it_is_for():
    """A module docstring, because the filename is not the reason it exists."""
    undocumented = []
    for path in _scripts(".py"):
        try:
            doc = ast.get_docstring(ast.parse(path.read_text()))
        except SyntaxError as exc:  # a tool that does not parse is worse
            raise AssertionError(f"{path} does not parse: {exc}") from exc
        if not doc or not doc.strip():
            undocumented.append(str(path.relative_to(REPO_ROOT)))
    assert not undocumented, (
        f"Python tools with no module docstring: {undocumented}")
