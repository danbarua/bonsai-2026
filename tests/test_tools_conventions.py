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

EVERY CHECK TAKES A ROOT. Raised by this project's own vacuous-test review:
reading `TOOLS` directly meant the only way to watch these guards fail was to
`chmod +x` a real file and undo it, which is a verification that lives in a
session transcript and fails nothing when it regresses. With the root
injectable, each check is run against a synthetic tree that violates it, and
the break-confirmation is committed. That is principle 20 applied to the
guard rather than to the finding.
"""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"


def _scripts(suffix: str, root: Path | None = None) -> list[Path]:
    return sorted(
        p
        for p in (root or TOOLS).rglob(f"*{suffix}")
        if "__pycache__" not in p.parts and p.is_file()
    )


# --------------------------------------------------------------------------
# The checks, each taking a root so it can be aimed at a violating tree
# --------------------------------------------------------------------------


def _check_python_not_executable(root: Path | None = None) -> None:
    executable = [p.name for p in _scripts(".py", root) if os.access(p, os.X_OK)]
    assert not executable, (
        f"Python tools marked executable: {executable}. Their shebang names "
        f"the SYSTEM python, which lacks this project's dependencies -- run "
        f"them as `uv run python <path>` instead, and drop the exec bit")


def _check_shell_runnable_and_documented(root: Path | None = None) -> None:
    problems = {}
    for path in _scripts(".sh", root):
        text = path.read_text()
        if not os.access(path, os.X_OK):
            problems[path.name] = "not executable"
        elif not text.startswith("#!"):
            problems[path.name] = "no shebang"
        else:
            body = text.splitlines()[1:]
            if not body or not body[0].startswith("#"):
                problems[path.name] = "no header comment saying what it is for"
    assert not problems, f"shell tools breaking the convention: {problems}"


def _check_python_documented(root: Path | None = None) -> None:
    undocumented = []
    for path in _scripts(".py", root):
        try:
            doc = ast.get_docstring(ast.parse(path.read_text()))
        except SyntaxError as exc:  # a tool that does not parse is worse
            raise AssertionError(f"{path} does not parse: {exc}") from exc
        if not doc or not doc.strip():
            undocumented.append(path.name)
    assert not undocumented, (
        f"Python tools with no module docstring: {undocumented}")


# --------------------------------------------------------------------------
# The live assertions, on tools/ as committed
# --------------------------------------------------------------------------


def test_the_scan_finds_tools_in_every_directory():
    """Anti-vacuity: an empty scan would pass every assertion below.

    Enumerated from the filesystem, so a new `tools/` subdirectory is covered
    the day it appears rather than when someone remembers a list.
    """
    subdirs = {p.parent.name for p in _scripts(".py") + _scripts(".sh")}
    assert len(subdirs) >= 3, (
        f"expected tool scripts in at least three subdirectories, found "
        f"{sorted(subdirs)} -- the scan is broken or the layout changed")
    assert len(_scripts(".py")) >= 5 and len(_scripts(".sh")) >= 3


def test_python_tools_are_not_executable_so_they_go_through_uv():
    """A `chmod +x` here is a bug, not a convenience."""
    _check_python_not_executable()


def test_shell_tools_are_executable_and_say_what_they_are_for():
    """`.sh` files here ARE invoked directly, by CI and by hand."""
    _check_shell_runnable_and_documented()


def test_every_python_tool_says_what_it_is_for():
    """A module docstring, because the filename is not the reason it exists."""
    _check_python_documented()


# --------------------------------------------------------------------------
# Break-confirmation, committed rather than performed by hand
# --------------------------------------------------------------------------


def _write(path: Path, text: str, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def clean_tree(tmp_path: Path) -> Path:
    """A synthetic `tools/` that satisfies every check."""
    root = tmp_path / "tools"
    _write(root / "alpha" / "thing.py", '#!/usr/bin/env python3\n"""Does a thing."""\n')
    _write(root / "alpha" / "run.sh", "#!/bin/bash\n# Runs the thing.\n", executable=True)
    return root


def test_the_synthetic_clean_tree_passes_every_check(clean_tree: Path):
    """Without this, a violation test could pass because the tree is broken
    in some unrelated way rather than in the way it names."""
    _check_python_not_executable(clean_tree)
    _check_shell_runnable_and_documented(clean_tree)
    _check_python_documented(clean_tree)


def test_an_executable_python_tool_is_caught(clean_tree: Path):
    (clean_tree / "alpha" / "thing.py").chmod(0o755)
    with pytest.raises(AssertionError, match="marked executable"):
        _check_python_not_executable(clean_tree)


def test_a_non_executable_shell_tool_is_caught(clean_tree: Path):
    (clean_tree / "alpha" / "run.sh").chmod(0o644)
    with pytest.raises(AssertionError, match="not executable"):
        _check_shell_runnable_and_documented(clean_tree)


def test_a_shell_tool_with_no_header_comment_is_caught(clean_tree: Path):
    _write(clean_tree / "alpha" / "run.sh", "#!/bin/bash\nset -u\n", executable=True)
    with pytest.raises(AssertionError, match="no header comment"):
        _check_shell_runnable_and_documented(clean_tree)


def test_a_shell_tool_with_no_shebang_is_caught(clean_tree: Path):
    _write(clean_tree / "alpha" / "run.sh", "# Runs the thing.\n", executable=True)
    with pytest.raises(AssertionError, match="no shebang"):
        _check_shell_runnable_and_documented(clean_tree)


def test_an_undocumented_python_tool_is_caught(clean_tree: Path):
    _write(clean_tree / "alpha" / "thing.py", "#!/usr/bin/env python3\nX = 1\n")
    with pytest.raises(AssertionError, match="no module docstring"):
        _check_python_documented(clean_tree)
