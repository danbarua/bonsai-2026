"""The executor-version probe, run against the real binary.

`executor` exists because a capture recording WHAT ran but not WHICH
VERSION ran it inherits the skew blindness it was built to close --
`c2c-implementation` lost a message to a deployment silently lacking a
feature the caller assumed present, and a GPU number from a different
mighty-colab or CUDA build is a different number.

A version probe that silently never resolves is **worse than no version
field**, because the record still carries the key and a reader reasonably
takes `version: null` to mean "this tool has no version" rather than "the
command we tried was wrong". The original entry here was
`mighty-colab --version`, written from documentation and never executed:
it is not a valid flag, exits non-zero with a usage error, and would have
recorded `probe_failed` indefinitely.

That is CLAUDE.md principle 21's shape once more -- a hand-written list
standing in for something checkable -- so the list is checked here against
the binaries themselves. Skips cleanly when a tool is absent rather than
failing, since these are developer-machine binaries and not dependencies.
"""
import importlib.util
import shutil
import subprocess
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


capture = _load("capture")


@pytest.mark.parametrize("tool", sorted(capture._VERSION_COMMANDS))
def test_the_probe_command_is_real_and_succeeds(tool):
    """Every entry must actually work against the installed binary.

    Not "returns something plausible" -- exits zero and prints a non-empty
    first line, which is what `executor_version` reads.
    """
    argv = capture._VERSION_COMMANDS[tool]
    if shutil.which(argv[0]) is None:
        pytest.skip(f"{argv[0]} not installed on this machine")
    proc = subprocess.run(argv, capture_output=True, text=True,
                          timeout=capture.VERSION_TIMEOUT_S)
    print(f"\n[executor] {' '.join(argv)} -> exit={proc.returncode} "
          f"{proc.stdout.strip().splitlines()[:1]}")
    assert proc.returncode == 0, (
        f"`{' '.join(argv)}` exited {proc.returncode}. The probe would "
        f"record probe_failed forever without saying why.\n"
        f"stderr: {proc.stderr.strip()[:300]}")
    assert proc.stdout.strip(), "probe produced no stdout to record"


@pytest.mark.parametrize("tool", sorted(capture._VERSION_COMMANDS))
def test_executor_version_resolves_and_caches(tmp_path, tool):
    """End-to-end through the function the hook actually calls."""
    if shutil.which(capture._VERSION_COMMANDS[tool][0]) is None:
        pytest.skip(f"{tool} not installed")
    first = capture.executor_version(tool, tmp_path)
    print(f"[executor] {tool} -> {first}")
    assert first["version"], f"{tool} version unresolved: {first}"
    assert first["resolved_from"] == "probe"

    # The cache exists so a hook does not shell out on every capture.
    second = capture.executor_version(tool, tmp_path)
    assert second["resolved_from"] == "cache"
    assert second["version"] == first["version"]


def test_an_unknown_tool_says_so_rather_than_guessing():
    """`no_probe_known` is an honest answer; a fabricated version is not."""
    result = capture.executor_version("Bash", REPO_ROOT / "nonexistent")
    assert result["version"] is None
    assert result["resolved_from"] == "no_probe_known"


def test_a_probe_that_fails_is_reported_as_failed(tmp_path, monkeypatch):
    """The state that must stay distinguishable from `no_probe_known`.

    A tool we know how to probe but could not is a different fact from a
    tool we never knew how to probe, and collapsing them would hide exactly
    the breakage this file was written after.
    """
    monkeypatch.setitem(capture._VERSION_COMMANDS, "fake-tool",
                        ["definitely-not-a-real-binary-xyz", "version"])
    result = capture.executor_version("fake-tool", tmp_path)
    assert result["version"] is None
    assert result["resolved_from"] == "probe_failed"


def test_the_timeout_exceeds_the_measured_probe_cost():
    """A timeout set at the measured cost is a timeout that will fire.

    `mighty-colab version` measures ~0.9-1.0s cold here; the guard is that
    the budget stays comfortably clear of it rather than being tightened to
    look efficient.
    """
    assert capture.VERSION_TIMEOUT_S >= 3, (
        "version probe timeout is too close to the measured ~1s cost; an "
        "ordinary slow start would leave the field permanently unresolved")
