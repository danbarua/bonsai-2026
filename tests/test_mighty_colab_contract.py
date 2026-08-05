"""What this repository assumes about the `mighty-colab` CLI, pinned.

Every GPU target in the `Makefile` is a shell recipe wrapped around this
CLI, and two of its behaviours are load-bearing in ways a reader of the
recipe would not guess:

1. `status -s <name>` on a session that does not exist exits **0** and
   announces the fact on **stdout**. The recipes' idempotency guard greps
   that message instead of trusting the exit status, so if a release ever
   made `status` fail, or moved the message to stderr, the guard would
   stop distinguishing "already up" from "not there" and would silently
   stop provisioning.
2. `exec --timeout` bounds the gap between outputs and defaults to **30
   seconds**, so a healthy remote script that simply goes quiet for longer
   than that dies with `TimeoutError: Timeout waiting for output`. Every
   long-running driver in this project goes quiet for far longer.

Both were established by hand -- (1) when the guard was written, (2) when
`stage2a-class0-classify-gpu` was run as a target for the first time and
died 30 seconds in, having never completed once since it was codified from
a hand-run session. CLAUDE.md principle 20: a hand-verified property that
lives only in a comment is not re-checkable and nothing fails when a
dependency upgrade invalidates it -- which is exactly what a `mighty-colab`
upgrade did to a neighbouring comment in this same file.

Tier 1 (always runs, no CLI, no network, nothing provisioned) parses the
`Makefile` and asserts every `exec` passes an explicit `--timeout` and
every session-creating recipe tears down unconditionally. Tier 2 asks the
installed CLI what it actually does; it skips cleanly when the `gpu`
dependency group is not installed. Nothing here creates a session or
bills.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "Makefile"

# Above this, an "explicit timeout" would be no better than the default.
MAX_TRUSTWORTHY_DEFAULT_TIMEOUT_S = 60.0


def _recipes():
    """Every Makefile recipe, as {target_name: recipe_text}.

    A recipe is the run of tab-indented lines following a `target:` line.
    Line continuations are left as-is; the assertions below are substring
    checks, so joining them would only obscure which line matched."""
    out, current, body = {}, None, []
    for line in MAKEFILE.read_text().splitlines():
        if line.startswith("\t"):
            if current:
                body.append(line)
            continue
        if current:
            out[current] = "\n".join(body)
            current, body = None, []
        m = re.match(r"^([A-Za-z0-9_-]+):(?!=)", line)
        if m:
            current, body = m.group(1), []
    if current:
        out[current] = "\n".join(body)
    return out


def _exec_recipes():
    return {name: body for name, body in _recipes().items()
            if "mighty-colab) exec" in body or "MIGHTY_COLAB) exec" in body}


def test_makefile_has_gpu_recipes_to_check():
    """Guards the two tests below against silently passing on zero recipes
    if the variable name or recipe shape ever changes."""
    found = sorted(_exec_recipes())
    print(f"\n[contract] Makefile recipes invoking `exec`: {found}")
    assert len(found) >= 3, (
        f"expected several GPU targets, parsed {found} -- if the Makefile still has "
        f"them, this test's parser has gone stale and the checks below are vacuous")


def test_every_exec_passes_an_explicit_timeout():
    """The 30-second default is far below what any driver here needs."""
    offenders = []
    for name, body in _exec_recipes().items():
        for line in body.splitlines():
            if ") exec " not in line:
                continue
            ok = "--timeout" in line
            if not ok:
                offenders.append((name, line.strip()))
            print(f"[contract] {name}: exec {'passes' if ok else 'OMITS'} --timeout")
    assert not offenders, (
        "these `mighty-colab exec` invocations rely on the 30s default and will die "
        "with `Timeout waiting for output` on any driver that goes quiet:\n"
        + "\n".join(f"  {n}: {l}" for n, l in offenders))


def test_every_session_creating_recipe_tears_down_unconditionally():
    """A recipe that only stops the session on the success path leaks a
    billing VM on every failure -- which is what `exec && download && stop`
    started doing the moment `exec` gained the ability to fail."""
    for name, body in _exec_recipes().items():
        assert ") stop -s " in body, f"{name} never stops its session"
        stop_lines = [l for l in body.splitlines() if ") stop -s " in l]
        for line in stop_lines:
            assert not line.strip().startswith("&&"), (
                f"{name} chains its teardown onto a previous command's success: {line.strip()}")
        print(f"[contract] {name}: teardown present and not chained on success")


@pytest.fixture(scope="module")
def cli():
    found = shutil.which("mighty-colab")
    if found is None:
        pytest.skip("the `mighty-colab` CLI is not on PATH; it ships in this project's "
                    "`gpu` dependency group (`uv run --group gpu pytest ...`)")
    return found


def test_status_of_unknown_session_exits_zero_on_stdout(cli):
    """Reads local session state only -- no network, nothing provisioned."""
    name = "bonsai-contract-probe-no-such-session"
    r = subprocess.run([cli, "status", "-s", name], capture_output=True, text=True,
                       timeout=120)
    print(f"\n[contract] `mighty-colab status -s {name}`")
    print(f"[contract]   exit={r.returncode}")
    print(f"[contract]   stdout={r.stdout.strip()!r}")
    print(f"[contract]   stderr={r.stderr.strip()!r}")
    assert r.returncode == 0, (
        "`status` now fails for an unknown session. The Makefile's idempotency guard "
        "greps for 'not found' precisely because it did not -- but the guard runs "
        "inside an `if`, so a non-zero exit does not break it. Re-read the guard and "
        "update the Makefile comment before relaxing this.")
    assert "not found" in (r.stdout + r.stderr).lower()
    assert "not found" in r.stdout.lower(), (
        "the 'not found' message moved off stdout. The Makefile guard redirects "
        "stderr into its grep, so it still works -- but the comment claiming stdout "
        "is now wrong.")


def test_exec_default_timeout_is_short_enough_to_need_overriding(cli):
    """Documents *why* every recipe passes `--timeout`: without one, a
    driver that computes quietly for a minute is killed."""
    r = subprocess.run([cli, "help", "exec"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"`mighty-colab help exec` failed: {r.stderr}"
    text = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    flat = " ".join(text.split())
    m = re.search(r"default:\s*([0-9]+(?:\.[0-9]+)?)", flat)
    assert m, f"could not find the documented default timeout in:\n{flat}"
    default_s = float(m.group(1))
    print(f"\n[contract] `exec --timeout` documented default: {default_s}s "
          f"(recipes pass EXEC_TIMEOUT instead)")
    assert default_s <= MAX_TRUSTWORTHY_DEFAULT_TIMEOUT_S, (
        f"the default is now {default_s}s. If it has genuinely become long enough for "
        f"this project's drivers, the explicit --timeout in each recipe is still "
        f"correct but this test's rationale needs rewriting rather than deleting.")
