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
3. `stop` on a session that is already absent exits **0**. That is what
   makes an unconditional teardown safe, because the recipes do reach
   `stop` on paths where provisioning never created a session. "Already
   absent" and "could not stop" mean opposite things for billing, so the
   recipes must be able to tell them apart -- `STOP_ABSENT_RC` names which
   code means absent, and everything else non-zero is treated as a
   possible leak and fails the target.

All three were established by hand -- (1) when the guard was written, (2)
when `stage2a-class0-classify-gpu` was run as a target for the first time
and died 30 seconds in, having never completed once since it was codified
from a hand-run session, (3) when weighing whether `stop` should start
erroring on a missing session. CLAUDE.md principle 20: a hand-verified
property that lives only in a comment is not re-checkable and nothing
fails when a dependency upgrade invalidates it -- which is exactly what a
`mighty-colab` upgrade did to a neighbouring comment in this same file.

Tier 1 (always runs, no CLI, no network, nothing provisioned) parses the
`Makefile`, and separately drives a real recipe end to end against a stub
CLI so the teardown logic is exercised rather than merely inspected. Tier
2 asks the installed CLI what it actually does; it skips cleanly when the
`gpu` dependency group is not installed. Nothing here creates a session or
bills.
"""
import re
import shutil
import subprocess

import pytest

from _makefile import REPO_ROOT, recipes as _recipes

# Above this, an "explicit timeout" would be no better than the default.
MAX_TRUSTWORTHY_DEFAULT_TIMEOUT_S = 60.0


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


STUB = """#!/bin/sh
# Stand-in for the mighty-colab CLI. Provisions nothing, bills nothing.
case "$1" in
  sessions) echo "[stub] no active sessions"; exit 0 ;;
  status)   echo "[stub] Session 'x' not found."; exit 0 ;;
  new)      echo "[stub] created"; exit 0 ;;
  install|reinstall|upload) echo "[stub] $1 ok"; exit 0 ;;
  exec)     echo "${STUB_SENTINEL:-GPU_VERIFY_OK}"; exit "${STUB_EXEC_RC:-0}" ;;
  stop)     echo "[stub] stop attempted"; exit "${STUB_STOP_RC:-0}" ;;
  *)        echo "[stub] unhandled: $*"; exit 0 ;;
esac
"""


@pytest.fixture(scope="module")
def stub_cli(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcstub") / "mighty-colab-stub"
    path.write_text(STUB)
    path.chmod(0o755)
    return path


def _run_target(stub, env_extra=None, make_vars=(), target="stage2b-verify-gpu"):
    """Run the real recipe against the stub.

    `-s` is required, not cosmetic: without it `make` echoes the recipe
    itself, so every assertion about output would match the recipe TEXT
    containing "LEAK WARNING" rather than the recipe having PRINTED it.

    `make` exits 2 for any recipe failure regardless of what the recipe
    exited with, so the recipe's own code is read from make's
    "*** [target] Error N" line on stderr -- that is where the leak code
    is actually distinguishable from the verdict code.

    `target` is a parameter because these four behaviours are a property of
    the RECIPE SHAPE, not of any one target. Hardcoding one meant the next
    GPU target added got only the three static checks above, which is how a
    recipe can look conformant and still mishandle a leak."""
    import os
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = ["make", "-s", target, f"MIGHTY_COLAB={stub}", *make_vars]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
                       env=env)
    m = re.search(r"\*\*\* \[[^\]]+\] Error (\d+)", r.stderr)
    recipe_rc = int(m.group(1)) if m else r.returncode
    print(f"\n[contract] {' '.join(cmd[2:])} env={env_extra or {}}")
    print(f"[contract]   make exit={r.returncode}, recipe exit={recipe_rc}")
    for line in (r.stdout + r.stderr).splitlines():
        if "LEAK" in line or "FAILED" in line or "Error" in line:
            print(f"[contract]   | {line}")
    return recipe_rc, r


def test_healthy_run_exits_zero(stub_cli):
    rc, r = _run_target(stub_cli)
    assert rc == 0, r.stdout + r.stderr
    assert "LEAK WARNING" not in r.stdout


def test_teardown_failure_fails_an_otherwise_successful_target(stub_cli):
    """The case the check exists for: the science passed, so nothing else
    would have failed, and an A100 may still be billing."""
    rc, r = _run_target(stub_cli, {"STUB_STOP_RC": "7"})
    assert rc == 7, (
        f"a failed teardown must fail the target, carrying stop's own code -- got {rc}\n"
        f"{r.stdout}{r.stderr}")
    assert "LEAK WARNING" in r.stdout
    assert "FAILED: the GPU ridge gate" not in r.stdout, (
        "a teardown failure must not be reported as a scientific failure")


def test_a_leak_never_masks_the_scientific_verdict(stub_cli):
    """Both wrong: the run's own failure stays the headline and exit code,
    and the leak is still reported rather than swallowed."""
    rc, r = _run_target(stub_cli, {"STUB_STOP_RC": "7", "STUB_SENTINEL": "NOTHING_USEFUL"})
    assert rc == 1, r.stdout + r.stderr
    assert "FAILED: the GPU ridge gate" in r.stdout
    assert "LEAK WARNING" in r.stdout


# ---- The same four behaviours, on the ladder target ----
#
# Not a copy for its own sake: these are properties of the recipe shape, and
# the whole reason they are re-run here is that a new target inherits the
# STATIC checks automatically but nothing about how it actually behaves when
# teardown fails. The ladder target also runs real money, so its leak
# handling is the last thing that should be assumed rather than exercised.

LADDER_TARGET = "stage2b-ladder-stage1"

# Reports a clean tree on a commit that exists on a remote, so the recipe
# gets past its pre-flight and into the part these tests are about.
GIT_STUB_READY = """#!/bin/sh
case "$1 $2" in
  "status --porcelain") exit 0 ;;
  "rev-parse HEAD") echo "0000000000000000000000000000000000000000"; exit 0 ;;
  "branch -r") echo "  origin/stage2b"; exit 0 ;;
esac
case "$1" in
  status) exit 0 ;;
  rev-parse) echo "0000000000000000000000000000000000000000"; exit 0 ;;
  branch) echo "  origin/stage2b"; exit 0 ;;
  *) exit 0 ;;
esac
"""

GIT_STUB_DIRTY = """#!/bin/sh
case "$1" in
  status) echo " M experiments/stage2b_denoising/run_ladder_stage1.py"; exit 0 ;;
  rev-parse) echo "0000000000000000000000000000000000000000"; exit 0 ;;
  branch) echo "  origin/stage2b"; exit 0 ;;
  *) exit 0 ;;
esac
"""

GIT_STUB_UNPUSHED = """#!/bin/sh
case "$1" in
  status) exit 0 ;;
  rev-parse) echo "0000000000000000000000000000000000000000"; exit 0 ;;
  branch) exit 0 ;;
  *) exit 0 ;;
esac
"""


@pytest.fixture(scope="module")
def git_stubs(tmp_path_factory):
    base = tmp_path_factory.mktemp("gitstub")
    made = {}
    for name, body in (("ready", GIT_STUB_READY), ("dirty", GIT_STUB_DIRTY),
                       ("unpushed", GIT_STUB_UNPUSHED)):
        path = base / f"git-{name}"
        path.write_text(body)
        path.chmod(0o755)
        made[name] = path
    return made


def _run_ladder(stub, git_stub, env_extra=None, make_vars=()):
    return _run_target(stub, env_extra, (f"GIT={git_stub}", *make_vars),
                       target=LADDER_TARGET)


def test_ladder_healthy_run_exits_zero(stub_cli, git_stubs):
    rc, r = _run_ladder(stub_cli, git_stubs["ready"], {"STUB_SENTINEL": "STAGE1_OK"})
    assert rc == 0, r.stdout + r.stderr
    assert "LEAK WARNING" not in r.stdout


def test_ladder_teardown_failure_fails_an_otherwise_successful_target(stub_cli, git_stubs):
    rc, r = _run_ladder(stub_cli, git_stubs["ready"],
                        {"STUB_SENTINEL": "STAGE1_OK", "STUB_STOP_RC": "7"})
    assert rc == 7, (
        f"a failed teardown must fail the target, carrying stop's own code -- got {rc}\n"
        f"{r.stdout}{r.stderr}")
    assert "LEAK WARNING" in r.stdout
    assert "FAILED: ladder stage 1" not in r.stdout, (
        "a teardown failure must not be reported as a scientific failure")


def test_ladder_missing_sentinel_fails_even_on_a_zero_exit(stub_cli, git_stubs):
    """The case the sentinel exists for: `exec` returns 0 because the script
    exited cleanly, but it never reached its own verdict."""
    rc, r = _run_ladder(stub_cli, git_stubs["ready"],
                        {"STUB_SENTINEL": "nothing useful here"})
    assert rc == 1, r.stdout + r.stderr
    assert "FAILED: ladder stage 1" in r.stdout


def test_ladder_absent_session_is_not_treated_as_a_leak(stub_cli, git_stubs):
    """"Already gone" is the goal, not a failure -- it is "could not stop"
    that costs money, and conflating them makes the check unadoptable."""
    rc, r = _run_ladder(stub_cli, git_stubs["ready"],
                        {"STUB_SENTINEL": "STAGE1_OK", "STUB_STOP_RC": "3"},
                        make_vars=("STOP_ABSENT_RC=3",))
    assert rc == 0, r.stdout + r.stderr
    assert "LEAK WARNING" not in r.stdout


def test_ladder_refuses_a_dirty_tree_before_provisioning(stub_cli, git_stubs):
    """The runtime fetches one pinned commit, so uncommitted work is simply
    absent from the run -- and the resulting failure would read as a
    scientific result rather than a mistake. Refuse before any billing."""
    rc, r = _run_ladder(stub_cli, git_stubs["dirty"], {"STUB_SENTINEL": "STAGE1_OK"})
    assert rc == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stdout and "dirty" in r.stdout
    assert "stub] created" not in r.stdout, "refused too late -- a session was provisioned"


def test_ladder_refuses_an_unpushed_head_before_provisioning(stub_cli, git_stubs):
    """The runtime can only fetch what a remote has."""
    rc, r = _run_ladder(stub_cli, git_stubs["unpushed"], {"STUB_SENTINEL": "STAGE1_OK"})
    assert rc == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stdout and "not on any remote" in r.stdout
    assert "stub] created" not in r.stdout, "refused too late -- a session was provisioned"


def test_a_distinct_absent_code_can_be_declared_without_rewriting_recipes(stub_cli):
    """If `stop` ever grows a separate exit code for "already absent",
    pointing STOP_ABSENT_RC at it must restore the current meaning: absent
    is the goal, not a leak."""
    rc, r = _run_target(stub_cli, {"STUB_STOP_RC": "3"}, make_vars=("STOP_ABSENT_RC=3",))
    assert rc == 0, (
        f"declaring 3 as the absent code should make it a success -- got {rc}\n"
        f"{r.stdout}{r.stderr}")
    assert "LEAK WARNING" not in r.stdout


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
