"""What this repository assumes about the `mighty-colab` CLI, pinned.

Every GPU target in the `Makefile` is a shell recipe wrapped around this
CLI, and three of its behaviours are load-bearing in ways a reader of the
recipe would not guess:

1. Under `--json`, the CLI's exit code answers only "did the client
   complete its transaction", and the remote job's own outcome moves into
   the envelope's `status` (`ok` / `job_raised` / `error`). A recipe that
   branches on the exit code alone therefore reads a CRASHED remote job as
   a success. This inverts the pre-0.4.0 contract, under which an uncaught
   remote exception was the thing that made `exec` exit non-zero.
2. `exec --timeout` bounds the gap between outputs and defaults to **30
   seconds**, so a healthy remote script that simply goes quiet for longer
   than that dies with `TimeoutError: Timeout waiting for output`. Every
   long-running driver in this project goes quiet for far longer.
3. `status` and `stop` answer an absent session differently, on purpose:
   `status` errors (`session_not_found`), `stop` succeeds
   (`already_stopped`). A query about something missing is a failure; a
   desired-state operation whose work is already done is not. Both are
   load-bearing -- the first keeps an unanswerable status from provisioning
   a second billable VM, the second keeps unconditional teardown adoptable.

All three were established by hand -- (1) and (3) by running the three
commands against an absent session before the recipes were rewritten, (2)
when `stage2a-class0-classify-gpu` was run as a target for the first time
and died 30 seconds in, having never completed once since it was codified
from a hand-run session. CLAUDE.md principle 20: a hand-verified property
that lives only in a comment is not re-checkable and nothing fails when a
dependency upgrade invalidates it -- which is exactly what a `mighty-colab`
upgrade did to a neighbouring comment in this same file, twice.

Tier 1 (always runs, no CLI, no network, nothing provisioned) parses the
`Makefile`, and separately drives a real recipe end to end against a stub
CLI so the teardown logic is exercised rather than merely inspected. Tier
2 asks the installed CLI what it actually does; it skips cleanly when the
`gpu` dependency group is not installed. Nothing here creates a session or
bills.
"""
import json
import re
import shutil
import subprocess

import pytest

from _makefile import REPO_ROOT, recipes as _recipes

# Above this, an "explicit timeout" would be no better than the default.
MAX_TRUSTWORTHY_DEFAULT_TIMEOUT_S = 60.0


# Matches `$(MIGHTY_COLAB) exec`, `$(MIGHTY_COLAB_JSON) exec`, and a literal
# `mighty-colab exec`, so the set stays derived when the variable is renamed
# again. It has been renamed once already: the 0.4.1 `--json` migration moved
# every call site to MIGHTY_COLAB_JSON, and the previous two-literal version
# of this function silently returned ZERO recipes -- which
# `test_makefile_has_gpu_recipes_to_check` caught, exactly the failure it was
# written for, before the timeout and teardown checks below could pass
# vacuously on an empty set.
# Matches `exec` AND `exec-async`. The trailing-space-after-`exec` version of
# this pattern silently excluded every async recipe from the contract checks
# below -- a narrowing that would have passed vacuously the day an async
# target was added, which is exactly principle 21's failure. Widened here, in
# the commit that adds the first such target rather than speculatively.
_EXEC_CALL = re.compile(
    r"(?:\$\(MIGHTY_COLAB(?:_JSON)?\)|mighty-colab\)?) exec(?:-async)? ")

# A detached recipe reaches the CLI through the submit_async template rather
# than naming `exec-async` itself, so matching only the literal missed it
# entirely -- caught by this file's own anti-vacuity assert, which reported
# zero async recipes the moment the call moved into a template.
_ASYNC_CALL = re.compile(r"\$\(call submit_async,")


def _exec_recipes():
    return {name: body for name, body in _recipes().items()
            if _EXEC_CALL.search(body) or _ASYNC_CALL.search(body)}


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
        if _is_async(body):
            continue   # covered by test_the_async_submit_template_passes_a_timeout
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


# The four templates that carry the whole `--json` contract. A recipe using
# some but not all of them is the dangerous state: e.g. one that captures an
# envelope but never reads its status looks migrated and silently treats a
# crashed remote job as a pass, because under `--json` the CLI still exits 0.
_REQUIRED_TEMPLATE_CALLS = ("ensure_session", "check_run_verdict",
                            "stop_session", "check_teardown")

# A detached recipe cannot use check_run_verdict: that template reads the
# envelope a SYNCHRONOUS exec returns in `$$out`, and exec-async returns
# immediately with {"status":"started","pid":...}. The job's outcome arrives
# later, so the verdict half of the contract is replaced -- not dropped -- by
# submit_async (refuses to proceed unless the CLI says it started) and
# await_async (follows to a terminal envelope status).
_ASYNC_REQUIRED_TEMPLATE_CALLS = ("ensure_session", "submit_async",
                                  "await_async", "stop_session",
                                  "check_teardown")

# What await_async itself must do. The CLI publishes a machine-readable
# completion oracle -- `log --tail --json` returns the sidecar's terminal
# status when the job is done, `running` while its pid lives, and
# `worker_terminated` when the pid is gone with no sidecar -- so the template
# must READ it. Grepping the log for a sentinel is not a substitute: a worker
# killed by OOM or by the backend prints no traceback, so a grep-driven loop
# sees nothing and polls until its ceiling while nothing is running.
_AWAIT_REQUIREMENTS = (
    (re.compile(r"log -s \$\(1\) --tail"), "polls `log --tail`"),
    (re.compile(r"jstat=.*\.status"), "reads the envelope's status"),
    (re.compile(r'if \[ "\$\$jstat" != "running" \]; then break'),
     "stops on any terminal status, not just success"),
    (re.compile(r'\[ "\$\$jstat" != "ok" \]'), "fails on a non-ok terminal status"),
    (re.compile(r"--since-offset"), "polls incrementally by byte offset"),
    (re.compile(r"grep -q '\$\(4\)'"), "still checks the driver's own sentinel"),
    (re.compile(r"while \[ \$\$waited -lt \$\(2\) \]"), "bounds its poll loop"),
)


def _is_async(body):
    return bool(_ASYNC_CALL.search(body)) or "exec-async " in body


def _template(name):
    """The body of a `define <name> ... endef` block in the Makefile."""
    text = (REPO_ROOT / "Makefile").read_text()
    m = re.search(rf"^define {re.escape(name)}$(.*?)^endef$", text,
                  re.M | re.S)
    assert m, f"no `define {name}` block in the Makefile"
    return m.group(1)


def test_the_async_await_template_reads_the_cli_completion_oracle():
    """await_async must derive completion from the CLI's own envelope.

    Checked against the template body rather than the recipe, because that is
    where the logic lives -- and by SHAPE rather than by token presence, after
    break-confirmation caught a token check passing vacuously (the word
    MAX_WAIT survived in a failure message when the loop became unbounded)."""
    body = _template("await_async")
    missing = [why for pat, why in _AWAIT_REQUIREMENTS if not pat.search(body)]
    print(f"\n[contract] await_async: {'complete' if not missing else 'MISSING ' + '; '.join(missing)}")
    assert not missing, ("await_async cannot reliably tell a finished job from "
                         "a dead one:\n  " + "\n  ".join(missing))


def test_the_async_submit_template_passes_a_timeout():
    """The 30s default bites hardest on exec-async -- it is exactly the
    command used for long, quiet jobs, and the value is forwarded unchanged
    to the detached child."""
    body = _template("submit_async")
    assert "--timeout $(3)" in body, (
        "submit_async does not forward a timeout, so every detached job "
        "inherits the 30-second default it was least suited to")


def test_the_async_submit_template_truncates_a_stale_log():
    """--output-log is a fixed path, and await_async proves success partly by
    grepping that file for the driver's sentinel. Without truncation, a log
    left by an earlier successful run satisfies the grep whatever this run
    does -- the sentinel check silently becomes "it worked once"."""
    body = _template("submit_async")
    assert ": > $(4);" in body, (
        "submit_async does not truncate the output log before submitting, so "
        "a previous run's sentinel can be mistaken for this run's")


def test_the_async_submit_template_refuses_a_job_that_did_not_start():
    body = _template("submit_async")
    assert 'astat=' in body and '.status' in body, (
        "submit_async ignores exec-async's envelope, so a refused submission "
        "would be followed by polling an empty log until the ceiling")
    assert '"$$astat" != "started"' in body, (
        "submit_async does not require status=started")


def test_every_async_recipe_carries_the_async_verdict_contract():
    """exec-async recipes are exempt from check_run_verdict and nothing else.
    Derived from the Makefile, so the next detached target is covered on the
    day it is written."""
    async_recipes = {n: b for n, b in _exec_recipes().items() if _is_async(b)}
    print(f"\n[contract] async recipes: {sorted(async_recipes)}")
    assert async_recipes, ("no async recipes parsed -- if the Makefile still "
                           "has one, _EXEC_CALL has gone stale and this test "
                           "is vacuous")
    offenders = {}
    for name, body in async_recipes.items():
        missing = [c for c in _ASYNC_REQUIRED_TEMPLATE_CALLS
                   if f"$(call {c}," not in body]
        print(f"[contract] {name}: {'complete' if not missing else 'MISSING ' + '; '.join(missing)}")
        if missing:
            offenders[name] = missing
    assert not offenders, (
        "these detached GPU recipes are only partly on the async contract:\n"
        + "\n".join(f"  {n}: missing {'; '.join(m)}" for n, m in offenders.items()))


def test_every_exec_recipe_uses_the_whole_json_contract():
    """Derived, so a GPU target added later is covered on the day it is
    written rather than whenever someone remembers this file exists.

    Half-migration is the failure this guards. The migration touched three
    branch points per recipe across thirteen recipes; a recipe that got the
    session guard and the teardown but not the verdict check would pass every
    other test here while reading a raised remote job as success."""
    offenders = {}
    sync = {n: b for n, b in _exec_recipes().items() if not _is_async(b)}
    assert sync, ("no synchronous exec recipes parsed -- this test has gone "
                  "vacuous, or every recipe became detached")
    for name, body in sync.items():
        missing = [c for c in _REQUIRED_TEMPLATE_CALLS
                   if f"$(call {c}," not in body]
        print(f"[contract] {name}: {'complete' if not missing else 'MISSING ' + ','.join(missing)}")
        if missing:
            offenders[name] = missing
    assert not offenders, (
        "these GPU recipes are only partly on the --json contract:\n"
        + "\n".join(f"  {n}: missing {', '.join(m)}" for n, m in offenders.items()))


def test_no_recipe_folds_stderr_into_a_captured_envelope():
    """`2>&1` on a `--json` capture is silently fatal.

    Under `--json` the CLI's chatter moves to stderr and stdout carries the
    envelope alone. Merging them puts log lines in front of the JSON, so
    every `jq` read returns "malformed" and the recipe fails for a reason
    unrelated to the run. Every recipe here did exactly this before the
    migration, because under the old contract it was correct."""
    offenders = []
    for name, body in _exec_recipes().items():
        for line in body.splitlines():
            if "MIGHTY_COLAB_JSON)" in line and "2>&1" in line:
                offenders.append((name, line.strip()))
    assert not offenders, (
        "a --json capture folds stderr into the envelope it parses:\n"
        + "\n".join(f"  {n}: {l}" for n, l in offenders))


def test_every_exec_recipe_expands_to_valid_shell():
    """Parses each recipe as the shell would, without running any of it.

    The stub-driven tests below execute two recipes end to end; the other
    eleven were only ever INSPECTED, and a static `$(call ...)` presence
    check cannot see an unbalanced quote or a missing `fi`. Those surface at
    runtime -- which for these targets means after a GPU is already
    provisioned and billing.

    `make -n` expands the recipe (including every `$(call ...)`) and prints
    it instead of running it; `bash -n` parses without executing. Nothing is
    provisioned and nothing bills."""
    targets = sorted(_exec_recipes())
    assert targets, "no exec recipes parsed -- this check would be vacuous"
    offenders = {}
    for target in targets:
        expanded = subprocess.run(["make", "-n", target], cwd=REPO_ROOT,
                                  capture_output=True, text=True, timeout=120)
        # Both checks matter, and the second is not obvious: `bash -n` on an
        # EMPTY string succeeds, so a `make` that failed to expand at all
        # would sail through as "valid shell". Found by break-confirmation --
        # deleting an `fi` from a template broke the `define` block, make
        # errored, stdout was empty, and this test stayed green.
        if expanded.returncode != 0 or not expanded.stdout.strip():
            offenders[target] = (
                f"`make -n` produced no recipe (exit {expanded.returncode}): "
                f"{expanded.stderr.strip()[:400]}")
            print(f"[contract] {target}: DID NOT EXPAND")
            continue
        parsed = subprocess.run(["bash", "-n"], input=expanded.stdout,
                                capture_output=True, text=True, timeout=120)
        ok = parsed.returncode == 0
        print(f"[contract] {target}: shell syntax {'ok' if ok else 'INVALID'}")
        if not ok:
            offenders[target] = parsed.stderr.strip()
    assert not offenders, (
        "these recipes do not expand to valid shell:\n"
        + "\n".join(f"  {t}: {e}" for t, e in offenders.items()))


def test_every_session_creating_recipe_tears_down_unconditionally():
    """A recipe that only stops the session on the success path leaks a
    billing VM on every failure -- which is what `exec && download && stop`
    started doing the moment `exec` gained the ability to fail."""
    for name, body in _exec_recipes().items():
        assert "$(call stop_session," in body, f"{name} never stops its session"
        stop_lines = [l for l in body.splitlines() if "$(call stop_session," in l]
        for line in stop_lines:
            assert not line.strip().startswith("&&"), (
                f"{name} chains its teardown onto a previous command's success: {line.strip()}")
        assert "$(call check_teardown," in body, (
            f"{name} stops its session but never reads the result -- a teardown "
            f"that failed would go unreported and the VM would keep billing")
        print(f"[contract] {name}: teardown present and not chained on success")


# Stand-in for the mighty-colab CLI. Provisions nothing, bills nothing.
#
# Python rather than the shell script this replaced, because the contract it
# now stands in for is a JSON envelope: emitting one from `sh` means hand-
# building the string, and a stub whose output is not valid JSON would fail
# the recipe for a reason that has nothing to do with what a test is asking.
#
# Every knob is an environment variable so a test names the ONE condition it
# varies. The envelope shapes below mirror `colab_cli.envelopes`, and the
# three defaults are the healthy path: session absent (so `new` is called),
# job ran, sentinel printed, teardown clean.
STUB = '''#!/usr/bin/env python3
import json, os, sys

ENVELOPE = {"schema_version": "1", "cli_version": "0.4.1-stub"}


def emit(**fields):
    print(json.dumps({**ENVELOPE, **fields}))


argv = sys.argv[1:]
while argv and argv[0].startswith("--"):
    argv.pop(0)
cmd = argv[0] if argv else ""

if cmd == "sessions":
    print("[stub] no active sessions")
    sys.exit(0)

if cmd == "status":
    mode = os.environ.get("STUB_STATUS_MODE", "absent")
    if mode == "present":
        emit(command="status", status="ok", exit_code=0)
        sys.exit(0)
    if mode == "unreachable":
        # Not "absent": the question could not be answered at all. The recipe
        # must refuse rather than provision a second billable VM.
        emit(command="status", status="error", exit_code=1,
             reason="auth_scope_missing", http_status=403)
        sys.exit(1)
    emit(command="status", status="error", exit_code=1,
         reason="session_not_found")
    sys.exit(1)

if cmd == "new":
    print("[stub] created")
    sys.exit(0)

if cmd in ("install", "reinstall", "upload", "download"):
    print(f"[stub] {cmd} ok")
    sys.exit(0)

if cmd == "exec":
    sentinel = os.environ.get("STUB_SENTINEL", "GPU_VERIFY_OK")
    status = os.environ.get("STUB_EXEC_STATUS", "ok")
    outputs = [{"output_type": "stream", "name": "stdout",
                "text": f"[stub] remote ran\\n{sentinel}\\n"}]
    if status != "ok":
        outputs.append({"output_type": "error", "ename": "RuntimeError",
                        "evalue": "stub failure",
                        "traceback": ["Traceback (most recent call last):",
                                      "RuntimeError: stub failure"]})
    fields = dict(command="exec", status=status,
                  exit_code=0 if status == "ok" else 1,
                  blocks=[{"code": "<stub>", "outputs": outputs}])
    if status != "ok":
        fields["reason"] = "remote_exception"
    emit(**fields)
    sys.exit(int(os.environ.get("STUB_EXEC_RC", "0")))

if cmd == "stop":
    status = os.environ.get("STUB_STOP_STATUS", "ok")
    reason = os.environ.get("STUB_STOP_REASON", "")
    fields = dict(command="stop", status=status,
                  exit_code=0 if status == "ok" else 1, session="stub")
    if reason:
        fields["reason"] = reason
    emit(**fields)
    sys.exit(int(os.environ.get("STUB_STOP_RC", "0")))

print(f"[stub] unhandled: {' '.join(argv)}")
sys.exit(0)
'''


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


def test_a_nonzero_exec_fails_the_target_even_when_the_sentinel_is_present(stub_cli):
    """The other half of `[ $rc -ne 0 ] || ! grep sentinel`, and the half a
    sentinel check alone cannot cover: a driver that printed its verdict and
    THEN died -- a crash in teardown, an upload that failed after the
    science ran, a non-zero exit from the exec transport itself.

    The sentinel is deliberately correct here. If the recipe consulted only
    the sentinel, this run would pass; the target's own exit code must
    instead carry `exec`'s, not be reset to 0 by a successful grep."""
    rc, r = _run_target(stub_cli, {"STUB_EXEC_RC": "5"})
    assert rc == 5, (
        f"a nonzero exec must propagate its own code -- got {rc}. A correct "
        f"sentinel must not rescue a run that exited nonzero.\n{r.stdout}{r.stderr}")
    assert "FAILED: the GPU ridge gate" in r.stdout
    assert "CLI itself exited 5" in r.stdout, (
        "the reported code must be the CLI's own, so the failure is diagnosable. "
        "Under --json a non-zero exit means the CLI did not complete its "
        "transaction at all -- distinct from a remote job that raised, which "
        "arrives as status=job_raised with the CLI exiting 0.")


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

GIT_STUB_UNPUSHED = """#!/bin/sh
case "$1" in
  status) exit 0 ;;
  rev-parse) echo "0000000000000000000000000000000000000000"; exit 0 ;;
  branch) exit 0 ;;
  *) exit 0 ;;
esac
"""


# The closure pre-flight is a python CLI into stage2b_fingerprint, so the
# git stubs above cannot reach it -- `git status` is no longer what the
# recipe asks. These stand in for that CLI at the two answers it can give.
# The real checker's own behaviour (which files count as the closure, what
# "dirty" means blob-by-blob against HEAD) is tested directly in
# tests/test_stage2b_fingerprint.py; what is tested HERE is that the recipe
# consults it and obeys it.
CLOSURE_STUB_CLEAN = """#!/bin/sh
echo "[closure] entrypoint $1: closure is clean"
exit 0
"""

CLOSURE_STUB_DIRTY = """#!/bin/sh
echo "[closure] REFUSING: these files in this driver's own source closure differ from HEAD:"
echo "[closure]   experiments/stage2b_denoising/run_ladder_stage1.py"
exit 1
"""


@pytest.fixture(scope="module")
def closure_stubs(tmp_path_factory):
    base = tmp_path_factory.mktemp("closurestub")
    made = {}
    for name, body in (("clean", CLOSURE_STUB_CLEAN), ("dirty", CLOSURE_STUB_DIRTY)):
        path = base / f"closure-{name}"
        path.write_text(body)
        path.chmod(0o755)
        made[name] = path
    return made


@pytest.fixture(scope="module")
def git_stubs(tmp_path_factory):
    base = tmp_path_factory.mktemp("gitstub")
    made = {}
    for name, body in (("ready", GIT_STUB_READY), ("unpushed", GIT_STUB_UNPUSHED)):
        path = base / f"git-{name}"
        path.write_text(body)
        path.chmod(0o755)
        made[name] = path
    return made


def _run_ladder(stub, git_stub, env_extra=None, make_vars=(), closure_stub=None):
    """`closure_stub` defaults to the clean one, so every test that is about
    something else gets past the pre-flight without saying so."""
    extra = list(make_vars)
    if closure_stub is not None:
        extra.append(f"CLOSURE_CHECK={closure_stub}")
    return _run_target(stub, env_extra, (f"GIT={git_stub}", *extra),
                       target=LADDER_TARGET)


def test_ladder_healthy_run_exits_zero(stub_cli, git_stubs, closure_stubs):
    rc, r = _run_ladder(stub_cli, git_stubs["ready"], {"STUB_SENTINEL": "STAGE1_OK"},
                        closure_stub=closure_stubs["clean"])
    assert rc == 0, r.stdout + r.stderr
    assert "LEAK WARNING" not in r.stdout


def test_ladder_teardown_failure_fails_an_otherwise_successful_target(stub_cli, git_stubs, closure_stubs):
    rc, r = _run_ladder(stub_cli, git_stubs["ready"],
                        {"STUB_SENTINEL": "STAGE1_OK", "STUB_STOP_RC": "7"},
                        closure_stub=closure_stubs["clean"])
    assert rc == 7, (
        f"a failed teardown must fail the target, carrying stop's own code -- got {rc}\n"
        f"{r.stdout}{r.stderr}")
    assert "LEAK WARNING" in r.stdout
    assert "FAILED: ladder stage 1" not in r.stdout, (
        "a teardown failure must not be reported as a scientific failure")


def test_ladder_missing_sentinel_fails_even_on_a_zero_exit(stub_cli, git_stubs, closure_stubs):
    """The case the sentinel exists for: `exec` returns 0 because the script
    exited cleanly, but it never reached its own verdict."""
    rc, r = _run_ladder(stub_cli, git_stubs["ready"],
                        {"STUB_SENTINEL": "nothing useful here"},
                        closure_stub=closure_stubs["clean"])
    assert rc == 1, r.stdout + r.stderr
    assert "FAILED: ladder stage 1" in r.stdout


def test_ladder_nonzero_exec_fails_the_target_even_when_the_sentinel_is_present(
        stub_cli, git_stubs, closure_stubs):
    """The converse case, on the target that spends real money: the driver
    printed its verdict and then died. The sentinel is deliberately correct,
    so only the `[ $rc -ne 0 ]` half of the disjunct can catch this."""
    rc, r = _run_ladder(stub_cli, git_stubs["ready"],
                        {"STUB_SENTINEL": "STAGE1_OK", "STUB_EXEC_RC": "5"},
                        closure_stub=closure_stubs["clean"])
    assert rc == 5, (
        f"a nonzero exec must propagate its own code -- got {rc}\n{r.stdout}{r.stderr}")
    assert "FAILED: ladder stage 1" in r.stdout
    assert "CLI itself exited 5" in r.stdout


def test_ladder_absent_session_is_not_treated_as_a_leak(stub_cli, git_stubs, closure_stubs):
    """"Already gone" is the goal, not a failure -- it is "could not stop"
    that costs money, and conflating them makes the check unadoptable."""
    rc, r = _run_ladder(stub_cli, git_stubs["ready"],
                        {"STUB_SENTINEL": "STAGE1_OK",
                         "STUB_STOP_REASON": "already_stopped"},
                        closure_stub=closure_stubs["clean"])
    assert rc == 0, r.stdout + r.stderr
    assert "LEAK WARNING" not in r.stdout


def test_ladder_refuses_a_dirty_source_closure_before_provisioning(
        stub_cli, git_stubs, closure_stubs):
    """The runtime fetches one pinned commit, so an uncommitted file THIS
    DRIVER IMPORTS is simply absent from the run -- and the resulting
    failure would read as a scientific result rather than a mistake. Refuse
    before any billing.

    The check is closure-keyed, not whole-tree: uncommitted work elsewhere
    cannot reach the computation, and refusing on it meant every GPU launch
    waited on a spotless checkout. `test_stage2b_fingerprint.py` tests what
    "dirty closure" means; this tests that the recipe asks and obeys."""
    rc, r = _run_ladder(stub_cli, git_stubs["ready"], {"STUB_SENTINEL": "STAGE1_OK"},
                        closure_stub=closure_stubs["dirty"])
    assert rc == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stdout and "closure" in r.stdout
    assert "stub] created" not in r.stdout, "refused too late -- a session was provisioned"


def test_ladder_proceeds_when_only_unrelated_files_are_uncommitted(
        stub_cli, git_stubs, closure_stubs):
    """The other direction, and the reason the guard was narrowed: a clean
    closure inside a dirty tree must RUN. Without this the change is
    untested in the direction that motivated it -- a guard that refuses
    everything also refuses everything wrong."""
    rc, r = _run_ladder(stub_cli, git_stubs["ready"], {"STUB_SENTINEL": "STAGE1_OK"},
                        closure_stub=closure_stubs["clean"])
    assert rc == 0, r.stdout + r.stderr
    assert "REFUSING" not in r.stdout
    assert "stub] created" in r.stdout, "the run should have reached provisioning"


def test_no_gpu_target_still_gates_on_whole_tree_porcelain():
    """The coarse check must not come back. If it did it would fire first,
    turning the closure check into dead code and restoring the behaviour
    this replaced -- and nothing about a green suite would reveal it,
    because the closure check would still be present and still correct."""
    offenders = []
    for name, body in _exec_recipes().items():
        if "status --porcelain" in body:
            offenders.append(name)
        print(f"[contract] {name}: "
              f"{'STILL GATES ON PORCELAIN' if 'status --porcelain' in body else 'no porcelain gate'}")
    assert not offenders, (
        f"these recipes gate on whole-tree `git status --porcelain`: {offenders}. "
        f"Uncommitted work outside a driver's import closure cannot reach the "
        f"computation -- the runtime executes one pinned commit. Use "
        f"$(CLOSURE_CHECK) instead.")


def test_every_repo_fetching_gpu_target_runs_the_closure_check():
    """Derived, not a hand-maintained list: any recipe that pins a commit
    for the runtime to fetch must ask whether that commit contains the
    driver's own sources. A new ladder target gets this on the day it is
    written rather than whenever someone remembers."""
    fetching = {name: body for name, body in _exec_recipes().items()
                if "BONSAI_COMMIT" in body}
    assert fetching, "no commit-pinning GPU recipes parsed -- this check is vacuous"
    missing = [name for name, body in fetching.items() if "CLOSURE_CHECK" not in body]
    for name in sorted(fetching):
        print(f"[contract] {name}: "
              f"{'runs the closure check' if 'CLOSURE_CHECK' in fetching[name] else 'NO CLOSURE CHECK'}")
    assert not missing, (
        f"these targets pin a commit for the runtime to fetch but never check that "
        f"the driver's own closure is in it: {missing}")


def test_ladder_refuses_an_unpushed_head_before_provisioning(stub_cli, git_stubs, closure_stubs):
    """The runtime can only fetch what a remote has."""
    rc, r = _run_ladder(stub_cli, git_stubs["unpushed"], {"STUB_SENTINEL": "STAGE1_OK"},
                        closure_stub=closure_stubs["clean"])
    assert rc == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stdout and "not on any remote" in r.stdout
    assert "stub] created" not in r.stdout, "refused too late -- a session was provisioned"


def test_already_stopped_is_a_success_not_a_leak(stub_cli):
    """"Already gone" is the GOAL, and it is the normal case whenever
    provisioning failed before a session existed.

    This replaces the STOP_ABSENT_RC test, and the replacement is the point
    of the migration. That variable existed because "absent" was encoded as
    a bare exit code, so the Makefile had to declare out-of-band which code
    meant which outcome, and a release changing it would silently invert the
    check. 0.4.1 returns `status=ok reason=already_stopped` as DATA, so the
    recipe reads the outcome instead of being told it."""
    rc, r = _run_target(stub_cli, {"STUB_STOP_REASON": "already_stopped"})
    assert rc == 0, (
        f"an already-absent session is not a leak -- got {rc}\n{r.stdout}{r.stderr}")
    assert "LEAK WARNING" not in r.stdout


def test_a_stop_that_reports_failure_is_a_leak_even_when_it_exits_zero(stub_cli):
    """The half a bare exit code could never express, and the reason the
    teardown check reads BOTH signals.

    `stop` completing its transaction (exit 0) while reporting that the
    teardown itself did not succeed is precisely the shape that leaves an
    A100 billing. Under the old contract this run was indistinguishable
    from a clean teardown and would have passed silently."""
    rc, r = _run_target(stub_cli, {"STUB_STOP_STATUS": "error",
                                   "STUB_STOP_REASON": "backend_error",
                                   "STUB_STOP_RC": "0"})
    assert rc == 1, (
        f"a stop reporting failure must fail the target even at exit 0 -- got {rc}\n"
        f"{r.stdout}{r.stderr}")
    assert "LEAK WARNING" in r.stdout


def test_a_raised_remote_job_fails_the_target_though_the_cli_exits_zero(stub_cli):
    """The flagship inversion. Under `--json` the CLI exits 0 whenever it
    completed its transaction, INCLUDING when the remote job raised -- so a
    recipe that still branched on the exit code alone would read a crashed
    run as a success. The verdict has to come from the envelope."""
    rc, r = _run_target(stub_cli, {"STUB_EXEC_STATUS": "job_raised"})
    assert rc == 1, (
        f"a raised remote job must fail the target -- got {rc}. The CLI exited 0 "
        f"because it completed its transaction; the job's outcome is in the "
        f"envelope.\n{r.stdout}{r.stderr}")
    assert "status=job_raised" in r.stdout


def test_a_clean_exit_with_no_sentinel_still_fails(stub_cli):
    """`status=ok` is not the verdict, and this is why the sentinel survived
    the migration. A script that exited cleanly without reaching its own gate
    -- truncated, short-circuited, an early `return` -- produces exactly this
    envelope, and nothing in it distinguishes that from a passed run."""
    rc, r = _run_target(stub_cli, {"STUB_SENTINEL": "NOTHING_USEFUL"})
    assert rc == 1, (
        f"a clean run that never printed its sentinel must fail -- got {rc}\n"
        f"{r.stdout}{r.stderr}")
    assert "never printed its success sentinel" in r.stdout


def test_an_unanswerable_status_refuses_instead_of_provisioning(stub_cli):
    """The guard the prose-grepping version could not express, and the one
    with a direct bill attached.

    `grep -q "not found"` treated ANY unparseable answer as "no session
    here" -- including an auth failure or a backend 5xx, where a session may
    well be up and billing. Provisioning a second one is the worst available
    response. The reason code separates "absent" from "unknown"."""
    rc, r = _run_target(stub_cli, {"STUB_STATUS_MODE": "unreachable"})
    assert rc == 1, f"an unanswerable status must refuse -- got {rc}\n{r.stdout}{r.stderr}"
    assert "REFUSING" in r.stdout
    assert "stub] created" not in r.stdout, (
        "refused too late -- a second billable session was provisioned")


def test_an_existing_session_is_reused_rather_than_duplicated(stub_cli):
    """The other side of the same branch: `status=ok` means it is already up."""
    rc, r = _run_target(stub_cli, {"STUB_STATUS_MODE": "present"})
    assert rc == 0, r.stdout + r.stderr
    assert "Reusing existing session" in r.stdout
    assert "stub] created" not in r.stdout, "provisioned a duplicate session"


@pytest.fixture(scope="module")
def cli():
    found = shutil.which("mighty-colab")
    if found is None:
        pytest.skip("the `mighty-colab` CLI is not on PATH; it ships in this project's "
                    "`gpu` dependency group (`uv run --group gpu pytest ...`)")
    return found


@pytest.mark.parametrize("command,expect_rc,expect_status,expect_reason", [
    ("status", 1, "error", "session_not_found"),
    ("stop", 0, "ok", "already_stopped"),
])
def test_json_envelope_contract_for_an_absent_session(
        cli, command, expect_rc, expect_status, expect_reason):
    """The three-line contract every migrated recipe branches on.

    Hand-measured against 0.4.1 before the recipes were rewritten; pinned
    here so it is re-checked rather than remembered (CLAUDE.md principle 20).
    Reads local session state only -- no network, nothing provisioned,
    nothing billed.

    `status` and `stop` differ deliberately and the difference is the whole
    design: a QUERY about a missing session is an error, while a
    desired-state operation that finds its work already done is a success.
    Collapsing them is what makes unconditional teardown either unsafe or
    unadoptable."""
    name = "bonsai-contract-probe-no-such-session"
    r = subprocess.run([cli, "--json", command, "-s", name],
                       capture_output=True, text=True, timeout=120)
    print(f"\n[contract] `mighty-colab --json {command} -s {name}`")
    print(f"[contract]   exit={r.returncode}")
    print(f"[contract]   stdout={r.stdout.strip()!r}")

    assert r.returncode == expect_rc, (
        f"`--json {command}` on an absent session exited {r.returncode}, expected "
        f"{expect_rc}. The Makefile branches on this; see $(ensure_session) and "
        f"$(stop_session).\n{r.stdout}{r.stderr}")

    envelope = json.loads(r.stdout)
    print(f"[contract]   status={envelope.get('status')!r} "
          f"reason={envelope.get('reason')!r} cli={envelope.get('cli_version')!r}")
    assert envelope["status"] == expect_status
    assert envelope.get("reason") == expect_reason, (
        f"the reason code changed. $(ensure_session) distinguishes "
        f"'session_not_found' from every other failure precisely so an "
        f"unanswerable status does not provision a second billable VM -- a "
        f"renamed reason silently turns that refusal into a provision.")


def test_json_stdout_carries_the_envelope_alone(cli):
    """Why every migrated recipe dropped its `2>&1`.

    Under `--json` the CLI's own chatter moves to stderr and stdout carries
    one envelope. The recipes capture stdout only; folding stderr back in --
    which every recipe did before this migration -- would put log lines in
    front of the JSON and break the parse for reasons unrelated to the run."""
    r = subprocess.run([cli, "--json", "status", "-s", "bonsai-contract-probe-absent"],
                       capture_output=True, text=True, timeout=120)
    print(f"\n[contract] stdout={r.stdout.strip()!r}")
    print(f"[contract] stderr carried {len(r.stderr.splitlines())} line(s)")
    json.loads(r.stdout)  # raises if anything else landed on stdout
    assert r.stdout.strip().count("\n") == 0, (
        "stdout carried more than the single envelope the recipes parse")


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
