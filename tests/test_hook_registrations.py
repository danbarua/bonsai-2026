""".claude/settings.json is a shared surface. Keep the provenance-capture
registrations intact across edits.

`.claude/settings.json` carries more than one thing at once: the
provenance-capture hooks (`PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
`SessionStart`), which this repository's own record-keeping depends on, and
optional per-session tooling -- c2c mail-awareness hooks, the CodeGraph MCP
client's own `UserPromptSubmit` registration, and whatever else a
contributor's coding agent adds for itself. Only the FIRST kind is required
here. c2c mail-awareness in particular is a multi-agent coordination
convenience some sessions use, not something anyone needs in order to clone
this repository and work with the code or data -- so `make test` must not
fail for a contributor (or a stripped-down working copy) that does not have
it registered. `REQUIRED` names only what this repo's own science tooling
actually depends on; nothing here should be added for a comfort, not a
dependency.

The failure this DOES guard is silent and real: an edit to the shared file
that drops the provenance-capture registrations leaves a green suite and
this repository's own audit trail quietly stopped. That is CLAUDE.md
principle 20 -- hand-verified functionality becomes an executable test once
confirmed -- applied to what this repo actually needs, not to every hook
any session happens to have registered for itself.

Asserted in both directions, per principle 21: every registration that must
exist does, and every hook script on disk is either registered or carries a
named exemption that is itself checked. Neither direction requires c2c-mail
or any other optional, per-session tooling to be present.
"""
import json
import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"

# (event, substring identifying the script) -- the registrations this
# repository's own tooling (provenance capture) depends on and that must
# survive any future edit to the shared file. Deliberately does NOT include
# c2c-mail or any other optional multi-agent-coordination convenience --
# nobody needs those to clone this repo and work with the code or data, so
# their absence must never fail `make test`.
REQUIRED = [
    ("SessionStart", "provenance-capture/capture.sh"),
    ("PreToolUse", "provenance-capture/capture.sh"),
    ("PostToolUse", "provenance-capture/capture.sh"),
    ("PostToolUseFailure", "provenance-capture/capture.sh"),
]

# Shell files under .claude/hooks/ that are deliberately NOT registered.
# Each needs a reason, and a test below asserts each still exists -- an
# exemption naming a deleted file is an exemption hiding a real gap.
NOT_REGISTERED = {
    "c2c-mail/lib/c2c_mail.sh":
        "sourced by the three c2c hooks, never invoked as a hook itself",
    "c2c-mail/test/break-tests.sh":
        "the c2c hooks' own break-test runner, invoked by a human",
    "c2c-mail/test/pre-c2c-mcp.sh":
        "test fixture for the pre-c2c-mcp hook, not a registration",
    "c2c-mail/test/bench-post-tool-use.sh":
        "benchmark for the PostToolUse mail hook, invoked by a human; "
        "committed so its numbers are reproducible rather than quoted "
        "from a heredoc",
}


def _settings() -> dict:
    return json.loads(SETTINGS.read_text())


def _commands_for(event: str, settings: dict | None = None) -> list[str]:
    out = []
    for group in (settings or _settings()).get("hooks", {}).get(event, []):
        for hook in group.get("hooks", []):
            if hook.get("command"):
                out.append(hook["command"])
    return out


def missing_registrations(settings: dict) -> list[str]:
    """Required registrations absent from `settings`.

    A pure function over a settings dict rather than a check welded to the
    real file, so the break-test below can feed it a deliberately damaged
    copy. Breaking the live `.claude/settings.json` by hand would mean
    editing the file that disables this repo's mail-awareness hooks -- a
    genuinely dangerous edit to make casually, and one the permission
    classifier correctly refuses.
    """
    return [f"{event} -> {fragment}" for event, fragment in REQUIRED
            if not any(fragment in c
                       for c in _commands_for(event, settings))]


def test_settings_file_parses_and_registers_something():
    """The vacuity guard: every assertion below is over an extracted set."""
    events = _settings().get("hooks", {})
    total = sum(len(_commands_for(e)) for e in events)
    print(f"\n[hooks] {len(events)} events, {total} registrations in "
          f"{SETTINGS.relative_to(REPO_ROOT)}")
    assert total >= len(REQUIRED), (
        f"only {total} registrations found; expected at least {len(REQUIRED)}")


def test_every_required_registration_is_present():
    """The provenance-capture hooks this repository's own record-keeping
    depends on, asserted together.

    Listed rather than derived because the mapping of script -> event is a
    design decision, not a fact about the filesystem: `capture.sh` on
    `Stop` would be wrong even though both exist.
    """
    missing = missing_registrations(_settings())
    for event, fragment in REQUIRED:
        if f"{event} -> {fragment}" not in missing:
            print(f"[hooks] ok  {event:20s} {fragment}")
    assert not missing, (
        "registrations missing from .claude/settings.json:\n"
        + "\n".join(f"  {m}" for m in missing)
        + "\nAn edit to this shared file dropped provenance-capture "
          "registrations this repository's own record-keeping depends on.")


def test_every_registered_script_exists_and_is_executable():
    """A registration naming a missing or non-executable file is a hook that
    silently never runs -- which for a fail-open hook looks like success.

    Only checks commands shaped as a project-local `.claude/` file
    reference. A registration can also be a bare CLI invocation owned by
    its own tool -- `codegraph prompt-hook` is one, installed by the
    CodeGraph MCP client rather than living under this repo's
    `.claude/hooks/` -- and treating every registered command as if it
    must resolve to a local script path would break `make test` for
    anyone running a coding agent whose own hooks this repository never
    pre-registered, which is a portability bug, not a real missing file."""
    problems = []
    for event in _settings().get("hooks", {}):
        for command in _commands_for(event):
            if ".claude/" not in command:
                continue
            # Commands are of the form "$CLAUDE_PROJECT_DIR"/.claude/hooks/...
            path = REPO_ROOT / ".claude" / command.split(".claude/")[-1].strip('"')
            if not path.exists():
                problems.append(f"{event}: missing {path}")
            elif not os.stat(path).st_mode & stat.S_IXUSR:
                problems.append(f"{event}: not executable {path}")
    assert not problems, "\n".join(problems)


def test_a_foreign_bare_command_registration_does_not_false_positive():
    """Break-confirmation, the other direction from
    `test_the_check_actually_fails_when_a_registration_is_dropped`: a
    third-party tool's own hook entry (no `.claude/` reference at all,
    e.g. a custom agentic setup registering its own command) must not be
    reported as a missing file. Constructs a synthetic settings dict with
    ONLY such an entry -- if this fails, the portability fix above
    regressed to requiring every command to be a local path again."""
    import ast
    import inspect

    source = inspect.getsource(test_every_registered_script_exists_and_is_executable)
    tree = ast.parse(source)
    # Confirms the guard is actually present in the function body, not
    # merely that this test's own belief about it happens to hold -- a
    # regression that deleted the `continue` would still pass a test that
    # only checked behavior on ONE synthetic foreign command below, if
    # that command happened not to collide with a real path.
    assert any(isinstance(node, ast.Continue) for node in ast.walk(tree)), (
        "the foreign-command skip (`if \".claude/\" not in command: continue`) "
        "is missing from test_every_registered_script_exists_and_is_executable")


def test_every_hook_script_is_registered_or_exempted():
    """The other direction, per principle 21.

    Derived from the filesystem: a new hook script that nobody registered
    is dead code, and a registration that lost its script is a dead hook.
    """
    on_disk = {p.relative_to(HOOKS_DIR).as_posix()
               for p in HOOKS_DIR.rglob("*.sh")}
    registered = {c.split("hooks/")[-1].strip('"')
                  for event in _settings().get("hooks", {})
                  for c in _commands_for(event)}
    unaccounted = on_disk - registered - set(NOT_REGISTERED)
    print(f"[hooks] {len(on_disk)} scripts on disk, {len(registered)} "
          f"registered, {len(NOT_REGISTERED)} exempted")
    assert not unaccounted, (
        f"hook scripts neither registered nor exempted: {sorted(unaccounted)}")


def test_the_check_actually_fails_when_a_registration_is_dropped():
    """The break-test, committed rather than performed once by hand.

    A guard you have not seen fail is not yet a guard (CLAUDE.md principle
    21's corollary). Two different provenance-capture events are dropped in
    turn -- not because a single missing registration is somehow
    insufficient evidence, but so the check cannot be satisfied by a
    function that happens to special-case whichever ONE event it was
    written and tested against.
    """
    for event, fragment in [("PreToolUse", "provenance-capture/capture.sh"),
                            ("PostToolUseFailure", "provenance-capture/capture.sh")]:
        damaged = json.loads(json.dumps(_settings()))
        damaged["hooks"][event] = [
            g for g in damaged["hooks"][event]
            if not any(fragment in h.get("command", "")
                       for h in g.get("hooks", []))]
        missing = missing_registrations(damaged)
        print(f"[hooks] break {event}/{fragment} -> detected {missing}")
        assert f"{event} -> {fragment}" in missing, (
            f"dropping {fragment} from {event} was NOT detected -- the "
            f"registration check is vacuous")

    # Non-vacuity: the undamaged settings must produce no findings, or the
    # assertions above would pass on a function that always reports missing.
    assert missing_registrations(_settings()) == []


def test_every_exemption_still_names_a_real_file():
    """An exemption for a deleted file is an exemption concealing a gap."""
    for relative, reason in NOT_REGISTERED.items():
        assert reason, f"exemption {relative} carries no reason"
        assert (HOOKS_DIR / relative).exists(), (
            f"{relative} is exempted as 'not a registration' but no longer "
            f"exists -- remove the exemption")
