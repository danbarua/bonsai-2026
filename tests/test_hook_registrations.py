""".claude/settings.json is a shared surface. Keep both features registered.

Two independent tracks now register hooks in one file: the c2c mail-awareness
hooks (`UserPromptSubmit`, `Stop`, `SessionStart`, and a `PreToolUse` on the
c2c MCP tools) and the provenance capture hooks (`PreToolUse`,
`PostToolUse`, `PostToolUseFailure`). Neither owns the file.

The failure this guards is silent and one-sided: a merge, rebase or edit
that keeps *your* registrations and drops the other track's leaves a green
suite, a working feature, and a peer whose tooling quietly stopped firing.
`stage2b-lead` asked for explicit confirmation that the capture merge left
the mail hooks intact -- they depend on mail-awareness to receive anything
at all. That confirmation is worth more as a test than as a sentence in a
reply, which is CLAUDE.md principle 20: hand-verified functionality becomes
an executable test once it is confirmed.

Asserted in both directions, per principle 21: every registration that must
exist does, and every hook script on disk is either registered or carries a
named exemption that is itself checked.
"""
import json
import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"

# (event, substring identifying the script) -- the registrations that must
# survive any future edit to the shared file.
REQUIRED = [
    ("UserPromptSubmit", "c2c-mail/user-prompt-submit.sh"),
    ("PostToolUse", "c2c-mail/post-tool-use.sh"),
    ("Stop", "c2c-mail/stop.sh"),
    ("SessionStart", "c2c-mail/session-start.sh"),
    ("PreToolUse", "c2c-mail/pre-c2c-mcp.sh"),
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
    """Both tracks' hooks, asserted together.

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
        + "\nTwo tracks share this file. Dropping the other track's hooks "
          "leaves a green suite and a peer whose tooling stopped firing.")


def test_every_registered_script_exists_and_is_executable():
    """A registration naming a missing or non-executable file is a hook that
    silently never runs -- which for a fail-open hook looks like success."""
    problems = []
    for event in _settings().get("hooks", {}):
        for command in _commands_for(event):
            # Commands are of the form "$CLAUDE_PROJECT_DIR"/.claude/hooks/...
            path = REPO_ROOT / ".claude" / command.split(".claude/")[-1].strip('"')
            if not path.exists():
                problems.append(f"{event}: missing {path}")
            elif not os.stat(path).st_mode & stat.S_IXUSR:
                problems.append(f"{event}: not executable {path}")
    assert not problems, "\n".join(problems)


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
    21's corollary). The specific scenario is the one that motivated this
    file: a future edit keeps the provenance hooks and silently drops the
    c2c-mail ones, leaving a green suite and a peer whose mail-awareness
    stopped firing.

    Both directions are asserted. Dropping the OTHER track's hook must
    fail -- that is the point -- and dropping one of MINE must fail too, so
    the check cannot be satisfied by whoever edits it last.
    """
    for event, fragment in [("Stop", "c2c-mail/stop.sh"),
                            ("PostToolUse", "provenance-capture/capture.sh")]:
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
