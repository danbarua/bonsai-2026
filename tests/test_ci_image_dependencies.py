"""Every command the CI scripts require must be installed in the CI image.

The incident, 2026-08-08: `tools/ci/review_delta.sh` uses `jq` four times and
guards it with `command -v jq || fail_open`. The Cloud Build image installed
`git` and `make` and nothing else. So on Linux the script failed open to
`mode=full` on every invocation -- correct behaviour, safe direction,
announced on stderr -- and `tests/test_review_delta.py`, which pins specific
modes, went red three ways.

The third failure is the one that matters. It was the break-confirmation:
restore the `.filename`-only filter and assert the original defect reappears.
Without `jq` the script returns `full` either way, so breaking it produced NO
observable change and the test reported exactly that -- *"these tests are not
pinning the field-semantics bug"*. A guard that cannot tell broken from
fixed, saying so out loud, is the best available version of this failure.

The fix is not "add jq". Adding jq fixes today's instance and leaves the next
one to be found the same way -- a hand-maintained package list standing in
for a derivable set, which is principle 21 and this project's most-repeated
bug. So the set is DERIVED: every `command -v X` in a script CI runs must be
installed, or carry a named exemption with a reason, and the exemption is
itself checked to still refer to something real.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUDBUILD = REPO_ROOT / "cloudbuild.yaml"
CI_SCRIPTS = sorted((REPO_ROOT / "tools" / "ci").glob("*.sh"))

# A command may legitimately be absent from the image when CI never invokes
# the real thing. Each entry says why, and each is checked below to still
# describe reality -- an exemption whose justification has evaporated is the
# stale-exemption shape principle 21 warns about.
EXEMPT = {
    "gh": (
        "CI never calls the real `gh`. The scripts that need it are exercised "
        "only through a stub on PATH (tests/test_review_delta.py, "
        "tests/test_review_run.py), because a build that could reach the "
        "GitHub API would also need a credential CI deliberately does not have."
    ),
}

# Command name and Debian package name are different things -- the same class
# of confusion as an import name versus a distribution name. They coincide for
# everything used here; anything that does not must be mapped explicitly
# rather than assumed.
COMMAND_TO_PACKAGE = {
    "git": "git",
    "make": "make",
    "jq": "jq",
}


def required_commands() -> dict[str, list[str]]:
    """{command: [scripts that guard on it]}, read from the scripts."""
    found: dict[str, list[str]] = {}
    for script in CI_SCRIPTS:
        for match in re.finditer(r"command -v (\w[\w.-]*)", script.read_text()):
            found.setdefault(match.group(1), []).append(script.name)
    return found


def _step_bodies() -> dict[str, str]:
    import yaml

    doc = yaml.safe_load(CLOUDBUILD.read_text())
    bodies = {}
    for step in doc.get("steps", []):
        body = step.get("script") or "\n".join(step.get("args") or [])
        bodies[step.get("id") or f"step{len(bodies)}"] = body
    return bodies


def suite_step_id() -> str:
    """The id of the step that runs the pytest suite.

    Derived from the step that INVOKES a test target, not named here. Each
    Cloud Build step is its own container, so a package installed in one step
    is absent in the next -- taking the union across steps would report a
    command as available in a container that never installed it.
    """
    matching = [
        step_id
        for step_id, body in _step_bodies().items()
        if re.search(r"^\s*make\s+\S*test", body, re.M) or re.search(r"^\s*pytest\s", body, re.M)
    ]
    assert len(matching) == 1, (
        f"expected exactly one step to run the test suite, found {matching}. "
        "If the suite genuinely moved or split, this derivation needs "
        "updating -- it must not silently pick the wrong container"
    )
    return matching[0]


def installed_packages() -> set[str]:
    """Packages installed by the step that runs the suite."""
    body = _step_bodies()[suite_step_id()]
    packages = set()
    for line in re.findall(r"apt-get install ([^\n]*)", body):
        for token in line.split():
            if token.startswith((">", "|", "&", "-")):
                continue
            if token == "install":
                continue
            packages.add(token)
    return packages


def test_the_scan_finds_the_guards_it_is_supposed_to_find():
    """Anti-vacuity: if the scan breaks, everything below passes for nothing.

    This whole file is an assertion about a derived set. A regex that stopped
    matching would report an empty requirement set and go green while the
    image was missing everything.
    """
    assert CI_SCRIPTS, "no shell scripts found under tools/ci/"
    required = required_commands()
    assert len(required) >= 2, (
        f"the `command -v` scan found only {sorted(required)}; the scripts do "
        "guard on more than that, so the scan is broken"
    )
    assert "jq" in required, (
        "no script guards on jq any more -- if that is genuinely true this "
        "file needs rewriting, not the assertion deleting"
    )


def test_every_command_the_ci_scripts_need_is_installed_or_exempt():
    """The guard proper.

    Derived from the scripts, not from a list somebody remembered to update.
    """
    installed = installed_packages()
    missing = {}

    for command, scripts in sorted(required_commands().items()):
        if command in EXEMPT:
            continue
        package = COMMAND_TO_PACKAGE.get(command, command)
        if package not in installed:
            missing[command] = (package, scripts)

    assert not missing, (
        "the CI image does not install command(s) the CI scripts guard on. "
        "Each will fail open at runtime -- the safe direction, but it means "
        "the affected tests pin nothing, and a break-confirmation cannot tell "
        f"a fixed script from a broken one: {missing}. "
        f"Installed: {sorted(installed)}"
    )


def test_each_exemption_still_describes_something_real():
    """An exemption for a command nothing requires is stale, not permissive."""
    required = required_commands()
    unused = sorted(set(EXEMPT) - set(required))
    assert not unused, (
        f"exemption(s) for command(s) no CI script requires any more: {unused}. "
        "Remove the exemption rather than leaving it to authorise a future "
        "absence nobody reasoned about"
    )


def test_the_gh_exemption_rests_on_a_stub_that_actually_exists():
    """The `gh` exemption is only sound while the tests really stub `gh`.

    If a test ever invoked the real `gh` in CI it would reach the network
    without a credential, and this exemption would be the reason nobody
    noticed it was missing from the image.
    """
    stubbing_tests = [
        path
        for path in (REPO_ROOT / "tests").glob("test_*.py")
        if '"gh"' in path.read_text() and "PATH" in path.read_text()
    ]
    assert stubbing_tests, (
        "the `gh` exemption claims the scripts are exercised through a stub on "
        "PATH, and no test appears to build one any more"
    )
