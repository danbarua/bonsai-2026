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


def required_commands(scripts: list[Path] | None = None) -> dict[str, list[str]]:
    """{command: [scripts that guard on it]}, read from the scripts."""
    found: dict[str, list[str]] = {}
    for script in (CI_SCRIPTS if scripts is None else scripts):
        for match in re.finditer(r"command -v (\w[\w.-]*)", script.read_text()):
            found.setdefault(match.group(1), []).append(script.name)
    return found


def _step_bodies(text: str | None = None) -> dict[str, str]:
    import yaml

    doc = yaml.safe_load(CLOUDBUILD.read_text() if text is None else text)
    bodies = {}
    for step in doc.get("steps", []):
        body = step.get("script") or "\n".join(step.get("args") or [])
        bodies[step.get("id") or f"step{len(bodies)}"] = body
    return bodies


def suite_step_id(text: str | None = None) -> str:
    """The id of the step that runs the pytest suite.

    Derived from the step that INVOKES a test target, not named here. Each
    Cloud Build step is its own container, so a package installed in one step
    is absent in the next -- taking the union across steps would report a
    command as available in a container that never installed it.
    """
    matching = [
        step_id
        for step_id, body in _step_bodies(text).items()
        if re.search(r"^\s*make\s+\S*test", body, re.M) or re.search(r"^\s*pytest\s", body, re.M)
    ]
    assert len(matching) == 1, (
        f"expected exactly one step to run the test suite, found {matching}. "
        "If the suite genuinely moved or split, this derivation needs "
        "updating -- it must not silently pick the wrong container"
    )
    return matching[0]


def installed_packages(text: str | None = None) -> set[str]:
    """Packages installed by the step that runs the suite."""
    body = _step_bodies(text)[suite_step_id(text)]
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


def missing_commands(
    scripts: list[Path] | None = None, cloudbuild_text: str | None = None
) -> dict[str, tuple[str, list[str]]]:
    """The composition: which required commands the suite's image lacks.

    Extracted so it can be run against SYNTHETIC scripts and a synthetic
    image, which is the only way a break-confirmation for this file can be
    committed rather than performed by hand and left in a transcript.

    The hand version was done -- drop `jq` from the install line, watch it
    fail, restore -- and this project's own review pointed out that it does
    not count: nobody can re-check it and nothing fails when it regresses.
    Principle 20, aimed at the guard rather than the finding.
    """
    installed = installed_packages(cloudbuild_text)
    missing: dict[str, tuple[str, list[str]]] = {}

    for command, using in sorted(required_commands(scripts).items()):
        if command in EXEMPT:
            continue
        package = COMMAND_TO_PACKAGE.get(command, command)
        if package not in installed:
            missing[command] = (package, using)
    return missing


def test_every_command_the_ci_scripts_need_is_installed_or_exempt():
    """The guard proper.

    Derived from the scripts, not from a list somebody remembered to update.
    """
    missing = missing_commands()
    assert not missing, (
        "the CI image does not install command(s) the CI scripts guard on. "
        "Each will fail open at runtime -- the safe direction, but it means "
        "the affected tests pin nothing, and a break-confirmation cannot tell "
        f"a fixed script from a broken one: {missing}. "
        f"Installed: {sorted(installed)}"
    )


_SYNTHETIC_CLOUDBUILD = """\
steps:
  - id: prep
    script: |
      apt-get install -y curl
  - id: verify
    script: |
      apt-get install -y -qq --no-install-recommends git make
      make stage2b-test
"""


def _synthetic_script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def test_a_required_command_missing_from_the_image_is_caught(tmp_path: Path):
    """Break-confirmation, committed rather than performed by hand.

    The hand version was done -- drop `jq` from the install line, watch the
    guard fail, restore -- and this project's own review pointed out that it
    proves nothing durable: nobody can re-check it, and nothing fails when a
    later change breaks the guard.

    Here the scripts and the image are both synthetic, so the composition
    runs against an input it must REJECT.
    """
    script = _synthetic_script(
        tmp_path, "needs_things.sh", "#!/bin/bash\ncommand -v yq || exit 1\n"
    )
    missing = missing_commands([script], _SYNTHETIC_CLOUDBUILD)

    assert "yq" in missing, (
        "a command required by a CI script and absent from the suite step's "
        "install line was not reported")
    assert missing["yq"] == ("yq", ["needs_things.sh"]), (
        "the finding must name the package and the script that needs it, or "
        "the failure cannot say what to install or why")


def test_a_required_command_present_in_the_image_is_not_caught(tmp_path: Path):
    """Anti-vacuity for the test above: the guard must not flag everything.

    Without this, `missing_commands` returning its whole input would satisfy
    the rejection test while being useless.
    """
    script = _synthetic_script(
        tmp_path, "needs_git.sh", "#!/bin/bash\ncommand -v git || exit 1\n"
    )
    assert missing_commands([script], _SYNTHETIC_CLOUDBUILD) == {}


def test_the_exemption_still_spares_its_command_under_synthetic_input(
    tmp_path: Path,
):
    """`gh` is exempt, and the exemption must be what spares it -- not luck."""
    script = _synthetic_script(
        tmp_path, "needs_gh.sh", "#!/bin/bash\ncommand -v gh || exit 1\n"
    )
    assert missing_commands([script], _SYNTHETIC_CLOUDBUILD) == {}
    assert "gh" in EXEMPT


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
