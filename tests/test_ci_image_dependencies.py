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

TWO sources, because there are two ways this repository reaches for a system
command. The CI shell scripts guard with `command -v`; the `Makefile`
declares its tools as overridable variables (`JQ ?= jq`) and invokes them at
command position. The second was added when the GPU recipes moved onto
`mighty-colab --json` and began parsing envelopes with `jq`. CI never runs a
GPU target, but `tests/test_mighty_colab_contract.py` drives those recipes
against a stub CLI inside the suite's own container -- so the requirement is
real, and scanning only `tools/ci/*.sh` would have missed it entirely.
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
        "tests/test_review_run.py, tests/test_publish_review.sh), because a "
        "build that could reach the GitHub API would also need a credential "
        "CI deliberately does not have."
    ),
    "claude": (
        "CI never invokes `tools/ci/vacuous_review_local.sh`. That script is a "
        "developer preflight (`make vacuous-review`); the Actions review runs "
        "claude-code-action on a GitHub runner, not this image. Installing the "
        "Claude CLI here would pull a second auth surface into a credential-free "
        "build for a path nothing in cloudbuild.yaml calls."
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


MAKEFILE = REPO_ROOT / "Makefile"

# The Makefile declares the external tools its recipes call as overridable
# variables (`JQ ?= jq`) rather than guarding on them with `command -v`, so
# the scan above cannot see them -- and it needs to. CI does not run the GPU
# targets, but `tests/test_mighty_colab_contract.py` DRIVES those recipes
# end to end against a stub CLI, in CI, in the suite's own container. A
# recipe that shells out to `jq` fails there if the image lacks it.
#
# Two conditions, and BOTH are needed -- either alone is wrong here:
#
#   1. The variable is EXPANDED AT A COMMAND POSITION in a recipe: at the
#      start of a recipe line, or straight after `&&`, `||`, `;`, `|`, `(`,
#      or a shell keyword. This is what separates a tool from a value.
#      Requiring only a bare-token definition matched `SESSION_CLASS0 ?=
#      class0-audit-gpu`, `HAIKU_MODEL ?= haiku` and the GCS bucket name --
#      fifteen "packages" that are nothing of the sort.
#   2. Its definition is a BARE SINGLE TOKEN. This excludes
#      `PYTHON ?= uv run python`, `MIGHTY_COLAB ?= uv run --group gpu
#      mighty-colab` and `CLOSURE_CHECK ?= uv run python ...` -- all invoked
#      at command position, all provided by the uv-managed environment
#      rather than by apt, so demanding them as system packages would fail
#      this file for something the image is right not to install.
_MAKE_TOOL_DEF = re.compile(r"^([A-Z][A-Z0-9_]*)\s*\?=\s*([a-z][\w.-]*)\s*$", re.M)
# A `$(VAR)` whose preceding character begins a new command.
_COMMAND_POSITION = r"(?:^\t|&&\s*|\|\|\s*|[;|(]\s*|\bthen\s+|\belse\s+|\bdo\s+)\$\({var}\)"


def makefile_required_commands(makefile: Path | None = None) -> dict[str, list[str]]:
    """{command: [Makefile]} for external tools its recipes actually invoke."""
    text = (MAKEFILE if makefile is None else makefile).read_text()
    found: dict[str, list[str]] = {}
    for var, command in _MAKE_TOOL_DEF.findall(text):
        pattern = _COMMAND_POSITION.format(var=re.escape(var))
        if re.search(pattern, text, re.M):
            found.setdefault(command, []).append("Makefile")
    return found


def required_commands(scripts: list[Path] | None = None,
                      makefile: Path | None = None) -> dict[str, list[str]]:
    """{command: [sources that need it]}, read from the scripts and Makefile."""
    found: dict[str, list[str]] = {}
    for script in (CI_SCRIPTS if scripts is None else scripts):
        for match in re.finditer(r"command -v (\w[\w.-]*)", script.read_text()):
            found.setdefault(match.group(1), []).append(script.name)
    for command, sources in makefile_required_commands(makefile).items():
        found.setdefault(command, []).extend(sources)
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


def test_the_makefile_scan_finds_the_tool_variables_it_is_supposed_to_find():
    """Anti-vacuity for the Makefile half, which has its own failure mode.

    The `command -v` scan and this one look at different files for different
    syntax, so the assertion above says nothing about whether the Makefile
    contributed anything. If `_MAKE_TOOL_VAR` stopped matching, the Makefile
    would contribute an empty set and this file would go green while the
    image was missing a command its GPU recipes shell out to.
    """
    found = makefile_required_commands()
    print(f"\n[ci-deps] Makefile tool variables: {sorted(found)}")
    assert "jq" in found, (
        "the Makefile's `JQ ?= jq` is no longer detected. Every GPU recipe "
        "parses `--json` envelopes with it, and the contract tests drive "
        "those recipes in CI -- so this is a real requirement, not decoration."
    )
    assert "git" in found, (
        "`GIT ?= git` is no longer detected, so the scan's shape has drifted"
    )


def test_the_makefile_scan_excludes_environment_provided_wrappers():
    """The other direction, and the reason the rule is 'bare single token'.

    `PYTHON ?= uv run python` and `MIGHTY_COLAB ?= uv run --group gpu
    mighty-colab` are not system packages -- demanding them from apt would
    make this file fail for something the image is right not to install.
    """
    found = makefile_required_commands()
    for wrapper in ("uv", "python", "python3", "mighty-colab"):
        assert wrapper not in found, (
            f"{wrapper!r} was picked up as a system package requirement. It is "
            f"provided by the uv-managed environment, not by apt."
        )


def test_a_makefile_tool_that_the_image_lacks_is_reported(tmp_path):
    """Break-confirmation, committed rather than performed by hand.

    A synthetic Makefile declaring a tool the image does not install must be
    reported as missing. Without this, the Makefile half of the derivation
    could silently never contribute to `missing_commands` and nothing would
    say so -- the exact shape of the `STAGE2B_TEST_FILES` incident that
    principle 21 is drawn from.
    """
    fake = tmp_path / "Makefile"
    fake.write_text("SOMETOOL ?= ripgrep\n\ntarget:\n\t$(SOMETOOL) --version\n")
    assert makefile_required_commands(fake) == {"ripgrep": ["Makefile"]}

    missing = missing_commands(scripts=[], makefile=fake)
    assert "ripgrep" in missing, (
        "a Makefile tool absent from the CI image was NOT reported missing, so "
        "the Makefile half of this guard does not actually gate anything"
    )
    print(f"\n[ci-deps] break-confirmation: synthetic tool reported as {missing}")


def missing_commands(
    scripts: list[Path] | None = None,
    cloudbuild_text: str | None = None,
    makefile: Path | None = None,
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

    for command, using in sorted(required_commands(scripts, makefile).items()):
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
    installed = sorted(installed_packages())
    assert not missing, (
        "the CI image does not install command(s) the CI scripts guard on. "
        "Each will fail open at runtime -- the safe direction, but it means "
        "the affected tests pin nothing, and a break-confirmation cannot tell "
        f"a fixed script from a broken one: {missing}. "
        f"Installed: {installed}"
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


def _empty_makefile(tmp_path: Path) -> Path:
    """A Makefile contributing no requirements.

    The synthetic-input tests below assert on a script and an image they
    construct themselves. Once the derivation grew a SECOND source, letting
    them read the real Makefile meant its genuine `jq` requirement leaked
    into assertions about a synthetic world -- the test then failing for a
    fact that has nothing to do with what it is checking.
    """
    path = tmp_path / "EmptyMakefile"
    path.write_text("# no tool variables\n")
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
    missing = missing_commands([script], _SYNTHETIC_CLOUDBUILD, _empty_makefile(tmp_path))

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
    assert missing_commands([script], _SYNTHETIC_CLOUDBUILD, _empty_makefile(tmp_path)) == {}


def test_the_exemption_still_spares_its_command_under_synthetic_input(
    tmp_path: Path,
):
    """`gh` is exempt, and the exemption must be what spares it -- not luck."""
    script = _synthetic_script(
        tmp_path, "needs_gh.sh", "#!/bin/bash\ncommand -v gh || exit 1\n"
    )
    assert missing_commands([script], _SYNTHETIC_CLOUDBUILD, _empty_makefile(tmp_path)) == {}
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


def test_the_claude_exemption_rests_on_ci_never_calling_the_local_runner():
    """`claude` is only required by vacuous_review_local.sh — keep it out of CI.

    If cloudbuild (or a Makefile target CI invokes) starts calling that
    script, this exemption becomes a missing-install bug disguised as policy.
    """
    cloudbuild = CLOUDBUILD.read_text()
    assert "vacuous_review_local" not in cloudbuild, (
        "cloudbuild.yaml now references vacuous_review_local.sh; either install "
        "claude in the suite image or stop calling the local runner from CI"
    )
    assert "command -v claude" not in cloudbuild
    # ci_targets is the spend/allow guard over make targets CI may run.
    ci_targets = (REPO_ROOT / "tools" / "ci" / "ci_targets.py").read_text()
    assert "vacuous-review" not in ci_targets and "vacuous_review" not in ci_targets, (
        "ci_targets.py now allows a vacuous-review make target; that would pull "
        "the local claude runner into CI without an image install"
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
