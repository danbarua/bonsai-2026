#!/usr/bin/env python3
"""What CI is allowed to invoke, and the derivation that keeps it honest.

Every GPU and GCS target in this project is a recipe in the same Makefile
the test suite reads. A build that can invoke one is an unattended spend
surface with no human in the loop, and release has been human-gated
throughout Stage 2B. So the restriction is structural rather than a
convention: a short allowlist, checked against a set DERIVED from the
Makefile instead of a second hand-written list of things to avoid.

CLAUDE.md principle 21 is the reason for the shape. A hand-maintained
"forbidden targets" list would silently under-cover the next GPU target
somebody writes. The spending set here is computed from each recipe's own
text -- a recipe that reaches `$(MIGHTY_COLAB)` provisions a billing
runtime; one that exports `$(GCS_ENV)` or `$(GCS_EXEC_ENV)` reaches the
science bucket -- so a target added tomorrow is covered on the day it is
written. `tests/_makefile.py` does the parsing, because there is exactly
one Makefile parser in this repository and adding a second is the
duplication that module exists to prevent.

Four checks, and the last two are what actually bind `cloudbuild.yaml`:

1. every allowlisted name is a real target (an allowlist naming a renamed
   target enforces nothing)
2. the allowlist is disjoint from the derived spending set
3. every allowlisted recipe runs pytest and nothing else -- so a future
   edit that makes `test` shell out to a driver is caught, not just a
   future target added to the allowlist
4. `cloudbuild.yaml` invokes only allowlisted targets, and does not
   mention a spending target anywhere in its text

Check 4 reads the raw file rather than parsing YAML: a substring match
over-flags relative to a semantic parse, which is the safe direction for
a spend guard, and it removes a YAML-library dependency from the one
check that must never fail to run. The cost is that `cloudbuild.yaml`
cannot name a spending target even in a comment. It refers readers here
and to `docs/proposals/CI_CLOUDBUILD.md` instead.

Two flavours of spend, both real: Colab/GCS billing, and Cloud Build
minutes. `stage2a-analyze` bills no GPU and takes about four hours of
CPU; it is excluded by being absent from a three-entry allowlist rather
than by a rule, which is what a short allowlist is for.

Usage:
    python3 tools/ci/ci_targets.py --check cloudbuild.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

import _makefile  # noqa: E402  (path set above; one parser, see module docstring)

# The complete set of Makefile targets a build may invoke. Kept explicit
# because "safe to run unattended" is a judgement, not a property any
# derivation can read off the recipe -- and kept to three so the judgement
# is reviewable at a glance. Checks 1-3 below are what stop it rotting.
CI_INVOCABLE_TARGETS = frozenset({"test", "stage2a-test", "stage2b-test"})

# Recipe text that means the target spends. `MIGHTY_COLAB` provisions a
# billing runtime; the GCS_ENV pair hands a script the bucket and the
# service-account key.
SPEND_MARKERS = ("MIGHTY_COLAB", "GCS_ENV", "GCS_EXEC_ENV")

# What an allowlisted recipe is permitted to do. Anything else -- a
# `$(PYTHON)` driver invocation, a dependency group that pulls in the
# cloud CLIs -- means the target stopped being a test runner.
_ALLOWED_RECIPE_REQUIRES = "uv run pytest"

# What makes a target a TEST SUITE, which is a different question from
# whether CI may invoke it, and conflating them under-covered exactly the
# way principle 21 warns.
#
# `_ALLOWED_RECIPE_REQUIRES` is the literal `uv run pytest`, so a recipe
# reading `uv run --group gpu pytest` did not match it -- and since the
# runner derivation used that same constant, a capability-requiring suite
# was INVISIBLE to the completeness check. Not excused, not dispositioned:
# never seen. The check that exists to ensure a new suite is either invoked
# by CI or explicitly declined could not see the one kind of suite CI is
# guaranteed not to run.
#
# So the two questions get two predicates. This one asks "is this a test
# suite whose coverage must be accounted for", and answers yes regardless of
# dependency group. `_ALLOWED_RECIPE_FORBIDS` still governs the separate
# question of what CI may be allowed to invoke.
_RUNS_PYTEST = re.compile(r"\buv run\b[^\n]*\bpytest\b")
_ALLOWED_RECIPE_FORBIDS = ("$(PYTHON)", "--group gpu", *(f"$({m})" for m in SPEND_MARKERS),
                           *SPEND_MARKERS)

# Anchored to the start of a line, because that is how a build config
# invokes make and because the unanchored form has a false positive that
# fires immediately: `apt-get install make git` reads as `make git`. An
# anchored pattern is still conservative -- the "no spending target named
# anywhere in the text" check below backs it up regardless of syntax.
_MAKE_INVOCATION = re.compile(r"^\s*make\s+((?:-[^\s]+\s+)*)([A-Za-z0-9_.-]+)",
                              re.MULTILINE)


# Targets a test-runner derivation finds but CI deliberately does not run.
# Each needs a reason, and a check below asserts each still exists and is
# still a test runner -- an entry for something that changed shape is an
# exemption concealing the thing it was written to excuse.
NOT_RUN_IN_CI: dict[str, str] = {
    "stage2a-test": "subsumed by `test`, which collects the same files; "
                    "running both would double the build for no coverage",
    "stage2b-test": "subsumed by `test` in the full tier, and invoked "
                    "directly as the fast tier",
    "stage2b-test-roundtrip": "provisions a real Colab runtime and writes to "
                              "the science bucket -- it BILLS while running "
                              "and needs the service-account key. Human-gated "
                              "throughout Stage 2B and must stay so; it is "
                              "also `slow`-marked and excluded from every "
                              "other target",
    "test-capabilities": "runs the same files as `test` with the optional "
                         "cloud group installed. CI must NOT invoke it: "
                         "`--group gpu` installs google-cloud-storage, which "
                         "assert_no_cloud_credentials.py fails the build for. "
                         "It exists for a machine that already has the "
                         "capability, and for the mighty-colab contract and "
                         "GCS-credentialed tests that skip without it",
}


def spending_targets(recipes: dict[str, str] | None = None) -> set[str]:
    """Targets whose own recipe text reaches billable infrastructure."""
    recipes = _makefile.recipes() if recipes is None else recipes
    return {name for name, body in recipes.items()
            if any(marker in body for marker in SPEND_MARKERS)}


def test_runner_targets(recipes: dict[str, str] | None = None) -> set[str]:
    """Targets whose recipe runs pytest and nothing else.

    **Derived, and that is the point.** An allowlist answers "may CI run
    this?" and fails safe: forget to add a target and nothing spends. But it
    fails UNSAFE in the other direction -- forget to add a new `stage3-test`
    and CI silently never runs that suite, which is under-coverage wearing a
    green badge, and the same shape as `STAGE2B_TEST_FILES`.

    So the safe set is derived from the same recipe text the spending set
    is, and `check` requires every derived test runner to be either invoked
    by the build config or dispositioned in `NOT_RUN_IN_CI` with a reason.
    A new suite is therefore covered on the day it is written, or visibly
    unclassified -- never silently absent.
    """
    recipes = _makefile.recipes() if recipes is None else recipes
    return {name for name, body in recipes.items() if _RUNS_PYTEST.search(body)}


def make_invocations(text: str) -> set[str]:
    """Every `make <target>` named in a build config, flags skipped."""
    return {match.group(2) for match in _MAKE_INVOCATION.finditer(text)}


def check(config_path: Path, allowlist: frozenset[str] = CI_INVOCABLE_TARGETS,
          recipes: dict[str, str] | None = None) -> list[str]:
    """Every violation found, as reader-facing sentences. Empty means clean."""
    recipes = _makefile.recipes() if recipes is None else recipes
    spending = spending_targets(recipes)
    problems: list[str] = []

    unknown = sorted(allowlist - set(recipes))
    if unknown:
        problems.append(
            f"allowlisted target(s) {unknown} do not exist in the Makefile. An "
            f"allowlist naming a renamed or deleted target enforces nothing.")

    overlap = sorted(allowlist & spending)
    if overlap:
        problems.append(
            f"allowlisted target(s) {overlap} reach billable infrastructure "
            f"({', '.join(SPEND_MARKERS)}). CI must not be able to spend.")

    for name in sorted(allowlist & set(recipes)):
        body = recipes[name]
        if _ALLOWED_RECIPE_REQUIRES not in body:
            problems.append(
                f"allowlisted target `{name}` no longer runs "
                f"`{_ALLOWED_RECIPE_REQUIRES}`. It was allowlisted as a test "
                f"runner; re-read the recipe before leaving it on the list.")
        for token in _ALLOWED_RECIPE_FORBIDS:
            if token in body:
                problems.append(
                    f"allowlisted target `{name}` now contains `{token}`, which is "
                    f"not something a test runner does.")

    if config_path is not None:
        text = config_path.read_text()
        invoked = make_invocations(text)
        not_allowed = sorted(invoked - allowlist)
        if not_allowed:
            problems.append(
                f"{config_path.name} invokes non-allowlisted target(s) {not_allowed}.")
        named = sorted(t for t in spending if re.search(rf"\b{re.escape(t)}\b", text))
        if named:
            problems.append(
                f"{config_path.name} mentions spending target(s) {named}. This check "
                f"reads raw text and cannot tell a comment from a command, so name "
                f"them in docs/proposals/CI_CLOUDBUILD.md instead.")

        # The under-coverage direction. A derived test runner that CI neither
        # invokes nor explicitly declines is a suite nobody runs -- silent,
        # and indistinguishable from a suite that passes.
        unrun = sorted(test_runner_targets(recipes) - invoked - set(NOT_RUN_IN_CI))
        if unrun:
            problems.append(
                f"target(s) {unrun} run pytest but are neither invoked by "
                f"{config_path.name} nor listed in NOT_RUN_IN_CI with a reason. A "
                f"new test suite that CI never runs is under-coverage with a green "
                f"badge -- disposition it either way, but do not leave it silent.")

    # Dispositions must still describe something real, in both respects: the
    # target exists, and it is still a test runner. An entry for a target
    # that became a driver would excuse exactly the thing worth catching.
    runners = test_runner_targets(recipes)
    for name, reason in NOT_RUN_IN_CI.items():
        if not reason:
            problems.append(f"NOT_RUN_IN_CI entry `{name}` carries no reason.")
        if name not in recipes:
            problems.append(
                f"NOT_RUN_IN_CI names `{name}`, which is not a Makefile target.")
        elif name not in runners:
            problems.append(
                f"NOT_RUN_IN_CI declines `{name}` as a test runner, but its recipe "
                f"no longer looks like one -- the exemption now conceals whatever "
                f"it became.")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path, default=REPO_ROOT / "cloudbuild.yaml",
                        help="build config to check (default: cloudbuild.yaml)")
    args = parser.parse_args(argv)

    spending = spending_targets()
    print(f"[spend-guard] Makefile targets: {len(_makefile.recipes())}, "
          f"of which {len(spending)} reach billable infrastructure")
    print(f"[spend-guard] CI may invoke: {', '.join(sorted(CI_INVOCABLE_TARGETS))}")

    problems = check(args.check)
    for problem in problems:
        print(f"[spend-guard] FAIL: {problem}")
    if problems:
        return 1
    print(f"[spend-guard] OK: {args.check.name} invokes only allowlisted, "
          f"non-spending targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
