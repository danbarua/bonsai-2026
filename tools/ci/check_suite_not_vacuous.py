#!/usr/bin/env python3
"""Fail a green suite that went green by skipping.

A CI machine has no `datasets/`, no gitignored `.pkl`/`.npz` caches, no GCS
credentials, no Colab session and no `claude` CLI. This project's two-tier
test convention skips cleanly on all of them, which is the correct
behaviour and also the failure mode: a build where forty more tests
started skipping is indistinguishable, from the exit code, from one where
they all ran.

So the exit code is not the whole verdict. This reads the JUnit XML the
run produced and compares the SET of skipped tests against a committed
baseline measured in the same capability-free environment. Both directions
are errors, per CLAUDE.md principle 21:

  * a skip not in the baseline  -- a test silently stopped running
  * a baseline entry that ran   -- the baseline is stale and now claims
                                   less coverage than the build has
  * a baseline entry not collected at all -- the test was renamed or
                                   deleted and the entry is now hiding a
                                   real gap rather than recording one

A count-based guard (floor on passed, ceiling on skipped) cannot see the
case where one test starts skipping while another stops: the two cancel
and both constants still hold. That is principle 12's "the rounded
statistic matched" one layer up, which is why this compares identities and
not totals. There is deliberately no minimum-test-count constant here --
a collapse in collection shows up as every baseline entry going missing,
so the count check would be a second, hand-maintained expression of
something the set already covers.

Skip reasons are printed on every run, passing or failing. An absence has
to announce itself; one that has to be inferred from a total is the same
as no report at all.

Usage
-----
    python3 tools/ci/check_suite_not_vacuous.py --junit junit.xml

    # after a deliberate, understood change in what CI can run:
    python3 tools/ci/check_suite_not_vacuous.py --junit junit.xml \
        --write-baseline

Baseline format: one `classname::name` key per line, sorted, `#` comments
ignored. The key is JUnit's own pair, used verbatim rather than
reconstructed into a pytest nodeid -- a transformation is a place for a
bug and buys nothing a human editing the file needs.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPO_ROOT / "tools" / "ci" / "ci_skip_baseline.txt"

BASELINE_HEADER = """\
# Tests that skip in the credential-free CI environment.
#
# Measured, not chosen. Regenerate with:
#
#     python3 tools/ci/check_suite_not_vacuous.py --junit junit.xml \\
#         --write-baseline
#
# Every entry is a capability CI deliberately does not have: no
# `datasets/`, no gitignored `.pkl`/`.npz` caches, no `google-cloud-storage`,
# no GCS credentials, no `mighty-colab`, no `claude` CLI. Adding an entry
# is a statement that CI has lost coverage and that this was intended;
# removing one is a statement that CI gained it. Neither should happen by
# accident, which is why both directions fail the build.
#
# Format: `classname::name`, one per line, sorted. The trailing comment on
# each line is the skip reason as pytest reported it, kept for the reader
# and ignored by the checker.
"""


class Outcome:
    """One testcase as JUnit recorded it."""

    def __init__(self, key: str, skipped: bool, reason: str) -> None:
        self.key = key
        self.skipped = skipped
        self.reason = reason


def parse_junit(path: Path) -> tuple[list[Outcome], dict[str, int]]:
    """Every testcase in the report, plus the suite's own totals.

    Handles both shapes pytest emits: a bare `<testsuite>` root and a
    `<testsuites>` wrapper around one or more suites.
    """
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise SystemExit(f"{path}: no <testsuite> element -- not a pytest JUnit report")

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    outcomes: list[Outcome] = []
    for suite in suites:
        for field in totals:
            totals[field] += int(suite.get(field, 0) or 0)
        for case in suite.iter("testcase"):
            key = f"{case.get('classname', '')}::{case.get('name', '')}"
            skipped = case.find("skipped")
            if skipped is not None:
                reason = (skipped.get("message") or "").strip().replace("\n", " ")
                outcomes.append(Outcome(key, True, reason))
            else:
                outcomes.append(Outcome(key, False, ""))
    return outcomes, totals


def read_baseline(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(
            f"{path}: no baseline. Seed one from a run in the environment CI "
            f"actually has, with --write-baseline, and read it before committing "
            f"-- a baseline written from a fully-populated checkout records zero "
            f"skips and would pass every build vacuously.")
    keys = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            keys.add(line)
    return keys


def write_baseline(path: Path, outcomes: list[Outcome]) -> None:
    lines = [BASELINE_HEADER]
    for outcome in sorted((o for o in outcomes if o.skipped), key=lambda o: o.key):
        reason = outcome.reason[:160]
        lines.append(f"{outcome.key}  # {reason}" if reason else outcome.key)
    path.write_text("\n".join(lines) + "\n")
    print(f"[vacuity] wrote {sum(o.skipped for o in outcomes)} entries to {path}")


def report(outcomes: list[Outcome], totals: dict[str, int]) -> None:
    """Print what happened, before any verdict.

    Unconditional: a build that passed is exactly when nobody looks, and
    the skip list is the part of the record that says how much the pass
    was worth.
    """
    skipped = [o for o in outcomes if o.skipped]
    passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    print(f"[vacuity] selected={totals['tests']} passed={passed} "
          f"skipped={totals['skipped']} failed={totals['failures']} "
          f"errored={totals['errors']}")
    if not skipped:
        print("[vacuity] no tests skipped")
        return
    print(f"[vacuity] {len(skipped)} skipped, with reasons:")
    for outcome in sorted(skipped, key=lambda o: o.key):
        print(f"[vacuity]   {outcome.key}")
        print(f"[vacuity]       {outcome.reason or '(no reason recorded)'}")


def check(outcomes: list[Outcome], totals: dict[str, int], baseline: set[str],
          baseline_path: Path, require_all_modules: bool = False) -> int:
    observed_skips = {o.key for o in outcomes if o.skipped}
    collected = {o.key for o in outcomes}
    # Modules this run actually touched. The fast tier runs a subset of the
    # files the full tier does, so a baseline entry from an untouched module
    # is out of scope here rather than missing -- scoping by module is what
    # lets one measured baseline serve both tiers instead of two files
    # drifting apart, which is the failure `tests/_makefile.py` was written
    # to end one layer down.
    modules = {key.split("::", 1)[0] for key in collected}
    in_scope = {key for key in baseline if key.split("::", 1)[0] in modules}

    unexpected = sorted(observed_skips - baseline)
    ran_instead = sorted((in_scope & collected) - observed_skips)
    vanished = sorted(in_scope - collected)

    problems = 0

    if totals["tests"] == 0:
        print("[vacuity] FAIL: the report contains no tests at all.")
        problems += 1

    if totals["failures"] or totals["errors"]:
        print(f"[vacuity] FAIL: {totals['failures']} failure(s) and "
              f"{totals['errors']} error(s) in the report. pytest's own exit code "
              f"normally stops the build before this runs, so reaching here means "
              f"something swallowed it.")
        problems += 1

    if require_all_modules:
        # A module that stops collecting takes its baseline entries out of
        # scope with it. pytest does emit a `classname=""` collection-skip
        # testcase for a module-scope importorskip, which lands in
        # `unexpected` above -- but that covers one cause, not the class.
        # Asserting the modules directly closes the rest, and is only
        # correct for a tier that runs the whole suite.
        missing_modules = sorted({key.split("::", 1)[0] for key in baseline} - modules)
        if missing_modules:
            problems += 1
            print(f"\n[vacuity] FAIL: {len(missing_modules)} module(s) named in the "
                  f"baseline were not collected at all. Their entries would "
                  f"otherwise fall silently out of scope:")
            for module in missing_modules:
                print(f"[vacuity]   ! {module}")

    if unexpected:
        problems += 1
        print(f"\n[vacuity] FAIL: {len(unexpected)} test(s) skipped that the baseline "
              f"does not account for. CI lost coverage it used to have:")
        reasons = {o.key: o.reason for o in outcomes if o.skipped}
        for key in unexpected:
            print(f"[vacuity]   + {key}")
            print(f"[vacuity]       {reasons.get(key) or '(no reason recorded)'}")
        print("[vacuity]   Either restore the capability, or -- if the skip is "
              "correct and intended -- regenerate the baseline and say why in the "
              "commit message.")

    if ran_instead:
        problems += 1
        print(f"\n[vacuity] FAIL: {len(ran_instead)} baseline entr(y/ies) did not skip. "
              f"The baseline claims less coverage than this build has, and a stale "
              f"baseline stops detecting the skips it exists to detect:")
        for key in ran_instead:
            print(f"[vacuity]   - {key}")
        # Name the likeliest cause, because it is TWO LAYERS from the
        # symptom and this message is otherwise actively misleading: it says
        # "stale baseline" when the baseline may be exactly right and the
        # ENVIRONMENT changed under it.
        #
        # The live example, caught by reading before this ever ran: a
        # Makefile target gained `uv run --group gpu` to fix a local skip.
        # cloudbuild.yaml invokes that target, so the group installs in CI
        # too, `google_crc32c` becomes importable, and a test the baseline
        # records as skipped now runs. Nothing about the baseline was wrong.
        # A build failing here with only "remove them from the baseline"
        # would have had the entry deleted -- destroying a correct capability
        # record to silence a message about the wrong thing.
        print("[vacuity]   BEFORE EDITING THE BASELINE, check whether CI's "
              "environment changed.")
        print("[vacuity]   A Makefile target that gained a dependency group "
              "(e.g. `uv run --group gpu`)")
        print("[vacuity]   installs it in CI too if cloudbuild.yaml invokes "
              "that target -- which makes")
        print("[vacuity]   an optional import succeed and its test run. That "
              "is an environment change,")
        print("[vacuity]   not a stale baseline, and the fix belongs in the "
              "Makefile: keep CI-invoked")
        print("[vacuity]   targets capability-free and put the capability run "
              "in a target CI does not")
        print("[vacuity]   call. Deleting the entry instead discards a "
              "correct record of what CI lacks.")
        print(f"[vacuity]   If the environment genuinely SHOULD have gained "
              f"this, regenerate {baseline_path}")
        print("[vacuity]   from a run in the new environment and say so in "
              "the commit message.")

    if vanished:
        problems += 1
        print(f"\n[vacuity] FAIL: {len(vanished)} baseline entr(y/ies) name tests that "
              f"were not collected. An entry naming a renamed or deleted test is an "
              f"exemption hiding a real gap:")
        for key in vanished:
            print(f"[vacuity]   ? {key}")

    if problems == 0:
        scope = "" if len(in_scope) == len(baseline) else (
            f" ({len(in_scope)} of {len(baseline)} baseline entries in scope for the "
            f"{len(modules)} modules this run collected)")
        print(f"[vacuity] OK: skip set matches {baseline_path.name} exactly "
              f"({len(observed_skips)} skipped){scope}.")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--junit", required=True, type=Path,
                        help="JUnit XML written by `pytest --junitxml=...`")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true",
                        help="record this run's skip set as the new baseline")
    parser.add_argument("--require-baseline-modules", action="store_true",
                        help="every module named in the baseline must have been "
                             "collected; for a tier that runs the whole suite")
    args = parser.parse_args(argv)

    outcomes, totals = parse_junit(args.junit)
    report(outcomes, totals)

    if args.write_baseline:
        write_baseline(args.baseline, outcomes)
        return 0

    return check(outcomes, totals, read_baseline(args.baseline), args.baseline,
                 require_all_modules=args.require_baseline_modules)


if __name__ == "__main__":
    sys.exit(main())
