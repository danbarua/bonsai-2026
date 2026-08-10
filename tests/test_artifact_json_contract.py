"""Committed artefact JSON must be portable, strict, and byte-reproducible.

THE SPEC. Every `.json` file tracked by git under `experiments/` is an
ARTEFACT: a result some FINDINGS.md cites, or a manifest that indexes such
results. Three invariants hold for all of them. They are not a structural
schema -- the four artefacts that exist today share no top-level keys, and
the two properties that actually bite (an absolute path in ANY string value,
at any depth; byte-exact serialization) cannot be expressed in JSON Schema
at all. So the contract is over form and portability, not shape.

  1. NO ABSOLUTE PATHS OR HOST IDENTITY in any string, at any depth.
     `/Users/...`, `/home/...`, `C:\\...`, `file://...`.

     Provenance: a CAUGHT DEFECT, 2026-08-10. `_write_provenance_sidecar`
     in `run_abs_conv_eps_sensitivity.py` builds `local_path` and
     `object_path` from `os.path.dirname(os.path.abspath(__file__))`, so a
     sidecar generated in one checkout embeds that checkout's absolute path.
     Committing it would bake one machine's layout into shared history and
     produce a spurious diff every time a different checkout regenerated it.
     A manifest that only validates on the machine that wrote it is not
     doing the job a manifest exists to do.

  2. STRICT RFC 8259 -- no `NaN`, `Infinity`, `-Infinity`.

     Provenance: PRINCIPLE, no paid incident in this repo yet, and this is
     recorded as such rather than dressed up as a catch. Python's
     `json.dumps` emits those three bare tokens by default and reads them
     back happily, so a violation is invisible from Python alone; `jq` and
     most non-Python parsers reject the file outright. The risk is real for
     artefacts full of floats, which is what these are.

  3. CANONICAL SERIALIZATION: the file is byte-identical to
     `json.dumps(obj, indent=2, sort_keys=True)`, optionally followed by a
     single newline.

     Provenance: CODIFIED FROM MEASUREMENT. All four artefacts already
     satisfy this exactly, so nothing needed changing to adopt it -- the
     convention was already in force and merely undocumented. It buys a
     real diff: regenerating an unchanged artefact produces no diff at all,
     so a diff always means the content moved.

DISCOVERY, NOT ENUMERATION. The file set comes from `git ls-files`, per
CLAUDE.md principle 21 -- a hand-maintained list would silently under-cover
the next artefact someone adds. There is deliberately no exemption
mechanism: every tracked artefact passes today, so an exemption list would
be machinery with no member, and the first exemption should have to be
argued for rather than slotted into a waiting hole.

Tier 1 throughout: reads tracked files and temporary files it writes
itself, touches no network, provisions nothing.
"""
import json
import re
import subprocess
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from _makefile import REPO_ROOT

# Splitting a JSON string into path-like tokens. A value is usually the whole
# path, but a path can also sit inside a sentence, so whitespace and the
# characters that commonly bracket a path are all separators.
_TOKENS = re.compile(r"[\s\"'(),\[\]{}<>;]+")

# The FIRST version of this check was a hand-written alternation of prefixes
# -- /Users/, /home/, /root/, a Windows drive, file:// -- and an external
# review found it accepts /tmp/x, /opt/x, /var/folders/... and BOTH forms of
# C:\Users\dan\x (the drive pattern demanded two separators where there is
# one). The break-tests below passed throughout, because they exercised the
# prefixes the alternation already knew about.
#
# That is CLAUDE.md principle 21 committed inside a file that cites principle
# 21: a hand-maintained list standing in for a derivable set, under-covering
# silently. So the set is no longer hand-maintained. `is_absolute()` is the
# standard library's own definition of the property being tested, it covers
# every root this alternation was enumerating one at a time, and it cannot
# fall behind a prefix nobody thought of.


def _looks_absolute(token: str) -> bool:
    """True for a rooted path that actually names something below the root.

    The `parts` length test is not decoration. A bare "/" is_absolute() --
    and a bare "/" is what tokenising a DIVISION SIGN produces. Stage 2B's
    own manifest carries `mean(s*d) / (SD(s*d, ddof=1) / sqrt(n))`, so
    without this an arithmetic formula in a committed artefact reads as a
    filesystem path. Requiring a segment after the root keeps every real
    path and drops every bare operator.
    """
    if not token:
        return False
    if token.lower().startswith("file://"):
        return True                      # an absolute path wearing a scheme
    for flavour in (PureWindowsPath, PurePosixPath):
        path = flavour(token)
        if path.is_absolute() and len(path.parts) > 1:
            return True
    return False

# Anti-vacuity anchors: files that MUST be in any correct discovery result.
# Their job is to fail loudly if the discovery predicate is ever narrowed to
# nothing -- the failure mode where a guard passes because it found no
# candidates at all (CLAUDE.md principle 21).
_ANCHORS = {
    "experiments/stage2a_dynamics_classification/results/ARTIFACT_MANIFEST.json",
    "experiments/stage2b_denoising/ARTIFACT_MANIFEST.json",
}

# THE ONE EXEMPTION, and the argument for it.
#
# Invariant 1 forbids absolute paths because they are CONFIGURATION that
# should have been portable -- the sidecar's `local_path` describes where a
# file may be found, and describing that with one machine's layout is simply
# wrong. But a path can also be EVIDENCE: a record of where a run physically
# happened. Stage 4's locked result records `/content/...`, the Colab VM it
# ran on. That is not an un-portable way of saying something portable; it is
# the true and only answer to "where did this execute". Rewriting it
# repo-relative would not improve portability, it would falsify the record.
#
# So the exemption is by ROLE, not by file: a field whose job is to say where
# execution occurred. It is deliberately keyed on exact leaf paths rather
# than a substring like "path", which would exempt half the corpus, and
# `test_every_exemption_still_matches_a_real_string` fails if any entry stops
# naming something real -- CLAUDE.md principle 21's requirement that an
# exemption carry a reason AND a test that it still refers to something.
_EXECUTION_PROVENANCE = {
    "$.frozen_results.stage4_official_result.run.clone_dir",
    "$.frozen_results.stage4_official_result.run.credentials_path",
    "$.frozen_results.stage4_official_result.run.driver_identity.path",
}


def artefact_json_paths():
    """Every .json tracked by git under experiments/, newest listing each run."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "experiments/*.json", "experiments/**/*.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p)


def _strings(node, trail="$"):
    """Yield (json_pointer_ish_path, string) for every string ANYWHERE in the doc.

    Keys as well as values: a key can carry a path just as easily, and a
    check that only walked values would miss it.
    """
    if isinstance(node, str):
        yield trail, node
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                yield f"{trail}.<key>", key
            yield from _strings(value, f"{trail}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _strings(value, f"{trail}[{i}]")


def absolute_path_violations(doc, *, exempt=frozenset()):
    out = []
    for where, s in _strings(doc):
        if where in exempt:
            continue
        for token in _TOKENS.split(s):
            if _looks_absolute(token):
                out.append((where, s))
                break
    return out


def strict_json_violation(raw):
    """Return the offending token if the text is not strict RFC 8259, else None."""
    def reject(token):
        raise ValueError(token)
    try:
        json.loads(raw, parse_constant=reject)
    except ValueError as exc:
        return str(exc)
    return None


def canonical_violation(raw):
    """Return (expected, actual) if `raw` is not the canonical form, else None."""
    doc = json.loads(raw)
    expected = json.dumps(doc, indent=2, sort_keys=True)
    if raw in (expected, expected + "\n"):
        return None
    return expected, raw


# ---- discovery is itself under test --------------------------------------


def test_discovery_finds_the_known_artefacts_and_is_not_empty():
    found = artefact_json_paths()
    print("\n[artefact-json] discovered:")
    for p in found:
        print(f"[artefact-json]   {p}")
    assert found, (
        "git ls-files matched no JSON under experiments/. Every check in this "
        "file would then pass vacuously, so this is a failure, not a pass.")
    missing = _ANCHORS - set(found)
    assert not missing, (
        f"discovery no longer finds known artefacts: {sorted(missing)}. Either "
        "they moved (update _ANCHORS) or the glob was narrowed and is now "
        "under-covering.")


def test_anchors_still_name_files_that_exist():
    """An anchor pointing at a deleted file would make the guard above unfalsifiable."""
    for rel in sorted(_ANCHORS):
        assert (REPO_ROOT / rel).is_file(), (
            f"_ANCHORS names {rel}, which no longer exists; the anti-vacuity "
            "check is pinned to a ghost.")


# ---- the three invariants ------------------------------------------------


def test_every_exemption_still_matches_a_real_string():
    """An exemption naming nothing is a hole, not an exemption.

    Principle 21: an exemption gets a named constant, a reason, and its own
    test that it still refers to something real. If Stage 4's manifest is
    ever regenerated without these fields, this fails and the entry gets
    deleted rather than sitting there silently widening the contract.
    """
    seen = set()
    for rel in artefact_json_paths():
        doc = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        seen.update(where for where, _ in _strings(doc))
    orphans = _EXECUTION_PROVENANCE - seen
    assert not orphans, (
        f"exemption(s) naming no string in any artefact: {sorted(orphans)}. "
        "Delete them; an exemption for a field that no longer exists only "
        "widens the contract for whatever takes that path next.")


def test_the_exemption_is_load_bearing_and_narrow():
    """It must actually suppress something, and only what it names.

    Without the first half the exemption could be deleted with no test
    noticing, which would make it indistinguishable from dead code.
    """
    doc = {"frozen_results": {"stage4_official_result": {"run": {
        "credentials_path": "/content/key.json",
        "some_other_path": "/content/key.json"}}}}
    unexempted = absolute_path_violations(doc)
    assert len(unexempted) == 2, "both should trip without the exemption"
    exempted = absolute_path_violations(doc, exempt=_EXECUTION_PROVENANCE)
    assert len(exempted) == 1, (
        "the exemption must suppress exactly the field it names and no other")
    assert exempted[0][0].endswith("some_other_path")


@pytest.mark.parametrize("rel", artefact_json_paths())
def test_no_absolute_paths_or_host_identity(rel):
    doc = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
    violations = absolute_path_violations(doc, exempt=_EXECUTION_PROVENANCE)
    for where, value in violations:
        print(f"[artefact-json] {rel}: {where} = {value!r}")
    assert not violations, (
        f"{rel} contains {len(violations)} absolute path(s) or host identifier(s). "
        "These bake one checkout's layout into shared history and re-diff on every "
        "machine. Store repo-relative paths, or the object path within the bucket.")


@pytest.mark.parametrize("rel", artefact_json_paths())
def test_is_strict_rfc_8259(rel):
    raw = (REPO_ROOT / rel).read_text(encoding="utf-8")
    offender = strict_json_violation(raw)
    assert offender is None, (
        f"{rel} contains the bare token {offender!r}, which Python emits but RFC 8259 "
        "forbids; jq and most non-Python parsers reject the whole file. Serialize "
        "with allow_nan=False and decide explicitly how a non-finite value is encoded.")


@pytest.mark.parametrize("rel", artefact_json_paths())
def test_is_canonically_serialized(rel):
    raw = (REPO_ROOT / rel).read_text(encoding="utf-8")
    result = canonical_violation(raw)
    if result is not None:
        expected, actual = result
        print(f"[artefact-json] {rel}: {len(actual)} bytes on disk, "
              f"{len(expected)} bytes canonical")
    assert result is None, (
        f"{rel} is not json.dumps(obj, indent=2, sort_keys=True). Rewrite it with "
        "those exact arguments. Without this, regenerating an unchanged artefact "
        "produces a diff, and a diff stops meaning the content moved.")


# ---- break-tests: each invariant shown FAILING on purpose ----------------
#
# CLAUDE.md principle 21's corollary -- a guard you have not seen fail is not
# yet a guard. Each check above is driven against a deliberately broken input
# and asserted to reject it for the SPECIFIC stated reason.


@pytest.mark.parametrize("path", [
    "/Users/dan/x/results/t.json",
    "/home/ci/out.json",
    # every one of the five below was ACCEPTED by the first version of this
    # check and found by external review, 2026-08-10. They are the regression
    # test for that miss, not decoration.
    "/tmp/x.json",
    "/opt/tools/x.json",
    "/var/folders/zz/T/tmpabc/x.json",   # macOS tempfile.mkdtemp() lives here
    "C:\\Users\\dan\\x.json",
    "C:/Users/dan/x.json",
    "file:///tmp/x.json",
    "\\\\server\\share\\x.json",
])
def test_absolute_path_check_rejects_every_absolute_form(path):
    assert absolute_path_violations({"local_path": path}), f"{path} not caught"


def test_absolute_path_check_looks_at_any_depth_and_at_keys():
    assert absolute_path_violations({"a": [{"b": "/home/ci/out.json"}]})
    assert absolute_path_violations({"/Users/dan/key": "harmless value"}), (
        "a path used as a KEY must be caught too")
    assert absolute_path_violations({"note": "scratch went to /tmp/x.json first"}), (
        "a path embedded in a sentence must be caught too")


@pytest.mark.parametrize("value", [
    "stage2b/train/stage3/common/table.json",       # bucket-relative object
    "experiments/stage2b_denoising/results/t.json",  # repo-relative
    "ratio a/b is fine",
    "a / b with spaces",
    # the real string from stage2b's ARTIFACT_MANIFEST.json that the first
    # attempt at this fix flagged: a division sign tokenises to a bare "/",
    # which is_absolute() calls absolute.
    "studentized: mean(s*d) / (SD(s*d, ddof=1) / sqrt(n))",
    "/",
    "//",
    "5ebded9ea78da1f66aa826683828c0990fbd57ab",
    "stage2b_local_manifest_v1",
    "",
])
def test_absolute_path_check_does_not_fire_on_legitimate_content(value):
    """The half that stops the fix above from being a blunt instrument.

    Widening a detector is only safe if its false-positive rate stays at
    zero on the content that must keep passing -- otherwise the next author
    silences it.
    """
    assert not absolute_path_violations({"k": value}), f"{value!r} wrongly flagged"


def test_strict_json_check_rejects_nan_and_infinity():
    assert strict_json_violation('{"x": NaN}') == "NaN"
    assert strict_json_violation('{"x": Infinity}') == "Infinity"
    assert strict_json_violation('{"x": -Infinity}') == "-Infinity"
    assert strict_json_violation('{"x": 1.5}') is None


def test_canonical_check_rejects_unsorted_and_recompacted_forms():
    doc = {"b": 1, "a": 2}
    assert canonical_violation(json.dumps(doc)) is not None, "compact form must fail"
    assert canonical_violation(json.dumps(doc, indent=2)) is not None, (
        "indented but unsorted must fail")
    assert canonical_violation(json.dumps(doc, indent=4, sort_keys=True)) is not None, (
        "wrong indent width must fail")
    canonical = json.dumps(doc, indent=2, sort_keys=True)
    assert canonical_violation(canonical) is None
    assert canonical_violation(canonical + "\n") is None, (
        "one trailing newline is permitted")


def test_the_sidecar_shape_that_prompted_this_would_be_rejected():
    """The concrete artefact this guard was written for.

    Mirrors `_write_provenance_sidecar`'s output with the absolute paths it
    currently produces. It is not committed today; this asserts the guard
    would reject it on the day someone commits it, rather than waiting to
    find out.
    """
    sidecar = {
        "format": "stage2b_local_manifest_v1",
        "local_path": "/Users/dan/Documents/Codex/2026-08-10/work/bonsai-2026"
                      "/experiments/stage2b_denoising/results/table.json",
        "object_path": "/Users/dan/Documents/Codex/2026-08-10/work/bonsai-2026"
                       "/experiments/stage2b_denoising/results/table.json",
        "payload_sha256": "0" * 64,
        "fingerprint": {"source_manifest_digest": "abc", "config_digest": "def"},
    }
    violations = absolute_path_violations(sidecar)
    assert len(violations) == 2, f"expected both path fields flagged, got {violations}"
    # the portable fields are exactly what should survive the fix
    assert not absolute_path_violations(
        {k: v for k, v in sidecar.items() if k not in ("local_path", "object_path")})
