"""Committed artefact JSON must be portable, strict, and byte-reproducible.

THE PROBLEM THIS PREVENTS, stated before any rule, because the first
version of this file got the rule right and the problem wrong: someone
clones the repo, runs `make`, and it does what the README says it does. An
artefact that names a location only one machine can reach breaks that, and
the person it breaks for cannot tell why.

THE SPEC. Every `.json` file tracked by git under `experiments/` is an
ARTEFACT: a result some FINDINGS.md cites, or a manifest that indexes such
results. Three invariants hold for all of them. They are not a structural
schema -- the four artefacts that exist today share no top-level keys, and
the two properties that actually bite (reachability from a fresh clone;
byte-exact serialization) cannot be expressed in JSON Schema at all.

  1. A DECLARED ARTEFACT THE REPOSITORY CARRIES IS ACTUALLY THERE, AND
     EVERY DECLARED PATH STAYS INSIDE THE REPOSITORY.

     The failure this exists to produce, in full: "the manifest says
     `experiments/.../some_result.json` is there; it isn't." Someone
     fat-fingered a filename, or the file never landed. Either way it is a
     GitHub issue in two sentences, filed by whoever ran `make` after a
     fresh clone.

     Existence is asserted only for paths GIT TRACKS. The rest are
     regenerable local caches -- Stage 2A's manifest declares eight
     gitignored `.pkl`s -- and demanding those exist would turn a fresh
     clone permanently red for files it is not supposed to have.

     `REPO_ROOT / s` lands outside REPO_ROOT exactly when `s` is absolute
     or climbs out with `..`, so one local question replaces a prohibition
     that would need every root every filesystem ever had. Whole values
     only, never fragments of prose: the harm is a field some code opens.

     Provenance: a CAUGHT DEFECT, 2026-08-10. `_write_provenance_sidecar`
     in `run_abs_conv_eps_sensitivity.py` builds `local_path` from
     `abspath(__file__)`, naming a directory no other checkout has.

     NOT asserted, and the reason is the same defect one level up: these
     manifests carry `present: true/false`, and all eight of Stage 2A's
     `present: true` entries are absent from a fresh clone. `present`
     records what the GENERATING machine had. Asserting it would fail on
     locked science for being honest about a sandbox that no longer
     exists, so this file reports the disagreement and does not fail on
     it.

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
import pytest

from _makefile import REPO_ROOT

def _escapes_repo(value: str) -> bool:
    """True if `value`, read as a path from the repo root, lands outside it."""
    if not value or value.lower().startswith("file://"):
        return bool(value)
    try:
        landed = (REPO_ROOT / value).resolve()
    except (OSError, ValueError):
        return False        # not path-shaped on this platform; not our problem
    return not (landed == REPO_ROOT or REPO_ROOT in landed.parents)

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


def unreachable_path_violations(doc, *, exempt=frozenset()):
    return [(where, s) for where, s in _strings(doc)
            if where not in exempt and _escapes_repo(s)]


def declared_artefacts(doc):
    """(name, path, present) for each entry in a manifest's `artifacts` map.

    Derived from the shape both manifests already use rather than from a
    list of field names, so a manifest added later is covered without being
    named here. Entries keyed `object` are bucket ids in a different
    namespace and carry no local path; they are skipped.
    """
    for name, entry in sorted(doc.get("artifacts", {}).items()):
        if isinstance(entry, dict) and "path" in entry:
            yield name, entry["path"], entry.get("present")


def _tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return {p for p in out.split("\0") if p}


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
    unexempted = unreachable_path_violations(doc)
    assert len(unexempted) == 2, "both should trip without the exemption"
    exempted = unreachable_path_violations(doc, exempt=_EXECUTION_PROVENANCE)
    assert len(exempted) == 1, (
        "the exemption must suppress exactly the field it names and no other")
    assert exempted[0][0].endswith("some_other_path")


@pytest.mark.parametrize("rel", artefact_json_paths())
def test_every_path_resolves_inside_the_repository(rel):
    doc = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
    violations = unreachable_path_violations(doc, exempt=_EXECUTION_PROVENANCE)
    for where, value in violations:
        print(f"[artefact-json] {rel}: {where} = {value!r}")
    assert not violations, (
        f"{rel} names {len(violations)} location(s) outside this repository, so a "
        "fresh clone cannot reach them and `make` will not do what the README says "
        f"it does: {[w for w, _ in violations]}. Store the path relative to the repo "
        "root, or the object path within the bucket.")


@pytest.mark.parametrize("rel", artefact_json_paths())
def test_report_what_each_manifest_claims_about_obtainability(rel):
    """REPORTS. Asserts nothing, and says so, because there is nothing yet to assert.

    A "declared artefacts exist" check was written here and removed the same
    day: measured against the only two manifests in the repository it has
    ZERO candidates -- Stage 2A's eight paths are all untracked, Stage 2B's
    seventeen entries carry bucket object ids and no local path. It passed
    by finding nothing, which is the failure `gate_inventory.py` exits 2 to
    avoid, committed inside the file that cites that rule.

    The check cannot be written because the decision it would enforce has
    not been made. `present: true` collapses four different situations --
    committed here, in the bucket, regenerable on demand, or gone -- into
    one word, and only the last is dangerous. Until a manifest says WHICH,
    any assertion built on it encodes a promise nobody made. See task #38.
    """
    doc = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
    rows = list(declared_artefacts(doc))
    tracked = _tracked_files()
    print(f"\n[artefact-json] {rel}: {len(rows)} entries carrying a local path")
    for name, path, present in rows:
        state = ("committed" if path in tracked
                 else "not in repo" if not (REPO_ROOT / path).exists()
                 else "untracked but on this disk")
        print(f"[artefact-json]   present={present!s:5} {state:26} {name}")


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
    "/Users/dan/x/results/t.json",   # the sidecar's own shape
    "/tmp/x.json",                   # accepted by the prefix-list version
    "/var/folders/zz/T/tmpabc/x.json",   # macOS mkdtemp; same miss
    "file:///tmp/x.json",
    "../../elsewhere/t.json",        # escapes without being absolute at all
])
def test_a_location_outside_the_repository_is_rejected(path):
    assert unreachable_path_violations({"local_path": path}), f"{path} not caught"


def test_it_looks_at_any_depth_and_at_keys():
    assert unreachable_path_violations({"a": [{"b": "/home/ci/out.json"}]})
    assert unreachable_path_violations({"/Users/dan/key": "harmless value"}), (
        "a path used as a KEY must be caught too")


@pytest.mark.parametrize("value", [
    "stage2b/train/stage3/common/table.json",        # bucket object path
    "experiments/stage2b_denoising/results/t.json",  # repo-relative
    "ratio a/b is fine",
    # real string from stage2b's ARTIFACT_MANIFEST.json. Checking whole values
    # rather than hunting path-shaped fragments is what keeps this passing.
    "studentized: mean(s*d) / (SD(s*d, ddof=1) / sqrt(n))",
    "5ebded9ea78da1f66aa826683828c0990fbd57ab",
    "stage2b_local_manifest_v1",
    "",
])
def test_ordinary_content_is_not_flagged(value):
    """A guard with false positives gets switched off by the next author."""
    assert not unreachable_path_violations({"k": value}), f"{value!r} wrongly flagged"


def test_no_manifest_yet_declares_an_obtainable_local_artefact():
    """The measurement that removed the existence check, pinned so it stays true.

    This is the anti-vacuity assertion the deleted check could not make. It
    fails the day a manifest first declares a path the repository actually
    carries -- which is the day an existence check becomes writable, and the
    day someone should write it. Until then it records WHY there isn't one.
    """
    tracked = _tracked_files()
    obtainable = []
    for rel in artefact_json_paths():
        doc = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        obtainable += [(rel, p) for _, p, _ in declared_artefacts(doc) if p in tracked]
    assert not obtainable, (
        f"a manifest now declares artefact(s) the repo tracks: {obtainable}. That "
        "is good news and it invalidates this test: replace it with the existence "
        "check it was standing in for -- declared + tracked must be on disk.")


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
    violations = unreachable_path_violations(sidecar)
    assert len(violations) == 2, f"expected both path fields flagged, got {violations}"
    # the portable fields are exactly what should survive the fix
    assert not unreachable_path_violations(
        {k: v for k, v in sidecar.items() if k not in ("local_path", "object_path")})
