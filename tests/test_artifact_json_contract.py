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

import pytest

from _makefile import REPO_ROOT

# `file://` is included because it is an absolute path wearing a scheme.
_ABSOLUTE = re.compile(r"(?:^|[\s\"'(=])(?:/Users/|/home/|/root/|[A-Za-z]:[\\/]{2}|file://)")

# Anti-vacuity anchors: files that MUST be in any correct discovery result.
# Their job is to fail loudly if the discovery predicate is ever narrowed to
# nothing -- the failure mode where a guard passes because it found no
# candidates at all (CLAUDE.md principle 21).
_ANCHORS = {
    "experiments/stage2a_dynamics_classification/results/ARTIFACT_MANIFEST.json",
    "experiments/stage2b_denoising/ARTIFACT_MANIFEST.json",
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


def absolute_path_violations(doc):
    return [(where, s) for where, s in _strings(doc) if _ABSOLUTE.search(s)]


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


@pytest.mark.parametrize("rel", artefact_json_paths())
def test_no_absolute_paths_or_host_identity(rel):
    doc = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
    violations = absolute_path_violations(doc)
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


def test_absolute_path_check_rejects_a_path_at_any_depth():
    assert absolute_path_violations({"local_path": "/Users/dan/x/results/t.json"})
    assert absolute_path_violations({"a": [{"b": "/home/ci/out.json"}]})
    assert absolute_path_violations({"a": "file:///tmp/x.json"})
    assert absolute_path_violations({"/Users/dan/key": "harmless value"}), (
        "a path used as a KEY must be caught too")
    # and does not fire on the shapes these artefacts legitimately contain
    assert not absolute_path_violations(
        {"object": "stage2b/train/stage3/common/table.json",
         "rel": "experiments/stage2b_denoising/results/table.json",
         "note": "ratio a/b is fine", "digest": "5ebded9ea78d"})


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
