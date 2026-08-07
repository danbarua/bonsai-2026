"""Artifact provenance for Stage 2B: what produced an artifact, recorded at
write time and checked before an existing artifact is ever trusted.

**An object merely existing is never sufficient evidence that it is
resumable.**

`stage2b_gcs.ensure_artifact` treats an object's existence as proof its
step is done -- that is what makes a dead session cost one step instead of
an afternoon. The same property is a hazard once configuration can change:
a run with a different encoder budget, a different corruption scheme, or a
different version of the code that computes any of it would silently
consume artifacts produced under the old one and report success. Only the
encoder step count was ever fingerprinted, and only by being spelled into
the object name.

This module supplies the missing half. Every artifact-generating run
computes a fingerprint; every consumer compares the recorded fingerprint
against what this run expects and refuses, by name, on disagreement.

## Why the source manifest is a union of two closures

The fingerprint records a SHA-256 per repo-local module that participates
in the computation. Establishing that set is the part with a real failure
mode, and one closure alone does not do it:

- **Runtime alone** (`sys.modules` filtered to repo paths) sees only what
  this particular execution happened to import. Conditional, lazy, and
  branch-specific imports that did not fire are invisible -- and Stage
  2B's own ladder drivers defer every repo import into `load_modules()`,
  so a runtime closure taken before that call is nearly empty.
- **Static alone** (AST-walking imports from the entrypoint) misses what
  is reached dynamically, and resolves package `__init__.py` files
  inconsistently -- measured on `encode_stage3_local.py`, static found 15
  modules and runtime found those same 15 plus four `__init__.py`.

So: the union, established BEFORE generation and revalidated AFTER. If
execution imported a repo module that neither closure predicted,
`revalidate_after_execution` refuses -- the artifact was produced by code
the fingerprint does not describe, which is exactly the condition the
fingerprint exists to make impossible to miss.

Every exclusion is frozen in `EXEMPT_PATHS` with a stated reason, and
`tests/test_stage2b_fingerprint.py` asserts each exemption still refers to
something real -- a dead exemption is one nobody notices has stopped
applying.

## Per-consumer field selection, and why it is not laxity

A fingerprint that demanded byte-identical producing code for every
consumed artifact would refuse this pipeline's own legitimate reuse: ladder
stage 2 and stage 3 deliberately consume stage 1's `topologies.npz` and the
KMNIST inputs staged once under stage 1, both written under earlier commits
and different run configs. Reading them is correct -- topology construction
does not depend on which images a later rung processes, and re-deriving
them would weaken the guarantee, not strengthen it.

So a consumer declares a `ConsumePolicy` naming which fields must match,
and the policy travels with the call site rather than being inferred.
`STRICT` is the default; `CONTENT_ONLY` is the documented relaxation, and
it still checks payload identity in full -- it relaxes only the claim that
the same code and config produced it.

## What this module does not do

It computes and compares fingerprints. It does not talk to GCS, does not
publish manifests, and does not decide what an artifact is. Publication,
the manifest sidecar, and the single validated consume path live in
`stage2b_gcs.py`, which imports this.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from typing import NamedTuple

FINGERPRINT_FORMAT = "stage2b-artifact-fingerprint/1"

DIGEST_BLOCK_SIZE = 4 * 1024 * 1024

# Directories a repo-local import can resolve into. Not every directory in
# the repo: these are the ones on the ladder drivers' `sys.path`, so this
# list is the same narrowing those drivers already perform.
SOURCE_ROOTS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
    "experiments/stage1b2_structured_transformation",
    "src",
)

# Frozen exemptions, each with a reason. A path here is deliberately absent
# from the scientific-source manifest. `tests/test_stage2b_fingerprint.py`
# asserts every entry still names an existing file, so an exemption cannot
# quietly outlive the thing it exempted.
EXEMPT_PATHS = {
    "experiments/stage2b_denoising/stage2b_fingerprint.py":
        "this module: it records provenance and performs no scientific "
        "computation, so including its own hash would invalidate every "
        "artifact whenever the provenance mechanism itself is edited",
}

# Directory prefixes excluded wholesale, each with a reason. Separate from
# EXEMPT_PATHS because these exclude a ROLE rather than a named file.
EXEMPT_PREFIXES = {
    "tests/":
        "test modules are not scientific source and are never loaded during "
        "an artifact-generating run. Without this, computing a fingerprint "
        "under pytest would fold the test harness into the manifest and give "
        "a different answer than the same code gives in production -- the "
        "manifest would describe the observer rather than the computation",
}


def _is_exempt(path):
    return path in EXEMPT_PATHS or any(path.startswith(p) for p in EXEMPT_PREFIXES)


class FingerprintError(OSError):
    """Base for every named refusal this module raises."""


class DirtyTreeError(FingerprintError):
    """An artifact-generating run was attempted from an uncommitted tree."""


class SourceClosureError(FingerprintError):
    """Execution imported repo code the pre-established fingerprint did not
    describe."""


class FingerprintMismatch(FingerprintError):
    """A recorded fingerprint disagrees with what this run expects."""


class ConsumePolicy(NamedTuple):
    """Which fingerprint fields a consumer requires to match.

    Declared at the call site, never inferred: the reason a given artifact
    may be consumed under a relaxed policy is a scientific judgement about
    that artifact, and it belongs next to the code making it."""
    name: str
    fields: tuple
    reason: str


STRICT = ConsumePolicy(
    name="strict",
    fields=("format", "source_manifest", "config", "git_commit"),
    reason="the artifact must have been produced by this exact code and "
           "configuration; the default for anything a run generates itself")

CONTENT_ONLY = ConsumePolicy(
    name="content_only",
    fields=("format",),
    reason="cross-stage reuse: payload identity is still verified in full, "
           "but the producing commit and config are deliberately not "
           "required to match. Ladder stages 2 and 3 consume stage 1's "
           "topologies and staged KMNIST inputs on purpose -- those do not "
           "depend on which images a later rung processes, and re-deriving "
           "them per stage would weaken the guarantee rather than strengthen "
           "it")


# ---------------------------------------------------------------- digests

def sha256_of_file(path):
    """Block-wise SHA-256, matching `stage2b_gcs.crc32c_of_file`'s shape.

    Block-wise rather than `handle.read()`: the artifacts this describes
    reach hundreds of megabytes, and the whole-file form already in
    `run_ladder_stage*.py` is sized for a source file, not a payload."""
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(DIGEST_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_of_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def canonical_digest(obj):
    """SHA-256 of a JSON structure, canonicalised so the digest depends on
    content and not on key order or whitespace."""
    return sha256_of_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                      default=str).encode("utf-8"))


# ------------------------------------------------------------- repo paths

def repo_relative(path, repo_root):
    """A repo-relative POSIX path, or None if `path` is outside the repo.

    Returns POSIX separators unconditionally so a manifest is comparable
    across platforms -- the fingerprint is a scientific record, and it
    should not change because it was written on a different filesystem."""
    try:
        rel = os.path.relpath(os.path.abspath(str(path)), os.path.abspath(repo_root))
    except ValueError:                       # different drive on Windows
        return None
    if rel.startswith(os.pardir):
        return None
    return rel.replace(os.sep, "/")


def _is_repo_source(path, repo_root):
    """Repo-local, a real `.py` file, and not inside a virtualenv.

    The `.venv` exclusion matters because an editable install puts the
    project's own package on `sys.path` twice -- once as `src/bonsai`, once
    resolved through site-packages -- and only the former is source."""
    if not path or not str(path).endswith(".py"):
        return False
    absolute = os.path.abspath(str(path))
    if not os.path.isfile(absolute):
        return False
    if f"{os.sep}.venv{os.sep}" in absolute or f"{os.sep}site-packages{os.sep}" in absolute:
        return False
    return repo_relative(absolute, repo_root) is not None


# ------------------------------------------------------------- closures

def _resolve_module(module, repo_root):
    """Repo-relative path for a dotted module name, or None if it is not
    repo-local (stdlib, third-party, or simply absent)."""
    parts = module.split(".")
    for root in SOURCE_ROOTS:
        candidate = os.path.join(repo_root, root, *parts) + ".py"
        if os.path.isfile(candidate):
            return repo_relative(candidate, repo_root)
        package = os.path.join(repo_root, root, *parts, "__init__.py")
        if os.path.isfile(package):
            return repo_relative(package, repo_root)
    return None


def static_import_closure(entrypoint, repo_root):
    """Transitive repo-local imports reachable by AST from `entrypoint`.

    Sees imports that this execution did not run -- conditional branches,
    functions never called, and the deferred `load_modules()` bodies the
    ladder drivers use. Cannot see imports formed dynamically; that is what
    the runtime closure is for."""
    start = repo_relative(entrypoint, repo_root)
    if start is None:
        raise FingerprintError(f"entrypoint {entrypoint!r} is outside the repository")
    found, pending = set(), [start]
    while pending:
        rel = pending.pop()
        if rel in found:
            continue
        found.add(rel)
        try:
            source = open(os.path.join(repo_root, rel), "r", encoding="utf-8").read()
            tree = ast.parse(source, filename=rel)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                # `from pkg import mod` may name a module, not an attribute
                names += [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                resolved = _resolve_module(name, repo_root)
                if resolved and resolved not in found:
                    pending.append(resolved)
    return found


def runtime_import_closure(repo_root):
    """Repo-local modules currently in `sys.modules`.

    Sees what actually loaded, including package `__init__.py` files a
    static walk resolves inconsistently, and anything imported dynamically.
    Blind to code paths this execution did not take."""
    found = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if _is_repo_source(path, repo_root):
            found.add(repo_relative(path, repo_root))
    return found


def scientific_source_paths(entrypoint, repo_root):
    """The union of both closures, minus the frozen exemptions."""
    union = static_import_closure(entrypoint, repo_root) | runtime_import_closure(repo_root)
    return {path for path in union if not _is_exempt(path)}


def source_manifest(paths, repo_root):
    """`{repo-relative path: sha256}`, sorted, for a set of source files."""
    return {path: sha256_of_file(os.path.join(repo_root, path))
            for path in sorted(paths)}


# ------------------------------------------------------------ git identity

def closure_dirty_paths(paths, repo_root):
    """Which of `paths` differ from their committed content at HEAD.

    A path absent from HEAD (untracked, or newly added and uncommitted) is
    dirty, which is the correct answer: there is no committed version to
    reproduce it from."""
    dirty = []
    for path in sorted(paths):
        full = os.path.join(repo_root, path)
        try:
            committed = subprocess.run(
                ["git", "rev-parse", f"HEAD:{path}"], cwd=repo_root,
                capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            dirty.append(path)
            continue
        try:
            working = subprocess.run(
                ["git", "hash-object", full], cwd=repo_root,
                capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            dirty.append(path)
            continue
        if committed != working:
            dirty.append(path)
    return dirty


def git_identity(repo_root, require_clean=True, paths=None):
    """The commit an artifact-generating run is pinned to.

    `require_clean` refuses rather than recording a dirty flag. A commit
    plus "there were also some uncommitted changes" is not an identity --
    it does not say WHICH changes, so it cannot support the comparison this
    fingerprint exists to make. Stage 3's Phase A was run from a dirty tree
    and that is precisely why its provenance had to be reconstructed rather
    than read.

    ## Two cleanliness claims, and which one the refusal keys on

    `clean` is the whole working tree, `git status --porcelain` empty. It
    is recorded always.

    `closure_clean`, present when `paths` is given, is the narrower and
    stronger claim: every file in the run's own source closure matches its
    committed content at HEAD. When `paths` is supplied that is what the
    refusal keys on.

    The narrower check is not a relaxation of the broader one, and the
    distinction is not cosmetic. `git status` reports the whole repository,
    including work belonging to efforts that cannot reach this artifact --
    a scratch file, a second concurrent branch of work, an editor's
    leftovers. None of those can change what the encoder computes. What
    CAN is a closure file differing from HEAD, and the whole-tree check
    detects that only as one entry among many, with no way to tell it
    apart. `closure_dirty_paths` compares blob for blob and names the file.

    A run whose closure is clean inside a dirty tree therefore proceeds,
    with `clean=False`, `closure_clean=True` and the offending paths
    recorded -- more information than the coarse check carried, not less.
    Pass no `paths` and the old whole-tree behaviour is exactly what
    happens."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        porcelain = git("status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise FingerprintError(f"could not read git state at {repo_root!r}: {exc}") from exc

    identity = {"commit": commit, "clean": not porcelain}
    if paths is None:
        if porcelain and require_clean:
            raise DirtyTreeError(
                f"refusing to fingerprint an artifact from a dirty tree at "
                f"{repo_root!r}. A commit plus unrecorded local edits is not a code "
                f"identity, so the artifact could not later be shown to have come "
                f"from known source. Commit or stash first.\n{porcelain}")
        return identity

    dirty = closure_dirty_paths(paths, repo_root)
    identity["closure_clean"] = not dirty
    identity["closure_dirty_paths"] = dirty
    identity["tree_dirty_porcelain"] = porcelain if porcelain else None
    if dirty and require_clean:
        raise DirtyTreeError(
            f"refusing to fingerprint an artifact whose own source closure is not "
            f"committed at {repo_root!r}. These files differ from HEAD (or are "
            f"untracked), so the artifact could not later be shown to have come "
            f"from known source:\n"
            + "\n".join(f"  {path}" for path in dirty)
            + "\nCommit them first. Other uncommitted work elsewhere in the tree "
              "is recorded but does not block, since it cannot reach this artifact.")
    return identity


# ------------------------------------------------------------- fingerprint

def compute(*, entrypoint, config, repo_root, parents=None, require_clean=True):
    """The fingerprint for a run about to generate artifacts.

    `config` is whatever the caller declares scientifically load-bearing
    (encoder steps, corruption seed scheme, dataset/split identity, dtypes,
    population). `parents` maps consumed artifact names to their recorded
    payload digests, so a derived artifact records what it was derived
    from.

    Establish this BEFORE generation and pass the same object to
    `revalidate_after_execution` afterwards."""
    paths = scientific_source_paths(entrypoint, repo_root)
    manifest = source_manifest(paths, repo_root)
    fingerprint = {
        "format": FINGERPRINT_FORMAT,
        "entrypoint": repo_relative(entrypoint, repo_root),
        # The closure is passed in, so the refusal keys on this run's own
        # source files rather than on the whole repository's tidiness.
        "git": git_identity(repo_root, require_clean=require_clean,
                            paths=sorted(manifest)),
        "source_manifest": manifest,
        "source_manifest_digest": canonical_digest(manifest),
        "config": dict(config),
        "config_digest": canonical_digest(dict(config)),
        "parents": dict(parents or {}),
        "environment": _environment(),
    }
    return fingerprint


def _environment():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "jax", "jaxlib", "diffrax",
                 "equinox", "optax", "google-cloud-storage", "google-crc32c"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "packages": packages}


def revalidate_after_execution(fingerprint, repo_root):
    """Refuse if execution imported repo code the fingerprint omits.

    The whole point of establishing the closure in advance is that it is a
    claim about what will run. This checks the claim against what did.
    Returns the set of newly-seen paths (empty on success) so a caller can
    log a clean revalidation rather than only a failure."""
    recorded = set(fingerprint["source_manifest"])
    seen = {path for path in runtime_import_closure(repo_root)
            if not _is_exempt(path)}
    new = seen - recorded
    if new:
        raise SourceClosureError(
            f"execution imported repo modules absent from the fingerprint's source "
            f"manifest: {sorted(new)}. The artifact was produced by code the "
            f"fingerprint does not describe. Either the static closure missed a "
            f"dynamic import, or a module was imported after the fingerprint was "
            f"established -- both mean the recorded provenance is incomplete.")
    changed = [path for path, digest in fingerprint["source_manifest"].items()
               if os.path.isfile(os.path.join(repo_root, path))
               and sha256_of_file(os.path.join(repo_root, path)) != digest]
    if changed:
        raise SourceClosureError(
            f"source files changed on disk during execution: {sorted(changed)}. "
            f"The fingerprint no longer describes the code that ran.")
    return new


# ------------------------------------------------------------- comparison

def compare(recorded, expected, policy=STRICT):
    """Mismatching fields between a recorded and an expected fingerprint,
    under `policy`. Empty list means acceptable.

    Compares digests rather than full structures for `source_manifest` and
    `config`, so a mismatch message stays readable; the full structures are
    in the artifact's manifest for anyone diagnosing one."""
    if not isinstance(recorded, dict):
        return [f"recorded fingerprint is {type(recorded).__name__}, not a mapping"]
    mismatches = []
    for field in policy.fields:
        if field == "git_commit":
            got = (recorded.get("git") or {}).get("commit")
            want = (expected.get("git") or {}).get("commit")
            key = "git.commit"
        elif field in ("source_manifest", "config"):
            got, want = recorded.get(f"{field}_digest"), expected.get(f"{field}_digest")
            key = f"{field}_digest"
        else:
            got, want = recorded.get(field), expected.get(field)
            key = field
        if got != want:
            mismatches.append(f"{key}: recorded {got!r} != expected {want!r}")
    return mismatches


def require_match(recorded, expected, *, name, policy=STRICT):
    """`compare`, raising `FingerprintMismatch` naming the object."""
    mismatches = compare(recorded, expected, policy=policy)
    if mismatches:
        raise FingerprintMismatch(
            f"{name!r} was produced under a different fingerprint than this run "
            f"expects, under the {policy.name!r} policy ({policy.reason}). "
            f"Refusing to consume it: an object merely existing is never "
            f"sufficient evidence that it is resumable.\n  "
            + "\n  ".join(mismatches))
    return True


# --------------------------------------------------------- array manifests

def array_manifest(npz_path):
    """Per-array dtype, shape and SHA-256 for an `.npz` payload.

    Per-array rather than whole-file because the whole-file comparison is
    the wrong test for this container: `np.savez` writes a zip whose headers
    embed timestamps, so two runs producing bit-identical arrays produce
    different files. Digesting `arr.tobytes()` also covers NaN bit patterns,
    which `np.array_equal` does not.

    C-contiguous byte order is forced so the digest describes the array's
    values and layout rather than whichever internal representation numpy
    happened to hand back."""
    import numpy as np
    manifest = {}
    with np.load(str(npz_path), allow_pickle=False) as handle:
        for key in sorted(handle.files):
            array = np.ascontiguousarray(handle[key])
            manifest[key] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": sha256_of_bytes(array.tobytes()),
            }
    return manifest


def compare_array_manifests(recorded, computed):
    """Differences between two array manifests, as readable strings.

    Reports missing, extra, and differing arrays separately -- a payload
    that gained an array is a different failure from one whose values
    moved, and collapsing them would hide which happened."""
    problems = []
    for key in sorted(set(recorded) - set(computed)):
        problems.append(f"{key}: present in the recorded manifest, absent from the payload")
    for key in sorted(set(computed) - set(recorded)):
        problems.append(f"{key}: present in the payload, absent from the recorded manifest")
    for key in sorted(set(recorded) & set(computed)):
        was, now = recorded[key], computed[key]
        for field in ("dtype", "shape", "sha256"):
            if was.get(field) != now.get(field):
                problems.append(f"{key}.{field}: recorded {was.get(field)!r} "
                                f"!= payload {now.get(field)!r}")
    return problems


# =====================================================================
# CLI: the pre-flight closure check the GPU Makefile targets gate on
# =====================================================================
#
# The GPU targets used to refuse on `git status --porcelain` being
# non-empty. That was the blunt first draft, written before the closure
# concept existed, and its own refusal message defeats it: "the runtime
# fetches one pinned commit; uncommitted work would not be in it." The
# remote executes the pinned commit BY CONSTRUCTION, so dirt outside the
# driver's source closure cannot reach the computation. What can is a
# closure file differing from HEAD -- which the porcelain check reports
# as one line among many, with no way to tell it apart from an unrelated
# scratch file.
#
# This entry point exists so the Makefile can ask the question in shell
# without reimplementing the answer. One definition of "dirty", owned by
# the module that already has it and its tests; principle 16 says the
# risk of reimplementing a helper is distinct from the helper being
# wrong, and a shell reimplementation of a blob-by-blob HEAD comparison
# is exactly that shape.
#
#     python stage2b_fingerprint.py --check-closure <driver.py>
#
# Exit 0: every file in the driver's closure matches HEAD. Exit 1: at
# least one does not, and each is named on stdout. The whole-tree state
# is printed either way and never gates -- it is recorded in the
# artifact's manifest, not enforced at the door.

def _cli(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Refuse a run whose own source closure is not committed.")
    parser.add_argument("--check-closure", metavar="ENTRYPOINT", required=True,
                        help="driver whose import closure must be committed at HEAD")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)

    entrypoint = os.path.abspath(args.check_closure)
    repo_root = os.path.abspath(
        args.repo_root or os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.isfile(entrypoint):
        print(f"[closure] REFUSING: no such entrypoint {entrypoint!r}")
        return 2

    paths = sorted(scientific_source_paths(entrypoint, repo_root))
    dirty = closure_dirty_paths(paths, repo_root)
    identity = git_identity(repo_root, require_clean=False)

    print(f"[closure] entrypoint {repo_relative(entrypoint, repo_root)}: "
          f"{len(paths)} files in closure, commit {identity['commit']}")
    if dirty:
        print("[closure] REFUSING: these files in this driver's own source "
              "closure differ from HEAD (or are untracked), so the runtime "
              "would fetch a commit that does not contain them:")
        for path in dirty:
            print(f"[closure]   {path}")
        print("[closure] Commit them first. Uncommitted work ELSEWHERE in the "
              "tree does not block -- it cannot reach the computation.")
        return 1
    if not identity["clean"]:
        print("[closure] closure is clean; the working tree is dirty elsewhere, "
              "which is recorded in the artifact manifest and does not block.")
    else:
        print("[closure] closure is clean; working tree is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
