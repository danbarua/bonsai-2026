"""Publish local-only, pre-Stage-2B artifacts to the public GCS bucket, once.

WHY THIS EXISTS. `experiments/stage1b2_structured_transformation/results/
class0_constructions.pkl` exists on exactly one disk and cannot be
regenerated: its `random` construction does not reproduce the cached
artifact's `random` key under any of 10 swept seeds, a structural mismatch
(`src/bonsai/dynamics/construction_bundle.py:12-25`), and
`tests/test_construction_driver.py:145-151` asserts that non-match as a
pinned regression check. Uploading a freshly-generated bundle under this
name would silently replace an irreplaceable historical artifact with a
reconstruction and flip that assertion. This script uploads the REAL
bytes, once, and never regenerates anything.

Object names live under a `historical/` root
(`stage2b_gcs.historical_object_path`), separate from Stage 2B's own
`stage2b/{split}/stage{n}/...` scheme -- these files predate Stage 2B and
have no stage/condition/split to encode.

SCOPE: Tier 1 only (the one artefact confirmed genuinely irreplaceable).
Tier 2 (~228 MB across 12 files, reproducible at hours of compute) is a
documented follow-up, not covered here.

Run:
    uv run python experiments/publish_historical_artifacts.py --dry-run
    uv run python experiments/publish_historical_artifacts.py
"""
import argparse
import hashlib
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments", "stage2b_denoising"))

import stage2b_gcs as gcs                                          # noqa: E402

# Repo-root-relative.
TIER1 = [
    "experiments/stage1b2_structured_transformation/results/class0_constructions.pkl",
]

INDEX_PATH = os.path.join(_THIS_DIR, "historical_artifacts_index.json")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _produce_already_present(local):
    """`ensure_artifact` wants a step that WRITES its local path. Here the
    bytes already exist at exactly that path -- this driver publishes
    existing irreplaceable artifacts, it never generates them -- so the
    step asserts identity rather than writing anything. Matches
    `stage_kmnist_inputs.py::stage`'s `produce`, same reason."""
    def produce(path, local=local):
        if not os.path.isfile(local):
            raise FileNotFoundError(
                f"{local} does not exist locally and this driver cannot regenerate "
                f"it -- see module docstring")
        if os.path.abspath(path) != os.path.abspath(local):
            raise RuntimeError(f"expected to publish {local}, asked for {path}")
    return produce


def plan(files):
    """[{repo_path, local, object, sha256, bytes}] -- no client, no network."""
    entries = []
    for repo_path in files:
        local = os.path.join(_REPO_ROOT, repo_path)
        if not os.path.isfile(local):
            raise FileNotFoundError(
                f"{local} is missing -- this driver publishes existing local "
                f"artifacts, it does not generate them")
        entries.append({
            "repo_path": repo_path,
            "local": local,
            "object": gcs.historical_object_path(repo_path),
            "sha256": sha256_of(local),
            "bytes": os.path.getsize(local),
        })
    return entries


def publish(entries, bucket_name=None, credentials=None):
    bucket = gcs.get_bucket(name=bucket_name, credentials=credentials)
    import stage2b_fingerprint as fingerprint                      # noqa: E402
    fp = fingerprint.compute(
        entrypoint=os.path.abspath(__file__), repo_root=_REPO_ROOT,
        require_clean=False,
        config={
            "role": "historical pre-Stage-2B artifact, published once for durability",
            "files": [e["repo_path"] for e in entries],
            "note": ("these predate Stage 2B's fingerprint/manifest contract and are "
                     "uploaded as-is, never regenerated -- construction_bundle.py's "
                     "'random' construction does not reproduce class0_constructions."
                     "pkl's cached 'random' under any swept seed, and "
                     "test_construction_driver.py asserts that non-match as a pinned "
                     "regression check"),
        })
    print(f"bucket: {bucket.name}")

    published = []
    for entry in entries:
        result = gcs.ensure_artifact(
            entry["object"], entry["local"],
            produce=_produce_already_present(entry["local"]),
            bucket=bucket, fingerprint=fp)
        print(f"  {entry['repo_path']}")
        print(f"    -> {entry['object']}")
        print(f"    {result.summary() if hasattr(result, 'summary') else result}")
        published.append(entry)
    return published


def _write_index(published):
    index = []
    if os.path.isfile(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    known = {e["object"] for e in index}
    for entry in published:
        if entry["object"] not in known:
            index.append({k: entry[k] for k in ("repo_path", "object", "sha256", "bytes")})
            known.add(entry["object"])
    index.sort(key=lambda e: e["object"])
    with open(INDEX_PATH, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")
    return INDEX_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned objects and digests; upload nothing")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--credentials", default=None)
    args = parser.parse_args(argv)

    entries = plan(TIER1)
    for entry in entries:
        print(f"{entry['repo_path']}")
        print(f"  -> {entry['object']}")
        print(f"  sha256={entry['sha256']} bytes={entry['bytes']}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    published = publish(entries, bucket_name=args.bucket, credentials=args.credentials)
    index_path = _write_index(published)
    print(f"\nwrote {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
