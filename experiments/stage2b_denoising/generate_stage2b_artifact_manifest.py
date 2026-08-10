"""Generates a manifest of the artifacts behind Stage 2B's locked results
-- the stage-4 official confirmatory result and the stage-5
amendment-impact audit -- GCS object paths, payload SHA256, producing
commit, and the frozen headline numbers, so provenance is checkable from
a git clone with no GCS credentials and no re-running anything. Mirrors
`experiments/stage2a_dynamics_classification/generate_artifact_manifest.py`'s
purpose exactly (external review's original ask: "an artifact manifest
recording hashes... the confirmatory run actually consumed"), adapted to
Stage 2B's architecture: Stage 2A hashes local scratch pickles because
that is where its artifacts live; Stage 2B's artifacts live in GCS with
their own manifest sidecars already published under the fingerprint
contract, so this script reads THOSE rather than downloading and
re-hashing multi-hundred-MB payloads.

Reads ONLY already-published GCS manifests and the two small locked
result JSONs, via an ANONYMOUS client -- the bucket is public-read
(`stage2b_gcs.DEFAULT_GCS_BUCKET`'s own comment: "public read (anonymous
objectViewer)"). No credentials, no new compute, no billing.

Run after a locked result changes (a new Stage 4 evaluation cannot
happen -- it is one-shot by construction -- but the audit is ordinarily
resumable and could produce a new report) to refresh the committed
`ARTIFACT_MANIFEST.json`.
"""
import argparse
import json
import os
import platform
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)

import stage2b_gcs as gcs                                                # noqa: E402

OUT_DEFAULT = os.path.join(_THIS_DIR, "ARTIFACT_MANIFEST.json")

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

# (label, stage, split, condition, kind, ext, allow_test_split) -- the
# artifacts behind the two locked/closed results this project currently
# treats as reproducible record. Listed rather than derived: which
# objects back a "locked result" is a scientific judgement (STAGE3_PLAN.md
# Freeze 5's own precedent for this project), not a fact the bucket
# listing can answer on its own.
ARTIFACTS = [
    # Stage 4: the ONE locked confirmatory evaluation.
    ("stage4_official_result_json", 4, TEST_SPLIT, None, "official_result", "json", True),
    ("stage4_official_result_txt", 4, TEST_SPLIT, None, "official_result", "txt", True),
    # Stage 5: the amendment-impact audit, closed 2026-08-09 (AUDIT_OK).
    ("stage5_audit_report_json", 5, TRAIN_SPLIT, None,
     "audit_report_20260809T192835Z", "json", False),
    ("stage5_audit_report_txt", 5, TRAIN_SPLIT, None,
     "audit_report_20260809T192835Z", "txt", False),
    ("stage5_trigger_verdict", 5, TRAIN_SPLIT, None, "trigger_verdict", "json", False),
    ("stage5_feature_distances", 5, TRAIN_SPLIT, None, "feature_distances", "json", False),
    ("stage5_oof_regimes", 5, TRAIN_SPLIT, None, "oof_regimes", "json", False),
    ("stage5_probe_sizing", 5, TRAIN_SPLIT, None,
     "probe_sizing_20260809T192835Z", "json", False),
]
# Per-graph 150-step evolved thetas and features, the audit's own new compute.
for _graph in ("T", "lattice", "rewired", "curr_random"):
    ARTIFACTS.append((f"stage5_theta_T_s150_{_graph}", 5, TRAIN_SPLIT,
                      f"evolved_{_graph}", "theta_T_s150", "npz", False))
    ARTIFACTS.append((f"stage5_features_s150_{_graph}", 5, TRAIN_SPLIT,
                      f"evolved_{_graph}", "features_s150", "npz", False))
ARTIFACTS.append(("stage5_features_s150_pre_evolution", 5, TRAIN_SPLIT,
                  "pre_evolution", "features_s150", "npz", False))


def get_environment_metadata():
    """Versions/platform of the environment THIS SCRIPT ran in (local,
    reading GCS only) -- not necessarily the remote A100 sessions that
    produced the artifacts themselves. Mirrors stage2a's own manifest
    generator's disclosed limitation."""
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:                    # noqa: BLE001
        git_sha = f"unavailable ({exc})"
    return {
        "git_commit_sha": git_sha,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "note": "Reflects the environment this script itself ran in (local, "
                "GCS-read-only) -- not the remote A100 sessions that produced the "
                "artifacts. Each artifact's own manifest carries ITS producing "
                "commit and fingerprint, which is the provenance that matters.",
    }


def entry_for(label, stage, split, condition, kind, ext, allow_test_split, *, bucket):
    object_name = gcs.object_path(stage=stage, condition=condition, kind=kind, ext=ext,
                                  split=split, allow_test_split=allow_test_split)
    manifest = gcs.read_manifest(object_name, bucket=bucket,
                                 allow_test_split=allow_test_split)
    if manifest is None:
        print(f"[{label}] {object_name}: no manifest (run-scoped report, or "
              f"pre-contract -- present=unknown from this script)")
        return {"object": object_name, "present": False, "reason": "no manifest"}
    entry = {
        "object": object_name,
        "present": True,
        "sha256": manifest.get("payload_sha256"),
        "generation": manifest.get("payload_generation"),
        "producing_commit": ((manifest.get("fingerprint") or {}).get("git") or {}).get("commit"),
        "config_digest": (manifest.get("fingerprint") or {}).get("config_digest"),
    }
    print(f"[{label}] {object_name}: sha256={entry['sha256'][:16]}..., "
          f"commit={(entry['producing_commit'] or '?')[:12]}")
    return entry


def read_json_object(object_name, *, bucket, allow_test_split):
    """The small locked-result JSONs themselves (not just their
    manifests) -- read through `consume_validated`, the one validated
    read path every other script in this directory uses, rather than a
    raw `download_file` that would bypass it. `require_manifest=False` is
    the named opt-out: the two report objects this function reads
    (`official_result.json`, `audit_report_*.json`) are run-scoped and
    carry no manifest by design (neither report step passes
    `fingerprint=`), the same situation `consume_pinned`'s pre-contract
    reads are in -- not a case for a byte pin (nothing else depends on
    these object's bytes being IMMUTABLE the way a lineage parent must
    be), just for skipping the manifest requirement a run-scoped kind was
    never going to satisfy."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "obj.json")
        gcs.consume_validated(object_name, path, bucket=bucket,
                              allow_test_split=allow_test_split,
                              require_manifest=False)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def strip_long_lists(obj, threshold=20):
    """Recursively drops list values longer than `threshold`, replacing
    them with a count -- e.g. a 20,000-element bootstrap resample array.
    Committed-manifest brevity, not data loss: the full arrays stay in
    the GCS artifacts this manifest points at; this keeps every headline
    scalar and summary field (CI bounds, observed means, Holm decisions,
    unique_winner) while dropping the bulk that made a first draft of
    this file 745 KB instead of a git-committed index's ~26 KB."""
    if isinstance(obj, dict):
        return {k: strip_long_lists(v, threshold) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > threshold:
            return f"<{len(obj)} values omitted for manifest brevity -- see the " \
                   f"GCS artifact itself>"
        return [strip_long_lists(v, threshold) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_DEFAULT,
                        help="Output path (default: the committed "
                             "experiments/stage2b_denoising/ARTIFACT_MANIFEST.json)")
    args = parser.parse_args()

    bucket = gcs.get_bucket(anonymous=True)
    manifest = {"artifacts": {}, "frozen_results": {},
               "environment": get_environment_metadata()}
    print(f"Environment: git={manifest['environment']['git_commit_sha'][:12]}, "
          f"anonymous GCS client, bucket={bucket.name!r}")

    for label, stage, split, condition, kind, ext, allow_test_split in ARTIFACTS:
        manifest["artifacts"][label] = entry_for(
            label, stage, split, condition, kind, ext, allow_test_split, bucket=bucket)

    # ---- Frozen headline numbers, read from the locked results themselves ----
    stage4_name = manifest["artifacts"]["stage4_official_result_json"]["object"]
    official = read_json_object(stage4_name, bucket=bucket, allow_test_split=True)
    manifest["frozen_results"]["stage4_official_result"] = strip_long_lists(official)

    trigger_name = manifest["artifacts"]["stage5_trigger_verdict"]["object"]
    trigger = read_json_object(trigger_name, bucket=bucket, allow_test_split=False)
    manifest["frozen_results"]["stage5_trigger_verdict_summary"] = {
        "combined_triggered": trigger.get("combined_triggered"),
        "combination_rule": trigger.get("combination_rule"),
        "fixed_primary_contrast": (trigger.get("fixed") or {}).get("primary_contrast"),
        "reselected_primary_contrast":
            (trigger.get("reselected") or {}).get("primary_contrast"),
        "fixed_triggered": (trigger.get("fixed") or {}).get("triggered"),
        "reselected_triggered": (trigger.get("reselected") or {}).get("triggered"),
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
