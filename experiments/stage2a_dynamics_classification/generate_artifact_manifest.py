"""
Generates a manifest of the artifacts behind Stage 2A's locked
confirmatory result -- SHA256 hashes, dimensions, image ordering, graph
hashes, and the selected `C` values the confirmatory run actually
consumed. Per the reproducibility gaps flagged in external review
(FINDINGS.md's "Reproducibility gaps" section): "an artifact manifest
recording hashes, dimensions, image ordering, graph hashes, and the
selected `C` values the confirmatory run actually consumed."

Reads only already-produced artifacts (`results/stage3_classifier_
conditions.pkl`, `results/stage4_confirmatory_results.pkl`, and the
scratch-directory encode/GPU-evolve pkls via `stage2a_paths`) -- no new
simulation, no new GPU time. Run after the full pipeline (stage 3
through the confirmatory evaluation) to produce `results/ARTIFACT_
MANIFEST.json`, committed alongside this script so the exact provenance
of the reported numbers is checkable without re-running anything.
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from stage2a_paths import train_scratch_dir, test_scratch_dir

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def file_entry(path):
    if not os.path.exists(path):
        return {"present": False, "path": path}
    return {
        "present": True, "path": path,
        "size_bytes": os.path.getsize(path),
        "sha256": sha256_of_file(path),
    }


def main():
    manifest = {"artifacts": {}, "graphs": {}, "selected_C": {}, "dimensions": {},
                "image_ordering": {}}

    train_dir = train_scratch_dir()
    test_dir = test_scratch_dir()

    # ---- File-level hashes of every artifact the confirmatory result depends on ----
    tracked_files = {
        "stage3_encode_local.pkl": os.path.join(train_dir, "stage3_encode_local.pkl"),
        "stage3_gpu_results.pkl": os.path.join(train_dir, "stage3_gpu_results.pkl"),
        "stage3_topologies.pkl": os.path.join(train_dir, "stage3_topologies.pkl"),
        "stage4_encode_local.pkl": os.path.join(test_dir, "stage4_encode_local.pkl"),
        "stage4_gpu_results.pkl": os.path.join(test_dir, "stage4_gpu_results.pkl"),
        "stage4_gpu_upload_topologies.pkl": os.path.join(test_dir, "stage4_gpu_upload_topologies.pkl"),
        "stage3_classifier_conditions.pkl": os.path.join(RESULTS_DIR, "stage3_classifier_conditions.pkl"),
        "stage4_confirmatory_results.pkl": os.path.join(RESULTS_DIR, "stage4_confirmatory_results.pkl"),
    }
    for label, path in tracked_files.items():
        manifest["artifacts"][label] = file_entry(path)
        print(f"[{label}] {'present' if manifest['artifacts'][label]['present'] else 'MISSING'}"
              + (f", sha256={manifest['artifacts'][label]['sha256'][:16]}..."
                 if manifest["artifacts"][label]["present"] else ""))

    # ---- Graph hashes, dimensions, image ordering: derived from the training
    # encode/GPU artifacts if present, skipped cleanly (not fabricated) if not ----
    encode_path = tracked_files["stage3_encode_local.pkl"]
    gpu_path = tracked_files["stage3_gpu_results.pkl"]
    if os.path.exists(encode_path) and os.path.exists(gpu_path):
        with open(encode_path, "rb") as f:
            train_encode = pickle.load(f)
        with open(gpu_path, "rb") as f:
            train_gpu = pickle.load(f)

        manifest["dimensions"] = {
            "n_train": int(train_encode["n_images"]),
            "n_active_nodes": int(len(train_encode["active_indices"])),
            "ref_idx": int(train_encode["ref_idx"]),
            "raw_feat_dim": int(train_encode["raw_feat"].shape[1]),
            "feat_pre_dim": int(train_encode["feat_pre"].shape[1]),
        }
        manifest["image_ordering"] = {
            "idx_is_arange": bool(np.array_equal(train_encode["idx"], np.arange(train_encode["n_images"]))),
            "labels_sha256": sha256_of_array(train_encode["labels"]),
            "labels_first10": train_encode["labels"][:10].tolist(),
            "labels_last10": train_encode["labels"][-10:].tolist(),
        }
        for name in TOPOLOGY_NAMES:
            if name in train_gpu.get("results", {}):
                theta_T = train_gpu["results"][name]["theta_T"]
                manifest["graphs"].setdefault(name, {})["theta_T_train_sha256"] = sha256_of_array(theta_T)
                manifest["graphs"][name]["theta_T_train_shape"] = list(theta_T.shape)
    else:
        print("Training encode/GPU artifacts not present locally -- "
              "dimensions/image_ordering/theta_T hashes skipped, not fabricated.")

    topo_path = tracked_files["stage3_topologies.pkl"]
    if os.path.exists(topo_path):
        with open(topo_path, "rb") as f:
            topologies = pickle.load(f)
        for name, W in topologies.items():
            manifest["graphs"].setdefault(name, {})["adjacency_sha256"] = sha256_of_array(W)
            manifest["graphs"][name]["n_edges"] = int(np.count_nonzero(np.triu(W, 1)))
            manifest["graphs"][name]["shape"] = list(W.shape)
    else:
        print("stage3_topologies.pkl not present locally -- adjacency hashes skipped.")

    # ---- Selected C values the confirmatory run actually consumed ----
    conditions_path = tracked_files["stage3_classifier_conditions.pkl"]
    if os.path.exists(conditions_path):
        with open(conditions_path, "rb") as f:
            stage3_conditions = pickle.load(f)
        manifest["selected_C"] = {
            label: c["selected_C"] for label, c in stage3_conditions["conditions"].items()
            if c.get("converged", False)
        }
    else:
        print("stage3_classifier_conditions.pkl not present locally -- selected_C skipped.")

    # ---- Frozen headline numbers, for the regression test to check against ----
    confirmatory_path = tracked_files["stage4_confirmatory_results.pkl"]
    if os.path.exists(confirmatory_path):
        with open(confirmatory_path, "rb") as f:
            confirmatory = pickle.load(f)
        manifest["frozen_primary_effect"] = {
            "observed_mean_d": confirmatory["primary"]["bootstrap"]["observed_mean"],
            "ci_low": confirmatory["primary"]["bootstrap"]["ci_low"],
            "ci_high": confirmatory["primary"]["bootstrap"]["ci_high"],
            "verdict": confirmatory["primary"]["verdict"],
        }
        manifest["image_ordering"]["y_test_sha256"] = sha256_of_array(confirmatory["y_test"])
        manifest["image_ordering"]["n_test"] = int(len(confirmatory["y_test"]))
    else:
        print("stage4_confirmatory_results.pkl not present locally -- "
              "frozen_primary_effect skipped.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ARTIFACT_MANIFEST.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
