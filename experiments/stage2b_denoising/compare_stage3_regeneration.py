"""Stage 3 Phase A regeneration acceptance test.

The regeneration spans two populations with different evidentiary status,
and this script exists so one "regeneration passed" cannot blur them:

- the **54,000** images the previous Phase A run encoded must come back
  BIT-EXACT from the new 60,000-image artifact;
- the **6,000** locked-validation images are NEW evidence with no prior
  artifact to compare against. They are fingerprinted at birth and their
  final-Delta tail is reported separately, never averaged into a single
  verdict.

## The join is on official indices, and the prefix shortcut is refused

`AUDIT_PROTOCOL.md`: all cross-artifact comparison is by official KMNIST
image index, never by positional prefix. Both artifacts carry their own
index arrays, so the comparison builds an index -> row map from each and
extracts the new artifact's rows in the OLD artifact's index order.

`new[:54000]` would give the right answer only if the new artifact
happened to be ordered fit-then-validation -- and it is not; it is in
ascending official order, so the fit rows are scattered through it. The
script asserts the extraction is a genuine permutation and NOT a prefix,
so the day someone reorders an artifact this fails loudly instead of
silently comparing the wrong 54,000 rows against each other.

`compare_array_manifests` is the right tool one level down but the wrong
one at the top: it compares per-array digests by key, so a (60000, 505)
array and a (54000, 505) array are reported as differing on shape, which
is true and useless. It is used here on the SUBSET, after the join.

## Why not a whole-file digest

`np.savez` writes a zip whose headers embed a timestamp at two-second
granularity, so two runs producing bit-identical arrays produce
different files. Array-level digests are what a regeneration is judged
on -- see `stage2b_fingerprint.array_manifest`.

Reads only; writes nothing to GCS.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)

import encode_stage3_local as enc                # noqa: E402
import stage2b_encoder_gate as encoder_gate      # noqa: E402
import stage2b_fingerprint as fingerprint        # noqa: E402
import stage2b_gcs as gcs                        # noqa: E402

LADDER_STAGE = 3
SPLIT = "train"

# The 54,000-image artifact this regeneration is checked against. Named
# explicitly rather than derived, because it is a historical object whose
# name encodes a population that will never be produced again.
BASELINE_KIND = "encoded_fit_s1200"
BASELINE_LOCAL = os.path.join(_THIS_DIR, "results", "stage3_encoded_fit_s1200.npz")

# The fit-side tail measured by that run, from FINDINGS.md. A bit-exact
# regeneration must reproduce it exactly; a payload comparison that
# passes while this moves means the comparison is not looking at what it
# thinks it is.
BASELINE_TAIL_NONZERO = 79
BASELINE_TAIL_N = 54_000


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def fetch(object_name, local_path, bucket, require_manifest):
    """Download unless already present, through the transport's own
    verification. The baseline predates the manifest contract, so it is
    read with the named opt-out rather than by weakening the check."""
    if os.path.exists(local_path):
        print(f"  using local copy: {local_path}")
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        gcs.download_file(object_name, local_path, bucket=bucket)
        print(f"  downloaded: {object_name} -> {local_path}")
    manifest, _ = gcs.consume_validated(object_name, local_path, bucket=bucket,
                                        require_manifest=require_manifest)
    print(f"  manifest: {'present' if manifest else 'ABSENT (pre-contract opt-out)'}")
    return load_npz(local_path), manifest


def align(old, new):
    """Rows of `new` corresponding to `old`'s images, in `old`'s order.

    Returns (row_positions, report). Raises if the overlap is not exactly
    `old`'s index set."""
    old_idx = np.asarray(old["fit_indices"])
    new_idx = np.asarray(new["train_indices"])

    if new_idx.size != np.unique(new_idx).size:
        raise ValueError("the new artifact's train_indices contain duplicates")
    position = {int(v): i for i, v in enumerate(new_idx)}
    missing = [int(v) for v in old_idx if int(v) not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} of the baseline's images are absent from the new "
            f"artifact (first few: {missing[:5]}). The regeneration does not "
            f"cover the population it is being compared against.")
    rows = np.array([position[int(v)] for v in old_idx], dtype=np.int64)

    # Vacuity guards. Without these a passing comparison could mean
    # "compared the right rows" or "compared a prefix that happened to
    # line up", and the two are indistinguishable from a green result.
    if rows.size != BASELINE_TAIL_N:
        raise ValueError(f"expected an overlap of {BASELINE_TAIL_N}, got {rows.size}")
    if np.unique(rows).size != rows.size:
        raise ValueError("the alignment maps two baseline images to one new row")
    is_prefix = bool(np.array_equal(rows, np.arange(rows.size)))
    report = {
        "n_overlap": int(rows.size),
        "n_new_total": int(new_idx.size),
        "alignment_is_a_prefix": is_prefix,
        "n_rows_moved": int(np.count_nonzero(rows != np.arange(rows.size))),
        "first_five_baseline_indices": [int(v) for v in old_idx[:5]],
        "their_rows_in_the_new_artifact": [int(v) for v in rows[:5]],
    }
    if is_prefix:
        raise ValueError(
            "the baseline's images map onto the new artifact's first 54,000 rows "
            "in order. Expected them scattered through ascending official order. "
            "Either the new artifact's row order changed, or the join silently "
            "degenerated into the positional prefix this script exists to avoid.")
    return rows, report


def compare(old, new, rows):
    """Per-array comparison over the aligned subset."""
    findings = []
    subset = {
        "thetas_505": np.asarray(new["thetas_505"])[rows],
        "deltas": np.asarray(new["deltas"])[rows],
    }
    for key, new_arr in subset.items():
        old_arr = np.asarray(old[key])
        entry = {
            "array": key,
            "shape_old": list(old_arr.shape), "shape_new": list(new_arr.shape),
            "dtype_old": str(old_arr.dtype), "dtype_new": str(new_arr.dtype),
            "sha256_old": _digest(old_arr), "sha256_new": _digest(new_arr),
        }
        entry["bit_exact"] = (entry["sha256_old"] == entry["sha256_new"]
                              and entry["shape_old"] == entry["shape_new"]
                              and entry["dtype_old"] == entry["dtype_new"])
        if not entry["bit_exact"] and old_arr.shape == new_arr.shape:
            diff = np.abs(old_arr.astype(np.float64) - new_arr.astype(np.float64))
            entry["max_abs_diff"] = float(np.max(diff))
            entry["n_differing"] = int(np.count_nonzero(diff))
        findings.append(entry)
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--credentials", default=None)
    parser.add_argument("--steps", type=int, default=encoder_gate.ENCODER_STEPS)
    parser.add_argument("--json-out", default=None,
                        help="write the full acceptance report here")
    args = parser.parse_args(argv)

    bucket = gcs.get_bucket(name=args.bucket, credentials=args.credentials)
    new_name = gcs.object_path(stage=LADDER_STAGE, condition=None,
                               kind=f"encoded_train_s{args.steps}", ext="npz",
                               split=SPLIT)
    old_name = gcs.object_path(stage=LADDER_STAGE, condition=None,
                               kind=BASELINE_KIND, ext="npz", split=SPLIT)

    print("=" * 72)
    print("STAGE 3 PHASE A REGENERATION -- ACCEPTANCE")
    print("=" * 72)
    print(f"\nbaseline (54,000, pre-contract): {old_name}")
    old, old_manifest = fetch(old_name, BASELINE_LOCAL, bucket, require_manifest=False)
    print(f"\nregenerated (60,000, under contract): {new_name}")
    new_local = os.path.join(_THIS_DIR, "results",
                             f"stage3_encoded_train_s{args.steps}.npz")
    new, new_manifest = fetch(new_name, new_local, bucket, require_manifest=True)

    print("\n-- Part 1: the 54,000 overlapping images, joined by official index")
    rows, alignment = align(old, new)
    for key, value in alignment.items():
        print(f"  {key:<34}: {value}")
    findings = compare(old, new, rows)
    for entry in findings:
        verdict = "BIT-EXACT" if entry["bit_exact"] else "DIFFERS"
        print(f"  {entry['array']:<12} {str(entry['shape_new']):<14} "
              f"{entry['dtype_new']:<8} {verdict}")
        print(f"    baseline sha256 {entry['sha256_old']}")
        print(f"    new      sha256 {entry['sha256_new']}")
        if "max_abs_diff" in entry:
            print(f"    max abs diff {entry['max_abs_diff']:.6e} over "
                  f"{entry['n_differing']} coordinates")
    part1_ok = all(entry["bit_exact"] for entry in findings)

    # The independent cross-check: a bit-exact subset must reproduce the
    # recorded tail count exactly. Same conclusion from a different
    # direction -- if the digests agree but this moves, the comparison is
    # not looking at the rows it believes it is.
    subset_deltas = np.asarray(new["deltas"])[rows]
    subset_nonzero = int(np.count_nonzero(subset_deltas > 0.0))
    tail_ok = subset_nonzero == BASELINE_TAIL_NONZERO
    print(f"\n  fit-side tail cross-check: {subset_nonzero} nonzero final-Delta "
          f"in the aligned subset, FINDINGS records {BASELINE_TAIL_NONZERO} "
          f"-> {'MATCH' if tail_ok else 'MISMATCH'}")

    print("\n-- Part 2: the 6,000 validation images, NEW evidence")
    summary = json.loads(new["summary_json"].item())
    val = summary["tail"]["validation"]
    fit = summary["tail"]["fit"]
    print(f"  no prior artifact exists for these images, so nothing is compared;")
    print(f"  they are fingerprinted at birth and their tail is measured:")
    print(f"    validation : {val['n_nonzero']}/{val['n']} = {val['rate']*100:.4f}% "
          f"[95% CI {val['ci95_lower']*100:.4f}%, {val['ci95_upper']*100:.4f}%]")
    print(f"    fit        : {fit['n_nonzero']}/{fit['n']} = {fit['rate']*100:.4f}% "
          f"[95% CI {fit['ci95_lower']*100:.4f}%, {fit['ci95_upper']*100:.4f}%]")
    print("  AUDIT_PROTOCOL.md sets NO expected-agreement criterion between these "
          "two rates.\n  A different proportion is a finding to report, not a "
          "reproducibility failure.")

    print("\n-- Provenance")
    if new_manifest:
        fp = new_manifest["fingerprint"]
        enc.print_git_state(fp["git"], indent="  ")
        print(f"  source files    : {len(fp['source_manifest'])}")
        print(f"  config digest   : {fp['config_digest']}")
        print(f"  payload sha256  : {new_manifest['payload_sha256']}")
        mismatches = fingerprint.compare(fp, fp)
        print(f"  self-consistency: {'ok' if not mismatches else mismatches}")
    baseline_state = "present" if old_manifest else (
        "none -- this artifact predates the contract, which is what the "
        "regeneration closes")
    print(f"  baseline manifest: {baseline_state}")

    ok = part1_ok and tail_ok
    print("\n" + "=" * 72)
    print(f"ACCEPTANCE: {'PASS' if ok else 'FAIL'}  "
          f"(54k bit-exact: {part1_ok}; tail cross-check: {tail_ok}; "
          f"6k: new evidence, reported not compared)")
    print("=" * 72)

    if args.json_out:
        report = {"alignment": alignment, "arrays": findings,
                  "fit_tail_crosscheck": {"observed": subset_nonzero,
                                          "recorded": BASELINE_TAIL_NONZERO,
                                          "match": tail_ok},
                  "validation_tail": val, "fit_tail": fit,
                  "manifest": new_manifest, "passed": ok}
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
        print(f"\nreport written to {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
