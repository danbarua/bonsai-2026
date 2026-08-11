"""ARM vs x86 agreement for the CNN forward pass.

Companion Protocol 1 measured the PROPAGATION chain across architectures
(encode, evolve, ridge). The CNN's forward pass was left unmeasured, for
a reason that has now gone away: no trained weights were persisted
anywhere, so any measurement had to retrain the network first. They are
persisted now (`cnn_weights.npz`), and this is that measurement.

## What makes the comparison mean anything

Both architectures must see BIT-IDENTICAL inputs, or the difference
measured is the input's and not the forward pass's. So the inputs are a
fixed slice of stage 3's stored `corruption.npz` -- the same object, the
same bytes, fetched over plain HTTPS on both sides. Nothing is corrupted,
sampled or seeded here: there is no RNG in this file, deliberately.

The weights arrive the same way, from `cnn_weights.npz`. So the ONLY
thing that differs between the two runs is the machine executing
`stage2b_cnn.forward`.

## Two phases, then a comparison

    --phase run       writes this machine's outputs + platform record
    --phase compare   reads two such files and reports the difference

`forward` carries the pinned `CNN_MATMUL_PRECISION`, so this measures the
production forward pass rather than a differently-configured one. That
pin was measured to bring CPU/GPU agreement on ONE machine from 1.058e-04
to 1.172e-07; whether it does the same across instruction sets is exactly
what is unmeasured and what this answers.

No threshold is applied. There is no measured basis for one, and
inventing a tolerance after seeing a number is what AUDIT_PROTOCOL.md's
Freeze 1 rejected. This reports; a gate can be written later by someone
who has these numbers in front of them.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import urllib.request

import numpy as np

REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)
ENV_COMMIT = "BONSAI_COMMIT"

BUCKET_URL = "https://storage.googleapis.com/bonsai-2026-stage2b-cache"
WEIGHTS_OBJECT = "stage2b/train/stage3/common/cnn_weights.npz"
CORRUPTION_OBJECT = "stage2b/train/stage3/common/corruption.npz"
N_PROBE = 512          # fixed; the slice is [:N_PROBE] of the stored corpus
OK_SENTINEL = "CNN_ARCH_OK"


def fetch_npz(object_name, cache_dir):
    local = os.path.join(cache_dir, object_name.replace("/", "__"))
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(f"{BUCKET_URL}/{object_name}", local)
    with np.load(local, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def ensure_importable():
    """Local runs import `stage2b_cnn` from this directory and this does
    nothing. A Colab run arrives as transmitted TEXT with no repository on
    disk, so the pinned commit is cloned first -- the same bootstrap the
    ladder drivers use, and the reason `BONSAI_COMMIT` is required there.

    Deliberately conditional on the import actually failing: a machine that
    already has the code must not have a clone imposed on it, or the file
    that runs stops being the file the operator is looking at."""
    try:
        import stage2b_cnn                                  # noqa: F401
        return None
    except ImportError:
        pass
    commit = os.environ.get(ENV_COMMIT)
    if not commit:
        raise SystemExit(
            f"stage2b_cnn is not importable and {ENV_COMMIT} is unset, so there "
            f"is nothing to bootstrap from. Run this from the stage2b directory, "
            f"or pass the commit to pin.")
    import subprocess
    if not os.path.isdir(os.path.join(CLONE_DIR, ".git")):
        os.makedirs(CLONE_DIR, exist_ok=True)
        for argv in (["git", "init", "-q", CLONE_DIR],
                     ["git", "remote", "add", "origin", REPO_URL],
                     ["git", "fetch", "--depth", "1", "origin", commit],
                     ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(argv, cwd=None if argv[1] == "init" else CLONE_DIR,
                           check=True, capture_output=True, text=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLONE_DIR, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"clone is at {head}, expected {commit}")
    for directory in (*EXPERIMENT_DIRS, "src"):
        entry = os.path.join(CLONE_DIR, directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    print(f"bootstrapped {CLONE_DIR} at {head}")
    return head


def phase_run(args):
    commit = ensure_importable()
    import stage2b_cnn as cnn

    weights = fetch_npz(WEIGHTS_OBJECT, args.cache_dir)
    meta = json.loads(weights["weights_json"].item())
    key = args.seed_key or meta["best_weights_key"]
    model = cnn.deserialise_model(weights[key])

    corr = fetch_npz(CORRUPTION_OBJECT, args.cache_dir)
    x = np.asarray(corr["x_t_clip"])[:N_PROBE]
    print(f"probe: {x.shape} from {CORRUPTION_OBJECT}, sha256 {digest(x)[:16]}")
    print(f"weights: {key}, sha256 {digest(weights[key])[:16]}")

    out = np.asarray(cnn.forward(model, cnn.as_image_batch(x, "probe")))
    record = {
        "machine": platform.machine(), "processor": platform.processor(),
        "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__,
        "weights_key": key,
        "weights_sha256": digest(weights[key]),
        "input_sha256": digest(x),
        "output_sha256": digest(out),
        "n_probe": int(x.shape[0]),
        "matmul_precision": cnn.CNN_MATMUL_PRECISION,
        "commit": commit,
    }
    try:
        import jax
        record["jax"] = jax.__version__
        record["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:                                    # noqa: BLE001
        pass

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, output=out,
                        record_json=np.array(json.dumps(record, indent=2, sort_keys=True)))
    print(f"wrote {args.out}")
    print(f"machine={record['machine']} output sha256 {record['output_sha256'][:16]}")
    print(OK_SENTINEL)
    return 0


def phase_compare(args):
    if not args.a or not args.b:
        raise SystemExit("--phase compare needs --a and --b")
    with np.load(args.a, allow_pickle=False) as h:
        out_a, rec_a = h["output"], json.loads(h["record_json"].item())
    with np.load(args.b, allow_pickle=False) as h:
        out_b, rec_b = h["output"], json.loads(h["record_json"].item())

    for field in ("input_sha256", "weights_sha256", "weights_key", "n_probe"):
        if rec_a[field] != rec_b[field]:
            raise SystemExit(
                f"{field} differs between the two runs ({rec_a[field]!r} vs "
                f"{rec_b[field]!r}). The comparison would measure that "
                f"difference, not the forward pass. Refusing to report it.")

    print(f"A: {rec_a['machine']:8s} {rec_a['platform']}")
    print(f"B: {rec_b['machine']:8s} {rec_b['platform']}")
    if rec_a["machine"] == rec_b["machine"]:
        print("NOTE: both records report the same machine type -- this is not "
              "a cross-architecture comparison.")

    diff = np.abs(out_a.astype(np.float64) - out_b.astype(np.float64))
    scale = np.maximum(np.abs(out_a), np.abs(out_b)).astype(np.float64)
    rel = np.where(scale > 0, diff / np.maximum(scale, 1e-300), 0.0)
    n_diff = int(np.count_nonzero(out_a != out_b))
    print(f"identical outputs: {out_a.size - n_diff} of {out_a.size} "
          f"({100.0 * (out_a.size - n_diff) / out_a.size:.4f}%)")
    print(f"max |difference|      : {diff.max():.6e}")
    print(f"max relative diff     : {rel.max():.6e}")
    print(f"mean |difference|     : {diff.mean():.6e}")
    print(f"output sha256 A / B   : {rec_a['output_sha256'][:16]} / "
          f"{rec_b['output_sha256'][:16]}")
    if args.json_out:
        summary = {
            "a": rec_a, "b": rec_b,
            "n_outputs": int(out_a.size), "n_differing": n_diff,
            "max_abs_difference": float(diff.max()),
            "max_relative_difference": float(rel.max()),
            "mean_abs_difference": float(diff.mean()),
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json_out}")
    print(OK_SENTINEL)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "compare"), default="run")
    parser.add_argument("--cache-dir", default="results/_plot_cache")
    parser.add_argument("--out", default="results/cnn_forward_arm.npz")
    parser.add_argument("--seed-key", default=None)
    parser.add_argument("--a", default=None)
    parser.add_argument("--b", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    return phase_run(args) if args.phase == "run" else phase_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
