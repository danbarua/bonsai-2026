"""Measure the COMBINED scaler->ridge operator norm per condition.

WHY. `run_abs_conv_eps_sensitivity.py`'s axis 4 bounds the whole chain at a
Lipschitz constant of 2. `measure_scaler_lipschitz.py` showed one link -- the
standardiser -- is ~1e3 for T, which refutes that DERIVATION but settles
nothing about the composed constant: composing operator norms bounds a
product from ABOVE, and the ridge is free to contract or annihilate exactly
the direction the scaler maximally amplifies. External review, 2026-08-10.

So measure the composition directly. Per condition:

    z = (x - mu) / s          standardiser
    pred = z @ W              ridge readout, W is (p, k) -- see below
    => pred = (x - mu) @ M    with M = diag(1/s) @ W, i.e. W / s[:, None]

`fit["W"][0]` is (p, k) = (n_features, n_targets) and multiplies on the
RIGHT (`stage2b_ridge.ridge_predict`, and the transpose in
`sklearn_ridge_predict`'s docstring). For a perturbation `dx` with
||dx||_inf <= B:

    |d_pred_j| = |sum_i dx_i M_ij| <= B * sum_i |M_ij|

so the induced infinity-norm is the MAX OVER OUTPUT COORDINATES OF THE SUM
OVER INPUT COORDINATES of |M| -- a column sum in this (p, k) layout. Named
that way in full here rather than as "row sum" or "column sum", because the
answer flips with layout convention and this file has already been wrong
about it once in each direction.

SECOND QUESTION, and the more interesting one. External review's hypothesis:
strongly synchronizing controls compress raw across-image variation most
severely, after which their independent scalers expand it by thousands --
which would explain why apparently collapsed dynamics still yield useful
ridge representations. This prints the raw feature spread alongside the
norms so the two can be read against each other.

Inputs are all public-read; no credentials, no GPU. ~460MB per condition,
streamed and released one at a time. `fit_final` is called unmodified
(CLAUDE.md principle 16 -- the production fit, not a reimplementation).

    uv run python experiments/stage2b_denoising/measure_combined_operator_norm.py
"""
import argparse
import io
import json
import os
import sys
import urllib.request

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2b_ridge as ridge          # noqa: E402
import stage2b_conditions as conditions  # noqa: E402

BUCKET = "bonsai-2026-stage2b-cache"
BASE = f"https://storage.googleapis.com/{BUCKET}/stage2b/train"
FINAL_FIT = f"{BASE}/stage3/common/ridge_final_g13_88edf9ac.npz"
CORPUS = f"{BASE}/stage3/common/corpus.npz"
TOPOLOGIES = f"{BASE}/stage1/common/topologies.npz"

FULL_GRID = 784
CLAIMED_END_TO_END_LIPSCHITZ = 2.0


def load_npz(url, cache_dir):
    """Fetch (once) and open an npz. Cached so a rerun costs no bandwidth."""
    local = os.path.join(cache_dir, url.rsplit("/", 2)[-2] + "__" + url.rsplit("/", 1)[-1])
    if not os.path.exists(local) or os.path.getsize(local) == 0:
        os.makedirs(cache_dir, exist_ok=True)
        with urllib.request.urlopen(url) as response, open(local, "wb") as handle:
            while True:
                chunk = response.read(8 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    return np.load(local, allow_pickle=False)


def induced_inf_norm(M):
    """max over output coordinates of sum over input coordinates of |M|.

    M is (n_inputs, n_outputs) and multiplies on the right, so the sum runs
    down axis 0 and the max across axis 1.
    """
    return float(np.abs(M).sum(axis=0).max())


def measure(condition, X, Y, alpha):
    fit, scaler = ridge.fit_final(X, Y, alpha)
    W = np.asarray(fit["W"][0], dtype=np.float64)      # (p, k)
    s = np.asarray(scaler.scale_, dtype=np.float64)    # (p,)

    M = W / s[:, None]                                 # diag(1/s) @ W
    raw_std = np.asarray(X, dtype=np.float64).std(axis=0)

    return {
        "condition": condition,
        "alpha": float(alpha),
        "n_images": int(X.shape[0]),
        "scaler_norm": float(1.0 / s.min()),
        "ridge_norm": induced_inf_norm(W),
        "combined_norm": induced_inf_norm(M),
        "raw_min_col_std": float(raw_std.min()),
        "raw_median_col_std": float(np.median(raw_std)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=os.path.join(_THIS_DIR, "results", "_operator_cache"))
    parser.add_argument("--conditions", nargs="*", default=list(conditions.ALL_CONDITIONS))
    parser.add_argument("--out", default=os.path.join(_THIS_DIR, "results",
                                                      "combined_operator_norms.json"))
    args = parser.parse_args(argv)

    with load_npz(FINAL_FIT, args.cache) as final:
        alphas = json.loads(str(final["summary_json"]))["alphas"]
    with load_npz(TOPOLOGIES, args.cache) as topo:
        active_indices = np.asarray(topo["active_indices"])
    with load_npz(CORPUS, args.cache) as corpus:
        images = np.asarray(corpus["images_01"], dtype=np.float64)
    n = images.shape[0]
    Y = images.reshape(n, FULL_GRID)[:, active_indices]
    del images
    print(f"corpus n={n}, Y={Y.shape}, active={active_indices.size}\n", flush=True)

    rows = []
    for condition in args.conditions:
        segment = conditions.path_segment(condition)
        url = f"{BASE}/stage3/{segment}/features.npz"
        print(f"[{condition}] fetching {url.rsplit('/', 2)[-2]}/features.npz ...", flush=True)
        with load_npz(url, args.cache) as handle:
            X = np.asarray(handle["X"], dtype=np.float64)
        print(f"[{condition}] X={X.shape}, fitting at alpha={alphas[condition]} ...", flush=True)
        row = measure(condition, X, Y, alphas[condition])
        del X
        rows.append(row)
        print(f"[{condition}] scaler={row['scaler_norm']:.3e} "
              f"ridge={row['ridge_norm']:.3e} COMBINED={row['combined_norm']:.3e}\n", flush=True)

    print(f"\n{'condition':18} {'raw med std':>12} {'scaler':>11} {'ridge':>11} "
          f"{'COMBINED':>12} {'vs 2':>10}")
    print("-" * 80)
    for row in rows:
        print(f"{row['condition']:18} {row['raw_median_col_std']:12.4e} "
              f"{row['scaler_norm']:11.3e} {row['ridge_norm']:11.3e} "
              f"{row['combined_norm']:12.4e} "
              f"{row['combined_norm'] / CLAIMED_END_TO_END_LIPSCHITZ:9.2e}x")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"rows": rows,
                                 "claimed_end_to_end_lipschitz": CLAIMED_END_TO_END_LIPSCHITZ,
                                 "note": "combined_norm is the induced infinity-norm of "
                                         "diag(1/s) @ W: max over output coordinates of the "
                                         "sum over input coordinates of |M|."},
                                indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
