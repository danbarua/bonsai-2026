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
import hashlib
import io
import json
import os
import platform
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


def sha256_of(url, cache_dir):
    """SHA-256 of the cached payload for `url`, so the committed result
    records WHICH bytes it was computed from. The objects carry no
    manifests to validate against, so this identifies rather than verifies
    -- a later reader can at least tell whether they hold the same input."""
    local = os.path.join(cache_dir, url.rsplit("/", 2)[-2] + "__" + url.rsplit("/", 1)[-1])
    digest = hashlib.sha256()
    with open(local, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def induced_inf_norm(M):
    """max over output coordinates of sum over input coordinates of |M|.

    M is (n_inputs, n_outputs) and multiplies on the right, so the sum runs
    down axis 0 and the max across axis 1.
    """
    return float(np.abs(M).sum(axis=0).max())


def measure(condition, X, Y, alpha, frozen_mse=None):
    """Refit and measure. `frozen_mse` is the production `mse_<condition>`
    array from the final-fit npz, used to check that this refit reproduces
    the frozen result.

    W is NOT persisted by run_ladder_stage3 (only mse_* and summary_json),
    so this operator is a RECONSTRUCTION, not the production matrix read
    back. The reconstruction is validated against the one production
    quantity that WAS persisted. Note what that does and does not buy:
    matching per-image MSE confirms corpus alignment and that the refit
    reproduces the frozen ERROR to ~1e-12. It does NOT prove the predictions
    are identical -- per-image MSE is a many-to-one summary of a 505-vector,
    so equal MSE is consistent with different predictions -- and it
    certainly does not prove coefficient identity. This project's own
    equivalence gate treats coefficient agreement as diagnostic rather than
    binding (stage2b_ridge.ridge_equivalence_check), so there is no
    established route from "same error" to "same operator".

    The honest statement is therefore: the norm below belongs to a refit
    that reproduces the frozen per-image error, on the same corpus, at the
    frozen alpha. Whether it is the production operator is untested and
    untestable from what was persisted.
    """
    fit, scaler = ridge.fit_final(X, Y, alpha)
    W = np.asarray(fit["W"][0], dtype=np.float64)      # (p, k)
    s = np.asarray(scaler.scale_, dtype=np.float64)    # (p,)

    M = W / s[:, None]                                 # diag(1/s) @ W
    raw_std = np.asarray(X, dtype=np.float64).std(axis=0)

    row = {
        "condition": condition,
        "alpha": float(alpha),
        "n_images": int(X.shape[0]),
        "scaler_norm": float(1.0 / s.min()),
        "ridge_norm": induced_inf_norm(W),
        "combined_norm": induced_inf_norm(M),
        "raw_min_col_std": float(raw_std.min()),
        "raw_median_col_std": float(np.median(raw_std)),
    }

    if frozen_mse is not None:
        prediction = ridge.ridge_predict(fit, scaler.transform(X), 0)
        refit_mse = ridge.clipped_per_image_mse(prediction, Y)
        frozen = np.asarray(frozen_mse, dtype=np.float64)
        row["refit_vs_frozen_max_abs_mse_diff"] = float(
            np.max(np.abs(refit_mse - frozen)))
        row["refit_vs_frozen_mean_abs_mse_diff"] = float(
            np.mean(np.abs(refit_mse - frozen)))
        row["frozen_mean_mse"] = float(np.mean(frozen))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=os.path.join(_THIS_DIR, "results", "_operator_cache"))
    parser.add_argument("--conditions", nargs="*", default=list(conditions.ALL_CONDITIONS))
    parser.add_argument("--out", default=os.path.join(_THIS_DIR, "results",
                                                      "combined_operator_norms.json"))
    args = parser.parse_args(argv)

    with load_npz(FINAL_FIT, args.cache) as final:
        alphas = json.loads(str(final["summary_json"]))["alphas"]
        frozen_mse = {c: np.asarray(final[f"mse_{c}"]) for c in args.conditions
                      if f"mse_{c}" in final.files}
    with load_npz(TOPOLOGIES, args.cache) as topo:
        active_indices = np.asarray(topo["active_indices"])
    with load_npz(CORPUS, args.cache) as corpus:
        images = np.asarray(corpus["images_01"], dtype=np.float64)
    n = images.shape[0]
    Y = images.reshape(n, FULL_GRID)[:, active_indices]
    del images
    digests = {name: sha256_of(url, args.cache)
               for name, url in (("final_fit", FINAL_FIT), ("corpus", CORPUS),
                                 ("topologies", TOPOLOGIES))}
    print(f"corpus n={n}, Y={Y.shape}, active={active_indices.size}\n", flush=True)

    rows = []
    for condition in args.conditions:
        segment = conditions.path_segment(condition)
        url = f"{BASE}/stage3/{segment}/features.npz"
        print(f"[{condition}] fetching {url.rsplit('/', 2)[-2]}/features.npz ...", flush=True)
        with load_npz(url, args.cache) as handle:
            X = np.asarray(handle["X"], dtype=np.float64)
        print(f"[{condition}] X={X.shape}, fitting at alpha={alphas[condition]} ...", flush=True)
        row = measure(condition, X, Y, alphas[condition],
                      frozen_mse=frozen_mse.get(condition))
        del X
        digests[f"features_{condition}"] = sha256_of(url, args.cache)
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
                                 "source_sha256": digests,
                                 "environment": {"numpy": np.__version__,
                                                 "platform": platform.platform()},
                                 "claimed_end_to_end_lipschitz": CLAIMED_END_TO_END_LIPSCHITZ,
                                 "note": "combined_norm is the induced infinity-norm of "
                                         "diag(1/s) @ W: max over output coordinates of the "
                                         "sum over input coordinates of |M|."},
                                indent=2, sort_keys=True))
    print(f"""
WHAT THIS IS AND IS NOT. `combined_norm` is the exact worst-case gain from
raw features to unclipped prediction, over ARBITRARY feature perturbations.
It is a MIDDLE SUBMAP. A value above {CLAIMED_END_TO_END_LIPSCHITZ:.0f} shows axis 4's published
derivation is invalid; it does NOT by itself show the full
ODE->features->readout->clipping->MSE->Delta_g composition exceeds {CLAIMED_END_TO_END_LIPSCHITZ:.0f}, because
the upstream map may never reach the maximising direction and clipping/MSE
may contract it. This file claimed otherwise once; external review, 2026-08-11.

THE COUNTEREXAMPLE IS ELSEWHERE, and already measured. Protocol 1
(FINDINGS.md:1535-1596) propagated a real ARM-vs-x86 perturbation through
the entire pipeline. The quotient is STAGE 1 -> STAGE 5: axis 4's
B(eps) = 2*sin(eps/2) is the IMMEDIATE cos/sin bound from an encoder phase
residual, so the input is the stage-1 encoding difference, NOT stage 2 --
stage 2 is post-ODE and an earlier version of this note wrongly called it
B. At the observed eps = 4.4408921e-16, B(eps) equals eps numerically.

Ratios of global maxima are conservative and do not require numerator and
denominator to fall on the same image:

    T               9.975e-14 / 4.441e-16 =  224.6    112x the claimed 2
    lattice         4.455e-13 / 4.441e-16 = 1003.1    502x
    rewired         5.483e-13 / 4.441e-16 = 1234.7    617x
    curr_random     1.830e-12 / 4.441e-16 = 4120.0   2060x

There is no pre_evolution row because Delta_g is DEFINED relative to
pre_evolution; its stage-1 -> stage-2 ratio is 1.0.

SCOPE, and it is narrower than "2B fails everywhere". This refutes the
COEFFICIENT-2 derivation for the implemented pipeline. It does not measure
a global Lipschitz constant, and it does not fail every swept envelope:
against the smallest tabulated one, 2B = 2e-13 at eps = 1e-13, T stays
below while lattice, rewired and curr_random exceed it -- and none exceeds
2B = 2e-12 at eps = 1e-12.""")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
