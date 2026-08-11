"""Is the gauge-sensitivity comparison DESIGN.md pre-registered non-trivial?

`DESIGN.md:198-206` pre-registers the reference-node gauge as primary and
circular-mean as the robustness check, and commits: if they "disagree
materially on the confirmatory result, that disagreement is reported as a
finding about gauge sensitivity, not resolved by picking whichever was
significant."

That comparison was never run. `circular_mean_features` exists in
`stage2a_core.py`, is unit-tested for its dimension only, and no pipeline
calls it. Stage 2A is CLOSED and LOCKED; this file does not reopen it and
computes no confirmatory statistic. It answers a narrower, prior question:
COULD the two gauges have disagreed, or are they equivalent for a linear
readout, in which case the missing comparison was a formality?

## The two things it measures

1. Is there a FIXED linear map from one feature set to the other? If so, a
   linear readout absorbs the difference and the gauges cannot disagree.
   They are not: `cos(theta_i - a)` is a rotation of each `(cos, sin)` pair
   by the SAME `a`, but `a` is `theta_ref` in one gauge and the circular
   mean in the other, and their difference is PER IMAGE. An image-dependent
   rotation is not a fixed linear map.

2. How much does that per-image offset actually vary ON REAL EVOLVED
   STATES? This is the part that decides whether (1) matters in practice.
   Evolved states are strongly synchronised, and if `theta_ref - mu` were
   near-constant across images the two gauges would agree up to a global
   rotation the readout could absorb after all.

Deliberately uses the TRAIN split: this produces numbers that may enter a
record, so the test split is not touched at all.

Reads `topologies.npz` from the public-read bucket, so it runs from a
fresh clone with no credentials.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
from scipy.integrate import solve_ivp

# The corruption lives in Stage 2B, whose directory is not on the path when
# this runs from Stage 2A's. Reused rather than reimplemented: the locked
# forward corruption is what makes these states the ones the pipeline sees
# (CLAUDE.md principle 16).
_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE2B = os.path.join(_HERE, "..", "stage2b_denoising")
if _STAGE2B not in sys.path:
    sys.path.insert(0, _STAGE2B)

BUCKET_URL = "https://storage.googleapis.com/bonsai-2026-stage2b-cache"
TOPOLOGIES_OBJECT = "stage2b/train/stage1/common/topologies.npz"

T_HORIZON = 2.5
RTOL, ATOL, MAX_STEP = 1e-6, 1e-8, 0.05
REF_IDX = 363          # T's median-degree node in 505-space, per DESIGN.md
ENCODER_SEED = 0
SPLIT = "train"


def fetch_topologies(cache_dir):
    local = os.path.join(cache_dir, TOPOLOGIES_OBJECT.replace("/", "__"))
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(f"{BUCKET_URL}/{TOPOLOGIES_OBJECT}", local)
    with np.load(local, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def fixed_linear_map_residual(theta, ref_idx, rng):
    """Relative residual of the best FIXED linear map between the gauges.

    Near zero would mean one gauge is a linear re-expression of the other
    and a linear readout could not tell them apart. Computed on random
    phases, where the per-image offset is unconstrained -- the upper bound
    on how different they can be."""
    import stage2a_core

    R = np.stack([stage2a_core.reference_node_features(t, ref_idx) for t in theta])
    C = np.stack([stage2a_core.circular_mean_features(t) for t in theta])
    A, *_ = np.linalg.lstsq(R, C, rcond=None)
    return float(np.linalg.norm(C - R @ A) / np.linalg.norm(C)), R.shape, C.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/kmnist")
    parser.add_argument("--cache-dir", default="results/_gauge_cache")
    parser.add_argument("--n", type=int, default=200,
                        help="real images to encode and evolve")
    args = parser.parse_args()

    import stage2b_corruption as corruption
    from bonsai.data.mnist_loader import load_mnist
    from bonsai.dynamics.learned_topology_construction import _local_converged_phases

    rng = np.random.default_rng(0)
    topo = fetch_topologies(args.cache_dir)
    active = np.asarray(topo["active_indices"])
    W = np.asarray(topo["W_T"])

    # (1) The structural question, on unconstrained phases.
    resid, r_shape, c_shape = fixed_linear_map_residual(
        rng.uniform(0, 2 * np.pi, (200, 8)), 3, rng)
    print(f"reference-node features {r_shape}, circular-mean {c_shape}")
    print(f"best FIXED linear map between the gauges: relative residual "
          f"{resid:.4e}")
    print("  (near 0 would mean a linear readout cannot tell them apart)")

    # (2) The practical question, on real evolved states.
    x_train, _, _, _ = load_mnist(args.data_dir, gz=False)
    idx = np.arange(args.n)
    clean = np.asarray(x_train)[idx].astype(np.float64).reshape(args.n, 784) / 255.0
    _x_t, noisy = corruption.corrupt_corpus(
        clean, SPLIT, idx, alpha_bar=corruption.ALPHA_BAR)

    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return np.sum(W * np.sin(diff), axis=1)

    states = []
    for i in range(args.n):
        theta0 = _local_converged_phases(
            noisy[i].reshape(28, 28), seed=ENCODER_SEED).flatten()[active]
        sol = solve_ivp(rhs, (0.0, T_HORIZON), theta0, method="RK45",
                        rtol=RTOL, atol=ATOL, max_step=MAX_STEP)
        if not sol.success:
            raise RuntimeError(f"evolution failed on image {i}: {sol.message}")
        states.append(sol.y[:, -1] % (2 * np.pi))
    theta_T = np.stack(states)

    order = np.abs(np.mean(np.exp(1j * theta_T), axis=1))
    mu = np.angle(np.mean(np.exp(1j * theta_T), axis=1))
    # Wrapped to (-pi, pi] before taking a spread: the raw difference of two
    # angles is discontinuous at the branch cut and its std is meaningless.
    offset = np.angle(np.exp(1j * (theta_T[:, REF_IDX] - mu)))

    print(f"\nreal evolved states, n={args.n}, T (learned), train split")
    print(f"  order parameter R(T)      : mean {order.mean():.4f}")
    print(f"  gauge offset theta_ref-mu : mean {offset.mean():+.4f} rad, "
          f"std {offset.std():.4f} rad")
    print(f"                              range [{offset.min():+.4f}, "
          f"{offset.max():+.4f}] rad")
    print("\nA near-constant offset would make the gauges equivalent up to a "
          "global rotation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
