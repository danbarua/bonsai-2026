"""
Diagnostic only -- does not touch the locked pipeline or any stage
driver. Investigates the mechanism behind the striking near-total
phase synchronization (R(theta) approx 1) observed for the rewired and
curr_random topologies in a stage-3 timing sub-test (T/lattice: R_post
mean ~0.86; rewired/curr_random: R_post mean ~0.99-1.0), before
committing ~4.2 hours of compute to the full stage-3 run.

Standard Kuramoto-synchronization theory: algebraic connectivity (the
Fiedler value -- second-smallest eigenvalue of the graph Laplacian
L = D - W) is a well-established predictor of synchronization strength
-- higher algebraic connectivity means faster, more complete
synchronization. Degree-preserving random rewiring is a classic way to
increase algebraic connectivity relative to a structured graph, since
rewiring destroys any community/clustering structure (which impedes
global synchronization) while holding the degree sequence fixed.
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_topologies as topo


def laplacian_spectrum(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals = np.linalg.eigvalsh(L)  # L is symmetric (W symmetric), ascending order
    return eigvals


def main():
    active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()

    print("=" * 70)
    print("LAPLACIAN SPECTRUM / ALGEBRAIC CONNECTIVITY (Fiedler value)")
    print("=" * 70)
    results = {}
    for name, W in topologies.items():
        eigvals = laplacian_spectrum(W)
        # eigvals[0] should be ~0 (connected graph -> one zero eigenvalue); eigvals[1] is the Fiedler value
        fiedler = eigvals[1]
        largest = eigvals[-1]
        spectral_gap = eigvals[1] - eigvals[0]
        mean_nonzero = eigvals[1:].mean()
        print(f"\n{name}:")
        print(f"  smallest eigenvalue (~0 expected): {eigvals[0]:.6f}")
        print(f"  Fiedler value (algebraic connectivity): {fiedler:.6f}")
        print(f"  largest eigenvalue: {largest:.6f}")
        print(f"  mean of nonzero-ish eigenvalues: {mean_nonzero:.6f}")
        print(f"  eigenvalue spread (max/Fiedler ratio): {largest / fiedler:.4f}")
        results[name] = {
            "smallest": float(eigvals[0]), "fiedler": float(fiedler),
            "largest": float(largest), "mean": float(mean_nonzero),
        }

    print("\n" + "=" * 70)
    print("SUMMARY: Fiedler value (algebraic connectivity) by topology")
    print("=" * 70)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["fiedler"]):
        print(f"  {name}: Fiedler={r['fiedler']:.4f}")

    return results


if __name__ == "__main__":
    main()
