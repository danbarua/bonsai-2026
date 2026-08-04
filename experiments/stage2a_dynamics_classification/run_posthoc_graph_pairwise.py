"""
Post hoc, exploratory graph-to-graph pairwise comparison among the four
evolved conditions (T, lattice, rewired, curr_random), per external
review's own explicit recommendation in FINDINGS.md's secondary-
comparisons section: "these could be computed from the already-saved
per-image losses, but any such comparison would now be explicitly post
hoc and should use multiplicity correction across however many pairwise
tests it involved."

NO NEW SIMULATION, NO NEW GPU TIME: reuses the per-image test-set losses
already computed and saved by run_confirmatory_evaluation.py
(results/stage4_confirmatory_results.pkl). This is a new bootstrap
computation on existing data only.

For each pair, d_i = ell_i(graph_A) - ell_i(graph_B). Two separate
statistics are computed, deliberately not conflated:

1. **Descriptive interval**: the same 20,000 paired class-stratified
   bootstrap procedure as the locked primary/secondary comparisons
   (DESIGN.md), reported as a pointwise 95% percentile interval on
   mean d_i -- kept exactly as before, per external review's own
   guidance that these intervals "can remain as pointwise descriptive
   intervals."
2. **Inferential p-value, for the Holm family**: a paired sign-flip
   permutation test (`stage2a_stats.paired_sign_flip_p`), NOT the
   bootstrap-derived `bootstrap_two_sided_p`. External review found the
   latter is not properly null-calibrated for a family-wise-error
   claim -- it is closely related to inverting the percentile CI
   (a distribution centred on the observed effect), not a genuine
   simulation of the null. Sign-flipping is: under H0 (no systematic
   difference between the two evolved conditions), each image's d_i is
   exchangeable with -d_i, so independently flipping signs directly
   simulates the null distribution (CLAUDE.md principle 10), unit-tested
   on synthetic data first (tests/test_stage2a_stats.py) per that same
   principle's explicit requirement.

This is new, un-pre-registered multiple-comparison territory (DESIGN.md
only locked the four graph-vs-pre-evolution tests) -- Holm-Bonferroni
correction across all six sign-flip-test p-values, as one family, is
applied here, not optional.
"""
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

from stage2a_stats import (  # noqa: E402
    N_RESAMPLES, paired_class_stratified_bootstrap, paired_sign_flip_p, holm_bonferroni,
)

EVOLVED_CONDITIONS = ["evolved_T", "evolved_lattice", "evolved_rewired", "evolved_curr_random"]


def main():
    print("Loading already-saved per-image test-set losses (no new simulation)...")
    with open(os.path.join(RESULTS_DIR, "stage4_confirmatory_results.pkl"), "rb") as f:
        d = pickle.load(f)
    ell_i = d["condition_ell_i"]
    y_test = d["y_test"]

    pairs = []
    for i in range(len(EVOLVED_CONDITIONS)):
        for j in range(i + 1, len(EVOLVED_CONDITIONS)):
            pairs.append((EVOLVED_CONDITIONS[i], EVOLVED_CONDITIONS[j]))
    assert len(pairs) == 6, f"expected 6 pairs, got {len(pairs)}"

    print(f"\nRunning all {len(pairs)} pairwise comparisons: {N_RESAMPLES} paired "
          f"class-stratified bootstrap resamples each (descriptive interval only) "
          f"plus a separate {N_RESAMPLES}-permutation paired sign-flip test "
          f"(the inferential p-value the Holm family below actually uses)...\n")
    results = {}
    for a, b in pairs:
        d_ab = ell_i[a] - ell_i[b]
        boot = paired_class_stratified_bootstrap(d_ab, y_test)
        p = paired_sign_flip_p(d_ab, n_perms=N_RESAMPLES)
        # Descriptive only -- which condition has the lower observed mean
        # loss, NOT a significance claim (that's what the Holm-corrected
        # sign-flip p-value below is for).
        direction = f"{a} lower" if boot["observed_mean"] < 0 else f"{b} lower"
        results[(a, b)] = {"observed_mean": boot["observed_mean"], "ci_low": boot["ci_low"],
                            "ci_high": boot["ci_high"], "raw_p": p, "direction": direction}
        print(f"[{a} vs {b}] mean_d={boot['observed_mean']:+.4f}, "
              f"95% CI (descriptive)=[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}], "
              f"sign-flip raw_p={p:.4e} -> {direction}")

    raw_p = {k: v["raw_p"] for k, v in results.items()}
    adjusted, rejected = holm_bonferroni(raw_p, alpha=0.05)

    print(f"\n{'='*70}\nHOLM-BONFERRONI CORRECTION, family of {len(pairs)}, alpha=0.05\n{'='*70}")
    order = sorted(raw_p.keys(), key=lambda k: raw_p[k])
    for rank, key in enumerate(order, start=1):
        a, b = key
        print(f"  rank {rank}: [{a} vs {b}] raw_p={raw_p[key]:.4e}, "
              f"holm_adj_p={adjusted[key]:.4e}, survives={rejected[key]}")

    n_survive = sum(rejected.values())
    print(f"\n{n_survive} of {len(pairs)} pairwise comparisons survive Holm correction at alpha=0.05.")

    with open(os.path.join(RESULTS_DIR, "stage4_posthoc_pairwise_results.pkl"), "wb") as f:
        pickle.dump({"results": results, "raw_p": raw_p, "holm_adjusted_p": adjusted,
                     "holm_rejected": rejected}, f)
    print(f"\nSaved results/stage4_posthoc_pairwise_results.pkl")


if __name__ == "__main__":
    main()
