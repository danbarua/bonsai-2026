"""
Stage 2A feasibility stage 1 (DESIGN.md, "Minimal feasibility pass"):
1,000 official-training images (100/class), end-to-end mechanical
correctness only -- explicitly NOT an early scientific result. This
script never touches the official KMNIST test set.

Checks, per DESIGN.md's locked go/no-go criteria:
- zero non-finite feature vectors;
- zero silent solver failures (every evolution ODE solve reports its
  own status; any non-recovered failure is a stop, not a rate);
- R(theta) diagnostic distribution reported (never a gauge trigger);
- the linear readout converges in every condition (stops, per the
  locked non-convergence gate, otherwise).

Three conditions (raw pixels, encoded pre-evolution, evolved on T),
each fit via the real, locked CV procedure (stage2a_classifier.py) on
this 1,000-image set -- reported descriptively, per DESIGN.md's "the
feasibility ladder may report raw differences descriptively, without
formal inference."

Pipeline mechanics (subsampling, per-image processing, go/no-go checks,
classifier-condition fitting) live in stage2a_pipeline.py, shared with
feasibility stage 2 rather than duplicated.
"""
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
import stage2a_pipeline as pipe

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
SEED = 42
N_PER_CLASS = 100


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
    print(f"Official training set: {X_train.shape[0]} images "
          f"(official test set NOT loaded -- not needed at stage 1)")

    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=SEED, n_per_class=N_PER_CLASS)
    print(f"Subsampled {len(images_01)} images ({N_PER_CLASS}/class, SEED={SEED})")

    print("\nRunning pipeline (encode -> restrict -> evolve -> gauge features)...")
    results, elapsed, active_indices, nodes_T = pipe.run_pipeline(images_01, labels)
    print(f"Pipeline complete: {len(results)} images in {elapsed:.1f}s "
          f"({elapsed/len(results)*1000:.1f} ms/image)")

    print("\n" + "=" * 70)
    print("GO/NO-GO MECHANICAL CHECKS")
    print("=" * 70)
    go_no_go = pipe.check_go_no_go(results)
    for k, v in go_no_go.items():
        if k not in ("R_pre_values", "R_post_values"):
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("CLASSIFIER FITTING (3 conditions, correctness check only)")
    print("=" * 70)
    conditions_out = pipe.run_classifier_conditions(results, labels, label_prefix="stage1_")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "stage1_feasibility_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "go_no_go": go_no_go,
            "conditions": conditions_out,
            "elapsed_seconds": elapsed,
            "n_images": len(results),
            "nodes_T": nodes_T,
            "seed": SEED,
            "n_per_class": N_PER_CLASS,
            "selected_idx": selected_idx,
        }, f)
    print(f"\nSaved to {out_path}")

    all_go = (go_no_go["solver_failure_rate_ok"] and go_no_go["non_finite_ok"]
              and all(c.get("converged", False) for c in conditions_out.values()))
    print(f"\n{'='*70}\nOVERALL: {'GO' if all_go else 'NO-GO'} "
          f"(mechanical criteria only -- not a scientific result)\n{'='*70}")
    return go_no_go, conditions_out


if __name__ == "__main__":
    main()
