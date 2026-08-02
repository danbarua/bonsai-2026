"""
Stage 2A feasibility stage 2 (DESIGN.md, "Minimal feasibility pass"):
up to 5,000 official-training images, plus the fixed training-derived
validation subset (this same 5,000-image set) -- throughput measurement
and development. Still not confirmatory. Never touches the official
KMNIST test set.

Two things this stage does that stage 1 did not:
1. Measures throughput at 5x stage 1's scale and extrapolates to stage
   3's full 60,000-image training set.
2. Runs the encoder-RNG robustness check (DESIGN.md, "Encoding"): reuses
   this exact 5,000-image subset, recomputes encoded-pre-evolution and
   evolved-T features with each image's encoder seed set to its
   immutable dataset index (rather than the shared primary seed 0),
   refits both readouts on identical folds/C-selection, and reports the
   change in validation log-loss difference -- explicitly descriptive,
   never able to replace the seed-0 primary analysis.
"""
import os
import pickle
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
import stage2a_pipeline as pipe

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
SEED = 42
N_PER_CLASS = 500  # 5,000 total
N_TRAIN_FULL = 60_000


def run_encoder_seed_robustness_check(images_01, labels, selected_idx, seed0_conditions):
    """DESIGN.md's locked robustness check: recompute pre-evolution and
    evolved-T features with per-image seed = each image's immutable
    dataset index, refit on identical folds/C-selection, report the
    validation-log-loss-difference change vs. the seed-0 primary run.
    Raw-pixel features don't depend on the encoder at all, so this check
    only applies to the two encoder-derived conditions."""
    print("\nRe-running pipeline with independent per-image encoder seeds "
          "(seed = each image's immutable dataset index)...")
    results_indep, elapsed, _active_indices, _nodes_T = pipe.run_pipeline(
        images_01, labels, encoder_seeds=selected_idx)
    print(f"Independent-seed pipeline complete: {len(results_indep)} images in {elapsed:.1f}s")

    conditions_indep = pipe.run_classifier_conditions(
        results_indep, labels, label_prefix="stage2_indepseed_")

    robustness = {}
    for cond in ("encoded_pre_evolution", "evolved_T"):
        seed0_loss = seed0_conditions[cond].get("mean_val_loss_at_selected_C")
        indep_loss = conditions_indep[cond].get("mean_val_loss_at_selected_C")
        robustness[cond] = {
            "seed0_mean_val_loss_at_its_own_selected_C": seed0_loss,
            "indep_seed_mean_val_loss_at_its_own_selected_C": indep_loss,
            "change": (indep_loss - seed0_loss) if (seed0_loss is not None and indep_loss is not None) else None,
            "seed0_selected_C": seed0_conditions[cond].get("selected_C"),
            "indep_seed_selected_C": conditions_indep[cond].get("selected_C"),
        }
    return robustness, conditions_indep


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
    print(f"Official training set: {X_train.shape[0]} images "
          f"(official test set NOT loaded -- not needed at stage 2)")

    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=SEED, n_per_class=N_PER_CLASS)
    print(f"Subsampled {len(images_01)} images ({N_PER_CLASS}/class, SEED={SEED}) "
          f"-- this is the fixed training-derived validation subset")

    print("\nRunning primary (seed=0) pipeline (encode -> restrict -> evolve -> gauge features)...")
    results, elapsed, active_indices, nodes_T = pipe.run_pipeline(images_01, labels)
    per_image_ms = elapsed / len(results) * 1000
    print(f"Pipeline complete: {len(results)} images in {elapsed:.1f}s ({per_image_ms:.1f} ms/image)")

    projected_stage3_seconds = per_image_ms / 1000 * N_TRAIN_FULL
    print(f"Projected stage-3 pipeline runtime ({N_TRAIN_FULL} images): "
          f"{projected_stage3_seconds:.0f}s ({projected_stage3_seconds/60:.1f} min)")

    print("\n" + "=" * 70)
    print("GO/NO-GO MECHANICAL CHECKS (5,000-image scale)")
    print("=" * 70)
    go_no_go = pipe.check_go_no_go(results)
    for k, v in go_no_go.items():
        if k not in ("R_pre_values", "R_post_values"):
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("CLASSIFIER FITTING (3 conditions, seed=0 primary)")
    print("=" * 70)
    conditions_out = pipe.run_classifier_conditions(results, labels, label_prefix="stage2_seed0_")

    # DESIGN.md's locked stop-gate: "any non-converged fit during a required
    # fold/C combination stops advancement to the next stage, pending
    # investigation." run_classifier_conditions() catches NonConvergenceError
    # per-condition so every condition gets attempted and reported -- but
    # this driver must not then proceed to a further phase (the robustness
    # check) as if nothing happened. Halt here, save what's known so far,
    # and stop -- do not silently continue.
    non_converged = {label: c for label, c in conditions_out.items() if not c.get("converged", False)}
    if non_converged:
        print("\n" + "=" * 70)
        print("HALTING: non-convergence in primary CV fitting -- per DESIGN.md's "
              "locked stop-gate, this stage does not advance further (no robustness "
              "check, no stage-3 go-ahead) until investigated.")
        print("=" * 70)
        for label, c in non_converged.items():
            print(f"  {label}: {c['error']}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "stage2_feasibility_results.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "go_no_go": go_no_go,
                "conditions_seed0": conditions_out,
                "halted_on_non_convergence": True,
                "elapsed_seconds": elapsed,
                "per_image_ms": per_image_ms,
                "projected_stage3_seconds": projected_stage3_seconds,
                "n_images": len(results),
                "nodes_T": nodes_T,
                "seed": SEED,
                "n_per_class": N_PER_CLASS,
                "selected_idx": selected_idx,
            }, f)
        print(f"\nSaved (partial, halted) results to {out_path}")
        print(f"\n{'='*70}\nOVERALL: NO-GO -- non-convergence, investigation required "
              f"before proceeding\n{'='*70}")
        return go_no_go, conditions_out, None

    print("\n" + "=" * 70)
    print("ENCODER-SEED ROBUSTNESS CHECK")
    print("=" * 70)
    robustness, conditions_indep = run_encoder_seed_robustness_check(
        images_01, labels, selected_idx, conditions_out)
    for cond, r in robustness.items():
        print(f"  {cond}: seed0={r['seed0_mean_val_loss_at_its_own_selected_C']:.4f} "
              f"(C={r['seed0_selected_C']}), indep_seed={r['indep_seed_mean_val_loss_at_its_own_selected_C']:.4f} "
              f"(C={r['indep_seed_selected_C']}), change={r['change']:+.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "stage2_feasibility_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "go_no_go": go_no_go,
            "conditions_seed0": conditions_out,
            "conditions_indep_seed": conditions_indep,
            "robustness_check": robustness,
            "elapsed_seconds": elapsed,
            "per_image_ms": per_image_ms,
            "projected_stage3_seconds": projected_stage3_seconds,
            "n_images": len(results),
            "nodes_T": nodes_T,
            "seed": SEED,
            "n_per_class": N_PER_CLASS,
            "selected_idx": selected_idx,
        }, f)
    print(f"\nSaved to {out_path}")

    all_go = (go_no_go["solver_failure_rate_ok"] and go_no_go["non_finite_ok"]
              and all(c.get("converged", False) for c in conditions_out.values())
              and all(c.get("converged", False) for c in conditions_indep.values()))
    print(f"\n{'='*70}\nOVERALL: {'GO' if all_go else 'NO-GO'} "
          f"(mechanical criteria only -- not a scientific result)\n{'='*70}")
    return go_no_go, conditions_out, robustness


if __name__ == "__main__":
    main()
