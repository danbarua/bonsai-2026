"""
Decomposes docs/report_visuals/generate_report_visuals.py's plot10
("early leader vs. final winner") into the two distinct transitions a
statistical review found it was conflating.

**The problem plot10 had.** It compared argmax(q_tangent at tau=0.95)
against argmax(fixed_time_q at tau=T) -- the cached FINITE (nonlinear)
endpoint. That bundles together two different things: (a) time evolution
WITHIN the linear tangent system (early tangent tau=0.95 -> final tangent
tau=T), and (b) the nonlinear modification at the SAME timepoint (final
tangent tau=T -> final finite tau=T). For seed=3000 specifically,
CONCENTRATION_REGIME_NOTE.md Part 3 already demonstrated genuine
tangent-system overtaking directly (time-resolved q_tangent(tau) for three
tracked nodes, node 152 leading until roughly tau=1.35-1.40 before node
103 overtakes it). For the other 4 seeds (3010, 3020, 3080, 3090) this was
never separated -- the mismatch plot10 reported for those 4 could in
principle be entirely due to the nonlinear step, not linear overtaking at
all.

**Pure re-analysis of already-cached data -- no new simulation.** Reuses:
  - stage1b2_frontier_visuals_data.pkl's q_tangent_full (already computed
    by generate_frontier_visuals_data.py, DISCLOSED NEW SIMULATION at the
    time IT was generated) for the early tangent leader (tau=0.95 argmax)
    -- exactly the same lookup plot10 already does.
  - each trial's own already-cached fixed_time_q_tangent (linear tangent
    response at tau=T) and fixed_time_q (finite/nonlinear response at
    tau=T) from stage1b2_results.pkl / the Stage 1C per-seed result files.
    Both fields are already saved per trial (confirmed: every one of the
    432 trials in stage1b2_results.pkl has both keys) -- plot9 already
    reads fixed_time_q_tangent the same way. No new simulation for this
    script at all.

Covers the same 87 concentrated trials (top1_finite > 0.5, in the
high-degree-node/t_p=0 cell) across the same 5 concentrating seeds (3000,
3010, 3020, 3080, 3090) that plot10 covers.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _THIS_DIR / "results"
sys.path.insert(0, str(_THIS_DIR))

STAGE1B2_RESULTS_PATH = RESULTS_DIR / "stage1b2_results.pkl"
STAGE1C_RESULTS_DIR = _THIS_DIR / ".." / "stage1c_trajectory_generalization" / "results"
FRONTIER_DATA_PATH = RESULTS_DIR / "stage1b2_frontier_visuals_data.pkl"
OUT_PATH = RESULTS_DIR / "stage1b2_early_leader_decomposition.pkl"

CONCENTRATING_SEEDS = [3000, 3010, 3020, 3080, 3090]
EARLY_TAU = 0.95


def _load_high_tp0_cell(seed):
    path = STAGE1B2_RESULTS_PATH if seed == 3000 else STAGE1C_RESULTS_DIR / f"stage1c_results_seed{seed}.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    return {k: v for k, v in results.items() if k[0] == 0 and k[2] == "high"}


def _per_seed_breakdown(rows, key):
    out = {}
    for seed in sorted({r["seed"] for r in rows}):
        seed_rows = [r for r in rows if r["seed"] == seed]
        n_match = sum(r[key] for r in seed_rows)
        out[seed] = (n_match, len(seed_rows))
    return out


def main():
    with open(FRONTIER_DATA_PATH, "rb") as f:
        frontier = pickle.load(f)
    extra_tangent = frontier["extra_tangent_by_seed_replica"]

    def early_tangent_leader(seed, replica):
        """Identical lookup to plot10's early_leader_for()."""
        if replica == 0:
            entry = frontier["per_seed"][seed]
        else:
            entry = extra_tangent.get((seed, replica))
            if entry is None:
                return None
        t = entry["t_eval"]
        idx = int(np.argmin(np.abs(t - EARLY_TAU)))
        return int(np.argmax(entry["q_tangent_full"][:, idx]))

    rows = []
    skipped_no_tangent = 0
    for seed in CONCENTRATING_SEEDS:
        cell = _load_high_tp0_cell(seed)
        for (t_p, replica, node_label, sign, amp), trial in cell.items():
            q_f = np.asarray(trial["fixed_time_q"])
            top1_f = float(q_f.max())
            if top1_f <= 0.5:
                continue  # not a concentrated trial, same threshold as plot10
            final_finite = int(np.argmax(q_f))
            q_t = np.asarray(trial["fixed_time_q_tangent"])
            final_tangent = int(np.argmax(q_t))
            early = early_tangent_leader(seed, replica)
            if early is None:
                skipped_no_tangent += 1
                continue
            rows.append({
                "seed": seed, "replica": replica, "sign": sign, "amplitude": amp,
                "early_tangent_leader": early,
                "final_tangent_winner": final_tangent,
                "final_finite_winner": final_finite,
                "match_a_linear_overtaking": early == final_tangent,
                "match_b_nonlinear_modification": final_tangent == final_finite,
            })

    breakdown_a = _per_seed_breakdown(rows, "match_a_linear_overtaking")
    breakdown_b = _per_seed_breakdown(rows, "match_b_nonlinear_modification")

    print(f"{len(rows)} concentrated trials across {len(CONCENTRATING_SEEDS)} seeds "
          f"({skipped_no_tangent} skipped -- no tangent solve at that replica)\n")
    print("Transition (a): early tangent (tau=0.95) -> final tangent (tau=T)  [LINEAR overtaking, within the tangent system]")
    for seed, (n_match, n) in breakdown_a.items():
        print(f"  seed={seed}: {n_match}/{n} match")
    print("\nTransition (b): final tangent (tau=T) -> final finite (tau=T)  [NONLINEAR modification, same timepoint]")
    for seed, (n_match, n) in breakdown_b.items():
        print(f"  seed={seed}: {n_match}/{n} match")

    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "rows": rows,
            "breakdown_a_linear_overtaking": breakdown_a,
            "breakdown_b_nonlinear_modification": breakdown_b,
            "skipped_no_tangent": skipped_no_tangent,
        }, f)
    print(f"\nSaved {OUT_PATH}")
    return rows, breakdown_a, breakdown_b


if __name__ == "__main__":
    main()
