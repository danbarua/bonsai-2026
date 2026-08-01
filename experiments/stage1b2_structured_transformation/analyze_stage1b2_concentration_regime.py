"""
Follow-up, scoped observation on already-frozen Stage 1B.2 data -- NOT a
revision of FINDINGS.md, which is not touched by this script or its
output. Prompted by an anecdotal observation while building a report
visual (docs/report_visuals/generate_report_visuals.py): two trial
choices (amplitude=0.8 at t_p=0; amplitude=0.2 at t_p=0) showed the
final-timepoint response energy relocating almost entirely onto a single
node OTHER than the stimulated one, rather than spreading broadly across
many -- unlike the trial eventually used for the visual (t_p=2.5,
amplitude=0.2), which does spread broadly.

This is PURE RE-ANALYSIS of results/stage1b2_results.pkl's already-
computed 432 trials (fixed_time_q, the tau=T=2.5 energy distribution
already saved per trial) -- no new simulation, no new trial.

Question: is "relocates onto one other specific node" a real,
characterizable regime tied to amplitude/t_p/sign/node, or were the two
dead-end examples an unrepresentative pair drawn from otherwise-uniform
behavior?

Concentration measures, per trial, on fixed_time_q:
- top1: fraction of energy held by the single largest node
- top2: fraction of energy held by the two largest nodes combined
- effective_n = 1 / sum(q_i^2) (inverse participation ratio, inverted so
  larger = more spread, smaller = more concentrated -- "effective number
  of participating nodes")

Reports group summaries by amplitude, t_p, sign, and node_label, plus
Spearman rank correlations of top1/effective_n against the two ordinal
factors (amplitude, t_p) that a "relocation regime" would most plausibly
track.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kruskal

_THIS_DIR = Path(__file__).resolve().parent
RESULTS_PATH = _THIS_DIR / "results" / "stage1b2_results.pkl"

AMPLITUDE_RANK = {0.025: 1, 0.2: 2, 0.8: 3}
TP_RANK = {0: 1, 0.833: 2, 1.667: 3, 2.5: 4}


def load_table():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    assert len(results) == 432, f"expected 432 trials, got {len(results)}"

    rows = []
    for (t_p, replica, node_label, sign, amplitude), trial in results.items():
        q = np.asarray(trial["fixed_time_q"])
        sorted_q = np.sort(q)[::-1]
        top1 = float(sorted_q[0])
        top2 = float(sorted_q[0] + sorted_q[1])
        ipr = float(np.sum(q ** 2))
        effective_n = 1.0 / ipr
        rows.append({
            "t_p": t_p, "replica": replica, "node_label": node_label, "sign": sign,
            "amplitude": amplitude, "top1": top1, "top2": top2, "effective_n": effective_n,
        })
    return rows


def group_summary(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    print(f"\n-- grouped by {key} --")
    for level in sorted(groups.keys(), key=lambda x: (isinstance(x, str), x)):
        g = groups[level]
        top1_vals = np.array([r["top1"] for r in g])
        eff_vals = np.array([r["effective_n"] for r in g])
        print(f"  {key}={level!r} (n={len(g)}): "
              f"top1 median={np.median(top1_vals):.3f} IQR=[{np.percentile(top1_vals,25):.3f},"
              f"{np.percentile(top1_vals,75):.3f}] | "
              f"effective_n median={np.median(eff_vals):.1f} IQR=[{np.percentile(eff_vals,25):.1f},"
              f"{np.percentile(eff_vals,75):.1f}]")
    return groups


def main():
    rows = load_table()
    print(f"Loaded {len(rows)} trials.")

    all_top1 = np.array([r["top1"] for r in rows])
    all_eff = np.array([r["effective_n"] for r in rows])
    print(f"\nOverall: top1 median={np.median(all_top1):.3f}, range=[{all_top1.min():.3f}, {all_top1.max():.3f}]")
    print(f"Overall: effective_n median={np.median(all_eff):.1f}, range=[{all_eff.min():.1f}, {all_eff.max():.1f}]")

    group_summary(rows, "amplitude")
    group_summary(rows, "t_p")
    group_summary(rows, "sign")
    group_summary(rows, "node_label")

    amp_rank = np.array([AMPLITUDE_RANK[r["amplitude"]] for r in rows])
    tp_rank = np.array([TP_RANK[r["t_p"]] for r in rows])

    print(f"\n-- Spearman rank correlations --")
    for factor_name, factor_vals in [("amplitude", amp_rank), ("t_p", tp_rank)]:
        for measure_name, measure_vals in [("top1", all_top1), ("effective_n", all_eff)]:
            rho, p = spearmanr(factor_vals, measure_vals)
            print(f"  {factor_name} vs {measure_name}: rho={rho:+.3f}, p={p:.5f}")

    print(f"\n-- Kruskal-Wallis across amplitude levels, across t_p levels --")
    for factor_name, key in [("amplitude", "amplitude"), ("t_p", "t_p")]:
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r["top1"])
        stat, p = kruskal(*groups.values())
        print(f"  top1 across {factor_name} levels: H={stat:.3f}, p={p:.5f}")

    # Cross-tab: how many "highly concentrated" trials (top1 > 0.5) fall into each
    # (t_p, amplitude) cell, vs the cell's total count -- directly answers whether
    # the two dead-end examples (t_p=0, amplitude in {0.2, 0.8}) sit in a cell that's
    # disproportionately concentrated, or whether concentration is scattered evenly.
    print(f"\n-- Fraction of trials with top1 > 0.5 (highly concentrated), by (t_p, amplitude) cell --")
    cells = {}
    for r in rows:
        key = (r["t_p"], r["amplitude"])
        cells.setdefault(key, []).append(r["top1"])
    for t_p in sorted(TP_RANK, key=TP_RANK.get):
        line = f"  t_p={t_p}: "
        parts = []
        for amp in sorted(AMPLITUDE_RANK, key=AMPLITUDE_RANK.get):
            vals = np.array(cells.get((t_p, amp), []))
            frac = np.mean(vals > 0.5) if len(vals) else float("nan")
            parts.append(f"amp={amp}: {frac:.2f} ({(vals>0.5).sum()}/{len(vals)})")
        print(line + ", ".join(parts))

    return rows


if __name__ == "__main__":
    main()
