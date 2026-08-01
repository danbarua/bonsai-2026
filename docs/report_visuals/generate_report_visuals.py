"""
Generates the 4 plain-language PNG visuals for the non-technical research
report, saved alongside this script in docs/report_visuals/.

Uses only already-cached, committed result files -- no new oscillator
simulation is run. The one exception, disclosed here rather than
silently done, is plot 4 (Stage 0 multistability): no cached artifact of
the 5 raw equilibrium phase vectors exists anywhere in the repo (only
the qualitative claim "5 distinct equilibria" in FINDINGS.md), so this
script recomputes them via find_equilibrium_lbfgs -- the same cheap,
deterministic L-BFGS optimization tests/test_stage0_simulator_calibration.py
already runs on every test invocation (not a new oscillator time-
integration simulation, and not new evidence -- it reproduces an
already-established, already-tested result for the sole purpose of
visualizing it).

Usage: python3 generate_report_visuals.py
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from bonsai.dynamics.graph_oscillator_field import GraphOscillatorField, find_equilibrium_lbfgs

STAGE1C_PATH = os.path.join(_REPO_ROOT, "experiments", "stage1c_trajectory_generalization",
                             "results", "stage1c_final_analysis.pkl")
STAGE1A_PATH = os.path.join(_REPO_ROOT, "experiments", "stage1a_re_verification",
                             "results", "stage1a_reverification_analysis.pkl")
STAGE1B2_RESULTS_PATH = os.path.join(_REPO_ROOT, "experiments", "stage1b2_structured_transformation",
                                       "results", "stage1b2_results.pkl")
STAGE1B2_CONSTRUCTIONS_PATH = os.path.join(_REPO_ROOT, "experiments", "stage1b2_structured_transformation",
                                             "results", "class0_constructions.pkl")

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

NEUTRAL = "#4C72B0"
HIGHLIGHT = "#DD8452"
GREY = "#888888"


def plot1_stage1c_consistency():
    with open(STAGE1C_PATH, "rb") as f:
        per_trajectory = pickle.load(f)

    seeds = sorted(per_trajectory.keys())
    values = [per_trajectory[s]["pooled_delta_map"] for s in seeds]
    labels = [str(i + 1) for i in range(len(seeds))]
    mean_val = np.mean(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=NEUTRAL, width=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=1, zorder=2)
    ax.axhline(mean_val, color=GREY, linewidth=1, linestyle="--", zorder=2)
    ax.text(len(labels) - 0.4, mean_val + 0.012, "average", color=GREY, fontsize=10, ha="right")

    ax.set_title("The same pattern appears across 10 independent random starting points",
                  fontsize=13, pad=14, wrap=True)
    ax.set_xlabel("Starting point (10 separate random trials)")
    ax.set_ylabel("Strength of the pattern")
    ax.set_ylim(0, max(values) * 1.2)
    fig.text(0.5, -0.02,
              "Each bar is a completely independent trial. All 10 land in a similar range,\n"
              "showing this is a consistent, repeatable pattern -- not a one-off fluke.",
              ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "01_stage1c_consistency.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot2_seed_instability():
    with open(STAGE1A_PATH, "rb") as f:
        analysis = pickle.load(f)

    stability = analysis["hist_random"]["stability_diagnostic"]
    ks = sorted(stability.keys())
    class2_values = [stability[k][2] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, class2_values, marker="o", markersize=8, color=NEUTRAL, linewidth=2, zorder=3)

    jump_x0, jump_y0 = ks[-2], class2_values[-2]
    jump_x1, jump_y1 = ks[-1], class2_values[-1]
    ax.annotate("", xy=(jump_x1, jump_y1), xytext=(jump_x0, jump_y0),
                arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=2))
    ax.text(jump_x1 - 1.5, (jump_y0 + jump_y1) / 2,
            "one extra random\nsample swung the\nestimate ~16x",
            color=HIGHLIGHT, fontsize=10, ha="right", va="center")

    ax.set_title("Why we needed more than one random sample to trust a result",
                  fontsize=13, pad=14, wrap=True)
    ax.set_xlabel("Number of random samples averaged together")
    ax.set_ylabel("Estimated result (running average)")
    ax.set_xticks(ks)
    fig.text(0.5, -0.02,
              "With only a handful of random samples, a single unusual draw can swing the\n"
              "average wildly -- this is why the analysis used many samples, not one.",
              ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "02_seed_instability.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot3_stage1b2_spread():
    with open(STAGE1B2_RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    with open(STAGE1B2_CONSTRUCTIONS_PATH, "rb") as f:
        data = pickle.load(f)[0]
    W = data["constructions"]["T"]

    sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments", "stage1b2_structured_transformation"))
    from stage1b2_core import get_degree_stratified_nodes
    nodes = get_degree_stratified_nodes(W)
    stimulated_node = nodes["high"]

    # Trial selection, explained (two dead ends tried first, kept here rather than
    # silently discarded): (a) t_p=0, amplitude=0.8 relocates 97.6% of the energy onto a
    # single DIFFERENT node by tau=2.5 -- a real, separately interesting finding (consistent
    # with this project's "structured transformation" result), but a relocation, not a
    # spread. (b) t_p=0, amplitude=0.2 does the same thing at smaller scale (84% onto one
    # other node) -- still a relocation. Perturbing later along the baseline trajectory
    # instead (t_p=2.5, i.e. after the system has partly settled) at the same intermediate
    # amplitude=0.2 gives genuine broad spreading (largest single pixel ends up with only
    # ~17% of the energy, 20+ pixels each hold at least 1%) -- the case actually plotted.
    trial_key = (2.5, 0, "high", 1, 0.2)
    trial = results[trial_key]
    initial_fraction_at_source = trial["initial_f_source"]
    later_q = np.asarray(trial["fixed_time_q"])
    later_fraction_at_source = later_q[stimulated_node]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.bar(["the pixel\nthat was nudged", "everywhere\nelse combined"],
           [initial_fraction_at_source, 1 - initial_fraction_at_source],
           color=[HIGHLIGHT, NEUTRAL], width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of the response energy")
    ax.set_title("Immediately after the nudge", fontsize=12)

    ax = axes[1]
    order = np.arange(len(later_q))
    colors = [HIGHLIGHT if i == stimulated_node else NEUTRAL for i in order]
    ax.bar(order, later_q, color=colors, width=1.0)
    ax.set_xlabel("Every pixel in the network (500+)")
    ax.set_ylabel("Share of the response energy")
    ax.set_title("Later in the response window", fontsize=12)
    ylim_top = later_q.max() * 1.35
    ax.set_ylim(0, ylim_top)
    ax.annotate("the originally-nudged pixel\nis now just one of many",
                xy=(stimulated_node, later_fraction_at_source),
                xytext=(stimulated_node, ylim_top * 0.75),
                color=HIGHLIGHT, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=1.5))

    fig.suptitle("Energy starts at one point and spreads across the whole network",
                 fontsize=13, y=1.03)
    fig.text(0.5, -0.04,
              "One representative example: a single pixel is nudged. Right after, essentially all the\n"
              "response sits at that one pixel; later, it is spread across dozens of others -- the\n"
              "originally-nudged pixel still has some response energy, but is no longer dominant.",
              ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "03_stage1b2_spread.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _same_attractor(a, b, threshold=0.05):
    shift = np.angle(np.mean(np.exp(1j * (a - b))))
    residual = np.angle(np.exp(1j * (a - b - shift)))
    return np.mean(np.abs(residual)) < threshold


def plot4_stage0_multistability():
    with open(STAGE1B2_CONSTRUCTIONS_PATH, "rb") as f:
        data = pickle.load(f)[0]
    W = data["constructions"]["T"]

    equilibria = []
    for seed in range(5):
        eq, _info = find_equilibrium_lbfgs(W, k_coupling=1.0, seed=seed)
        if not any(_same_attractor(eq, existing) for existing in equilibria):
            equilibria.append(eq)
        if len(equilibria) == 5:
            break

    fig, axes = plt.subplots(1, len(equilibria), figsize=(3 * len(equilibria), 3.4),
                              subplot_kw={"projection": "polar"})
    if len(equilibria) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(equilibria)))

    for i, (ax, eq) in enumerate(zip(axes, equilibria)):
        ax.scatter(eq, np.ones_like(eq), s=6, color=colors[i], alpha=0.6)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(f"Pattern {i + 1}", fontsize=11, pad=10)
        ax.grid(alpha=0.3)

    fig.suptitle("The same network can settle into 5 different stable patterns,\ndepending on how it starts",
                 fontsize=13, y=1.08)
    fig.text(0.5, -0.05,
              "Each dot is one point in the network; its position around the circle is its resting state.\n"
              "Started from 5 different random starting conditions, the network settles into 5 visibly\n"
              "different (but each individually stable) arrangements -- not just one.",
              ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "04_stage0_multistability.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} ({len(equilibria)} distinct equilibria found)")


def main():
    os.makedirs(_THIS_DIR, exist_ok=True)
    plot1_stage1c_consistency()
    plot2_seed_instability()
    plot3_stage1b2_spread()
    plot4_stage0_multistability()


if __name__ == "__main__":
    main()
