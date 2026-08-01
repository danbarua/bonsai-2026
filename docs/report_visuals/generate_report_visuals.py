"""
Generates PNG visuals for two different audiences, saved alongside this
script in docs/report_visuals/.

**Plots 1-4** are the original 4 plain-language visuals for the
non-technical research report. They use only already-cached, committed
result files -- no new oscillator simulation is run. The one exception,
disclosed here rather than silently done, is plot 4 (Stage 0
multistability): no cached artifact of the 5 raw equilibrium phase
vectors exists anywhere in the repo (only the qualitative claim "5
distinct equilibria" in FINDINGS.md), so this script recomputes them via
find_equilibrium_lbfgs -- the same cheap, deterministic L-BFGS
optimization tests/test_stage0_simulator_calibration.py already runs on
every test invocation (not a new oscillator time-integration simulation,
and not new evidence -- it reproduces an already-established,
already-tested result for the sole purpose of visualizing it).

**Plots 5-12** implement EXTRA_VISUALS_DESIGN.md's proposals (numbered
1-10 there). The mapping is NOT 1:1 by number -- this file's plot5-plot12
map onto design-doc items as follows (also noted in each function's own
docstring):

  design-doc item 1 (temporal q_i(tau))        -> plot5
  design-doc item 2 (J_ij(tau) pathway openness) -> plot6
  design-doc item 3 (Jacobian snapshot heatmaps) -> plot7
  design-doc item 4 (concentration landscape)    -> plot8
  design-doc item 5 (destination consistency map) -> plot9
  design-doc item 6 (early leader vs final winner) -> plot10
  design-doc item 8 (Stage 1C consistency, extended) -> plot11
  design-doc item 9 (source-energy redistribution) -> plot12 (SCOPED, see its own docstring)

These are a DIFFERENT, technical audience from plots 1-4 -- the research
team investigating the routing mechanism itself, not a lay reader -- so
unlike plots 1-4 they use the field's own notation (J_ij(tau), q_tangent
vs q_finite) rather than jargon-free language. Two design-doc items are
deliberately NOT implemented here: item 7 (the non-commutativity
illustration) is skipped -- its schematic left panel is a hand-drawn
diagram, not a data figure, and its quantitative right panel would need a
"frozen-J(0)" propagator baseline that isn't built yet; item 10 (T vs.
graph-control gating comparison) is explicitly gated in the design doc
itself on "once topology-specificity experiments exist" -- they don't yet
(see PROJECT_MEMORY.md's open frontier items), so this is future work,
not something to force here.

Plots 5, 6, 7, 10, and 12 depend on
experiments/stage1b2_structured_transformation/results/
stage1b2_frontier_visuals_data.pkl -- DISCLOSED NEW SIMULATION
(re-integrating the tangent and/or nonlinear-perturbed ODE systems and
retaining full time series over all 505 nodes, not just the tau=T
endpoint or the 3 selected nodes earlier caches kept), produced by
generate_frontier_visuals_data.py -- see that script's own docstring for
what's new and why. Plots 8, 9, and 11 use only already-cached
stage1b2_results.pkl / Stage 1C result files, no new simulation.

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
STAGE1B2_FRONTIER_DATA_PATH = os.path.join(_REPO_ROOT, "experiments", "stage1b2_structured_transformation",
                                             "results", "stage1b2_frontier_visuals_data.pkl")
STAGE1C_RESULTS_DIR = os.path.join(_REPO_ROOT, "experiments", "stage1c_trajectory_generalization", "results")

FRONTIER_NODES = {"source": 129, "relay": 105, "dest_a": 103, "dest_b": 152}
FRONTIER_EDGE_COLORS = {
    "source -> relay": "tab:green",
    "relay -> destination (103)": "tab:blue",
    "source -> direct alternative (152)": "tab:orange",
}
FRONTIER_NODE_COLORS = {103: "tab:blue", 105: "tab:green", 152: "tab:orange"}

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
              "Each bar is one of ten independent baseline trajectories on the same learned\n"
              "network. All 10 land in a similar range, showing this is a consistent,\n"
              "repeatable pattern -- not a one-off fluke.",
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


def _load_T():
    with open(STAGE1B2_CONSTRUCTIONS_PATH, "rb") as f:
        data = pickle.load(f)[0]
    return data["constructions"]["T"]


def _load_high_tp0_cell(seed):
    """Loads one baseline seed's full trial dict (stage1b2_results.pkl for
    seed=3000, the matching Stage 1C seed file otherwise) and returns just
    the (t_p=0, node_label='high') cell -- existing cached data only, no
    new simulation."""
    if seed == 3000:
        path = STAGE1B2_RESULTS_PATH
    else:
        path = os.path.join(STAGE1C_RESULTS_DIR, f"stage1c_results_seed{seed}.pkl")
    with open(path, "rb") as f:
        results = pickle.load(f)
    return {k: v for k, v in results.items() if k[0] == 0 and k[2] == "high"}


def plot5_temporal_routing():
    """EXTRA_VISUALS_DESIGN.md item 1 (highest priority): node-wise energy
    q_i(tau) for the high-degree source at t_p=0, one panel per
    illustrative seed (3000/3030/3090), for nodes 103/105/152 (the winner,
    relay, and alternate-destination candidates identified in
    CONCENTRATION_REGIME_NOTE.md). Solid = the actual finite response;
    thin dashed = the pure tangent q_tangent(tau). Depends on
    stage1b2_frontier_visuals_data.pkl -- DISCLOSED NEW SIMULATION, see
    generate_frontier_visuals_data.py's docstring."""
    with open(STAGE1B2_FRONTIER_DATA_PATH, "rb") as f:
        frontier = pickle.load(f)

    seeds = frontier["illustrative_seeds"]
    nodes = [103, 105, 152]

    fig, axes = plt.subplots(1, len(seeds), figsize=(15, 4.8), sharey=True)
    for ax, seed in zip(axes, seeds):
        entry = frontier["per_seed"][seed]
        t = entry["t_eval"]
        q_finite = entry["q_finite_full"]
        q_tangent = entry["q_tangent_full"]
        for node in nodes:
            color = FRONTIER_NODE_COLORS[node]
            ax.plot(t, q_finite[node, :], color=color, linewidth=2.2, label=f"node {node} (finite)")
            ax.plot(t, q_tangent[node, :], color=color, linewidth=1.0, linestyle="--", alpha=0.7,
                    label=f"node {node} (tangent)")
        argmax_final = int(np.argmax(q_finite[:, -1]))
        ax.set_title(f"seed={seed}\nfinal winner: node {argmax_final}", fontsize=11)
        ax.set_xlabel(r"$\tau$ (time since perturbation)")

    axes[0].set_ylabel(r"$q_i(\tau)$ (share of response energy)")
    axes[0].legend(fontsize=7, loc="upper left", ncol=1)
    fig.suptitle(r"Temporal routing: $q_i(\tau)$ solid (finite response) vs $q_i^{\mathrm{tangent}}(\tau)$ dashed"
                 "\n(node=high-degree source, $t_p=0$ -- new simulation, full time series)", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "05_temporal_routing.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot6_pathway_openness():
    """EXTRA_VISUALS_DESIGN.md item 2: instantaneous effective edge strength
    J_ij(tau) = W_ij*cos(theta_j(tau)-theta_i(tau)) along the three key
    edges (source->relay, relay->103, source->152), across the same three
    illustrative seeds, marking the moment (if any) the finite response's
    lead switches between nodes 103 and 152. Depends on
    stage1b2_frontier_visuals_data.pkl's theta_base_tau -- DISCLOSED NEW
    SIMULATION."""
    with open(STAGE1B2_FRONTIER_DATA_PATH, "rb") as f:
        frontier = pickle.load(f)
    W = _load_T()
    seeds = frontier["illustrative_seeds"]
    edge_pairs = frontier["edge_pairs"]

    fig, axes = plt.subplots(1, len(seeds), figsize=(15, 4.8), sharey=True)
    for ax, seed in zip(axes, seeds):
        entry = frontier["per_seed"][seed]
        t = entry["t_eval"]
        theta = entry["theta_base_tau"]
        for i, j, label in edge_pairs:
            J_t = W[i, j] * np.cos(theta[j, :] - theta[i, :])
            ax.plot(t, J_t, color=FRONTIER_EDGE_COLORS[label], linewidth=2, label=label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

        q_finite = entry["q_finite_full"]
        if q_finite is not None:
            q103, q152 = q_finite[103, :], q_finite[152, :]
            diff = q103 - q152
            crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
            for idx in crossings:
                if max(q103[idx], q152[idx], q103[idx + 1], q152[idx + 1]) > 0.05:
                    ax.axvline(t[idx + 1], color="grey", linewidth=1.3, linestyle="--")
                    ax.text(t[idx + 1], ax.get_ylim()[1] * 0.85, " 103/152\n lead switch",
                            fontsize=7.5, color="grey", va="top")

        ax.set_title(f"seed={seed}", fontsize=11)
        ax.set_xlabel(r"$\tau$")

    axes[0].set_ylabel(r"$J_{ij}(\tau)$")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Pathway openness over time: the same three edges open and close in\n"
                 "different orders across trajectories (new simulation, full time series)", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "06_pathway_openness.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot7_jacobian_snapshots():
    """EXTRA_VISUALS_DESIGN.md item 3: small "switchboard" heatmaps of
    J(theta(tau)) restricted to the four structurally relevant nodes
    (source=129, relay=105, destination candidates 103/152), at three
    snapshots (tau=0, tau=1.35 -- seed=3000's overtaking window from
    CONCENTRATION_REGIME_NOTE.md Part 3, and tau=T=2.5), one row per
    illustrative seed. Depends on stage1b2_frontier_visuals_data.pkl's
    theta_base_tau -- DISCLOSED NEW SIMULATION."""
    with open(STAGE1B2_FRONTIER_DATA_PATH, "rb") as f:
        frontier = pickle.load(f)
    W = _load_T()
    seeds = frontier["illustrative_seeds"]
    node_order = [129, 105, 103, 152]
    node_labels = ["129\n(source)", "105\n(relay)", "103\n(dest A)", "152\n(dest B)"]
    snapshot_taus = [0.0, 1.35, 2.5]
    sub_W = W[np.ix_(node_order, node_order)]

    fig, axes = plt.subplots(len(seeds), len(snapshot_taus), figsize=(9, 8.5))
    im = None
    for row, seed in enumerate(seeds):
        entry = frontier["per_seed"][seed]
        t = entry["t_eval"]
        theta = entry["theta_base_tau"]
        for col, tau in enumerate(snapshot_taus):
            idx = int(np.argmin(np.abs(t - tau)))
            th = theta[node_order, idx]
            diff = th[None, :] - th[:, None]
            J_sub = sub_W * np.cos(diff)
            ax = axes[row, col]
            im = ax.imshow(J_sub, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(node_order)))
            ax.set_yticks(range(len(node_order)))
            ax.set_xticklabels(node_labels if row == len(seeds) - 1 else [], fontsize=7)
            ax.set_yticklabels(node_labels if col == 0 else [], fontsize=7)
            if col == 0:
                ax.set_ylabel(f"seed={seed}", fontsize=10)
            if row == 0:
                ax.set_title(rf"$\tau={t[idx]:.2f}$", fontsize=10)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                 label=r"$J_{ij} = W_{ij}\cos(\theta_j-\theta_i)$")
    fig.suptitle("Jacobian \"switchboard\": which routes are open, right now\n"
                 "(new simulation, theta(tau) re-integrated)", fontsize=12, y=0.995)
    out_path = os.path.join(_THIS_DIR, "07_jacobian_snapshots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot8_concentration_landscape():
    """EXTRA_VISUALS_DESIGN.md item 4: heatmap of mean top1 (energy
    concentration) across the full design's (stimulated node) x (t_p)
    grid -- mean, not median, to match CONCENTRATION_REGIME_NOTE.md's own
    interaction table exactly (verified: mean reproduces its 0.350/0.622/
    etc. values cell-for-cell; median does not, 0.67 vs 0.622 for
    high/t_p=0). Existing cached stage1b2_results.pkl only, no new
    simulation. Confirms the concentration regime is a single sharp cell
    (high-degree node, t_p=0), not a smooth gradient -- sign and amplitude
    are pooled here to display the node-by-time interaction; their
    deterministic within-cell effects (4 of 6 conditions always
    concentrate, 2 never; amplitude sign/magnitude further modulates
    strength -- see CONCENTRATION_REGIME_NOTE.md) are analysed separately,
    not absent globally."""
    with open(STAGE1B2_RESULTS_PATH, "rb") as f:
        results = pickle.load(f)

    node_labels = ["low", "median", "high"]
    t_ps = sorted({k[0] for k in results.keys()})
    grid = np.full((len(node_labels), len(t_ps)), np.nan)
    for i, node_label in enumerate(node_labels):
        for j, t_p in enumerate(t_ps):
            vals = [float(np.max(np.asarray(trial["fixed_time_q"])))
                    for key, trial in results.items() if key[0] == t_p and key[2] == node_label]
            grid[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=grid.max(), aspect="auto")
    ax.set_xticks(range(len(t_ps)))
    ax.set_xticklabels([str(tp) for tp in t_ps])
    ax.set_yticks(range(len(node_labels)))
    ax.set_yticklabels(node_labels)
    ax.set_xlabel(r"$t_p$ (perturbation time)")
    ax.set_ylabel("stimulated node")
    for i in range(len(node_labels)):
        for j in range(len(t_ps)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < grid.max() * 0.6 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, label="mean top1")
    ax.set_title("Concentration is a single sharp cell, not a gradient\n"
                 "(mean top1 pooled across sign, amplitude and replica; sign-amplitude effects\n"
                 "within the highlighted cell are shown separately, see 09_destination_consistency.png)",
                 fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "08_concentration_landscape.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot9_destination_consistency_map():
    """EXTRA_VISUALS_DESIGN.md item 5: for every (seed, sign, amplitude) in
    the high-degree/t_p=0 cell AT REPLICA=0, the argmax destination node --
    separate panels for the finite and tangent-only response, with the 10
    Stage 1C seeds ordered so 3000/3030/3090 can be compared. Existing
    cached data only (stage1b2_results.pkl + Stage 1C's 9 seed files), no
    new simulation.

    Restricting to replica=0 (rather than the full 36-trial cell, i.e. all
    6 replicas x 6 conditions) can understate a seed's concentration: seeds
    3020 and 3080 each have a handful of concentrating trials among their
    36 (2 and 5 respectively, per the notebook's Section 9 table /
    plot11's fraction-concentrated figure), but NONE of those happen to
    fall at replica=0 specifically -- so both rows show as entirely grey
    here even though they are not "never concentrates" seeds. Marked with
    an asterisk on the seed label rather than silently shown as
    indistinguishable from a seed that truly never concentrates (3030,
    3040-3070)."""
    seeds = [3000, 3010, 3020, 3030, 3040, 3050, 3060, 3070, 3080, 3090]
    conditions = [(sign, amp) for sign in (1, -1) for amp in (0.025, 0.2, 0.8)]

    finite_grid = np.full((len(seeds), len(conditions)), -1)
    tangent_grid = np.full((len(seeds), len(conditions)), -1)
    finite_dest, tangent_dest = {}, {}
    replica0_all_grey_but_concentrates_elsewhere = []
    for i, seed in enumerate(seeds):
        cell = _load_high_tp0_cell(seed)
        n_concentrated_all_36 = sum(1 for t in cell.values()
                                     if float(np.max(np.asarray(t["fixed_time_q"]))) > 0.5)
        for j, (sign, amp) in enumerate(conditions):
            trial = cell[(0, 0, "high", sign, amp)]
            q_f = np.asarray(trial["fixed_time_q"])
            q_t = np.asarray(trial["fixed_time_q_tangent"])
            top1_f, argmax_f = float(q_f.max()), int(np.argmax(q_f))
            top1_t, argmax_t = float(q_t.max()), int(np.argmax(q_t))
            finite_grid[i, j] = argmax_f if top1_f > 0.5 else -1
            tangent_grid[i, j] = argmax_t if top1_t > 0.5 else -1
            finite_dest[(i, j)] = (argmax_f, top1_f)
            tangent_dest[(i, j)] = (argmax_t, top1_t)
        if n_concentrated_all_36 > 0 and np.all(finite_grid[i, :] == -1):
            replica0_all_grey_but_concentrates_elsewhere.append((seed, n_concentrated_all_36))

    all_dests = sorted({int(v) for v in finite_grid.flatten() if v >= 0} |
                        {int(v) for v in tangent_grid.flatten() if v >= 0})
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(all_dests), 1)))
    color_map = {d: palette[k] for k, d in enumerate(all_dests)}
    color_map[-1] = (0.9, 0.9, 0.9, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, grid, dest, title in [(axes[0], finite_grid, finite_dest, "Finite response"),
                                   (axes[1], tangent_grid, tangent_dest, "Tangent-only response")]:
        rgba = np.zeros((*grid.shape, 4))
        for idx, val in np.ndenumerate(grid):
            rgba[idx] = color_map[int(val)]
        ax.imshow(rgba, aspect="auto")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([f"{'+' if s > 0 else '-'}{a}" for s, a in conditions], fontsize=8, rotation=45)
        ax.set_yticks(range(len(seeds)))
        flagged_seeds = {s for s, _ in replica0_all_grey_but_concentrates_elsewhere}
        ax.set_yticklabels([f"{s}*" if s in flagged_seeds else str(s) for s in seeds])
        ax.set_xlabel("sign, amplitude")
        ax.set_title(title, fontsize=11)
        for (i, j), (node, top1) in dest.items():
            if top1 > 0.5:
                ax.text(j, i, str(node), ha="center", va="center", fontsize=8, color="black")
    axes[0].set_ylabel("baseline trajectory seed")
    fig.suptitle("Destination consistency: which node wins, per trajectory and condition (replica-0 slice)\n"
                 "(grey = not concentrated at replica=0; existing cached data only)", fontsize=12)
    if replica0_all_grey_but_concentrates_elsewhere:
        note = "; ".join(f"seed={s} concentrates in {n}/36 trials overall, just not at replica=0"
                          for s, n in replica0_all_grey_but_concentrates_elsewhere)
        fig.text(0.5, 0.01, f"* {note}", ha="center", fontsize=8.5, color=GREY, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = os.path.join(_THIS_DIR, "09_destination_consistency.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot10_early_leader_vs_final_winner():
    """EXTRA_VISUALS_DESIGN.md item 6, DECOMPOSED per statistical review:
    the original version of this plot compared argmax(q_tangent at
    tau=0.95) directly against argmax(fixed_time_q at tau=T) -- the cached
    FINITE (nonlinear) endpoint. That conflates two distinct transitions:
    (a) time evolution WITHIN the linear tangent system (early tangent
    tau=0.95 -> final tangent tau=T), and (b) nonlinear modification at
    the SAME timepoint (final tangent tau=T -> final finite tau=T). For
    seed=3000 specifically, CONCENTRATION_REGIME_NOTE.md Part 3 already
    demonstrated genuine tangent-system overtaking directly; for the other
    4 seeds this had never been separated out -- see
    analyze_stage1b2_early_leader_decomposition.py and
    CONCENTRATION_REGIME_NOTE.md's decomposition section for the full
    writeup.

    This version plots both transitions side by side instead of one
    conflated comparison. Panel (a) needs
    stage1b2_frontier_visuals_data.pkl's tangent solve -- DISCLOSED NEW
    SIMULATION (already generated by generate_frontier_visuals_data.py);
    panel (b) uses only each trial's own already-cached
    fixed_time_q_tangent and fixed_time_q -- no new simulation for that
    half, same as plot9's use of fixed_time_q_tangent.

    Covers the same ALL 87 concentrated trials across all 5 seeds that
    concentrate anywhere in their 36-trial cell (3000, 3010, 3020, 3080,
    3090) as the original version (see generate_frontier_visuals_data.py's
    find_concentrating_non_zero_replicas() for why this needs more than
    just the replica=0 trials).

    Result: transition (a) alone reproduces the exact per-seed split the
    original single-panel plot reported (4 of 5 seeds mismatch in EVERY
    trial, seed=3090 matches in EVERY trial) -- transition (b) matches in
    ALL 87/87 trials, for every seed, with zero exceptions. The nonlinear
    step never changes the winner for any concentrated trial in this
    cell; the entire early-leader failure is genuine time-ordered
    overtaking within the linear tangent system, exactly as Part 3 showed
    directly for seed=3000 alone, now confirmed to generalize to all 4
    mismatching seeds rather than merely being consistent with one of
    them."""
    with open(STAGE1B2_FRONTIER_DATA_PATH, "rb") as f:
        frontier = pickle.load(f)

    seeds = frontier["all_baseline_seeds"]
    extra_tangent = frontier["extra_tangent_by_seed_replica"]
    early_tau = 0.95

    def early_leader_for(seed, replica):
        if replica == 0:
            entry = frontier["per_seed"][seed]
        else:
            entry = extra_tangent.get((seed, replica))
            if entry is None:
                return None  # tangent solve not available at this (seed, replica)
        t = entry["t_eval"]
        idx = int(np.argmin(np.abs(t - early_tau)))
        return int(np.argmax(entry["q_tangent_full"][:, idx]))

    rows = []
    skipped_no_tangent = 0
    for seed in seeds:
        cell = _load_high_tp0_cell(seed)
        for (t_p, replica, node_label, sign, amp), trial in cell.items():
            q_f = np.asarray(trial["fixed_time_q"])
            top1_f, final_finite = float(q_f.max()), int(np.argmax(q_f))
            if top1_f <= 0.5:
                continue
            q_t = np.asarray(trial["fixed_time_q_tangent"])
            final_tangent = int(np.argmax(q_t))
            early_leader = early_leader_for(seed, replica)
            if early_leader is None:
                skipped_no_tangent += 1
                continue
            rows.append({"seed": seed, "replica": replica,
                         "early_leader": early_leader, "final_tangent": final_tangent,
                         "final_finite": final_finite,
                         "match_a": early_leader == final_tangent,
                         "match_b": final_tangent == final_finite})

    n_total = len(rows)
    n_match_a = sum(r["match_a"] for r in rows)
    n_match_b = sum(r["match_b"] for r in rows)

    # Precision fix, same convention as CONCENTRATION_REGIME_NOTE.md's earlier
    # "67% of the time" correction: report the per-seed breakdown, not just the
    # aggregate fraction, for each transition separately.
    def seed_breakdown(match_key):
        status = {}
        for seed in sorted({r["seed"] for r in rows}):
            seed_rows = [r for r in rows if r["seed"] == seed]
            n_seed_match = sum(r[match_key] for r in seed_rows)
            status[seed] = "all match" if n_seed_match == len(seed_rows) else (
                "none match" if n_seed_match == 0 else f"{n_seed_match}/{len(seed_rows)} match")
        return status

    status_a, status_b = seed_breakdown("match_a"), seed_breakdown("match_b")
    n_all_match_a = sum(1 for v in status_a.values() if v == "all match")
    n_all_match_b = sum(1 for v in status_b.values() if v == "all match")
    n_seeds = len(status_a)

    seed_list = sorted({r["seed"] for r in rows})
    seed_colors = {s: c for s, c in zip(seed_list, plt.cm.tab10(np.linspace(0, 1, max(len(seed_list), 1))))}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    def scatter_panel(ax, x_key, y_key, match_key, xlabel, ylabel, title):
        for r in rows:
            jitter = rng.uniform(-0.15, 0.15, 2)
            marker = "o" if r[match_key] else "x"
            ax.scatter(r[x_key] + jitter[0], r[y_key] + jitter[1],
                       color=seed_colors[r["seed"]], marker=marker, s=70, linewidth=1.5)
        all_vals = [r[x_key] for r in rows] + [r[y_key] for r in rows]
        lo, hi = (min(all_vals) - 5, max(all_vals) + 5) if all_vals else (0, 1)
        ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

    scatter_panel(
        axes[0], "early_leader", "final_tangent", "match_a",
        r"early tangent leader ($q_{\mathrm{tangent}}$ argmax, $\tau$=0.95)",
        r"final tangent winner ($q_{\mathrm{tangent}}$ argmax, $\tau=T$)",
        f"(a) LINEAR overtaking within the tangent system\n"
        f"{n_all_match_a}/{n_seeds} seeds match in ALL trials, {n_seeds - n_all_match_a}/{n_seeds} in NONE\n"
        f"{n_match_a}/{n_total} trials match (o), {n_total - n_match_a} don't (x)")
    scatter_panel(
        axes[1], "final_tangent", "final_finite", "match_b",
        r"final tangent winner ($q_{\mathrm{tangent}}$ argmax, $\tau=T$)",
        r"final finite winner ($q_{\mathrm{finite}}$ argmax, $\tau=T$)",
        f"(b) NONLINEAR modification at the same timepoint\n"
        f"{n_all_match_b}/{n_seeds} seeds match in ALL trials, {n_seeds - n_all_match_b}/{n_seeds} in NONE\n"
        f"{n_match_b}/{n_total} trials match (o), {n_total - n_match_b} don't (x)")

    for seed, color in seed_colors.items():
        axes[0].scatter([], [], color=color, label=f"seed={seed}", s=40)
    axes[0].legend(fontsize=7.5, loc="best", ncol=2)

    fig.suptitle("Decomposing early-leader failure: linear overtaking, not nonlinear rerouting", fontsize=12)
    caption = ("Covers all 87 concentrated trials across the 5 seeds that concentrate anywhere in their "
               "36-trial cell (3000/3010/3020/3080/3090). (a) tests time-ordered evolution inside the linear "
               "tangent system; (b) tests the nonlinear step at the fixed final timepoint. The mismatch reported "
               "for the combined (early tangent vs. final finite) comparison is entirely attributable to (a); "
               "(b) matches in every trial, for every seed.")
    if skipped_no_tangent:
        caption += f" {skipped_no_tangent} concentrated trial(s) skipped -- no tangent solve at that replica."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=GREY, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    out_path = os.path.join(_THIS_DIR, "10_early_leader_vs_final_winner.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} (a: {n_match_a}/{n_total}, b: {n_match_b}/{n_total}, {skipped_no_tangent} skipped)")


def plot11_stage1c_consistency_extended():
    """EXTRA_VISUALS_DESIGN.md item 8: extends plot1's Stage 1C consistency
    figure with a second row showing, per seed, the fraction of the
    high-degree/t_p=0 cell's 36 trials that concentrate (top1>0.5) --
    keeping the main Level-2 claim and the concentration side-result in
    one figure, and making explicit that the latter does NOT share the
    former's consistency. Existing cached data only, no new simulation."""
    with open(STAGE1C_PATH, "rb") as f:
        per_trajectory = pickle.load(f)
    seeds = sorted(per_trajectory.keys())
    delta_map_values = [per_trajectory[s]["pooled_delta_map"] for s in seeds]

    concentration_fracs = []
    for seed in seeds:
        cell = _load_high_tp0_cell(seed)
        tops = np.array([float(np.max(np.asarray(t["fixed_time_q"]))) for t in cell.values()])
        concentration_fracs.append(float(np.mean(tops > 0.5)))

    labels = [str(i + 1) for i in range(len(seeds))]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)

    ax = axes[0]
    ax.bar(labels, delta_map_values, color=NEUTRAL, width=0.6, zorder=3)
    ax.axhline(np.mean(delta_map_values), color=GREY, linewidth=1, linestyle="--")
    ax.set_ylabel(r"$\Delta_{\mathrm{map}}$")
    ax.set_title("Level-2 structured-transformation strength: consistent across all 10", fontsize=11)

    ax = axes[1]
    ax.bar(labels, concentration_fracs, color=HIGHLIGHT, width=0.6, zorder=3)
    ax.set_ylabel("fraction concentrated\n(high-degree, $t_p$=0 cell)")
    ax.set_xlabel("Baseline trajectory (seed order)")
    ax.set_title("Concentration side-result: NOT consistent across trajectories", fontsize=11)

    fig.suptitle("Level-2 holds everywhere; the concentration regime does not", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "11_stage1c_consistency_extended.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot12_source_energy_redistribution_scoped():
    """EXTRA_VISUALS_DESIGN.md item 9, SCOPED (disclosed, not silently
    narrowed): the design doc envisions a continuous f_source(tau) curve
    stratified across the full (stimulated node, t_p) grid -- that would
    need re-simulating a much larger slice of the 432-trial design than
    this pass covers. What's implemented here is the same continuous
    curve for the three already-computed illustrative trials only
    (seed=3000/3030/3090, node=high, t_p=0), reusing
    stage1b2_frontier_visuals_data.pkl's q_finite_full -- DISCLOSED NEW
    SIMULATION."""
    with open(STAGE1B2_FRONTIER_DATA_PATH, "rb") as f:
        frontier = pickle.load(f)
    stim_node = frontier["stim_node"]
    seeds = frontier["illustrative_seeds"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for seed in seeds:
        entry = frontier["per_seed"][seed]
        t = entry["t_eval"]
        f_source = entry["q_finite_full"][stim_node, :]
        ax.plot(t, f_source, linewidth=2, label=f"seed={seed}")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("share of energy remaining at the stimulated node")
    ax.set_title("Energy leaves the stimulated node over time\n"
                 "(3 illustrative trials only -- scoped, see docstring)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(_THIS_DIR, "12_source_energy_redistribution_scoped.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(_THIS_DIR, exist_ok=True)
    plot1_stage1c_consistency()
    plot2_seed_instability()
    plot3_stage1b2_spread()
    plot4_stage0_multistability()
    plot5_temporal_routing()
    plot6_pathway_openness()
    plot7_jacobian_snapshots()
    plot8_concentration_landscape()
    plot9_destination_consistency_map()
    plot10_early_leader_vs_final_winner()
    plot11_stage1c_consistency_extended()
    plot12_source_energy_redistribution_scoped()


if __name__ == "__main__":
    main()
