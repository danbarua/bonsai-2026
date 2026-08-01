# Visualisation Design Proposals

Here are concrete visualisation directions, ordered by how directly they illuminate the current frontier (state-dependent linear routing via \(\Phi(T,0)\), concentration regime, and Level-2 structure). Most can be built from already-cached Stage 1B.2 / 1C artefacts plus light re-integration of the tangent system.

---

### 1. Temporal routing trajectories (highest priority)

**What:** For the three illustrative seeds (3000, 3030, 3090), plot node-wise energy \(q_i(\tau)\) over the response window for the high-degree source at \(t_p=0\).

- One panel per seed.
- Highlight the eventual winner, the early leader, and the candidate relay (105).
- Overlay a second thin line for the pure tangent \(q_i^\text{tangent}(\tau)\).

**Why it matters:** This is the visual that makes the mid-course overtaking (seed 3000), monotonic domination (seed 3090), and failed late rise (seed 3030) immediately legible. It is the single most important figure for the \(\Phi(T,0)\) story.

---

### 2. Pathway openness over time

**What:** Along the same three trajectories, plot the instantaneous effective edge strengths

\[
J_{ij}(\tau) = W_{ij}\cos(\theta_j(\tau)-\theta_i(\tau))
\]

for the key edges: \(129\leftrightarrow105\), \(105\leftrightarrow103\), \(129\leftrightarrow152\).

- Absolute value or signed; both are informative.
- Mark the moments when the eventual winner overtakes.

**Why it matters:** Shows the continuous gating that the final propagator integrates. Makes non-commutativity tangible: the same edges open and close in different orders across seeds.

---

### 3. Snapshot Jacobian “switchboard” heatmaps

**What:** Small heatmaps (or chord diagrams restricted to the high-degree node’s neighbourhood) of \(J(\theta)\) at a few key times: \(\tau=0\), the overtaking window, and \(\tau=T\).

- One row of heatmaps per seed.
- Annotate the structurally important edges.

**Why it matters:** Instant visual of which routes are open *right now*. Complements the time-series in (2).

---

### 4. Concentration landscape across the full design

**What:** Heatmap or grouped strip plot of `top1` (or effective participation ratio) with axes:

- rows = stimulated node (low / median / high)
- columns = \(t_p\)
- colour or point position = `top1`
- optional small multiples by sign or amplitude

**Why it matters:** Shows that the concentration regime is a sharp, localised cell (high-degree + \(t_p=0\)), not a smooth gradient. Already partially quantified; a single figure locks it in.

---

### 5. Destination consistency map

**What:** For every (seed, sign, amplitude) in the high-degree / \(t_p=0\) cell, a small glyph or coloured cell showing the argmax destination node.

- Separate panels for tangent-only vs finite response.
- Stage 1C seeds arranged so that 3000 / 3030 / 3090 sit next to each other for comparison.

**Why it matters:** Makes the trajectory-dependence of the *destination* (not just the existence of concentration) visible at a glance.

---

### 6. Early-leader vs final-winner scatter

**What:** For all concentrated trials, scatter:

- x = identity or energy of the leader at an early time (e.g. \(\tau=0.5\) or \(0.95\))
- y = identity or energy of the final argmax
- colour = seed or sign

**Why it matters:** Directly shows how often the early leader is *not* the final winner (the overtaking phenomenon). Quantifies the failure of “\(J(0)\) + early leadership” as a predictor.

---

### 7. Propagator non-commutativity illustration (conceptual + quantitative)

**What:** Two small panels:

- Left: schematic of two different orderings of the same set of open/closed edges.
- Right: actual \(\|q(\tau)-q_\text{tangent}(\tau)\|\) or cosine similarity between finite and a “frozen-\(J(0)\)” predictor across the three seeds.

**Why it matters:** Communicates why matrix time-ordering matters without requiring the reader to accept the abstract \(\mathcal{T}\exp\) formula on faith.

---

### 8. Stage 1C consistency overview (already partly done)

**What:** The existing Stage 1C consistency plot can be extended:

- \(\Delta_\text{map}\) per seed with error bars or replica spread.
- Optional second row: fraction of high-degree / \(t_p=0\) trials that concentrate, per seed.

**Why it matters:** Keeps the main Level-2 claim and the concentration side-result in the same visual frame.

---

### 9. Source-energy redistribution curves

**What:** Mean / median fraction of response energy still on the stimulated node, as a function of \(\tau\), stratified by node degree and \(t_p\).

**Why it matters:** Complements concentration: even when energy does not collapse onto one other node, it still leaves the source. Already measured at three time points; a continuous curve is stronger.

---

### 10. (Later) T vs control gating comparison

Once topology-specificity experiments exist:

- Same temporal \(q_i(\tau)\) and \(J_{ij}(\tau)\) plots for rewired / lattice / random under matched conditions.
- Side-by-side \(\Delta_\text{map}\) and concentration frequency.

**Why it matters:** Directly addresses the next open question.

---

### Practical suggestions

| Priority | Plot | Data needed | New simulation? |
|----------|------|-------------|-----------------|
| 1 | Temporal \(q_i(\tau)\) for 3 seeds | Re-integrate tangent on existing baselines | Light |
| 2 | \(J_{ij}(\tau)\) on key edges | Same | Light |
| 3 | Jacobian snapshot heatmaps | Same | Light |
| 4 | Concentration landscape | Existing `fixed_time_q` | None |
| 5 | Destination map | Existing + Stage 1C | None / light |
| 6 | Early vs final leader | Time-resolved \(q\) | Light |
| 8–9 | Consistency + source decay | Mostly existing | None |

Plots 1–3 form a natural single figure or short figure sequence that carries the entire mechanistic conclusion. Everything else supports or contextualises it.

---

# Implementation Response

**Done.** 8 new plots implementing 8 of `EXTRA_VISUALS_DESIGN.md`'s 10 items, mapped as plot5-plot12 in `generate_report_visuals.py`:

| Design item                       | Plot                                                  |
| --------------------------------- | ----------------------------------------------------- |
| 1. Temporal q_i(τ)                | `05_temporal_routing.png`                             |
| 2. Pathway openness J_ij(τ)       | `06_pathway_openness.png`                             |
| 3. Jacobian snapshot heatmaps     | `07_jacobian_snapshots.png`                           |
| 4. Concentration landscape        | `08_concentration_landscape.png`                      |
| 5. Destination consistency map    | `09_destination_consistency.png`                      |
| 6. Early leader vs final winner   | `10_early_leader_vs_final_winner.png`                 |
| 8. Stage 1C consistency, extended | `11_stage1c_consistency_extended.png`                 |
| 9. Source-energy redistribution   | `12_source_energy_redistribution_scoped.png` (scoped) |

**Skipped, both disclosed in the docstring**: item 7 (non-commutativity illustration — its left panel is a hand-drawn schematic, not a data figure; its right panel needs a "frozen-J(0)" baseline that doesn't exist yet) and item 10 (T vs. graph-control comparison — explicitly gated in the design doc itself on topology-specificity experiments that haven't been run).

**New simulation, disclosed**: plots 5, 6, 7, 10, 12 depend on a new script, `experiments/stage1b2_structured_transformation/generate_frontier_visuals_data.py`, which re-integrates the tangent ODE for all 10 Stage 1C baseline seeds (all 505 nodes, not just 3) plus the nonlinear perturbed system for the 3 illustrative seeds — 19 seconds total, doesn't touch any frozen cache. Plots 8, 9, 11 use only already-cached data.

**Two things I caught and fixed while checking the outputs against existing numbers, rather than just trusting the first render:**
1. Plot 8 initially used median top1, giving 0.67 for the high-degree/t_p=0 cell — didn't match `CONCENTRATION_REGIME_NOTE.md`'s own table (0.622). Verified the note uses the mean, switched to match.
2. Plots 9 and 10 restrict to replica=0 (since that's what the tangent solve used), which makes seeds 3020 and 3080 look like they never concentrate — they actually do (2/36 and 5/36), just not at replica=0. Rather than leave that silently misleading, both plots now carry an asterisk/caption disclosing the restriction and the true counts.

Also fixed `08_concentration_landscape.png`'s colorbar label to match.
