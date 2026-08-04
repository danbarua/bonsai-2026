# Visualisation Design Proposals - ✅ implemented (see below)

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

# Implementation Response ✅ Done

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

---

# Extra-Extra Visuals Design Proposals

---

### 1. Item 10 — still the right visual?

**Original gate:** “Once topology-specificity experiments exist” → same \(q_i(\tau)\) / \(J_{ij}(\tau)\) for rewired/lattice/random, plus side-by-side Δ_map and concentration frequency.

**What Stage 1D actually found:** T ≈ lattice ≈ rewired ≈ hist_random ≈ curr_random on Δ_map (means ~0.327–0.338, spread ~0.012 < T’s own trajectory SD). Null is closed after confirmatory; not “we haven’t looked.”

**Verdict: keep a comparison visual, but change the job.**

| Original intent | Fit after Stage 1D |
|-----------------|-------------------|
| Hunt for T-specific routing signatures in \(q_i\) / \(J\) | **Weak** — invites hunting differences the primary endpoint did not find |
| Show Δ_map / concentration side-by-side | **Strong** — the right visual is **equivalence / tight cluster**, not “T wins” |

**Recommended rewrite of item 10:**

**What:**  
(a) Strip or forest plot of mean Δ_map (± trajectory spread) for all five constructions tested in Stage 1D — same y-scale, no “winner” highlighting.  
(b) Optional small multiples: concentration frequency (high-degree / \(t_p{=}0\)) per construction, same style.  
(c) **Do not** lead with full temporal-routing panels for controls unless a *secondary*, pre-registered routing endpoint is defined; exploratory \(q_i\) overlays risk overclaiming structure Stage 1D did not detect.

**Why:** Communicates the actual result — structured transformation is real and **shared**, learned wiring is not required for Level 2.

**Data needed:** Stage 1D analysis outputs (per-trajectory Δ_map by construction). Prose already lists means; live arrays are gitignored — generation must run from existing analysis pickles locally or recompute from locked design.

**Priority:** High for any Level-2 / “what is established” figure set.

---

### 2. Proposed new items (13+)

### 13. Stage 1D Δ_map equivalence (forest / strip)

**What:** Five constructions on one axis; Δ_map on the other; points = trajectories or mean±spread; optional Holm-null annotation (“no pair separated after correction”).

**Why:** Single figure for “Level 2 is real; T is not special on this measure.”

**Data needed:** Stage 1D per-trajectory Δ_map (from confirmatory analysis artefacts — not in git).

**Priority:** High.

**Overclaim risk:** Low if caption says *internal* mapping strength, not task performance.

---

### 14. Stage 2A primary: evolution vs pre-evolution

**What:** Paired comparison only — `encoded_pre_evolution` vs `evolved_T` (and optionally all four evolved vs shared pre-evolution baseline): test accuracy and/or mean log-loss; bootstrap CI on paired \(d_i\) as a simple interval graphic.

**Why:** This is the locked Level 3 claim. One visual, one contrast.

**Data needed:** Confirmatory per-image losses / metrics (gitignored pickles); numbers also in FINDINGS tables for a static annotated bar chart without re-touching arrays.

**Priority:** Highest among Stage 2A figures.

**Overclaim risk:** **High if merged with ranking.** Caption must not say “learned topology helps classification.” Claim is **graph evolution** helps vs pre-evolution encoding.

---

### 15. Stage 2A graph ranking (four evolved instances)

**What:** Ordered bars or forest plot of the four evolved conditions only (curr_random, rewired, T, lattice) on test log-loss or accuracy; annotate Holm-surviving pairwise separations from the post-hoc family.

**Why:** Separates “dynamics help” from “which wiring ranked where on this task.”

**Data needed:** Same confirmatory outputs; pairwise results in FINDINGS.

**Priority:** High, **as a separate figure from 14.**

**Overclaim risk:** **High** if titled “random beats learned topology” without “these four prespecified instances” and “classification endpoint, not Δ_map.”

---

### 16. Two-endpoint dissociation (1D vs 2A)

**What:** Two-panel or linked graphic: (left) Stage 1D Δ_map cluster — flat; (right) Stage 2A evolved ranking — ordered. Same four construction names where they overlap.

**Why:** Prevents the most likely misread: that Stage 1D and Stage 2A contradict each other. They measure different things.

**Data needed:** Summary statistics already in PROJECT_MEMORY / FINDINGS (can be drawn without pickles if using published means only).

**Priority:** High for any external narrative; medium if only internal lab slides.

**Overclaim risk:** Low if panels are explicitly labeled by endpoint; **high** if a single headline spans both panels.

---

### 17. Stage 2A baselines context (optional, careful)

**What:** Pre-evolution, best evolved, parameter-matched MLP (H=13), competent MLP (H=128) on one accuracy axis.

**Why:** Honest envelope: beats matched MLP, loses to larger MLP.

**Data needed:** FINDINGS baseline table.

**Priority:** Medium — useful, easy to misuse as “oscillators are worse than MLPs” without the matched-parameter frame.

**Overclaim risk:** Medium; keep parameter counts in the legend.

---

### 18. Sync strength vs task gain (exploratory, interpretive)

**What:** Scatter of graph-level sync / Fiedler (or R_post) vs classification improvement — four points only.

**Why:** Touches the multistability-vs-sync hypothesis; **not** a locked causal claim.

**Data needed:** FINDINGS sync diagnostics + confirmatory deltas.

**Priority:** Low until a designed test exists.

**Overclaim risk:** **Very high** with n=4 points — caption must say exploratory / hypothesis-generating.

---

### 3. Overclaim flags (cross-cutting)

| Collapse | Safer split |
|----------|-------------|
| “T doesn’t matter” | Stage 1D: T not special on **Δ_map**. Stage 2A: evolution helps; **among instances**, T not best on **classification**. |
| “Random graphs are better” | Four **prespecified** instances; not a family inference. |
| “Dynamics = learned topology” | Dynamics help even when wiring is lattice/random; learned T is optional for both Level 2 and best Level 3 score here. |
| Item 10-style routing diffs for controls | Only with a secondary endpoint; Stage 1D does not license “T routes differently.” |
| Item 18 sync→accuracy | Hypothesis sketch, not a result figure. |

Existing Stage 2A PNGs in `results/` (phase-state grids, ink correlation, topology structure, cost) are useful mechanism/context visuals; they should not substitute for **14** and **15** as the claim figures.

---

### 4. Suggested priority order for next builds

1. **14** — Level 3 primary (evolution vs pre-evolution)  
2. **15** — ranking, separate slide/figure  
3. **13** / rewritten **10** — Level 2 equivalence  
4. **16** — dissociation panel if communicating both programmes  
5. **17** — baselines context  
6. **18** — only if explicitly marked exploratory  

I did not read any `.pkl`/`.npz`; anywhere a visual needs per-trajectory or per-image arrays, generation has to run where those caches exist locally. Summary means in FINDINGS/PROJECT_MEMORY are enough for static claim figures **13–17** if you annotate from prose.

---

## Claude Expansion Context re: Extra-Extra-Visuals

Context (you won't have this from any prior conversation): Grok has been 
acting as the plain-English visual-design reviewer for this project — it 
drafted the original EXTRA_VISUALS_DESIGN.md (items 1–10), you implemented 
8 of them as plots 5–12 in docs/report_visuals/generate_report_visuals.py. 
That pattern continues here: Grok proposes, you implement.

I've appended Grok's new proposals to EXTRA_VISUALS_DESIGN.md — a rewrite of 
item 10 plus new items 13–18, covering Stage 1D (topology specificity) and 
Stage 2A (dynamics-classification), which had no visual coverage before now. 
Read that file in full before starting; it has priority ordering and 
overclaim-risk notes on every item that matter as much as the "what to build" 
part.

One correction to make on top of what's written there, and this is the part 
most likely to get lost without conversation context: item 15 (the four-graph 
ranking figure) needs to visually distinguish the five decisive pairwise 
separations from the one marginal one. Specifically: the post-hoc Holm-
corrected pairwise comparison among the four evolved graphs (T, lattice, 
rewired, curr_random) originally used a bootstrap that wasn't properly 
null-calibrated; that was caught and fixed with a paired sign-flip permutation 
test (see FINDINGS.md and the commit that replaced it). The corrected result: 
all 6 of 6 pairs still survive Holm at α=0.05, but rewired-vs-curr_random is 
explicitly flagged in FINDINGS.md as genuinely marginal (p≈0.046, stable 
across reruns and permutation counts, but with essentially no margin) — 
unlike the other five, which are at the Monte Carlo floor. If item 15's 
figure shows all six pairs with the same significance annotation, it 
misrepresents how fragile that one gap is. Give it a visibly different 
treatment (e.g. dashed vs. solid bracket, or an explicit footnote naming the 
marginal pair) rather than one uniform "Holm-surviving" style.

Two things to hold as guardrails throughout, not just for item 15:

1. Stage 1D and Stage 2A are different endpoints and must never share a 
   headline. Stage 1D: T is statistically indistinguishable from all four 
   controls on Δ_map (a null result, closed). Stage 2A: graph evolution 
   improves classification vs. pre-evolution encoding — for all four graphs, 
   not just T — and separately, T is not the best-ranked of the four evolved 
   graphs on this task. "T doesn't matter," "random graphs are better," and 
   "learned topology helps classification" are all overclaims of what's 
   actually established — see the "Overclaim flags" table in the design doc 
   for the specific collapses to avoid.

2. Grok explicitly did not read any .pkl/.npz (they're gitignored, never 
   committed — it worked entirely from FINDINGS.md/PROJECT_MEMORY.md prose 
   and approximated some numbers accordingly). You have real filesystem 
   access to those caches — compute exact figures from the actual data 
   rather than trusting Grok's paraphrased numbers as final. If anything you 
   compute meaningfully disagrees with what's written in FINDINGS.md or the 
   design doc, disclose it rather than silently reconciling — same standard 
   as the rest of this project.

Build order per the design doc's own priority list: item 14 (Level 3 primary: 
evolution vs. pre-evolution) first, then 15 (ranking, with the marginal-pair 
fix above), then 13/rewritten-10 (Level 2 equivalence), then 16 (two-endpoint 
dissociation panel), then 17 and 18 only if you have time — 18 is explicitly 
exploratory with n=4 points and must be captioned as hypothesis-generating, 
not a result.

Match whatever file/script convention already fits — docs/report_visuals/ 
for the Level-2 style plots, or alongside the existing Stage 2A visualize_*/
plot_*.py scripts in experiments/stage2a_dynamics_classification/, whichever 
the existing repo structure suggests fits each item best. You have better 
visibility into that than this handoff note does.

---

# Implementation Response (Extra-Extra) ✅ Done

**Done.** 5 of the 6 proposed items (13-17), mapped 1:1 as plot13-plot17
in `generate_report_visuals.py` (unlike plots 5-12, which didn't map
1:1 onto the original design doc's numbering, these do):

| Design item                          | Plot                                  |
| ------------------------------------- | -------------------------------------- |
| 13 / rewritten 10. Δ_map equivalence | `13_stage1d_delta_map_equivalence.png` |
| 14. Evolution vs pre-evolution        | `14_stage2a_evolution_vs_preevolution.png` |
| 15. Four-graph ranking                | `15_stage2a_graph_ranking.png`         |
| 16. Two-endpoint dissociation         | `16_two_endpoint_dissociation.png`     |
| 17. Baselines context                 | `17_stage2a_baselines_context.png`     |

**Skipped, disclosed in the docstring**: item 18 (sync/Fiedler vs.
task-gain scatter) — per this doc's own priority ordering ("only if
explicitly marked exploratory"), and n=4 with no designed test behind
it. Left for a future pass if that test gets designed.

**Data, all real, none paraphrased**: every number came from the
underlying cached analysis pickles (`stage1c_final_analysis.pkl`,
`stage1d_lattice_analysis.pkl`, `stage1d_confirmatory_analysis.pkl`,
`stage4_confirmatory_results.pkl`, `stage4_posthoc_pairwise_results.pkl`),
not from this doc's own prose. Checked against it anyway, per the
"don't trust Grok's paraphrased numbers as final" instruction above:
plot13's five construction means (T=0.3296, lattice=0.3381,
rewired=0.3283, hist_random=0.3288, curr_random=0.3266) matched
PROJECT_MEMORY.md exactly — no discrepancy to disclose.

**The item 15 marginal-pair fix, done as specified**: `raw_p` for all
six pairwise comparisons pulled directly from
`stage4_posthoc_pairwise_results.pkl` — five sit between 5.0e-05 and
0.0075, one (`rewired` vs `curr_random`) sits at 0.046. The figure
draws all six Holm-corrected brackets (stacked by rank-span) but the
marginal pair gets a dashed line, a distinct colour, bold text, and an
explicit "(marginal)" label — the other five are solid black with a
bare p-value. Stage 1D and Stage 2A never share a headline anywhere in
13/15/16 — each figure's title names its own endpoint (Δ_map vs.
classification log-loss) explicitly.

**Two layout bugs caught by actually looking at the rendered PNGs, not
just checking for exceptions**: plot13's per-construction annotations
were first placed just below the x-axis and collided with the tick
labels; plot15's pairwise-bracket spacing was first sized off the
~0.13 inter-bar spread rather than the 0-0.8 axis range, crowding all
three stacked bracket levels into a sliver near the top. Both fixed,
both re-rendered, both re-inspected before being called done.
