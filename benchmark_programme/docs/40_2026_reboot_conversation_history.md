# 40 — 2026 Reboot Conversation History

**Source:** Markdown export of the Claude chat that reopened Bonsai after early 2025 and continued through Stage 2A housekeeping (`AI_ML developments since early 2025.md`, Claudify export, last updated 2026-08-04).

**Purpose:** A single reference note, in the same sequential series as `benchmark_programme/docs/XX_*.md`, that maps that conversation to what actually exists in the repo — without treating the chat as ground truth.

**How this was built:** Chronological index (Phase 1) approved with corrections; detail pass (Phase 2) checked against `benchmark_programme/`, `tarballs/`, `experiments/`, `src/bonsai/`, and `docs/PROJECT_MEMORY.md`. Where chat and disk disagree, the disagreement is stated.

**Naming traps (read first):**

| Label | Programme | Path / notes | Do not confuse with |
|-------|-----------|--------------|---------------------|
| **Stage 0 / 0.5 (MNIST baselines)** | Topology-as-representation / classification | `benchmark_programme/docs/04_mnist_baselines.md`, tarball `mnist_stage0_25_and_writeup.tar.gz` | Dynamics Stage 0 |
| **Stage 0 (simulator calibration)** | Dynamics-as-computation | `experiments/stage0_simulator_calibration/` | MNIST baseline Stage 0 |
| **Stage 1A (infinitesimal response)** | Dynamics | `experiments/stage1a_infinitesimal_response/` then `experiments/stage1a_re_verification/` | Not a `benchmark_programme/` stage number |
| **Stage 1D** | Dynamics | `experiments/stage1d_topology_specificity/` (+ GPU sibling) | Stage 1A controls; Stage 2A ranking |
| **“Rewired”** | Three different experimental objects | See § Naming traps detail below | A single graph family used everywhere |

**Stage 1A revision (explicit):** The original Stage 1A write-up closed as a clean negative (no reliable T-vs-controls difference on the infinitesimal endpoint). The later re-verification **revised** one comparison: after robustness checks and a pre-committed log-scale pass, **T vs degree-preserving rewiring was demoted from clean null to genuinely inconclusive**. Historical/current random were resolved toward trustworthy nulls. Narrate this as a correction of the original finding, not as two independent confirmations of the same verdict.

**“Rewired” disambiguation:**

1. **Stage 1A / re-verification** — degree-preserving rewiring of class topologies; endpoint = infinitesimal / AUC-style response vs T.  
2. **Stage 1D** — rewired construction among lattice / hist_random / curr_random; endpoint = Δ_map.  
3. **Benchmark causal ablation** (`13_causal_ablation_findings.md`) — degree-preserving rewiring; endpoint = classifier accuracy drop.  

Same scrambling *idea*; different graphs, metrics, and decision rules. Verdicts may differ without contradiction.

---

## Table of contents (Phase 1, locked)

| # | Phase | One-line summary |
|---|--------|------------------|
| 0 | Cold start / field survey | AI/ML since 2025 → neuro-inspired / oscillator niche → Sakana CTM |
| 1 | Reopen old playground | Inspect `danbarua/bonsai`; state-of-project; three model families |
| 2 | Diagnose predictive-Hebbian cost | “223× slower” mostly dead diagnostics; non-stationary coherence |
| 3 | First autonomous coding turn | User green-lights patches and re-runs in the sandbox |
| 4 | Housekeeping + branch `2026-07` | Tests, `uv`/`beartype`, C-kernel removal, recovery branch on **old** repo |
| 5 | Bronski / theory vs classification | Align code with Bronski; realize classification was a drift |
| 6 | MNIST baselines & oscillator field | Real MNIST; Stage 0/0.5 **classification** baselines; early field experiments |
| 7 | Topology-as-representation | Graphs from images; capacity, ablation, transfer datasets |
| 8 | Spectral / E–R deep dive | Spectral scores → E and R; long control series; closed |
| 9 | Methodology freeze & split signal | Controls-as-next-experiment; early seeds of programme split |
| 10 | Pivot to dynamics-as-computation | What does the system *do* while it runs? (**Dynamics** Stage 0) |
| 11 | Structured transformation (Δ_map) | Level 2 established; concentration / Jacobian switchboard |
| 12 | Trajectory generalization & topology specificity | 1C + 1D; learned graph not special on Δ_map |
| 13 | Tooling crisis & multi-agent workflow | Sandboxes, tarballs, Claude Code, recovery after reset |
| 14 | GPU path for dynamics | JAX/diffrax; A100; correctness traps |
| 15 | Stage 2A design → confirmatory | Evolution helps classification on held-out test set |
| 16 | Ranking, mechanism, cost, hygiene | Pairwise Holm; cost no-crossover; audits; near merge |

---

## Phase 0 — Cold start / field survey

**What was tried / why / what happened**

You had not touched the research since early 2025. The chat opened as a general “what’s new in AI/ML?” survey, then you steered it to computational-neuroscience-inspired architectures, Kuramoto networks, and phase-field ideas (Nadasdy, cortical stack). Sakana AI’s Continuous Thought Machine was noted as conceptually close (internal “tick,” synchronization matrix as latent) but not a true phase-ODE substrate.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Field survey only | — | No code artefacts yet |
| CTM discussion | External (Sakana) | Conceptual neighbour only |

**Pivot?** No — pure discussion.

---

## Phase 1 — Reopen the old playground

**What was tried / why / what happened**

You pointed Claude at `https://github.com/danbarua/bonsai` and asked for a state-of-project read. Claude reported that the character-processing benchmark still ran, documented three model families (Hebbian Kuramoto, Predictive Hebbian, AKOrN / “deluxe”), and a messy test suite (~141 pass / 28 fail / 5 error; last activity ~April 2025). Focus was still undefined.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Old playground | Historical `danbarua/bonsai` (not this monorepo’s sole history) | `bonsai-2026` later became the clean-room / integration repo |
| State-of-project prose | Echoed later in session notes | Not the same file as `docs/PROJECT_MEMORY.md` (that is dynamics-era) |

**Verify:** The current `bonsai-2026` README describes the *dynamics* programme, not the old Hebbian/Predictive/AKOrN trio. Old model code is not the primary tree here; recovery of early work lives largely in `tarballs/` and `benchmark_programme/` notes.

---

## Phase 2 — Diagnose predictive-Hebbian cost

**What was tried / why / what happened**

The README claim (~3× coherence at ~223× runtime) was profiled. Most of the cost was diagnostic overhead (`eigvals` and related metrics computed every tick and then discarded by a type filter in the metrics collector). Coherence traces wandered rather than settling — so “didn’t converge by t=1000” was real non-stationarity (or a regime without a stable fixed point), not only a tight threshold.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Profiling narrative | Pre-`benchmark_programme` old-repo work | May appear in early tarballs (`bonsai_phase_a*.tar.gz`, session tarballs); not a numbered note solely about this bug |

**Pivot?** Still user-directed analysis; Claude runs code in sandbox with permission.

---

## Phase 3 — First autonomous coding turn

**What was tried / why / what happened**

You explicitly allowed Claude to apply fixes and re-run benchmarks. Interaction shifts from “explain / plan” to “edit, execute, report numbers.”

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Patches / re-runs | Early `tarballs/` (e.g. phase_a, updated_files) | Sandbox outputs packaged for handoff to your machine |

**The pivot moment (flagged):** This is the first clear hand-off of implementation agency to the model. Later multi-agent structure (chat Claude + Claude Code + you in PyCharm) grows from this pattern, not from a single dramatic announcement.

**Narrative beat (optional for blog):** Accidental engineering team — one human, one discussant model, one coding agent, artefacts moved by tarball and git because no shared filesystem was reliable.

---

## Phase 4 — Housekeeping + branch `2026-07`

**What was tried / why / what happened**

Patch application, test triage, environment friction (`uv`, `beartype`, NumPy), removal of the unused C `cognitive-kernel` tree, push of a recovery branch on the **old** `bonsai` remote. Laptop → desktop continuity problems show up.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Branch `2026-07` on old repo | Historical remote | Current `bonsai-2026` is a later clean-room / GitHub integration point |
| Test reorg plans | `benchmark_programme/docs/01_test_suite_reorg_plan.md`, `02_phase_a_changelog.md`, `03_phase_b_changelog.md` | Numbering matches early consolidation |

---

## Phase 5 — Bronski / theory vs classification

**What was tried / why / what happened**

Effort to tie Hebbian operators to Bronski et al. stability criteria; Laplacian bugs; growing honesty that “classification architecture” was not what the pure sync theory specified. Changelog discipline begins (`00_bronski_verification_changelog.md`).

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Bronski verification | `benchmark_programme/docs/00_bronski_verification_changelog.md` | |
| | `tarballs/bonsai_bronski_verification.tar.gz` | |
| Theory vs task tension | Carries into later methodology notes | |

---

## Phase 6 — MNIST baselines & oscillator field

**What was tried / why / what happened**

Real MNIST download and baselines: raw pixels vs phase-style encodings, untrained nearest-centroid vs trained linear readout. This is **classification Stage 0 / 0.5**, not dynamics simulator Stage 0. Early oscillator-field experiments and related checkpoints follow.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| MNIST Stage 0 / 0.25 / 0.5 / 0.75 | `benchmark_programme/docs/04_mnist_baselines.md` | **Not** `experiments/stage0_simulator_calibration/` |
| | `tarballs/mnist_stage0_25_and_writeup.tar.gz`, `mnist_classifier_sweep.tar.gz`, `mnist_few_shot_harness.tar.gz` | |
| Oscillator field checkpoint | `benchmark_programme/docs/05_oscillator_field_checkpoint.md` | |
| | `tarballs/bonsai_oscillator_field_experiments.tar.gz`, `oscillator_field_model.tar.gz` | |
| Complex Hopf / audio side quests | `06`–`09` notes; `bonsai_complex_hopf_field.tar.gz`, `bonsai_audio_fresh_start.tar.gz` | Side path; not the main topology programme |

**Naming:** Always say “MNIST baseline Stage 0” vs “dynamics Stage 0 (simulator calibration).”

---

## Phase 7 — Topology-as-representation programme

**What was tried / why / what happened**

Graphs built from image populations; topology-matching scores as features for a linear (or hybrid) head; capacity experiments; causal ablations (including degree-preserving rewiring as a **classification** control); transfer to Fashion-MNIST, Kuzushiji-MNIST, notMNIST. Methodology hardens: positive results trigger searches for simpler explanations.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Core topology-as-rep | `benchmark_programme/docs/10_topology_as_representation_findings.md` | |
| | `tarballs/bonsai_topology_as_representation.tar.gz` | |
| Capacity | `12`, `18`, `22` notes; capacity tarballs | |
| Causal ablation (rewiring as accuracy control) | `13_causal_ablation_findings.md` | **Rewired object #3** |
| | `tarballs/bonsai_causal_ablation_study.tar.gz` | |
| Fashion / KMNIST / notMNIST | `14`–`18` notes; matching tarballs | |
| Session / state wrap | `11`, `19`–`21` | |

**“Rewired” here:** ablation control for classifier accuracy, not Stage 1A/1D Δ_map objects.

---

## Phase 8 — Spectral / E–R deep dive

**What was tried / why / what happened**

Spectral graph features; decomposition into energy-like (E) and structural (R) pieces; random ensembles; residualization; ink and spatial controls. Investigation closes: auxiliary graph features do not survive as independent classifier gold after proper controls.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Spectral series | `24`–`30` notes | |
| Dependence / residualization | `31`–`33` | |
| Regularization / ink / spatial | `34`–`37` | |
| Methodology synthesis + R closed | `38_methodology_synthesis.md`, `39_R_investigation_closed_findings.md` | |
| Matching tarballs | `bonsai_spectral_*.tar.gz`, `bonsai_ER_*.tar.gz`, `bonsai_R_investigation_closed.tar.gz`, etc. | |

**Programme status:** This is the end of the main **benchmark-feature** arc as an active science programme. `benchmark_programme/` remains the frozen record.

---

## Phase 9 — Methodology freeze & split signal

**What was tried / why / what happened**

Codified habits: every positive result triggers simpler nulls; controls are the next experiment, not decoration. Conversation and artefacts start to separate “historical feature programme” from “what the dynamics do.”

**Origins of `experiments/` (honest answer)**

**Not cleanly visible as a single birth event in the early chat alone.** In the export, explicit `experiments/stage…` paths show up strongly once the dynamics stages are underway (Stage 1A re-verification, 1B.2, 1D, 2A). The directory split is real on disk today:

- `benchmark_programme/` — closed feature/classification programme + numbered notes `00`–`39`
- `experiments/` — dynamics-as-computation stages `stage0` … `stage2a`
- `src/bonsai/` — shared library code (dynamics constructions, stats) consolidated from stage drivers

Whether the first `experiments/` commit happened inside this same chat thread or in a parallel Claude Code / local session is **not fully recoverable from the export alone**. Phase 2 treats “unclear / emerges with dynamics stages” as the accurate statement.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Methodology | `38_methodology_synthesis.md`, `CLAUDE.md`, `docs/PROJECT_MEMORY.md` | Living docs vs frozen notes |
| Split | Directory layout above | |

---

## Phase 10 — Pivot to dynamics-as-computation

**What was tried / why / what happened**

Question changes: not “can frozen features classify?” but “does the running oscillator system organize information?” **Dynamics Stage 0** calibrates the simulator, graph constructions, multistability checks — unrelated to MNIST baseline Stage 0.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Dynamics Stage 0 | `experiments/stage0_simulator_calibration/` | |
| | Related: `tarballs/bonsai_diffusion_stage0_locked.tar.gz` (naming reflects an intermediate framing) | |
| Shared dynamics code | `src/bonsai/dynamics/*` | Consolidates constructions (T, lattice, rewiring, random) |

---

## Phase 11 — Structured transformation (Δ_map)

**What was tried / why / what happened**

Finite-amplitude responses; replica designs; permutation tests; Δ_map as primary structure metric. Level 2 (structured internal transformation) established. Side work on concentration regimes, pathway openness, Jacobian-as-switchboard (state-dependent effective edges).

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Stage 1B pilot | `experiments/stage1b_pilot/`; `tarballs/bonsai_stage1b_pilot.tar.gz` | |
| Stage 1B.2 | `experiments/stage1b2_structured_transformation/`; `tarballs/bonsai_stage1b2_execution_package.tar.gz` | |
| Stats helpers | `src/bonsai/stats/` | |

**Stage 1A in this arc (dynamics only):**

| Pass | Path | Verdict evolution |
|------|------|-------------------|
| Original | `experiments/stage1a_infinitesimal_response/` | Closed as clean negative (no reliable T-vs-controls difference on infinitesimal endpoint) |
| Re-verification | `experiments/stage1a_re_verification/` | **Revision:** rewiring demoted to **genuinely inconclusive** under pre-committed stopping rule; other stochastic controls resolved toward null; effort closed |

Do not write “Stage 1A confirmed twice as clean negative.”

Also: `tarballs/bonsai_diffusion_stage1a.tar.gz` may reflect intermediate naming; prefer `experiments/stage1a_*` paths as canonical.

---

## Phase 12 — Trajectory generalization & topology specificity

**What was tried / why / what happened**

**Stage 1C:** structured mapping generalizes across trajectories.  
**Stage 1D:** does *learned* topology T beat matched controls **on Δ_map**?

Stage 1D is one investigation in two parts (not two Stage 1Ds):

| Part | Comparison | Result |
|------|------------|--------|
| Part 1 | T vs lattice | No detectable difference |
| Part 2 (pilot → locked confirmatory) | T vs rewired, hist_random, curr_random | All Holm-adjusted to p = 1.0 — no detectable difference |

**Final word on the Δ_map endpoint:** T sits in a tight cluster with controls. “No advantage for learned wiring” on this *internal* measure is the closed result after confirmatory, not an early pilot overclaimed as final.

This does **not** close Stage 1A’s inconclusive rewiring comparison (different endpoint). It does **not** predict Stage 2A ranking (different endpoint again).

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Stage 1C | `experiments/stage1c_trajectory_generalization/` | |
| Stage 1D | `experiments/stage1d_topology_specificity/` | |
| Stage 1D GPU | `experiments/stage1d_topology_specificity_gpu/` | Infrastructure sibling |
| Memory summary | `docs/PROJECT_MEMORY.md` § Stage 1D | |

**“Rewired” here:** Stage 1D construction/endpoint (object #2 in the naming trap table).

---

## Phase 13 — Tooling crisis & multi-agent workflow

**What was tried / why / what happened**

Sandbox resets, lost MCP connections, tarball handoffs, Claude Code in PyCharm, Colab/GPU sessions, human as message broker. Post-hoc recovery of artefacts after sandbox wipe is explicit in the chat (tarball manifest searches, reconstruction of stage drivers into `src/bonsai/`).

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Handoff packages | Many `tarballs/*.tar.gz` | Chat often treated these as source of truth after resets |
| Consolidation note | `experiments/stage0_simulator_calibration/NOTE.md` (and similar) | Documents code moved into `src/bonsai/dynamics/` with diff checks for completeness, not always behavioral equivalence |
| Living memory | `docs/PROJECT_MEMORY.md`, `docs/GLOSSARY.md`, `CLAUDE.md` | |

**Verify vs chat:** Where chat claims “byte-exact recovery” or “sandbox still has X,” disk may show only later consolidated modules. Prefer stage `FINDINGS.md` + `NOTE.md` over chat memory for what was actually preserved.

**Narrative beat:** The “accidental engineering team” is structural here — not a metaphor.

---

## Phase 14 — GPU path for dynamics

**What was tried / why / what happened**

JAX/diffrax ports of evolution; float64 requirements; T4 vs A100; Delta_map bugs found in ports; large batch feasibility for 1D confirmatory and later 2A.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| GPU Stage 1D support | `experiments/stage1d_topology_specificity_gpu/` | |
| Stage 2A evolution kernels | `experiments/stage2a_dynamics_classification/evolve_on_graph_jax.py` (and related) | |
| Learnings | Chat + later FINDINGS sections | Some Colab notebooks were ephemeral by convention |

---

## Phase 15 — Stage 2A design → confirmatory

**What was tried / why / what happened**

Level 3 question: does **graph-level evolution** after local encoding improve classification vs encoded-pre-evolution alone? Design locked across review rounds; official KMNIST test set touched once for the confirmatory evaluation.

**Headline (locked):** Primary contrast (evolved-T vs pre-evolution) improves held-out log-loss; 95% CI entirely below zero; McNemar agrees. **Dynamics help.**

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Design / findings | `experiments/stage2a_dynamics_classification/DESIGN.md`, `FINDINGS.md` | |
| Pipeline / classifier | `stage2a_*.py` in that directory | |
| Shared data helpers | `src/bonsai/data/mnist_loader.py` | |

---

## Phase 16 — Ranking, mechanism, cost, hygiene

**What was tried / why / what happened**

Post-hoc Holm-corrected pairwise graph comparisons (all six pairs among T, lattice, rewired, curr_random): ranking holds; curr_random and rewired can beat T on this task. Mechanistic context (Fiedler / multistability) kept interpretive, not causal claim. Compute-cost accounting: no deployment crossover vs competent MLP. Class-0 support audit. cuML corroboration vs JAX classifier port (latter not adopted). Reproducibility passes, unit tests, README, GitHub issues for non-blocking debt. Stage 2A near merge to `main`.

**Repo mapping**

| Conversation | Repo / artefact | Notes |
|--------------|-----------------|-------|
| Pairwise / audits / cost | Sections in `FINDINGS.md`; `COMPUTE_COST_*.md`; `CUML_ACCEL_FINDINGS.md`; `JAX_CLASSIFIER_PORT_FINDINGS.md` | |
| Housekeeping | `README.md` in stage2a dir; tests; issues on GitHub | Branch `stage2a` in chat; check current default branch on pull |

**Important dissociation (for any summary):**

- Stage 1D: T ≈ controls on **Δ_map**  
- Stage 2A: evolution helps on **classification**; **T is not best** among the four graph instances  

Do not collapse these into “T doesn’t matter” or “T wins.”

---

## Cross-cutting: conversation → numbered notes (quick index)

| Notes range | Primary conversation phases | Programme |
|-------------|----------------------------|-----------|
| `00`–`03` | 4–5 | Old-repo recovery / Bronski / test reorg |
| `04`–`09` | 6 (+ Hopf/audio side quests) | MNIST baselines & oscillator field; audio/Hopf detour |
| `10`–`21` | 7 | Topology-as-representation core + transfers + wrap |
| `22`–`23` | 7–8 boundary | Capacity II / smoothness |
| `24`–`39` | 8–9 | Spectral / E–R / methodology close |
| `experiments/stage0`–`stage1d` | 10–12, 14 | Dynamics Levels 1–2 + topology specificity |
| `experiments/stage2a` | 15–16 | Dynamics Level 3 classification |
| `tarballs/*` | 3–14 especially | Handoff / recovery medium after sandbox limits |
| `src/bonsai/` | 10+ consolidation | Library extracted from stage drivers |
| **This file `40_…`** | All | Meta-history of the reboot chat |

---

## What this conversation does *not* settle

- Family-level claims about “all random graphs” or “all rewirings” (Stage 2A is instance-scoped).  
- Controllable metastability / Jacobian switchboard as a *task* result (interpretive thread; not Stage 2A’s locked claim).  
- Exact git birth commit of `experiments/` from chat text alone.  
- Identity of every ephemeral Colab notebook mentioned in chat with a file still in git.

---

## Suggested one-paragraph abstract (optional reuse)

In 2026 a Claude chat reopened a dormant Kuramoto/oscillator playground, first as code archaeology and test repair, then as a disciplined feature-benchmark programme on image-derived graphs, and finally as a dynamics-as-computation programme in which structured internal responses were established, learned topology was found unnecessary for that internal effect, and graph evolution was shown to improve held-out classification — while ranking among specific graph instances favored some non-learned wirings. The workflow itself evolved from rubber-ducking into an accidental multi-agent loop (human, chat model, coding agent) mediated by git and tarballs after repeated sandbox resets; the repo’s split between `benchmark_programme/` and `experiments/` records that scientific turn more cleanly than the chat log alone.

---

*End of note. Update if a later audit finds a chat↔disk mismatch not captured here.*
