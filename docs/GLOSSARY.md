# Bonsai Domain Glossary

If you have ever had to stop and ask, “What does *replica* mean in this
context?”, you are in the right place. This glossary records the project’s
domain-specific language so that the code, experiments, and findings remain
readable to people joining the project later—including readers encountering
Bonsai without the context in which a term was first introduced.

Some terms have precise technical meanings here that differ from their
everyday meanings. Each entry therefore gives the definition used by the
project, points to its scope where that matters, and—where useful—includes a
plain-English explanation. The glossary is intentionally a living document:
when a term starts doing important work in the code or documentation, define
it here rather than relying on shared background knowledge.

The definitions are grounded in the implementation and experiment documents,
not just in informal usage. If you find a term that is ambiguous, undefined,
or used inconsistently, please open an issue or submit a pull request with the
smallest clarification that makes it precise.

## Glossary

Recurring domain-specific terms, for anyone new to the project or a session
picking this up cold. Not exhaustive -- add to this as new terms accumulate,
rather than letting them go undefined across scripts and findings docs.

Each term is tagged with where it applies, verified against actual imports/
definitions rather than assumed from familiarity:

- **[general]**: a framework-level concept (`src/bonsai/`), usable in any
  stage.
- **[Stage 1B pilot onward]**: introduced in `stage1b_pilot`'s design and
  `src/bonsai/stats/permutation.py`'s statistical framework; reused by
  Stage 1B2 and Stage 1C. Does not apply to Stage 0, Stage 1A, or
  `benchmark_programme/`.
- **[Stage 1B2 onward]**: introduced specifically in
  `stage1b2_core.py`; reused by Stage 1C (which imports Stage 1B2's own
  functions rather than redefining them). Narrower than the above -- also
  does not apply to Stage 1B pilot.
- **[concentration-regime follow-up only]**: narrower still -- specific to
  the side-investigation in `analyze_stage1b2_concentration_regime.py` /
  `CONCENTRATION_REGIME_NOTE.md`, not part of Stage 1B2's original locked
  design or its primary metrics.
- **[Stage 1D]**: introduced in
  `experiments/stage1d_topology_specificity/` for the T-vs-matched-controls
  comparison on the Delta_map endpoint. `lattice` and `curr_random` are
  reused unchanged in Stage 2A (same construction functions, different
  endpoint); `rewired` and `hist_random` as defined here are specific to
  Stage 1D's own comparison -- see the "rewired" disambiguation entry
  below before assuming a `rewired` mention elsewhere means this one.
- **[Stage 2A]**: introduced in
  `experiments/stage2a_dynamics_classification/` for the Level 3
  (external classification task) design.

---

- **trajectory** *[general]*: the full time-evolved solution `theta(t)` of
  the oscillator system over the integration window, starting from one
  initial condition -- i.e. the whole solution curve, not a single point on
  it. "10 independent trajectories" = 10 different runs from 10 different
  starting conditions, each producing its own such curve.
- **baseline seed / baseline trajectory** *[general]* (e.g. "seed=3000"):
  the RNG seed for the initial phase draw (`theta0 =
  rng(seed).uniform(0, 2*pi, n)`) that generates one full, deterministic,
  *unperturbed* trajectory. A name for an initial condition, not a count of
  simulation runs. (The specific seed *values* used, like the 3000s, are a
  Stage 1B2/1C convention, not a general rule.)
- **replica** *[Stage 1B pilot onward]*: one of a small number (6, in
  Stage 1B2/1C) of *fixed, pre-generated* nearby-state perturbation
  directions, used to check that a finding isn't an artifact of one
  specific phase alignment. Each trial's actual starting state is
  `(baseline_state_at_t_p + scale * direction) mod 2*pi` for one of these
  directions. NOT a repeated/duplicate run of the same trial, and not used
  to average out noise -- each replica is a genuinely different (but
  nearby) starting configuration.
- **t_p (perturbation time)** *[Stage 1B pilot onward]*: the point along
  the *baseline* trajectory, in absolute time, at which a perturbation is
  injected.
- **tau** *[general]*: elapsed time *since the perturbation* (tau=0 is the
  moment of perturbation), as opposed to t or t_p, which are absolute
  baseline time.
- **finite response** *[general]* (also "actual"): the real, nonlinear
  difference between a perturbed trajectory and the baseline. Defined in
  `src/bonsai/stats/tangent_departure.py`; used from Stage 1A onward.
- **tangent response** *[general]*: the linearized (first-order,
  Jacobian-based) approximation to that same difference -- cheaper to
  compute, and used to check whether an effect is already present in the
  linear part of the dynamics. Same provenance as finite response.
- **residual** *[general]*: finite response minus tangent response -- the
  purely nonlinear leftover.
- **q_i(tau)** *[Stage 1B2 onward]*: the fraction of total response energy
  sitting at node i, at time tau. Always normalized (sums to 1 across all
  nodes at a given tau). Defined in `stage1b2_core.py`'s
  `normalized_energy` -- Stage 1A's own tangent-departure diagnostic (E/C)
  does not use this q-formalization.
- **top1** *[concentration-regime follow-up only]*: the largest single
  node's share of q at a given time -- "how concentrated is the response
  onto one node." Not one of Stage 1B2's original locked-design metrics
  (those are `Delta_map`, `J_tan`, `f_source`, etc.) -- introduced later,
  specifically to characterize the concentration-regime phenomenon.
- **effective_n** *[concentration-regime follow-up only]*: the inverse
  participation ratio, `1 / sum(q_i^2)` -- informally, "the effective
  number of nodes meaningfully participating," as opposed to `top1`'s "how
  dominant is the single largest." Same scope as `top1`.
- **J_ij(tau)** *[general]*: the phase-dependent Jacobian's (i,j) entry
  (`W_ij * cos(theta_j(tau) - theta_i(tau))`) -- the *effective*,
  time-varying coupling between nodes i and j, as opposed to the static
  learned graph weight `W_ij`, which does not change over time. Defined as
  `force_jacobian` in `src/bonsai/dynamics/graph_oscillator_field.py`;
  used wherever tangent linearization appears (Stage 1A onward).
- **node_label (low / median / high)** *[Stage 1B pilot onward]*: which of
  3 stimulated nodes, chosen by weighted-degree *rank* in the learned
  topology (e.g. 10th percentile, median, 90th percentile), not a literal
  degree count or threshold.
- **Delta_map** *[Stage 1B pilot onward]*: the project's primary Level-2
  (structured-transformation) strength metric, defined in
  `src/bonsai/stats/permutation.py` -- see
  `experiments/stage1b2_structured_transformation/FINDINGS.md` for its
  exact definition. First appears in `experiments/stage1b_pilot/FINDINGS.md`; also the
  endpoint Stage 1D (below) uses to compare T against matched controls --
  T sits in a tight cluster with all four, no detectable advantage for
  learned wiring on this endpoint specifically (a different, narrower
  claim than Stage 2A's classification result, below -- see the "rewired"
  disambiguation entry for why the two must not be collapsed).
- **T (learned topology)** *[general]*: the graph learned from a class's
  image population (`src/bonsai/dynamics/learned_topology_construction.py`),
  the one construction with real data-derived structure among the four
  Stage 1D/2A graph instances -- as opposed to `lattice`, `rewired`,
  `hist_random`, and `curr_random` below, which are all generic,
  structure-matched controls with no dependence on what the images
  actually look like beyond coarse statistics (edge count, degree
  sequence, or ink-active support).
- **lattice** *[Stage 1D]*: a deterministic 4-connectivity pixel-grid
  graph (`src/bonsai/dynamics/lattice_construction.py`) restricted to T's
  active support, edge weight rescaled so total coupling weight matches
  T's -- the "not even random, just adjacent pixels" control. Reused
  unchanged in Stage 2A.
- **rewired** *[ambiguous -- disambiguate before using]*: THREE different
  objects share this name across the project, not one graph family
  reused everywhere -- confirmed and tabulated in
  `benchmark_programme/docs/40_2026_reboot_conversation_history.md`'s
  naming-traps section:
  1. Stage 1A / re-verification's degree-preserving rewiring of class
     topologies, endpoint = infinitesimal/AUC-style response vs T.
  2. Stage 1D's `rewired` construction
     (`src/bonsai/dynamics/degree_preserving_rewiring.py`, double-edge-swap,
     preserves T's unweighted degree sequence, not exactly its weighted
     degree), endpoint = Delta_map. Reused unchanged in Stage 2A, where
     the endpoint is classification.
  3. The closed benchmark-feature programme's causal ablation
     (`benchmark_programme/docs/13_causal_ablation_findings.md`), endpoint
     = classifier accuracy drop.
  Same scrambling *idea* (degree-preserving edge swaps) in all three, but
  different graphs, metrics, and decision rules -- a verdict about one is
  not a verdict about the others.
- **hist_random (historical random)** *[Stage 1D]*: a structurally
  reconstructed match for a historical, pre-this-project cached random
  control (`src/bonsai/dynamics/historical_matched_sparsity_random.py`,
  "half-edge style," coupling-budget normalized). Structural equivalence
  established (correct rescaling formula, independently-sampled support,
  weights drawn from T's own pool); the exact historical edge-count rule
  and RNG seed remain unrecovered despite a 600-way sweep -- not the same
  algorithm as `curr_random`, below, even though both are informally
  "the random control" in loose conversation. Not used in Stage 2A (only
  `lattice`, `rewired`, and `curr_random` are Stage 2A's four instances,
  alongside T itself).
- **curr_random (current random)** *[Stage 1D onward]*: a different,
  intentional edge-count-matched random construction
  (`src/bonsai/dynamics/matched_sparsity_ablation.py`) -- keeps T's exact
  edge count and redistributes weights, rather than `hist_random`'s
  half-edge-style resampling. Reused unchanged in Stage 2A. Always use
  one of "hist_random" or "curr_random" explicitly; "random" or
  "matched-sparsity random" unqualified is ambiguous between the two.
- **evolved_X / encoded_pre_evolution** *[Stage 2A]*: the six named
  feature conditions compared in Stage 2A's confirmatory result --
  `encoded_pre_evolution` is the locally-encoded phase state *before* any
  graph-level evolution runs; `evolved_T` / `evolved_lattice` /
  `evolved_rewired` / `evolved_curr_random` are that same state after
  evolving on each of the four graph instances above. The primary locked
  result is `evolved_T` vs. `encoded_pre_evolution`; the other three
  `evolved_*` feed the secondary comparisons and the post hoc pairwise
  ranking.
- **go/no-go (mechanical checks)** *[Stage 1D onward]*: cheap, purely
  structural validity checks (no NaN/Inf, expected shapes, solver success
  flags, finite loss) run before trusting a batch of results enough to
  feed a locked statistical analysis -- distinct from the actual
  scientific result; a go/no-go pass means "safe to analyze," not
  "significant."
- **C_GRID** *[Stage 2A]*: the locked, prespecified 9-value logistic
  regression regularization grid (`1e-4` through `1e4`, log-spaced) that
  `select_C_via_cv` searches -- fixed before any Stage 2A result existed,
  per `DESIGN.md`.
- **R_post / feat_post** *[Stage 2A]*: `R_post` is the post-evolution
  Kuramoto order parameter (how synchronized the oscillator population is
  after graph evolution); `feat_post` is the post-evolution feature
  vector fed to the classifier. Both computed per (topology, image) pair
  by `analyze_stage3_results.py` / its JAX port.
- **sign-flip p (paired sign-flip test)** *[Stage 2A]*:
  `stage2a_stats.paired_sign_flip_p` -- a permutation test that
  independently flips each image's paired log-loss difference `d_i`
  sign to build a null distribution that genuinely destroys the effect
  under test (`CLAUDE.md` principle 10), used for the post hoc
  graph-to-graph pairwise comparison's Holm-corrected p-values. Exact
  validity requires a symmetry assumption on `d_i` stronger than "no
  systematic difference" -- see the function's own docstring and
  `FINDINGS.md`'s "Exactness caveat" for where that matters (the one
  marginal comparison) and where it doesn't (the five floor-level ones).
- **theta_static** *[Stage 2A, planned -- not yet run]*: a proposed
  control encoding (`theta = pi * x`, direct phase-from-pixel-intensity,
  no local-convergence encoding step at all) that would test whether the
  local encoding step already carries most of Stage 2A's classification
  value, independent of graph evolution. Named in `DESIGN.md`'s scope
  exclusions and `FINDINGS.md`'s "not settled" list; tracked as GitHub
  issue #10. Not yet implemented -- unlike `evolved_X` above, this is a
  planned comparison, not a completed one.

## Glossary (plain English)

Everyday readings of the same terms. Use this when you want the idea
without the formal definition; use the technical glossary when you need
the exact meaning. Scope tags match the technical glossary above --
*[general]* works everywhere, the rest are progressively narrower slices
of the project (see the tag key above for exactly what each one covers).

- **trajectory** *[general]* — One full run of the oscillator network from
  a starting point through time. Like watching a single path unfold, not a
  snapshot.

- **baseline seed / baseline trajectory** *[general]* — The particular
  random starting point that defines one unperturbed run. “Seed 3000” is
  just the name of that starting point.

- **replica** *[Stage 1B pilot onward]* — A slightly nudged version of the
  same moment on the same path. Used to check “does this still happen if
  we start *almost* in the same place?” Not a repeat of the same trial,
  and not noise-averaging.

- **t_p (perturbation time)** *[Stage 1B pilot onward]* — *When* along the
  unperturbed run we give the network a kick.

- **tau** *[general]* — How long has passed *since* that kick. tau = 0 is
  the moment of the kick itself.

- **finite response** *[general]* — What actually happened after the kick
  (the real, nonlinear difference from the unperturbed run).

- **tangent response** *[general]* — The best linear guess at what would
  happen. Cheap to compute; used to see whether the interesting behaviour
  is already present in the linear part.

- **residual** *[general]* — What’s left after you subtract the linear
  guess from the real response. The purely nonlinear piece.

- **q_i(tau)** *[Stage 1B2 onward]* — “What fraction of the disturbance is
  sitting on this node right now?” Always adds up to 100% across all
  nodes.

- **top1** *[concentration-regime follow-up only]* — How much of the
  disturbance is piled on the single busiest node. High top1 = highly
  concentrated.

- **effective_n** *[concentration-regime follow-up only]* — Roughly, “how
  many nodes are meaningfully involved?” Low number = concentrated; high
  number = spread out.

- **J_ij(tau)** *[general]* — How strongly two nodes are *effectively*
  coupled at this moment. The static wiring (W) is fixed; this effective
  strength opens and closes as their phases drift.

- **node_label (low / median / high)** *[Stage 1B pilot onward]* — Which
  node we kicked, chosen by how heavily connected it is in the learned
  graph (lightly, middling, or
  heavily connected).

- **Delta_map** *[Stage 1B pilot onward]* — The main score for “does the
  network turn different kicks into reliably different spatial patterns?”
  Higher = clearer structured transformation. Also the score Stage 1D
  used to ask “does the *learned* graph do this better than a generic
  one?” — answer: no, not detectably, on this specific score. That’s a
  different, narrower question than Stage 2A’s “does it help
  classification?”, below, where the answer is a clear yes.

- **T (learned topology)** *[general]* — The one graph among the four
  compared in Stage 1D/2A that’s actually derived from what the images
  look like. The other three (`lattice`, `rewired`, `hist_random`,
  `curr_random`) are all generic stand-ins that don’t know anything
  about the specific images beyond crude statistics like edge count.

- **lattice** *[Stage 1D]* — The simplest possible control: just connect
  each pixel to its neighbours on the grid, no learning involved at all.

- **rewired** *[ambiguous — disambiguate before using]* — Watch out: this
  word means three genuinely different graphs in three different parts
  of the project (an early infinitesimal-response check, Stage 1D’s
  Delta_map comparison, and the old benchmark programme’s classifier
  ablation). Same basic idea (scramble the wiring but keep the same rough
  connectivity) each time, but different graphs and different questions
  being asked of them — see the technical entry above for exactly which
  is which.

- **hist_random (historical random)** *[Stage 1D]* — A best-effort
  rebuild of an older “random control” graph from before this project’s
  current tooling. It behaves the same way structurally, but the exact
  recipe (how many edges, which random seed) was never fully pinned
  down, so it’s not a byte-for-byte match to the original.

- **curr_random (current random)** *[Stage 1D onward]* — This project’s
  own, cleanly defined “random control”: same edge count as the learned
  graph, weights shuffled around. A different recipe from `hist_random`
  — don’t assume “random” alone means one or the other.

- **evolved_X / encoded_pre_evolution** *[Stage 2A]* — `encoded_pre_evolution`
  is “what the network looks like right after encoding an image, before
  letting the graph dynamics run.” `evolved_T` etc. is “what it looks
  like after letting it run on graph X.” Stage 2A’s headline result
  compares these two for the learned graph and finds evolving genuinely
  helps classification.

- **go/no-go (mechanical checks)** *[Stage 1D onward]* — Quick sanity
  checks (nothing broke, no missing values, solvers actually finished)
  run before trusting a batch of results enough to do real statistics on
  it. Passing go/no-go doesn’t mean the result is interesting — just that
  it’s safe to look at.

- **C_GRID** *[Stage 2A]* — The fixed list of regularization strengths
  the classifier’s cross-validation searches over, decided in advance so
  nobody can quietly pick a value after seeing what looks good.

- **R_post / feat_post** *[Stage 2A]* — “How synchronized is the network
  after it’s done evolving” (`R_post`) and “what feature vector do we
  actually hand to the classifier afterward” (`feat_post`).

- **sign-flip p (paired sign-flip test)** *[Stage 2A]* — A way of asking
  “if there really were no difference between two graphs, how often
  would random chance alone produce a gap this big?” by literally
  flipping the sign of each image’s result at random and seeing how often
  that matches what was actually observed. Trustworthy for the clearly
  one-sided comparisons; a bit more of an approximation for the one
  genuinely close call (`rewired` vs. `curr_random`).

- **theta_static** *[Stage 2A, planned — not yet run]* — A proposed
  “no dynamics at all” control: turn each pixel straight into a phase
  value and stop there. Would tell us how much of Stage 2A’s
  classification value comes from the encoding step alone, independent
  of any graph evolution. Not built yet.
