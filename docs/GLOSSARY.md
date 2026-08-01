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
  exact definition. First appears in `stage1b_pilot/FINDINGS.md`; also
  referenced in `experiments/stage1d/DESIGN.md` as a planned future stage.

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
  Higher = clearer structured transformation.
