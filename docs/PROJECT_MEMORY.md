# Bonsai Project Memory: State, Methodology, and the Pivot to Dynamics-as-Computation

*A durable reference for this project. Written to be readable cold, by a
future session, agent, or collaborator with no other context. This is a
LIVING document -- update it when findings materially change, not just
when convenient. It lives in `docs/`, separate from
`benchmark_programme/docs/`, which is a fixed, sequentially-numbered
historical record (00-42) of findings documents in the order they were
produced.*

## What Bonsai is

An investigation into whether coupled-oscillator dynamics, evolved over
topologies learned from image populations, produce useful representations
of images -- and, since this document was last substantially rewritten,
whether the dynamics themselves perform structured computation, independent
of anything exported as a static classifier feature.

## Part 1: The benchmark-feature programme (fully closed)

**Topology-as-representation baseline**: pairwise phase-correlation
matching against class-conditioned topologies, scored via a 20-dimensional
hybrid feature, reaches 90.89% on full MNIST test set (95% CI
90.33-91.45%). Several trivial baselines beat this number outright -- the
result's significance rests on mechanism (causal ablation showing real
dynamics matter), not benchmark accuracy.

**Causal ablation** (6 controls): confirmed performance depends on
oscillator evolution, class-conditioned aggregation, AND specific learned
pairwise connectivity, not any single marginal property.

**Cross-dataset transfer**: topology density scales with image
morphology (sparse handwritten digits < printed glyphs < dense filled
objects), not dataset identity.

**Capacity Experiments I & II** (extra pruning threshold; graph
smoothness): clean negative results -- real independent signal, but no
combination benefit survived duplicate/random/shuffled controls.

**Capacity Experiment III (spectral projection) -> the E/R decomposition
-> full closure of both E and R as benchmark features**:

- The spectral score S decomposed as S = E x R (class-conditioned active-
  support energy x normalized low-frequency allocation). Separating E and
  R beat the product.
- **E investigation, closed**: E beat non-spatial ink statistics, but
  matched or lost to genuine spatial statistics (quadrant energy,
  centroid, moments). A trivial, non-oscillator "class-template" support
  (top-N pixels by class-mean intensity) reproduced E's near-total linear
  recoverability from that spatial control (93.6-98.8% block R²), and
  matched E's predictive behavior on every test, including E's last
  nominal residual. **No supported evidence oscillator dynamics
  contribute anything to E's classification value beyond a class-
  conditional mean image.**
- **R investigation, closed**: bandwise ablation showed the original low-
  frequency hypothesis failed. A DCT image-domain control matched or
  exceeded R directly on every dataset. A non-oscillator regular-lattice
  graph control (same spectral construction, zero learned topology)
  produced no significant difference from oscillator-derived R anywhere,
  with positive (nominal) evidence the surviving Kuzushiji-MNIST signal
  is graph-generic, not oscillator-specific.
- **The critical scoping distinction, preserved throughout**: "no
  supported advantage over generic controls" is a claim about *exported
  static features read by a linear classifier on cached digital data*. It
  is not a claim that E or R have no role in any architecture. In a
  biologically constrained, analog, or neuromorphic substrate, the
  relevant comparison may be "this emerges locally from ongoing dynamics"
  versus "the generic alternative requires explicit coordinates, global
  moments, and matrix operations." Both E and R remain open as *intrinsic
  properties of a physical dynamical system* even after being retired as
  *preferred exported classifier features*.

## Part 2: Methodological principles -- moved to CLAUDE.md

The numbered list of methodological principles this project holds
itself to now lives in `CLAUDE.md`'s "Methodological discipline this
project holds itself to" section -- kept there, not here, so a fresh
one-shot agent gets them without needing this document's full findings
history first. The numbering is unchanged by the move (still 1-17);
references elsewhere as "principle N" mean that list.

## Part 3: The dynamics-as-computation programme

### The reframed central question

What computations become native when information is represented in the
collective state of a coupled oscillator network -- not whether a frozen
snapshot of that state can be decoded by a conventional classifier
afterward.

**The three-way distinction, now load-bearing across the whole dynamics
programme**:
1. Nonlinear response (does the system behave nonlinearly at all).
2. Structured internal transformation (does that nonlinearity organize
   information reproducibly, not just sensitively).
3. Useful computation (can that structure be linked to an externally
   defined task).

These cannot be collapsed into each other. A result at level 1 is not
evidence for level 2; a result at level 2 is not evidence for level 3.

### Stage 0 (simulator + calibration, closed)

Built and validated a general weighted-graph oscillator simulator
(`GraphOscillatorField`, distinct from the fixed-local-grid
`LocalOscillatorField` used for the benchmark programme -- the learned
topology T had never previously been used as a coupling structure for
dynamics, only read as a static classifier feature).

**Multistability discovered and correctly diagnosed**: this system has a
rich multistable landscape -- five distinct stable equilibria found from
five initializations of one class's topology, confirmed via Jacobian
analysis (single connected component, one zero mode, all other
eigenvalues positive -- genuine stable equilibria, not artifacts).
Tangent-linear formulation validated against finite differences at small
epsilon, cross-checked against an independent solver family (RK45 vs.
DOP853, agreement to 4 decimal places).

### Stage 1A (infinitesimal response, closed -- clean negative)

Tangent-linear response to a local perturbation, T vs. three matched
controls (degree-preserving rewiring, matched-sparsity random, regular
lattice), across all 10 class topologies. **No statistically supported
evidence that learned topology produces distinguishable finite-time
infinitesimal perturbation dynamics from the matched controls.** Genuine
dynamical diversity was visible in the raw data (T "wins" on some
classes, controls on others) -- the null reflects inconsistency in which
construction leads, not absence of variation. This is a different
substantive hypothesis than E's or R's closures (first-order propagation
dynamics, not spatial organization or graph spectra) -- the commonality
across all three closures is methodological rigor, not a repeated
failure of the same claim.

### Stage 1B (finite-amplitude nonlinear response, closed)

Pilot across amplitude/node/sign grid found nonlinear behavior is real
(genuine directional reversal confirmed via tangent-departure
diagnostics E(t)/C(t)), rejected a state-independent response law, but
did not establish reproducible structure -- a striking pattern in one
initial condition did not replicate in a second, and in one case
partially inverted. **Capability remained unresolved: nonlinear behavior
established (Level 1), structured transformation not yet shown.**

### Stage 1B.2 (structured internal transformation, ESTABLISHED LOCALLY -- the project's current frontier)

The most significant milestone in the dynamics programme to date.
Controlled state-conditioning design: one prespecified baseline
trajectory (KMNIST class 0, T topology, seed=3000), four perturbation
times along it, six fixed nearby-state replicas per time, 3 nodes x 2
signs x 3 amplitudes = 18 inputs = 432 total trials. Primary statistic
Delta_map = B - W (between-input vs. within-input output-space distance,
d_q = sqrt(JSD)), corrected permutation test (independent per-replica
label shuffling -- an earlier "common relabeling" scheme was caught as
degenerate before being used for inference).

**Result: all four response representations (finite, tangent-only,
nonlinear residual, common-support-excluded) show a significant,
floor-level mapping (p_MC ~ 0.0001).** Critically:

- The mapping is not reducible to the directly stimulated coordinate.
  This required THREE rounds of correction to establish properly: (1)
  a coordinate-alignment bug (deleting rather than zeroing the excluded
  coordinate) was caught and fixed; (2) even after that fix, zeroing
  only the *actually* stimulated coordinate leaked node identity through
  the *position* of the zero itself (a deterministic, per-condition
  signature unrelated to genuine propagation); (3) the corrected
  common-support construction zeros ALL candidate source coordinates
  identically in every trial, closing that leak. The result held at
  each stage (Delta_map ~0.34 throughout), but that stability was
  reassurance only after being verified, not before -- an audit
  confirmed the corrected q object differs elementwise from the earlier,
  leakier version in all 432 trials (max diff 1.9e-4), even though the
  rounded omnibus statistic matched to 4 decimals.
- Source energy genuinely redistributes: from ~99.8% at the source node
  immediately after the impulse to a median ~7.5% by the end of the
  response window.
- Both tangent-linear propagation and the specifically nonlinear
  residual (finite minus tangent) separately carry significant,
  input-sensitive spatial organization -- not merely inherited from each
  other, since x_finite = x_tangent + z by construction (mathematically
  related, not independent quantities).
- All three input factors (node, sign, amplitude) are separately
  significant under Holm-corrected, factor-restricted permutation tests
  -- not just node dominance with sign/amplitude riding along.
- The node-specific claim on the common-support representation was
  directly tested (not inferred from the omnibus result):
  Delta_node^(-S) = 0.8074, p ~ 0.0001.

**Capability hierarchy, current status**:
- Level 1 (nonlinear behavior): established.
- Level 2 (structured internal transformation): **established, across
  ten independent baseline trajectories** on one class-specific learned
  topology (Stage 1B.2 established it on one trajectory; Stage 1C, below,
  confirmed it generalizes across nine further independent trajectories
  -- see that subsection for the numbers).
- Level 3 (useful computation): not established.

**Scope, essential** (Stage 1B.2's own scope, as originally run -- the
trajectory-count dimension is superseded by Stage 1C, immediately below;
the rest still holds): one baseline trajectory [**superseded by Stage
1C's ten**], one class, four repeated states along that single trajectory
(not four independent trajectories), T only (no graph controls yet
compared within the Stage 1B.2/1C mapping design specifically -- distinct
from Stage 1A's own T-vs-controls comparison, which used a different
design and is now closed, see Part 4), no external task.

**What remains open, in priority order** (generalization across
independent trajectories -- the original item 1 here -- is now resolved;
see Stage 1C, immediately below):
1. Topology specificity -- does T produce this mapping more strongly,
   efficiently, or differently than the matched controls established
   throughout the E/R/Stage-1A closures? Not yet tested for Stage 1B.2.
2. External usefulness (Level 3) -- can the structured mapping be linked
   to an externally defined task?

Full details: `experiments/stage1b2_structured_transformation/FINDINGS.md`.

A follow-up addendum characterises a deterministic, sign/amplitude-dependent 
routing regime specific to (highest-degree node, t_p=0) — confirmed as 
first-order linear routing (reproduced by q_tangent alone), not nonlinear 
attractor-switching. Does not alter the frozen finding above.
See: `experiments/stage1b2_structured_transformation/CONCENTRATION_REGIME_NOTE.md` 


### Stage 1C (trajectory generalization, CONFIRMED -- resolves Stage 1B.2 open item 1)

Tests whether Stage 1B.2's structured internal transformation is
specific to its one baseline trajectory (seed=3000) or generalizes
across independent trajectories on the same topology. Identical design
to Stage 1B.2 (432 trials: 3 nodes x 2 signs x 3 amplitudes x 4 t_p x 6
replicas, same permutation test), applied to 10 baseline trajectories:
seed=3000 (Stage 1B.2's own frozen reference, read from its already-
committed results, not re-run) plus 9 new, independent seeds
(3010-3090).

**Result: consistent generalization, not partial and not
trajectory-dependent.** All 10 trajectories hit the Monte Carlo
permutation floor (p_MC = 0.00010, the smallest attainable value at
10,000 permutations) -- zero failures. Pooled Delta_map: mean 0.3296,
range 0.2964-0.3505, coefficient of variation ~5.2%. Every one of the 40
individual per-trajectory, per-t_p Delta_map values is positive and
comfortably significant; no t_p in any trajectory weakens to
non-significance or reverses direction.

**Scope, unchanged from Stage 1B.2**: one class (KMNIST class 0), T only
(topology specificity vs. matched controls still untested for this
design), no external task (Level 3 still untested). What Stage 1C adds
is trajectory-generalization evidence for T on this class specifically --
not a broader capability claim.

Full details: `experiments/stage1c_trajectory_generalization/FINDINGS.md`.

### Stage 1D (topology specificity, IN PROGRESS -- resolves Stage 1B.2/1C open item 1)

Tests whether learned topology T shows any advantage over the matched
graph controls (lattice, rewired, historical-random, current-random)
under the Stage 1B.2/1C mapping design specifically -- distinct from
Stage 1A's own T-vs-controls comparison (different design, already
closed, Part 4 below).

**Part 1 (T vs. lattice), complete and confirmatory**: no detectable
difference across T's 10 Stage-1C-matched trajectories (paired t-test
p=0.28, exact sign-flip p=0.29, Wilcoxon p=0.43 -- all agree). Lattice's
point estimate is nominally slightly higher, well within trajectory-to-
trajectory noise. Consistent with Stage 1A's re-verification (no
surviving advantage for historical- or current-random; rewiring
genuinely inconclusive) -- lattice now joins the "no detected advantage"
side using this design instead. Full detail:
`experiments/stage1d_topology_specificity/FINDINGS.md`.

**Part 2 (T vs. the three stochastic controls -- rewired, hist_random,
curr_random), still a sizing pilot, not yet a confirmatory result.**
A 3-realization x 3-trajectory pilot (plus a follow-up refit of
hist_random on 2 more realizations) locked a common design of R=15
realizations x K=3 trajectories (80% power, delta_min=0.05,
alpha=0.0125/comparison) for rewired and curr_random -- but hist_random's
own reliably-refit requirement is (R=25, K=3), larger than the locked
common design. Whether to raise the common design to (25,3) or handle
hist_random separately is an **explicit open decision, not yet made**.
The pilot also surfaced a real, disclosed phenomenon: hist_random can
(rarely, ~20% of draws so far) isolate one of T's own fixed low-degree
intervention nodes, degenerating Delta_map for that realization -- not a
code bug, a genuine construction-density effect. Full detail:
`experiments/stage1d_topology_specificity/PILOT_RESULTS.md`.

**The confirmatory run itself (T vs. all three stochastic controls, at
whatever (R,K) gets locked) has not yet been executed.** A JAX/GPU port
of `run_one_trial` was built to make this computationally feasible at
scale (a locked (25,3) design across 3 controls is 225 trajectories x
432 trials each -- hours on CPU, minutes on GPU). The port itself is
verified correct (field-by-field vs. numpy, 1e-6 to 1e-8 precision) and
delivers a real, confirmed 112x speedup over the measured CPU baseline
(32.67s vs. 3660s for a 37-trajectory pilot workload, A100 vs. this
project's 10-core M1 Max). One nontrivial bug was found and fixed in the
surrounding batch-construction code (not the simulator itself) along the
way -- see CLAUDE.md's principle 16. **This is infrastructure for the still-
open confirmatory run, not a topology-specificity finding in itself** --
no claim about T vs. the stochastic controls should be drawn from the
GPU work to date. Full detail:
`experiments/stage1d_topology_specificity_gpu/FINDINGS.md`.

## Part 4: Infrastructure and execution environment

**This project now runs in three places, and the distinction matters:**

1. **Claude's own sandboxed computer-use environment** (ephemeral,
   resets between sessions, single CPU core). Used for initial
   development, prototyping, and everything through Stage 1B's pilot.
   Findings and code produced there are periodically packaged and handed
   off to the PyCharm project (this repository) for persistence and
   faster execution.
2. **This PyCharm project, on Dan's M1 Max, accessed via PyCharm's MCP
   server** (`execute_run_configuration`, `get_console_output`,
   `execute_terminal_command`, file-editing tools, etc.). Used for
   Stage 1B.2 onward -- genuine multi-core parallelism (10 cores) made
   previously slow analyses (hours single-threaded) complete in minutes.
3. **Ephemeral cloud GPU sessions via `mighty-colab`** (an MCP server
   wrapping a Colab-runtime CLI), introduced for Stage 1D's GPU/JAX port.
   New failure modes relative to the first two environments, learned the
   hard way in the session that fixed Stage 1D's Delta_map bug:
   - **A session (and the agent driving it) can die mid-task, silently,
     with nothing recoverable.** An ephemeral Claude Code instance
     spawned to run the GPU pilot lost its session (and was itself torn
     down) before writing any findings anywhere -- not on the VM, not
     locally, not to git. Evidence trail: the VM's own execution-history
     timestamp showed a script had run; the local `.pyc` import cache
     showed a second, local fallback attempt had *started*; no results
     file existed anywhere. The actual diagnosis had to be re-derived
     from scratch by a second, independent session, not recovered.
   - **`mighty-colab`'s read tools have different scopes -- don't assume
     they all see the same thing.** `sessions`/`status` query the
     backend/API directly (visible across any client on the account).
     `log`/`ls`/`exec` operate through *this local process's own*
     session tracking (`~/.config/colab-cli/sessions.json`) -- a
     session created or driven by a different agent process is invisible
     to these until explicitly `adopt`-ed, even though `status` already
     shows it. A `File or directory not found` from `ls` on an otherwise-
     valid-looking session can mean the local tracking never had it, not
     that the remote path doesn't exist -- confirmed by an `exec` call
     that got a definitive `Session appears to be lost (404/401)` rather
     than a path error.
   - **A local, throwaway "ephemeral instance" is not the same filesystem
     as the Colab VM.** A path like `/tmp/gpu_experiment/` reported by
     `status` as "last execution" is local to whichever machine is
     running the CLI, not the remote VM's `/content/` -- conflating the
     two wastes time searching the wrong side.
   - **Nothing survives on a dead session except what was explicitly
     committed to git before it died.** Treat any ephemeral-GPU-session
     finding as provisional and unsaved until it's actually in a commit
     -- not "will write it up at the end," since there may be no end.

**Practical lessons from operating across both**:
- `execute_terminal_command` with `reuseExistingTerminalWindow=true` can
  silently kill a long-running foreground process in that same window --
  always use a fresh window (`reuseExistingTerminalWindow=false`) for
  any command that isn't meant to interact with an already-running
  process.
- Long-running computation should be launched via
  `execute_run_configuration` (or `start_run_configuration`) with
  `waitForExit=false`, then monitored via `get_console_output` --
  raw terminal `sleep`+`cat` polling has shown occasional unexplained
  timeouts in this environment; the console-output tool is more
  reliable.
- Killing all terminal windows in the IDE can also kill run-configuration
  processes if their output is displayed in a console/terminal-like tool
  window, even though they weren't launched via a raw terminal command.
- Deferred MCP tools (pycharm, and others) must be loaded via
  `tool_search` before use -- they are not available by name until
  searched for at least once per session.
- `pip`/`python` in this project use a `uv`-managed `.venv`
  (`/Users/dan/Code/pycharm/bonsai-2026/.venv/bin/python`) -- bare
  `python3` on the host will not have the project's dependencies
  (scipy, etc.) installed.
- Python 3.14 on this machine's `.venv` occasionally emits a benign
  `RuntimeWarning: invalid value encountered in sqrt` from
  `scipy.spatial.distance.jensenshannon` internals during permutation
  tests -- confirmed via direct NaN-checking that this does not
  propagate into any actual output value; cosmetic only.
- Multiprocessing correctness for permutation tests requires independent
  per-worker random streams (`SeedSequence.spawn`), not a shared seed --
  verified this explicitly after finding a single-threaded version had
  been used by oversight in one script that wasn't caught until a
  parallelization pass.

**Directory structure (fully restructured, this housekeeping pass)**:
- `benchmark_programme/` -- the CLOSED benchmark-feature programme
  (Part 1 above). Deliberately isolated, per-milestone code snapshots
  (`00_...` through `39_...`, folder name matching its findings doc),
  not a shared library -- these results are settled and isolated
  snapshots protect against silent drift from later shared-code changes.
  `benchmark_programme/docs/` holds the sequential findings documents.
- `experiments/` -- the ACTIVE dynamics-as-computation lineage (Part 3
  above): `stage0_simulator_calibration/`, `stage1a_infinitesimal_response/`,
  `stage1b_pilot/`, `stage1b2_structured_transformation/`. Cumulative,
  unlike `benchmark_programme` -- stage-specific scripts import shared,
  evolving code from `src/bonsai` rather than duplicating it.
- `src/bonsai/` -- the shared, actively-developed package: `dynamics/`
  (oscillator simulator, graph-construction/control utilities),
  `data/` (dataset loading), `stats/` (the generic permutation-test
  framework, tangent-departure diagnostics). Editable-installed
  (`uv pip install -e .`) so `from bonsai.x.y import z` resolves
  properly from anywhere -- verified from an unrelated directory, not
  just assumed to work from proximity.
- `docs/` -- living/reference documents, currently just this file.
  Update in place; do not version-number this folder's contents.
- `datasets/` -- MNIST, Fashion-MNIST, KMNIST, notMNIST raw data.
- `tarballs/` -- original packaged deliverables that `benchmark_programme`
  and parts of `experiments` were decanted from. Kept for provenance.
- Fixed seed convention: `SEED = 42` for population-level/classifier
  work; Stage 1B.2 uses distinct, explicitly-chosen seeds per role
  (baseline trajectory seed=3000, replica-direction seed=3001, and
  numbered seeds per permutation test for reproducible, non-overlapping
  streams).

**Construction-pipeline reproducibility, substantially closed for class
0 (stress test largely passed)**: the cached intermediate pickle files
(topology models, etc.) from Claude's own sandbox environment were never
committed to this PyCharm project by design -- the open question was
whether a fresh, context-less agent could reconstruct the pipeline from
raw data using only what's in this repository. For KMNIST class 0, this
is now settled for three of the four matched graph constructions:
`src/bonsai/dynamics/learned_topology_construction.py` (T),
`degree_preserving_rewiring.py` (rewired), and `lattice_construction.py`
(lattice) all reconstruct their respective constructions from
`datasets/kmnist/` byte-exact (to float64 machine epsilon) against the
historical cached artifact, confirmed by
`tests/test_learned_topology_construction.py`,
`tests/test_construction_driver.py`, and
`tests/test_lattice_construction.py`'s Tier-2 tests.
`matched_sparsity_ablation.py` ("current edge-count-matched random") is
the exception: it implements a different, intentional algorithm that
does not match the historical cached 'random' construction. A separate
reconstruction, `historical_matched_sparsity_random.py` ("historical
half-edge random, coupling-budget normalized"), is structurally verified
but not byte-exact -- see the open items below. Report and docstring
usage should always use one of these two explicit labels, not "random"
or "matched-sparsity random" unqualified -- they are different null
models, not two names for the same thing (CLAUDE.md's principle 16-
adjacent naming discipline).
`src/bonsai/dynamics/construction_bundle.py` ties all four together into
one per-class bundle (using the current edge-count-matched random by
default), but has only been built and verified for class 0.

**Construction-recovery effort, open items** (priority order):
1. **Done.** `experiments/stage0_simulator_calibration/build_all_class_topologies.py`
   extended `construction_bundle.py` to all 10 KMNIST classes (seed=1
   for both rewired and random, n_per_class=200 for T, matching the
   historically recovered hyperparameters). Verified: all 10 classes
   present with correct structure, class 0's T/rewired/lattice
   byte-exact against the historical cache, n_active per class matching
   the real historical `stage1a_all_classes.pkl` exactly. Cached to
   `experiments/stage0_simulator_calibration/results/stage1a_all_classes.pkl`
   (gitignored, regenerable in ~15 seconds). Tested in
   `tests/test_build_all_class_topologies.py`.
2. The historical half-edge random's exact edge-count rule and RNG seed
   remain unrecovered. Two known realizations (552 and 545 unique edges,
   out of T's 1051) don't exactly match any deterministic candidate rule
   tested (floor/round/ceil of half T's edge count, a fixed fraction of
   the eligible pool, a count tied to ink-active-node count). Structural
   equivalence IS established (the correct rescaling formula, an
   independently-sampled support, values drawn from T's own weight
   pool) -- byte-exact reproduction is not, despite a 600-way
   seed/call-order sweep against the known raw artifact. Full account in
   `historical_matched_sparsity_random.py`'s docstring.
3. **Done.** The 10-class re-verification ran
   (`experiments/stage1a_re_verification/DESIGN.md`, 770 instances: T +
   lattice deterministic, 3 stochastic controls x 25 seeds x 10 classes).
   Primary mean-aggregated analysis showed 2 of 4 Holm-corrected
   comparisons nominally significant (historical random, rewiring), but
   this was traced to raw-scale AUC's heavy right tail (a single seed
   draw more than doubling a 20-seed-converged class mean in one
   documented case) and did not survive median aggregation or the
   design's own MCSE gate -- correctly, per the pre-registered decision
   rule, the tertiary mixed model was skipped for those.

   A second, separately pre-registered log-scale iteration
   (`DESIGN_v2_log_scale.md`, pure re-analysis of the same 770 values, no
   new simulation) resolved 2 of the 3 stochastic comparisons cleanly: T
   vs. historical random and T vs. current random both show no
   significant difference under log-mean aggregation, confirmed
   consistent across primary, median, sign-flip, and
   (now consistency-gated-in) mixed model, with 95% CIs bracketing 1.0 on
   the multiplicative scale. T vs. rewiring remains genuinely
   inconclusive even under log scale (p=0.037 vs p=0.084, narrowed from
   v1 but not resolved) -- per the design's explicit pre-commitment, no
   third transformation was attempted; this is reported as an honest open
   result, not chased further.

   Overall: none of the four original Stage 1A controls shows a
   Holm-significant difference from T surviving this project's full
   robustness battery. Two of three stochastic comparisons now rest on a
   trustworthy null (not merely an untested one); T-vs-rewiring and
   T-vs-lattice's status is otherwise unchanged from the original Stage
   1A finding. This effort is now closed -- any further pursuit of the
   rewiring comparison specifically (e.g. extending seed count) would be
   a new, separately-justified follow-up, not a continuation of this one.

## How to use this document

Read this first in any future session touching Bonsai, before reading
any individual findings document in `benchmark_programme/docs/` or
`experiments/`. It should answer "what has been
established, what hasn't, why we're not still testing E/R/Stage-1A, and
what the current frontier is" without needing to reconstruct the
reasoning from conversation history. If new findings materially change
any claim in Parts 1 or 3, update this document alongside them -- it is
meant to stay current, not to freeze any single session's state
permanently. New methodological lessons go in `CLAUDE.md`'s principle
list instead (see Part 2 above), not here -- that's what keeps them
reachable by a one-shot agent that never opens this file.
