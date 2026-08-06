# Bonsai: Oscillator-Network Research Project

**First thing to do in any session: read `docs/PROJECT_MEMORY.md` in
full.** It is the living, current-state record of this project --
what's established, what's open, and what the active frontier is. This
file is a map to the repository, plus the methodological principles
this project holds itself to (kept here rather than in
`PROJECT_MEMORY.md` so a one-shot agent gets working discipline without
reading the full findings history first). This file does not restate
`PROJECT_MEMORY.md`'s findings or current status -- read that document
for those.

## What this project is

An investigation into coupled-oscillator dynamics evolved over
topologies learned from image populations (MNIST and variants), asking
first whether the resulting representations are useful for
classification (closed, largely negative -- see below), and now whether
the dynamics themselves perform structured computation independent of
anything exported as a static feature (open, active, currently at its
strongest positive result to date).

## Directory structure

Two philosophies, deliberately split, because the two halves of this
project are different kinds of thing: one is finished science that just
needs to stay reproducible, the other is active, cumulative development
where a shared, hardened library actually earns its cost.

- **`docs/PROJECT_MEMORY.md`** -- the living project memory. Read this
  first, always. Update it when findings materially change.
- **`benchmark_programme/`** -- the CLOSED benchmark-feature programme
  (does oscillator dynamics produce useful exported classifier
  features? -- answered, largely negatively, see PROJECT_MEMORY.md Part
  1). Deliberately isolated, per-milestone snapshots, not a shared
  library: each numbered subfolder (`00_...` through `39_...`, name
  matching its findings doc) is a frozen decant of exactly the code that
  produced that milestone's result, with no cross-dependencies on other
  subfolders or on `src/bonsai`. This is intentional -- these results
  are settled, and isolated snapshots protect against exactly the
  failure mode this project has hit before: a later "improvement" to
  shared code silently changing what a historical experiment would now
  produce if rerun. Some of this code depends on the OLD `bonsai` repo's
  supporting modules (`dynamics.oscillators`, `maths.*`) which are not
  part of this project -- these are historical record, not standalone-
  runnable code. `benchmark_programme/docs/` holds the sequential
  findings documents themselves (00-39, read in order for full history;
  numbering is a best-effort chronological reconstruction from session
  history, not verified against exact per-file timestamps). Folders with
  no dedicated code tarball (pure planning/narrative docs) contain just
  a `NOTE.md` explaining that and pointing to where the real work landed.
- **`experiments/`** -- the ACTIVE dynamics-as-computation lineage
  (Stage 0 through Stage 1B.2 and beyond -- see PROJECT_MEMORY.md Part
  3). Unlike `benchmark_programme`, this is cumulative: each stage
  builds on the last, and stage-specific driver/analysis scripts here
  import shared, evolving code from `src/bonsai` rather than duplicating
  it. `stage0_simulator_calibration/` and `stage1a_infinitesimal_response/`
  contain only `FINDINGS.md` and a `NOTE.md` -- their original code was
  consolidated into `src/bonsai` after confirming (via diff, not
  assumption) that the shared version was a strict superset.
  `stage1b_pilot/` and `stage1b2_structured_transformation/` contain the
  stage-specific scripts (taxonomy, trial grid, analysis) that are
  reused nowhere else, plus each stage's own `FINDINGS.md` and a
  `results/` subfolder for cached `.pkl` artifacts (gitignored,
  regenerable).
- **`src/bonsai/`** -- the shared, actively-developed package
  underlying the `experiments/` lineage: `dynamics/` (the oscillator
  simulator and graph-construction/control utilities),
  `data/` (`mnist_loader.py`), `stats/` (the generic permutation-test
  framework and tangent-departure diagnostics). Editable-installed via
  `uv pip install -e .` so `from bonsai.dynamics.graph_oscillator_field
  import ...` resolves properly.
- **`datasets/`** -- MNIST, Fashion-MNIST, KMNIST, notMNIST, all in a
  consistent IDX format (dash-separated filenames, `.gz` where the
  original distribution was compressed, uncompressed where it wasn't --
  `src/bonsai/data/mnist_loader.py` handles both transparently). Load
  with `load_mnist(data_dir, gz=True/False)`. **Note: this PyCharm
  project does not have the cached intermediate pickle files
  (topology models, etc.) that exist in Claude's own sandbox
  environment** -- regenerating those from the datasets here, via the
  code in `experiments/` and `benchmark_programme/`, is an open,
  useful stress test of whether this repository is genuinely
  self-sufficient for a fresh agent.
- **`tarballs/`** -- the original packaged deliverables (one per major
  milestone across the whole project) that `benchmark_programme/` and
  parts of `experiments/` were decanted from. Kept for provenance; not
  needed for day-to-day work once decanted. A newer sub-pattern,
  `tarballs/*_handoff/` (e.g. `lattice_construction_handoff/`,
  `random_control_handoff/`), holds small recovery/verification
  packages -- a design-doc excerpt plus historical cached data, used to
  reconstruct or verify already-committed `src/bonsai/` code against a
  known result -- distinct from the frozen milestone deliverables above.
  The large binary data in these (`.pkl`/`.npz`) is gitignored and kept
  local only; only the small provenance document gets committed. This
  data is for verification of existing, committed code, not a shortcut
  for generating new, independently-unreproducible claims -- see
  principle 17 below.
- **`tests/`** -- pytest verification of specific, already-documented
  quantitative claims (a `FINDINGS.md` claim, a construction's byte-exact
  match against a historical cached artifact), not a general test suite
  for `src/bonsai/` in the ordinary sense. Two-tier convention,
  established starting with `test_stage0_simulator_calibration.py`: Tier
  1 is self-contained structural tests on small synthetic data, always
  run; Tier 2 is historical-artifact verification, using
  `pytest.mark.skipif` to skip cleanly when the needed local-only data
  (gitignored `.pkl`/`.npz` files, or `datasets/*/` itself) isn't
  present, rather than failing or fabricating a result. Slow, full-grid
  reproduction checks (minutes, not seconds) are tagged
  `@pytest.mark.slow` and excluded by default
  (`pytest -m "not slow"`); omit that flag for the full suite.

## Running things

- Python environment: `uv`-managed `.venv` at the project root, with
  `src/bonsai` editable-installed (`uv pip install -e .`) so
  `from bonsai.dynamics... import ...` resolves. Use `.venv/bin/python`,
  not bare `python3` -- the latter won't have scipy/numpy/tqdm/the
  bonsai package installed.
- Existing named run configurations may reference stale paths from
  before this project's restructuring -- prefer creating a fresh one
  from `filePath`+`line` against the actual script you want to run
  (e.g. `experiments/stage1b2_structured_transformation/analyze_stage1b2.py`)
  rather than trusting an old configuration name.
- Long-running computation should be launched with `waitForExit=false`
  and monitored via `get_console_output` (or, for a terminal-launched
  process, `execute_terminal_command` in a **fresh, non-reused** terminal
  window -- reusing a window that has a foreground process running in
  it will kill that process).
- This machine has 10 CPU cores. Scripts in `experiments/` already use
  `multiprocessing.Pool` with `max(1, cpu_count()-1)` workers and
  `SeedSequence.spawn` for independent per-worker random streams where
  Monte Carlo correctness depends on it -- follow this pattern for any
  new parallelized analysis rather than writing a single-threaded loop
  and parallelizing later as an afterthought (this has happened twice
  in this project already). The generic version of this machinery lives
  in `src/bonsai/stats/permutation.py` -- import and reuse it rather
  than writing new permutation-test code from scratch.

## Documentation style

Reader-facing docs (READMEs, findings docs) state current facts and
point to the authoritative source for detail -- they don't narrate the
editing process that produced them. Two concrete failures this
convention exists to prevent, both caught in the same session
(2026-08-04): a README that led with raw commands and only revealed a
newly-added Makefile 70 lines later, reading as an after-action report
rather than instructions for a reader arriving cold; and an AI-voice
meta-comment ("don't duplicate this here, this is exactly the X
convention applied to Y") that had leaked into README prose, narrating
an editing decision instead of stating a fact. "Amended by external
review" / "fixed in commit X" framing belongs in commit messages and
`git log`, not in the document a reader consults to use the code
today -- if a paragraph reads as explaining why *this* edit was made
rather than what's true right now, cut it.

## Where the project actually is right now

One closed programme (benchmark-feature, Part 1), one open one
(dynamics-as-computation, Part 3, currently at Stage 2A -- Level 3
established under a bounded classification design; topology-family
generality and other follow-ons remain open, tracked as GitHub issues).
See
`docs/PROJECT_MEMORY.md` Parts 1 and 3 for current status, what's
closed vs. open, and the priority-ordered open questions -- that status
changes as findings land and is kept there, not duplicated here.

## Methodological discipline this project holds itself to

Read these before designing any new experiment. Kept in this file
(reusable beyond Bonsai, and stable -- these don't change as findings
land) rather than in `PROJECT_MEMORY.md`, so a one-shot agent gets them
without reading that document's full findings history first.

1. **Hygiene audits are distinct from null models.** A null model asks
   "could something simpler explain this?" A hygiene audit asks "could my
   analysis pipeline itself explain this?" A result can survive every
   null model and still be a hygiene artifact.
2. **Non-significance is not equivalence.** Absence of evidence is not
   evidence of absence -- stated and re-stated across nearly every review
   round in this project.
3. **Multiplicity must be corrected within well-defined, prespecified
   families**, and exploratory additions to an existing family should be
   flagged as nominal, not folded into an already-corrected result.
   Holm-Bonferroni (step-down) is generally preferable to plain
   Bonferroni when multiple tests in a family are compared -- more
   powerful while still controlling family-wise error rate.
4. **Choose one primary control in advance; don't search multiple
   candidates for the strongest story.**
5. **Coordinate-wise statistics are not automatically block-level
   claims**, and comparing a statistic computed in one normalized space
   against the same-named statistic in a *different* normalized space
   (e.g. Delta_map computed on the finite response vs. on the tangent-
   only response vs. on the residual) does not tell you which space's
   underlying quantity is "stronger" -- each is normalized separately
   against its own replica dispersion.
6. **Report exact p-values in scientific notation; never "p=0.00000."**
   For permutation tests, report as a Monte Carlo floor
   (p = (1+exceedances)/(N+1)) when zero permutations exceed the
   observed statistic, not as if that were the exact tail probability.
7. **Provenance metadata prevents catastrophic normalization-state
   bugs.** Multiple incidents in this project traced to files with no
   explicit record of raw-vs-normalized state.
8. **A fixed random seed across all stochastic operations** both
   verifies genuine reproducibility and surfaces findings a single
   arbitrary seed had been obscuring. For parallelized Monte Carlo work,
   use `numpy.random.SeedSequence(seed).spawn(n_workers)` to give each
   worker a genuinely independent stream -- a shared or repeated seed
   across workers produces correlated or duplicate draws and silently
   invalidates the test.
9. **The evidence hierarchy is a table of specific, falsifiable claims**,
   each with its own status, not a single overall confidence level.
10. **A permutation scheme must actually destroy the effect it's testing
    for under the null.** A "common relabeling" that preserves which
    outputs are grouped together (even if applied consistently) can leave
    the test statistic literally unchanged under every permutation --
    caught once in this project before being reported as a result. Always
    unit-test a new permutation scheme on synthetic data first: identical
    input maps should give the test statistic ~0; maximally separated,
    reproducible maps should give a clearly positive, significant result.
11. **A deterministic exclusion mask must be identical across every
    condition being compared, not merely internally consistent per
    condition.** Zeroing "the stimulated coordinate" (correct per-trial,
    coordinate-aligned) still leaks input identity if *which* coordinate
    is zeroed differs by condition -- the position of the mask becomes a
    trivial, deterministic signature of the input, independent of any
    genuine propagated response. The fix: a common mask (e.g. all
    candidate source coordinates removed identically from every trial,
    not just the one actually used), so no exclusion pattern alone can
    reveal which condition produced which output.
12. **When re-running an analysis after fixing a subtle bug, verify the
    corrected object is genuinely different, not just assume the fix
    worked because the headline statistic matched.** A rounded statistic
    matching to several decimal places across a buggy and a corrected
    version is not evidence the bug didn't matter -- it can simply mean
    the specific dataset's structure happened not to expose it. Check
    elementwise differences directly.
13. **An omnibus test across combined factors does not establish a
    claim about one factor specifically**, even when that factor
    dominates the omnibus effect size. If a narrower, single-factor claim
    is going to be stated, run the corresponding factor-restricted test
    directly rather than inferring it from the omnibus result.
14. **Independent AI review, applied consistently across a long
    investigation, functions as an effective adversarial-collaboration
    methodology.** Nearly every substantive correction in this project's
    later stages (E/R closure precision, the Stage 1B permutation bug,
    the three-layer q_excl_node fix) came from a second model reviewing
    Claude's own output without the same investment in a particular
    conclusion. The pattern that made it work: the reviewer consistently
    pushed toward *narrower, more precisely scoped* claims rather than
    more impressive ones, and Claude's role was to absorb corrections
    fully (including reversing its own prior recommendations when a
    reviewer's counter-argument was sound) rather than defend earlier
    work reflexively.
15. **A diff confirming a code consolidation lost nothing establishes
    completeness, not behavioral equivalence.** Two `NOTE.md` files
    (`stage0_simulator_calibration/`, `stage1a_infinitesimal_response/`)
    stated the consolidation into `src/bonsai/dynamics/` was "confirmed
    by diff... not assumed," phrased in a way that read as uniform
    behavioral verification across all three consolidated files. Checked
    file by file: `graph_oscillator_field.py` and
    `degree_preserving_rewiring.py` were independently verified
    (multistability/spectral-gap/solver-cross-validation reproduction;
    byte-exact rewired-construction reproduction, respectively);
    `matched_sparsity_ablation.py` was not, and turned out to actively
    disagree with the historical cached result it had been assumed
    equivalent to. A completeness check (nothing lost) and a behavioral
    check (still does the same thing) are different claims and need
    separate verification, not one standing in for the other.
16. **A component verified field-by-field against its trusted reference
    can still feed a wrong result, if the caller-side glue code around it
    quietly reimplements something instead of calling the already-imported
    real function.** Stage 1D's GPU/JAX port of `run_one_trial` was
    verified correct to 1e-6-1e-8 precision, per-field, against the numpy
    reference -- and the bug was still real: the benchmark script's batch
    construction redrew replica directions via raw `uniform(-1,1)` instead
    of calling the already-imported `generate_fixed_replica_directions()`
    (which draws normal, projects out the rotation-invariant component,
    and unit-normalizes), and separately dropped the E_min validity gate
    when reformatting results for the real analysis functions. Confirmed
    via a 4-way factorial (correct/buggy directions x correct/buggy
    gating) that the direction bug alone fully reproduced the
    discrepancy. The same reimplemented-uniform-directions bug was found
    independently, a second time, in an unrelated draft notebook in the
    same folder -- not a one-off typo but a recurring failure mode worth
    naming: reimplementing a helper instead of importing it is a distinct
    risk from the helper itself being wrong, and passing per-field
    verification of the simulator does not clear the glue code around it.
    Full account: `experiments/stage1d_topology_specificity_gpu/FINDINGS.md`.
17. **Historical data recovered specifically to verify an
    already-committed claim must not be repurposed as generative input
    for a new, unverified one.** Caught mid-session: after recovering
    `stage1a_all_classes.pkl` and `kmnist_class_topologies_200.pkl`
    (sandbox-originated, explicitly scoped by their own handoff README
    as "for independent verification only -- keep local, do not
    commit") to check the newly-written `lattice_construction.py`
    against a historical number, the same files were about to be used to
    generate a NEW, seemingly-official "Stage 1B pilot across all 10
    classes" result -- exactly the "inline code / hand-fed external
    data, nobody else can reproduce it" pattern this project's whole
    restructuring exists to close, recreated fresh rather than avoided.
    Interrupted before any such result was produced. The distinction
    that matters is verification of existing, committed code versus
    generation of a new claim -- not merely whether the data happens to
    be gitignored.
18. **Extrapolating one pipeline stage's small-scale timing to full
    scale does not license assuming *other* stages scale the same way.**
    Stage 2A's own history is the concrete example: data generation
    (embarrassingly parallel, genuinely linear in image count) was
    correctly extrapolated from n=1,000 all the way to n=60,000;
    classifier CV fitting (iterative optimization, not guaranteed to
    stay flat as n grows, especially for ill-conditioned features) was
    dismissed as "a few seconds, not separately timed" at that same
    n=1,000 and turned out to dominate total runtime by ~79x at full
    scale. Different computations need their own timing checks, even
    within the same pipeline -- one stage behaving linearly is not
    evidence another does.
19. **A chunked or batched RNG draw is not automatically the same stream
    as the unchunked one, and the difference can be silent.**
    `numpy.random.Generator.integers` at sub-64-bit widths (e.g.
    `dtype=uint8`) buffers bits from the underlying bit generator, so the
    same seed yields a *different* sequence depending on how many
    elements are requested per call -- a chunked implementation and an
    unchunked one diverge, and both still produce entirely plausible
    p-values with no error raised anywhere. `Generator.random` consumes
    exactly one 64-bit draw per element and does not have this behavior;
    `Generator.integers` at int64 width also does not. Found in Stage
    2B's studentized sign-flip test, where sign matrices are drawn in
    chunks of 512-4096: drawing signs by thresholding `random()` floats
    is chunk-stream-invariant, while the "obvious simplification" to
    `integers(0, 2, dtype=uint8)` is not. The guard is a test, not a
    comment: sweep several chunk sizes (include a non-divisor of the
    total and the degenerate chunk-of-1) and assert a bit-identical
    p-value, so a later simplification fails loudly instead of quietly
    changing the number. Applies to any chunked or vectorized Monte
    Carlo work here, not just this one test -- and note it is a distinct
    failure from principle 8's shared-seed-across-workers problem:
    correct per-worker seeding does not protect against a within-worker
    chunking change altering the stream.
20. **Hand-verified functionality becomes an executable test once it is
    confirmed and locked.** A property established by running commands
    interactively is real, but it lives only in a session transcript:
    nobody can re-check it, and nothing fails when a later change breaks
    it. Convert it into a unit or integration test so it becomes living
    documentation of what was verified and how. Concretely, this project
    hand-checked that the Stage 2B GCS bucket's public-read grant works
    (an anonymous client fetching an object a Colab session wrote) --
    a claim genuinely distinct from the authenticated readback the
    round-trip test already made, since "readable from outside the
    session" and "readable without credentials" are different
    properties. That check is now an assertion in
    `tests/test_stage2b_gcs_roundtrip.py` rather than a paragraph in a
    chat log. The same applies to any manual spike that ends in "yes,
    that works": if it was worth verifying by hand, it is worth pinning,
    and the pin is what stops it silently regressing. Corollary for
    tests that exercise real infrastructure: have them REPORT their
    evidence (object names, byte counts, which credential path was
    used), not merely assert. A bare green PASS records that assertions
    held, not what happened on the wire -- and for a slow test run
    deliberately and rarely, that transcript is most of its value.
21. **A hand-maintained list standing in for a derivable set will
    silently under-cover, and the broader tool you verify with is what
    hides it.** Derive the set, or assert the list equals it. This
    project has now produced the same bug four times, each time as a
    list that looked authoritative: `stage2a-verify` gated on nothing;
    all three Stage 2A GPU targets omitted `exec --timeout` and so could
    never complete; `STAGE2B_TEST_FILES` omitted two test files; and a
    `gcs_scripts` allowlist would have passed vacuously for the next
    script that touched GCS rather than flagging it. The second half of
    the principle is the part that keeps biting: `STAGE2B_TEST_FILES`
    was verified with `pytest tests/` -- a glob -- so both missing files
    ran, the suite was green, and the gap was invisible from the very
    command used to check it. **When the artifact under test is a
    narrowing (an explicit list, a filtered target, a subset), verifying
    it with the broader form proves the code works and says nothing
    about the narrowing.** The fix is mechanical: enumerate from the
    filesystem or the AST instead of by hand, and where a list must stay
    explicit, test that it matches the derived set in both directions
    (missing entries, and entries naming things that no longer exist).
    Any exemption gets a named constant and a reason, plus its own test
    that the exemption still refers to something real. Concretely,
    `tests/test_stage2b_gcs_makefile.py` discovers GCS-touching files by
    walking each file's AST for a `get_bucket` call that does not inject
    a client, so a future ladder driver is guard-railed on the day it is
    written rather than whenever someone remembers the allowlist.
    Corollary, and the direct analogue of principle 10 one layer down:
    **a guard you have not seen fail is not yet a guard.** Both checks
    added here were confirmed by deliberately breaking what they watch
    (drifting the Makefile's bucket value; dropping in a GCS-touching
    script with no target) and observing the specific expected failure,
    exactly as a permutation scheme must be shown to destroy the effect
    it tests for.
22. **A tolerance on a statistic must scale with the statistic's
    MEASURED growth law, not sit at a constant the statistic will
    outgrow.** A threshold is a claim about a quantity's magnitude, and
    if that quantity grows with n while the threshold does not, the
    threshold stops meaning what it was set to mean -- it eventually
    halts on healthy data and, worse, it was never calibrated for the
    regime it now fires in. The live precedent is `GRAD_NORM_REL` in
    `experiments/stage2a_dynamics_classification/stage2a_classifier_jax.py`:
    the L-BFGS convergence threshold is applied as
    `GRAD_NORM_REL * C * n_train`, LINEAR in n, because the objective is
    an unweighted sum over samples scaled by C. The anti-precedent is
    `GRAD_NORM_TOL=1e-6`, the fixed absolute threshold it REPLACED --
    removed precisely because it did not scale, and measured to be three
    to eleven orders off across the C grid. Stage 2B's scaler-centering
    guard is the second instance and a different growth law: sqrt(n),
    because the quantity is `||mean(X_scaled)||` and the mean of n
    float64 values carries ~sqrt(n) accumulated rounding. Same principle,
    different exponent -- which is the point. Do not carry an exponent
    across problems; derive it from the mechanism and check it against
    measurement. Two further requirements, both of which this instance
    had to satisfy: (a) prefer the exponent the MECHANISM implies when it
    upper-bounds the measured growth (0.5 over the measured 0.405), so
    the margin widens with n instead of eroding; (b) never mix an anchor
    measured on one pipeline with a slope measured on another -- Stage 2B
    had a 0.66 exponent from a CPU-evolved table and an anchor from a
    GPU-evolved one, and the combination described neither, however
    reasonable each half looked alone. A scaling tolerance also needs the
    detection power it existed for shown intact at every scale, not
    assumed: the guard must still fire on a genuinely broken input at the
    largest n, or widening it has simply switched it off.
23. **A ratio gate between two quantities that each decay to a numerical
    floor measures which one floored first, not the mechanism.** Once
    either side of a ratio underflows toward zero, a division-by-zero
    guard (a small absolute floor in the denominator, or an equivalent
    protection) silently converts what reads as a RATIO test into an
    ABSOLUTE test against that floor -- and the two sides can cross their
    own floors at different points, so the ratio swings by orders of
    magnitude for reasons unrelated to whatever it was meant to measure.
    Found in Stage 2B's encoder-on-noisy-inputs gate
    (`stage2b_encoder_gate.py`): at ENCODER_STEPS=600 (a diagnostic step
    count, not the locked one), clean's median final-Delta had already hit
    exact float64 zero while noisy's sat at 1.776e-14 -- nine orders below
    the smallest meaningful measured value (2.177e-07) -- yet
    `rho = noisy / max(clean, 1e-15)` reported FAIL at rho=17.76. The full
    trajectory across five step counts (14.98, 169.9, 1.915e4, 17.76, 0.0)
    was non-monotone for exactly this reason, diagnosed in
    `diagnose_encoder_gate_failure.py` before being mistaken for evidence
    the mechanism itself was unstable. The fix is not a bigger floor --
    that only moves where the same failure recurs -- but a separate
    absolute-convergence escape: when BOTH quantities are already below a
    threshold chosen 5+ orders under the smallest meaningful measured
    value and well above observed numerical dust, the gate passes on that
    basis, and the ordinary ratio test applies everywhere else. Near a
    floor, a ratio is the wrong statistic; switch to an absolute one.

# IntelliJ MCP Server Companion
This project is open in Pycharm IDE (IntelliJ IDEA platform). 
Call `get_mcp_companion_overview` to discover available tools and how to use them.


