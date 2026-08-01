# Stage 1A Re-Verification Design Document

*Prepared ahead of implementation; to be committed at
`experiments/stage1a_re_verification/DESIGN.md` once the 10-class
construction pipeline is confirmed working.*

## Motivation

Stage 1A's original design used exactly one random-construction seed per
class for the stochastic controls (matched-sparsity random, and to a
lesser extent degree-preserving rewiring). A class-0 pilot (post-hoc
robustness note in `experiments/stage1a_infinitesimal_response/FINDINGS.md`)
found that the T-vs-random AUC comparison's direction is not stable
across seeds for class 0: 7 of 20 seeds produced the opposite sign under
the historical random control, and 2 of 20 under the current one. This
does not contradict the original null conclusion -- a control this
seed-sensitive would struggle to produce a robust aggregate signal -- but
it revealed a source of within-class variance that the original design
did not account for and did not document.

This re-verification corrects that by sweeping multiple graph seeds per
class and aggregating within class before testing. The primary
inferential structure (paired Wilcoxon across 10 class-level differences)
is otherwise identical to Stage 1A, preserving direct comparability.

---

## Notation

- c ∈ {0, ..., 9}: KMNIST class
- g ∈ {T, rewired, hist_random, curr_random, lattice}: construction
- s ∈ {0, ..., 24}: graph seed (stochastic controls only)
- A_{cgs}: Stage 1A AUC for class c, construction g, seed s
- A_cT: AUC for deterministic T (one value per class, no seed subscript)

---

## Planned comparisons (4, Holm-corrected across all four)

1. T vs. historical half-edge random, coupling-budget normalized
   (`generate_historical_matched_sparsity_random` from
   `src/bonsai/dynamics/historical_matched_sparsity_random.py`)
2. T vs. current edge-count-matched random
   (`generate_matched_sparsity_topology` from
   `src/bonsai/dynamics/matched_sparsity_ablation.py`)
3. T vs. degree-preserving rewiring
   (`degree_preserving_rewire` from
   `src/bonsai/dynamics/degree_preserving_rewiring.py`)
4. T vs. lattice
   (`build_lattice_topology` from
   `src/bonsai/dynamics/lattice_construction.py`)

These are kept separate. The two random controls represent different null
models and must not be combined into one seed ensemble.

---

## Seeds

**S = 25** independent graph seeds per class, for each STOCHASTIC
control: seeds 0 through 24, applied identically across all classes and
stochastic constructions (same published seed list for every class).
Matching seed numbers aids auditability but does not imply statistical
pairing across construction types.

**Deterministic controls** (T and lattice): one value per class, no seed
sweep. Do not generate 25 identical copies -- they would all be equal by
construction.

**Semi-deterministic note on rewiring**: `degree_preserving_rewire` is
technically stochastic (takes a seed parameter), but Stage 1A's original
design used seed=1 fixed. Include a 25-seed sweep for rewiring in the
same pattern as the random controls, for consistency and to test whether
the rewired comparison is similarly seed-sensitive. Report this
explicitly.

---

## Within-class aggregation (primary)

For each stochastic control g and class c:

  Ā_{cg} = (1/25) * Σ_{s=0..24} A_{cgs}   [arithmetic mean]

For deterministic controls (T, lattice), the "aggregate" is just the
single value.

**Seed stability diagnostic (descriptive only):**
After computing all 25 seeds, also compute Ā_{cg} using the first
5, 10, 15, 20 seeds. Show how the class-level mean estimate converges.
This is a descriptive stability check only -- it must NOT be used to
stop early or select a seed count based on the observed result.

**Within-class MCSE:**
  MCSE_{cg} = SD_s(A_{cgs}) / sqrt(25)
Report this for every class × stochastic control combination.

---

## Class-level differences

  d_{cg} = A_cT - Ā_{cg}

One value per class per planned comparison.

---

## Primary test

Exact two-sided paired Wilcoxon signed-rank test across the 10
class-level differences {d_{cg} : c = 0..9}, applied separately to
each of the 4 planned comparisons, with Holm correction across the 4.

Report for each comparison:
- All 10 class-level differences
- Median difference
- Hodges-Lehmann estimate
- Sign count (positive differences / 10)
- Wilcoxon W statistic
- Exact p-value (use exact distribution, not normal approximation -- N=10
  is small enough)
- Holm-corrected p-value

---

## Robustness analyses (in this order, not the other way around)

### 1. Median seed aggregation (sensitivity analysis)

Repeat the primary test using within-class median instead of mean.
Report whether conclusions change.

### 2. Exact class-level sign-flip test

For each comparison, apply all 2^10 = 1024 sign flips to the 10 d_{cg}
values, using the mean class difference as the test statistic. This is
exact at N=10 and assumption-free. Report the two-sided p-value.

### 3. Hierarchical bootstrap

Resample:
  1. classes with replacement (draw 10 classes with replacement from {0..9})
  2. seeds within each selected class with replacement (draw 25 from the
     25 available)
Recompute class means and overall mean class difference on each draw.
Use B=10,000 bootstrap draws. Report 95% CI for the mean class-level
difference and note whether zero is inside or outside the interval.

### 4. Mixed model (tertiary only)

If the above three are consistent, also fit:
  d_{cgs} = A_cT - A_{cgs} = mu + u_c + e_{cgs}
where u_c ~ N(0, sigma^2_c) is a class random effect and e_{cgs} is
residual error. Report the fixed-effect estimate mu and its 95% CI.

Note explicitly in the report that 10 classes is a thin basis for
estimating the class-level random-effects distribution, so this is a
tertiary check, not the primary analysis.

---

## Prerequisites

1. `experiments/stage0_simulator_calibration/results/stage1a_all_classes.pkl`
   must exist, containing all 10 KMNIST classes' T/rewired/random/lattice
   constructions in the historical format. This is produced by
   `build_all_class_topologies.py` (the 10-class pipeline currently
   being built).
2. All four construction modules (`historical_matched_sparsity_random.py`,
   `matched_sparsity_ablation.py`, `degree_preserving_rewiring.py`,
   `lattice_construction.py`) must be importable from `src/bonsai/dynamics`.
3. `joint_tangent_matrix_response` must be importable from
   `src/bonsai/dynamics/graph_oscillator_field.py`.

---

## Computational scope

**Stochastic control constructions:**
  3 stochastic controls (hist_random, curr_random, rewired)
  × 25 seeds × 10 classes = 750 construction runs

**Deterministic constructions:**
  T + lattice, 10 classes each = 20 (already in the pipeline output)

**AUC computations:**
  For each of 750 + 20 = 770 construction instances:
  Run Stage 1A's `joint_tangent_matrix_response` per class.
  Estimate runtime per class from Stage 1A's original methodology
  BEFORE committing to the full run.

**Checkpoint and parallelize:** These runs are embarrassingly parallel
across (class, construction, seed) triples. Use the same SeedSequence
pattern as Stage 1B2's parallel Monte Carlo. Checkpoint to pkl after
each batch in case of interruption.

---

## Decision rule

The historical Stage 1A conclusion (no significant T-vs-controls
difference) should be considered robust only if consistent across:
- Primary mean-aggregated Wilcoxon
- Median seed aggregation
- Exact class-level sign-flip test
- Reasonable within-class seed variability (MCSE small relative to
  class-level differences)

If the historical random and current random controls disagree on the
direction or significance of the T-vs-random comparison, report this as a
genuinely informative scientific finding (the two null models are not
interchangeable) -- do not average over or ignore the disagreement.

---

## Files to create

```
experiments/stage1a_re_verification/
  DESIGN.md           <- this document
  run_stage1a_reverification.py   <- driver
  analyze_stage1a_reverification.py  <- analysis + all 4 tests + robustness
  FINDINGS.md         <- populated after analysis
  results/            <- gitignored pkls
```

---

## What this does NOT do

- It does not extend Stage 1C (trajectory generalization) -- that is a
  separate question about Stage 1B2's result, not Stage 1A's.
- It does not re-run the original Stage 1A computation -- it re-runs the
  comparison against controls with proper seed accounting.
- It does not replace the original Stage 1A FINDINGS.md -- it lives
  in its own experiment folder and cross-references the original.