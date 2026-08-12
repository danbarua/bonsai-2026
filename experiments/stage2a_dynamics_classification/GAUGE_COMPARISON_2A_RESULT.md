# The gauge comparison on Stage 2A: result

Pre-registered in `GAUGE_COMPARISON_2A_PREREGISTRATION.md` (`917088f`,
amended `1bbfc89`), both committed before any classifier was fitted under
either gauge. Driver: `run_gauge_comparison_2a.py`. Run 2026-08-12, train
split only, 34.5 core-hours across ten arms.

## Verdict

**Class C: the condition ranking changed under the gauge swap.**

The inversion is `evolved_rewired` and `evolved_T` exchanging second and
third place. It reproduces in both the primary (re-selected C) and companion
(frozen C) arms.

| ranking | order |
|---|---|
| reference-node | curr_random > **rewired > T** > lattice > pre_evolution |
| circular-mean | curr_random > **T > rewired** > lattice > pre_evolution |

Direction flag: **mixed** in the primary arm, **uniform favouring
circular-mean** in the companion arm. That disagreement is itself a finding
and is treated below rather than averaged away.

## The reproduction gate

Registered as: the reference arm must reproduce the locked record's
`selected_C` exactly for every condition, with `mean_val_loss_per_C`
compared descriptively.

**PASSED, 5/5 conditions.** Four of the five reproduced *bit-exactly* —
`max|dloss| = 0.000e+00` across all nine C values, identical float64.

This is stronger than the registration expected. The document said exact
agreement "is not expected" because the states are regenerated rather than
consumed. In fact the whole train-side pipeline — local encode, A100
evolution, gauge, 45 sklearn CV fits — landed on the same bits as a run from
months earlier, on a fresh GPU allocation whose wall time differed by 0.9%
(113.96s locked vs 114.96s regenerated).

### The one condition that did not

`evolved_lattice` differs, and the differences grow monotonically with C:

| C | 1e-4 | 1e-2 | 1e0 | 1e1 | 1e3 | 1e4 |
|---|---|---|---|---|---|---|
| difference | -1.2e-11 | -1.0e-05 | -5.4e-04 | **-1.3e-03** | +4.7e-04 | +9.3e-04 |

The argmin is unaffected, so `selected_C = 1000` still matches and the gate
passes on its registered criterion honestly.

`evolved_lattice` is the worst-conditioned condition in the study — it needed
4,123 L-BFGS iterations at C=100 where `pre_evolution` needed 1,603 — and the
growth of the discrepancy with C is what amplification of a tiny input
perturbation through an ill-conditioned optimisation looks like.

**The cause is not established and is left standing as an open discrepancy.**
Two candidates, not distinguished by anything measured here:

- the regenerated lattice states differ from the originals in their last
  bits (GPU kernel scheduling differing for one topology's sparsity), or
- the fits differ (a BLAS reduction-order difference between this run's
  pinned `OMP_NUM_THREADS=1` and whatever the locked run used).

The second would ordinarily perturb every condition, and four are bit-exact,
which argues against it — but not decisively, since a well-conditioned fit
can absorb a perturbation that an ill-conditioned one amplifies. The original
states are gone (their symlinks dangle), so the direct check is not available
without re-running the locked pipeline.

## The comparison

Accuracy is the mean of five per-fold validation accuracies at each arm's own
selected C. C selection remains the locked log-loss rule.

| condition | ref | circ | M_g | C ref/circ |
|---|---|---|---|---|
| encoded_pre_evolution | 0.835917 | 0.857700 | **+0.021783** | 0.01 / 0.01 |
| evolved_T | 0.905167 | 0.908433 | +0.003267 | 1000 / 100 |
| evolved_lattice | 0.882900 | 0.887717 | +0.004817 | 1000 / 1000 |
| evolved_rewired | 0.905783 | 0.906733 | +0.000950 | 10 / 0.1 |
| evolved_curr_random | 0.910667 | 0.909517 | **-0.001150** | 1 / 0.1 |

## What the registered prediction got right and wrong

The registration predicted: circular-mean ≥ reference for **every** condition
(uniform direction), the gain largest for the **least synchronised**
condition, and any ranking change coming from **`pre_evolution` moving up**.

- **Uniform direction: FAILED** as registered. `curr_random` is negative in
  the primary arm, so the flag reads mixed.
- **Gain largest for the least synchronised: HELD.** `pre_evolution` gains
  +0.0218, roughly 4.5x the largest evolved gain and 20x the median. The
  (1-R) mechanism found in Stage 2B transfers to this task.
- **Ranking change from `pre_evolution` moving up: FAILED.**
  `pre_evolution` stayed last under both gauges. The inversion came from two
  evolved conditions I had not flagged as candidates at all.

Two of three predictions wrong is the useful outcome here: the mechanism
generalised, the specific consequences I drew from it did not.

## Why the two arms disagree on direction

The companion (frozen-C) arm holds C at the locked run's selection for both
gauges. Under it, **every** condition favours circular-mean:

| condition | C | ref | circ | M_g |
|---|---|---|---|---|
| encoded_pre_evolution | 0.01 | 0.835917 | 0.857700 | +0.021783 |
| evolved_T | 1000 | 0.905167 | 0.911367 | +0.006200 |
| evolved_lattice | 1000 | 0.882900 | 0.887717 | +0.004817 |
| evolved_rewired | 10 | 0.905783 | 0.906733 | +0.000950 |
| evolved_curr_random | 1 | 0.910667 | **0.911933** | **+0.001267** |

`curr_random`'s sign flips between arms. The reason is that **C is selected
by log-loss and the comparison is accuracy, and the two do not agree**: for
`curr_random` under circular-mean, C=0.1 minimises log-loss but C=1 gives
higher accuracy. So the primary arm's single negative M_g is a consequence of
re-selection, not of the gauge.

Under matched C, the gauge helps every condition without exception. That is
the registered prediction holding — in the companion arm, which is not
primary. Reported as such: the primary arm is primary, the prediction failed
as registered, and the mechanism it encoded is visible next door.

## The verdict is real but its margin is not comfortable

The inversion that produces class C is between conditions separated by
**0.000616** under reference — well below theta = 0.002, the threshold
registered as the resolution worth discussing at all.

The registered rule gives ranking inversions absolute precedence over margin
size, so a noise-scale gap flips the reported class. This is stated rather
than smoothed: re-reading the rule after seeing the numbers is what
pre-registration exists to prevent. Two facts bear on how much weight the
inversion carries:

- It reproduces in **both** arms, under different C selections. A pure
  coin-flip on noise would not have to.
- Both conditions sit inside a ~0.005 band with `T`, `rewired` and
  `curr_random` mutually within noise of each other, so "second place" is not
  a well-resolved quantity in this design regardless of gauge.

The honest summary: the ranking is genuinely gauge-dependent among the three
leading evolved topologies, because those three are not separated by enough
to have a stable order under any perturbation.

## Other observations

**Optimal regularisation moves under the gauge swap, for evolved conditions
only.** `T` 1000 -> 100, `rewired` 10 -> 0.1, `curr_random` 1 -> 0.1;
`lattice` and `pre_evolution` unchanged. Three of four evolved conditions
shift by one to two decades.

**The leading topology differs from Stage 2B.** 2B found `T` first under both
gauges; here `curr_random` leads under both, with `T` second or third. Same
topologies, different task and readout, so not a contradiction — recorded
because the two stages are otherwise closely matched and a reader would
reasonably expect agreement.

**No stop-gate fired.** Every one of 450 (fold, C) fits converged inside
`max_iter=10000`. The registration flagged the circular-mean arm as the
plausible non-convergence risk, since `sum_i sin(theta_i - mu) = 0` makes its
feature matrix rank deficient by exactly one. The opposite happened: the
circular-mean arms consistently converged *faster* (`evolved_T` needed 2,667
iterations at C=100 against reference's 3,997).

**Cost (principle 18).** 34.5 core-hours total, ~5.5h wall across ten
parallel single-threaded arms. Per-fit cost is dominated by high C and is
strongly condition-dependent: a fold of `pre_evolution` costs ~30 min, a fold
of `evolved_T` ~90 min. Iteration counts plateau above C=100 for
well-conditioned conditions and keep climbing for ill-conditioned ones.

## Scope

Train split only. `stage2a-prepare-test` and `stage2a-evolve-test-gpu` were
not run and no path here reads the 10,000-image official test set; that
evaluation was one-shot and remains spent on the locked result.

**Stage 2A's locked verdict is unchanged by this run.** This measures an
alternative gauge on the training split; it does not substitute one, and a
train-side ranking is not a test-side claim. Descriptive and nominal
throughout — no new confirmatory statistic, no member of any corrected
family.

---

## Amendment 1 — 2026-08-12, after a GPU replication

The whole ten-arm comparison was re-run on an A100 with the JAX classifier
(`run_gauge_comparison_2a_gpu.py`, 21.6 minutes against 34.5 core-hours).
Everything else was held fixed: same states, same gauges, same folds, same
seed, same C grid, same registered rule.

**It returns class B, not class C.**

| | sklearn (this record) | JAX |
|---|---|---|
| reference ranking | curr_random > **rewired > T** > lattice > pre_evolution | same |
| circular-mean ranking | curr_random > **T > rewired** > lattice > pre_evolution | curr_random > **rewired > T** > lattice > pre_evolution |
| class | **C** (inversion) | **B** (preserved) |

The `rewired`/`T` inversion that produces class C does not occur under the
other optimiser.

**This does not overturn the verdict above, and the record does not change.**
The reproduction gate requires the reference arm to reproduce the locked
`selected_C` for every condition. sklearn does, five for five; JAX misses on
two (`evolved_T` 1000 -> 100, `evolved_rewired` 10 -> 1), so it cannot serve
as the record. The gate is doing exactly what it was registered to do.

What the replication does establish is the strength of the class C call. The
section above already said the margin was not comfortable -- an inversion of
0.000616, below theta -- and defended it on the grounds that it reproduced in
both the re-selected and frozen-C arms. That defence was weaker than it read:
**both arms shared a classifier.** Varying the one factor they held fixed
removes the inversion. The honest statement is the one this document already
made a paragraph later, now with direct evidence: *second place is not a
well-resolved quantity in this design*, and the C-versus-B distinction rests
on a margin thin enough that a legitimate change of optimiser moves it.

Unchanged by the replication: `pre_evolution` last under both gauges by a
wide margin, `curr_random` first under both, `lattice` fourth under both, and
the large uniform `pre_evolution` gain (M_g = +0.021783 sklearn, +0.021784
JAX -- agreeing to 1e-06). The finding that survives is the mechanism, not
the ordering of the three leading topologies.

### Why the two classifiers differ, measured rather than argued

Agreement tracks the selected C:

| selected C | arms | \|delta accuracy\| |
|---|---|---|
| 0.01 | pre_evolution x2 | 1.7e-05 |
| 0.1 | rewired:circ, curr_random:circ | 6.2e-04, 8.2e-04 |
| 1 | curr_random:ref, rewired:ref | 1.7e-03, 8.0e-03 |
| 100-1000 | T x2, lattice x2 | 7.0e-03 to 1.16e-02 |

Five of ten arms agree within theta; the five that do not are the high-C
ones. C is not the whole story -- two arms at C=1 differ by 4.7x -- so
conditioning modulates it, but the trend is three orders of magnitude.

A three-way factorial on the most divergent cell (`evolved_lattice`,
reference, C=1000, fold 0) separates the two candidate causes:

| fit | accuracy | iterations |
|---|---|---|
| sklearn CPU | 0.882250 | 5,309 |
| JAX GPU | 0.871083 | 1,955 |
| JAX CPU | 0.871000 | 2,003 |

JAX-on-CPU is 135x closer to JAX-on-GPU than to sklearn, so **GPU numerics
are not the cause and the platform is irrelevant** (x64 was enabled and every
operand is float64, so TF32 never applied). The cause is the convergence
criterion. `stage2a_classifier_jax` stops on `GRAD_NORM_REL * C * n_train`,
which at C=1000 and n_train=48,000 is **2.88e+05**; the fit halted at
`grad_norm = 2.814e+05`. It converged by its own rule, while sklearn's fixed
`tol=1e-4` ran 2.6x longer.

That scaling is deliberate and correct for its purpose -- the objective is an
unweighted sum scaled by C, and this project removed a fixed absolute
threshold for exactly that reason (principle 22). The consequence is simply
that the two classifiers are not interchangeable at high C, and the JAX one
is systematically the pessimistic side.

**Operating rule, now measured rather than assumed: JAX for exploration
(69-183x faster), sklearn for anything that must reconcile with the locked
record.**
