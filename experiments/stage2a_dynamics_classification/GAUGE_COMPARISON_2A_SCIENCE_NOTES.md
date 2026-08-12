# Stage 2A gauge comparison: what the replications changed

Companion to `GAUGE_COMPARISON_2A_RESULT.md` (the pre-registered run and its
amendment) and `docs/MIGHTY_COLAB_ENGINEERING_NOTES.md` (the infrastructure
this cost). 2026-08-12.

**Scope, stated first because it is the thing most likely to drift.** The
locked, gate-passing result is the sklearn run committed at `d692a81`. sklearn
reproduces the locked `selected_C` for all five conditions; the JAX
replication misses on two, so it cannot serve as the record. Nothing below
changes the record. What follows is evidence about how much weight the
record's conclusions carry.

---

## 1. What survived everything

One result reproduced under every configuration tried:

| configuration | M_g for `encoded_pre_evolution` |
|---|---|
| sklearn, re-selected C (the record) | +0.021783 |
| sklearn, frozen C (companion arm) | +0.021783 |
| JAX, default tolerance | +0.021784 |
| JAX, deep (10,000 iterations) | +0.021816 |

Two optimisers, three convergence depths, both C-selection arms: agreement to
**~3e-05** on a difference of two independently fitted numbers, against a
registered theta of 2e-03.

The un-evolved condition gains ~+0.0218 accuracy under the circular-mean
gauge, an order of magnitude more than any evolved condition, and the gain is
largest where synchronisation is lowest. That is the (1-R) mechanism Stage 2B
found, transferring to a different task and readout. **It is the finding.**

## 2. What did not survive

The class C verdict -- the `evolved_rewired`/`evolved_T` inversion between
gauges -- rests on a margin of **0.000616**, below the registered theta. It
does not reproduce under the JAX classifier, which returns class B with the
ranking preserved under both gauges.

The result document defended the inversion on the grounds that it appeared in
both the re-selected and frozen-C arms. Both arms shared a classifier. Varying
that one factor removes it.

The three leading evolved topologies sit inside a ~0.005 band. Their ordering
has now failed to reproduce under a change of optimiser, and the accuracies
themselves move by more than the inversion margin when the optimiser is simply
allowed to run longer (section 4). **The ordering of the three leading
topologies is not identified by this design.** The gauge comparison should
report the `pre_evolution` result and treat the evolved ranking as unresolved.

## 3. Where the two classifiers agree, and where they do not

Ten arms, sklearn vs JAX at default tolerance:

| selected C | arms | \|delta accuracy\| |
|---|---|---|
| 0.01 | pre_evolution x2 | 1.7e-05 |
| 0.1 | rewired:circ, curr_random:circ | 6.2e-04, 8.2e-04 |
| 1 | curr_random:ref, rewired:ref | 1.7e-03, 8.0e-03 |
| 100-1000 | T x2, lattice x2 | 7.0e-03 to 1.16e-02 |

Five of ten agree within theta; the five that do not are the high-C arms, and
JAX is the pessimistic side in every one. C is not sufficient on its own --
two arms at C=1 differ by 4.7x -- so conditioning modulates it, but the trend
spans three orders of magnitude.

**Cause, isolated rather than argued.** A three-way factorial on the most
divergent cell (`evolved_lattice`, reference, C=1000, fold 0):

| fit | accuracy | iterations |
|---|---|---|
| sklearn CPU | 0.882250 | 5,309 |
| JAX GPU | 0.871083 | 1,955 |
| JAX CPU | 0.871000 | 2,003 |

JAX-on-CPU is **135x closer to JAX-on-GPU than to sklearn**, so GPU numerics
and TF32 are ruled out entirely (x64 was enabled; all operands float64, so
TF32 never applied to these matmuls). The cause is the convergence criterion:
`stage2a_classifier_jax` stops at `GRAD_NORM_REL * C * n_train`, which at
C=1000 and n_train=48,000 is 2.88e+05, and the fit halted at 2.814e+05.

## 4. Neither classifier was converged

Sweeping the threshold on that same cell:

| stopping point | iterations | accuracy |
|---|---|---|
| `rel=6e-3` (shipped default) | 1,758 | 0.869917 |
| sklearn `tol=1e-4` | 5,309 | 0.882250 |
| `rel<=1e-3` -> hits `max_iter` | 10,000 | 0.887333 |

Below `rel=1e-3` the tolerance stops binding and **`max_iter=10000` becomes
the ceiling**: the last three sweep rows are identical, ending at gradient
norm 9.24e+04 against a 4.8e+04 target. Not converged -- out of budget.

Accuracy is monotone in iterations. The deepest fit beats sklearn by +0.005
and is still climbing. So there is no tolerance at which JAX "matches"
sklearn, because sklearn is not a fixed target: it stops at
`||grad||/(C*n_train)` ~ 2.6e-03 on this condition, under-converged itself,
just less so.

`GRAD_NORM_REL=6e-3` is not a defect. It was calibrated as ~2.2x the
`[1.338e-3, 2.771e-3]` range sklearn achieves across the C grid -- a
deliberately scale-free threshold, correct in form (principle 22), set about a
factor of two loose. The consequence is measurable and one-directional.

**`max_iter=10000` is a locked parameter and it binds on the hardest
conditions.** It was raised from 1000 during feasibility stage 2 to stop the
non-convergence gate firing; whether it was *enough* was never asked. On
`evolved_lattice` at C=1000 it is not, and the partial deep re-run found
C=10000 folds failing to converge in `evolved_T` as well. The locked result's
high-C accuracies are budget-limited, not optimum-limited.

This does not invalidate the locked run: every condition converged by
sklearn's own criterion, which is the criterion the pipeline declares. It does
mean the gap between conditions at high C is partly a gap between how far each
fit got.

## 5. The deep re-run, partial

Four of ten arms completed at `GRAD_NORM_REL=1e-3` before an output-gap
timeout, with retries blocked by Colab 503s on the A100 assign endpoint:

| arm | default JAX | deep JAX | sklearn |
|---|---|---|---|
| pre_evolution:ref | 0.835933 | 0.836017 | 0.835917 |
| pre_evolution:circ | 0.857717 | 0.857833 | 0.857700 |
| evolved_T:ref | 0.894033 | **0.906667** | 0.905167 |
| evolved_T:circ | 0.901400 | **0.912950** | 0.908433 |

Well-conditioned arms move by ~1e-04 and cost 2.6x more time -- the extra
budget buys accuracy only where the fit was unfinished. `evolved_T` at depth
**exceeds** its sklearn counterpart on both gauges, consistent with section 4.

`M_g` for `pre_evolution` at depth is +0.021816, the fourth configuration to
agree to ~3e-05.

**Left open:** whether the class C inversion reappears at this third
convergence depth. It needs `lattice`, `rewired` and `curr_random`, which were
not obtained. Given that the ordering already failed to reproduce under one
perturbation and that the accuracies move by more than the inversion margin
under another, a third measurement would be unlikely to rescue it -- but that
is an expectation, not a result, and it is recorded as unfinished rather than
inferred.

## 6. Operating rule

Stated from measurement rather than instinct:

> **JAX for exploration, sklearn for anything that must reconcile with the
> locked record.** They are interchangeable to 1.7e-05 where the fit is
> well-conditioned and up to 1.2e-02 apart where it is not, with JAX
> systematically pessimistic. JAX is 69-183x faster per arm; the whole
> ten-arm comparison ran in 22 minutes on one A100 against 34.5 core-hours
> locally.

The venue caution that produced the locked run was **correct on the merits**
and taken without evidence. The reproduction gate genuinely required sklearn:
`evolved_T` and `evolved_rewired` select different C under JAX, and that
mismatch would have read as a regeneration failure. But the argument was never
tested, and a 96x speedup was left unused for a day to honour it.
