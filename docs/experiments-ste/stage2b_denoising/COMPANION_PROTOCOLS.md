Simplified Technical English version of `experiments/stage2b_denoising/COMPANION_PROTOCOLS.md`.

# Stage 2B companion protocols — FROZEN

**Status: frozen. The team committed this document before either
protocol had produced a number.** A companion protocol is one of two
extra check plans that `AUDIT_PROTOCOL.md` names as its own partners. A
frozen document is one the team agrees not to change once a result from
it exists. The same rule applies here: nothing in this document may
change once a result from it has been seen. What each protocol measures,
how its input data is built, and what happens as a result, are all fixed
in advance, so none of these can be chosen afterward to fit a
convenient outcome.

`AUDIT_PROTOCOL.md` is the authoritative source wherever the two
documents overlap — for example, the roles of each image population,
the sign convention, and the analytic resolution limit. Nothing in this
document restates a number from that one, except by pointing to it.

---

# Protocol 1 — ARM/x86 propagation stress set

## What this measures, and what it does not

Phase A moved image encoding onto Apple Silicon chips (ARM); ladder
stages 1 and 2 had both encoded on Colab's x86 chips. The recorded
cross-chip-type comparison (see `FINDINGS.md`, "Measured, before the
decision was acted on") found that 93.4% of coordinates were identical,
with a **maximum difference of 3 ULP** (4.441e-16). A ULP, unit in the
last place, is the smallest possible difference between two numbers as a
computer stores them. This compares against within-chip-type results,
which were bit-exact in both directions tested — the same machine
running twice, and two separate Colab sessions.

That comparison was made only at the **encoding** step, on the stage-1
test corpus. It is one input to this protocol, not a substitute for it.
The open question this protocol answers, that the earlier comparison
does not, is propagation: whether a 3-ULP difference at the encoding
step stays small after it passes through the ODE solve, the ridge
prediction, and the evolved-minus-pre-evolution contrast — which is the
actual quantity the audit's trigger checks.

This protocol is deliberately a **worst-case** test. The stress set is
built to be as hard a test as possible, so its numbers form an upper
bound on how much propagation can happen, and they are not
representative of the whole corpus. It is not a random sample, and it
must never be reported as one.

## How the stress set is built (deterministic, frozen)

Four parts, combined together, then any duplicate KMNIST training index
removed, then sorted in ascending order. This sorted, deduplicated list
of indices IS the stress set; it is written into the artifact, so the
set can be checked, rather than rebuilt from a description each time.

| # | part | rule | size |
|---|---|---|---|
| A | largest cross-chip-type encoding difference | the images with the largest per-image **maximum** absolute coordinate difference between the ARM-chip and x86-chip encodings, from the existing comparison; ties are broken by official index, ascending | 100 |
| B | convergence-tail stress cases | **every** image whose Phase A final-Delta value is strictly greater than 0.0, taken from the regenerated 60,000-image artifact's `deltas` array | all, capped (see below) |
| C | per-class floor | for any KMNIST class with fewer than 20 images after combining A, B, and D, the team adds the lowest-indexed images of that class until it reaches 20 | as needed |
| D | seeded stratified random sample | 20 images per class, 200 in total, drawn using `numpy.random.default_rng(42)` from each class's official-index list, in ascending order, via `rng.choice(indices, size=20, replace=False)`, visiting classes 0 through 9 in order | 200 |

**A cap on part B**, frozen here because the true count is not yet
known: if more than 500 images have a final-Delta value above 0.0, the
team takes the 500 largest by final-Delta value, breaking ties by
official index ascending, and records both the cap and the true count.
The 54,000-image run measured 79 such images (0.146%), so the team
expects this cap never to actually apply; it exists so the set size
cannot be decided only after seeing the number.

**Naming, following `AUDIT_PROTOCOL.md`.** These are called
*convergence-tail stress cases*. An earlier name, "basin-boundary
candidates," was withdrawn as unsupported: a residual value from the
encoder pulling toward a fixed point is a different kind of quantity
from being close to a boundary in the evolution dynamics, and nothing
measured here speaks to that second, different quantity.

**Where part A comes from if the recorded comparison is unavailable.**
It was measured on the stage-1 test corpus, not the stage-3 one. If the
per-image differences cannot be recovered from the existing record, part
A is instead rebuilt by encoding the stress set's other parts on both
chip types, and this substitution is stated explicitly wherever
reported — never silently replaced by a different rule.

## Comparison procedure

**First comparison — frozen ridge coefficients, both chip types.**
Identical, already-fitted ridge coefficients are applied to both the
ARM-derived and the x86-derived features. This isolates the effect of
numerical propagation from the effect of refitting a new model: a model
refit on ARM features would produce different coefficients, and any
resulting difference would then mix together the two effects with no way
to separate them afterward.

The team reports the maximum absolute difference at each stage of the
chain:

1. encoding (the 505-value restricted phase vector)
2. evolved features, **per graph** (`T`, `lattice`, `rewired`,
   `curr_random`)
3. prediction
4. per-image MSE
5. the evolved-minus-pre-evolution contrast `Delta_g`

All five stages are always reported, no matter what any one of them
shows. Reporting only the stages that stayed small would make the whole
chain hard to interpret.

## Consequence rule (frozen, before any number exists)

The propagation measurement is **descriptive only** at stages 1 through
4. Stage 5 has one pre-committed halt condition:

> If the maximum absolute cross-chip-type difference in `Delta_g`, for
> any graph, **exceeds the audit's frozen contrast threshold**
> (`4.604761e-10`, from `AUDIT_PROTOCOL.md` Freeze 1 at M=100), then
> cross-chip-type numerical noise cannot be told apart from the effect
> the audit is testing for, and this triggers a renewed interpretation
> review before Stage 4.

The threshold used here is borrowed, not invented for this purpose: it
is already frozen, already derived analytically from the metric's
algebra and a recorded value of `d`, and it is already the line below
which a difference is called implementation noise.

Deliberately **not** claimed in advance: that the measurement will
actually come in below this line. The encoding-stage result (3 ULP,
eight or more orders of magnitude below the solver's relative tolerance
of 1e-6) is a reason to expect this, but it is not evidence of it — in
the same way `AUDIT_PROTOCOL.md` declines to claim robustness across
orders of magnitude of `M` before measuring it.

---

# Protocol 2 — `ABS_CONV_EPS` sensitivity

## What this measures

`ABS_CONV_EPS = 1e-12` is the absolute-convergence escape value added to
the encoder gate, after the team found the ratio-near-a-floor problem
(see `FINDINGS.md` Part 2-3, and `CLAUDE.md` working rule 23). A
threshold introduced specifically to fix a failure is exactly the kind
of constant that needs its sensitivity checked, rather than simply
asserted, because its value was chosen after the team already saw the
failure it was meant to resolve.

## How this is recomputed (frozen)

The gate's verdict is recomputed **from the stored per-image final-Delta
arrays** — with no re-encoding — at

    eps in {1e-10, 1e-11, 1e-12, 1e-13}

crossed with every step count for which a stored array already exists:
the five step counts in `diagnose_encoder_gate_failure.py`'s
`STEP_COUNTS` list (75, 150, 300, 600, 1200), and the ladder's own
`encoder_gate_s{steps}` artifacts. The function `evaluate_rho_gate` is
called unmodified, varying only its existing `abs_conv_eps` parameter —
this protocol never reimplements the gate itself, following working
rule 16.

The output is one table. Its rows are step counts, its columns are eps
values, and each cell reports PASS or FAIL, along with `rho`,
`median_delta_clean`, and `median_delta_noisy`. Every cell is reported,
including the ones that agree with each other.

**The result the team is most interested in is where the verdict
FLIPS**, both at the locked `ENCODER_STEPS=1200` value, and at every
other step count. If the verdict at 1,200 steps stays the same across
all four eps values, that itself is the result. If it does not, the
value of eps where it changes, and the direction of the change, are
reported as the result.

## The four justification axes (all four required)

`AUDIT_PROTOCOL.md` requires the value 1e-12 to be justified against
four separate factors. Each one gets its own written paragraph, with its
own measured numbers. Three of these four already existed; the fourth
was dropped by mistake during an earlier consolidation, and is
reinstated here.

1. **float64 precision.** The observed numerical noise in the final-
   Delta series spans 1e-14 to 1e-16. 1e-12 sits above this band, so a
   value below it cannot be told apart from ordinary numerical noise.
2. **The scale of the phase-update diagnostic.** The smallest meaningful
   measured final-Delta value anywhere in this project is 2.177e-07
   (clean images, 150 steps, stage 1). 1e-12 is five or more orders of
   magnitude below that, so this escape cannot fire on a signal that is
   genuinely still converging.
3. **The encoder's own implementation.** This factor looks at what
   magnitudes the update rule can actually produce at all. The encoder
   pulls values toward a fixed point (a stable value it settles into),
   and the measured residual against the 1,200-step result falls from
   8.370e-07, to 8.062e-13, to exactly 0.0, at 300, 600, and 1,200 steps
   respectively. The point of writing this factor up is the relationship
   between that decay and the threshold — specifically, at which step
   count the residual first crosses below 1e-12, and whether the escape
   is therefore describing genuine convergence, or just describing the
   float64 numerical floor.
4. **Downstream feature sensitivity.** This factor covers how a final-
   Delta value of a given magnitude, eps, propagates into the cosine and
   sine features, and then into the ODE solve. The reference points are
   the solver's relative tolerance of 1e-6, and the 54,000-image worst
   case of 2.468e-10, which sits 4,053 times below that tolerance. The
   point of writing this factor up is what a phase residual of 1e-10,
   1e-12, and 1e-13 actually does to the evolved features and to
   `Delta_g` — in other words, whether any eps value in the swept range
   is even large enough to matter downstream at all.

## Consequence rule (frozen)

**`ABS_CONV_EPS` does not change as a result of this table**, on the
same footing as `AUDIT_PROTOCOL.md`'s statement that "the 1,200-step
budget stays frozen regardless of every number this audit produces."
What may change is the scope of the claim the team makes about the gate.

There is one halt condition: if the verdict at the locked
`ENCODER_STEPS=1200` value differs across the swept eps range, then the
gate's PASS result depends on a constant chosen after the team already
saw the failure it was meant to resolve, and this triggers a renewed
interpretation review before Stage 4.

---

## Where this data comes from, and how it is reported

Both protocols run under the fingerprint contract
(`stage2b_fingerprint.py`). A fingerprint is a record of exactly which
code produced a given result, used to prove a result is trustworthy. The
stress-set index array, every reported maximum value, and the full
sensitivity table are all published together with a manifest, so the
numbers always carry a record of the exact code that produced them.

Both protocols are reported in `FINDINGS.md`, with how they were built
stated inline. The stress set is adversarial by design, and the
sensitivity table is a recomputation of an existing result, not a new
measurement — a reader who encounters either number without this framing
would read it as a plain property of the whole corpus, which neither
number actually is.
