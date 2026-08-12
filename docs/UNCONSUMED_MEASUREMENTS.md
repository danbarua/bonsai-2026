# Measurements this project made, recorded, and never read back

The catalogue of a failure mode distinct from the one `VACUOUS_TESTS.md`
collects. Nothing here is wrong. Each entry was computed for one purpose,
recorded faithfully, and then not consulted when a later question needed
exactly it. The apparatus has extensive machinery for checking that records
are CONSISTENT and none for noticing that a record ANSWERS an open question.

Opened 2026-08-11, after a measurement that would have settled a live
question was found sitting unused in a table committed the previous day.

## The entries

### 1. Protocol 1's five-stage propagation table — CONSUMED

`experiments/stage2b_denoising/FINDINGS.md:1535-1596`. A real ARM-vs-x86
perturbation propagated through the whole pipeline and tabulated at every
stage, recorded 2026-08-10 and used only to check a halt threshold.

Read back 2026-08-11 (`cf14ea9`). The correct quotient is STAGE 1 -> STAGE 5:
axis 4's `B(eps) = 2*sin(eps/2)` is the immediate cos/sin bound at ENCODING,
so stage 2 (post-ODE) is not B — the first version of this reading got that
wrong. Gains: T 224.6, lattice 1003.1, rewired 1234.7, curr_random 4120.0 —
112x to 2060x the claimed coefficient of 2. This refutes the coefficient-2
derivation. It does NOT fail every swept envelope: T stays under 2B=2e-13 and
none exceeds 2B=2e-12.

Cost of not reading it back: two review rounds and a 2.3GB refit to
rediscover something already committed.

### 2. `min_col_std` — CONSUMED

Stored in the frozen ridge npz by `run_ladder_stage3.py:1144-1150` as the
scaler CENTRING MARGIN. Its INVERSE is the standardiser's Lipschitz
constant — the naming matters, because the stored quantity is not itself
the constant. Read back 2026-08-10 (`90410e4`).

### 3. Stage 2B's own condition numbers — CONSUMED

`DESIGN.md:241` ordered a rung-2 diagnostic on Stage 2B's features precisely
because the Stage 2A transfer was "plausible, not established". It was run
and recorded: `FINDINGS.md:276-279` gives fold condition numbers
(pre_evolution ~6e14, T/lattice ~1.2-1.6e14, rewired/curr_random ~2-7e13,
raw_505/raw_784 14-490), and `FINDINGS.md:470-471` the rung-3 values
(T 4.78e6, lattice 4.82e8, with curr_random/rewired ~3x better than T).

So the in-domain measurement exists and no inference from Stage 2A is needed.
An earlier draft called Stage 2A's ~2e6 "very likely the same phenomenon" as
T's alpha collapse; that is too strong — different inputs, sample scale,
target and readout — and it is superseded rather than softened, because the
right number had already been measured.

**The ordering is not clean, and saying so is the point.** `lattice` is worst
on both conditioning (4.82e8) and combined norm (7.60e5), but `T` has WORSE
conditioning than `rewired`/`curr_random` while having the SMALLEST norm. The
two extremes agree; the middle does not. Reading a clean story into that is
the mistake that produced the withdrawn operator-norm claim.

### 4. Stage 1A's S(t) AUC — UNCONSUMED, correctly

`experiments/stage1a_infinitesimal_response/FINDINGS.md`, same graphs and the
same T=2.5 window. Range roughly 0.51-458.71. It is an integrated
TANGENT-RESPONSE measurement, not a terminal operator norm, so it is
qualitative evidence about the ODE link and cannot be composed into a chain
bound. Listed so the next reader does not try.

### 5. Fold-level Frobenius norms — UNCONSUMED, low value

`stage2b_ridge.py:520,541-542`, stored for CV fits. The wrong norm
(Frobenius, not induced L-infinity) on the wrong fits (folds, not the final
refit). A weak clue rather than an answer.

### 6. `ARTIFACT_MANIFEST.json`'s `frozen_results` — CONSUMED

The sharpest entry. It contains the locked Stage 4 verdict, the primary
bootstrap CI, per-graph effect sizes, `one_graph_wins`, and a disclosed
p-underflow. Every key of that file was walked at every depth TWICE, hunting
for absolute-path strings, without a single value being read. Reading it took
90 seconds and withdrew a published claim (2026-08-11).

## The adjacent failure: measurements that never reached the repository

The entries above were all recorded HERE and then not read. This one is the
neighbouring case, and it is worth separating: a measurement that was
computed and verified, but whose only home is a chat transcript and a
workspace file that was never landed. Everything above can be re-read by
anyone with the repo. This cannot be read at all.

### Accessible target energy (ρ=−1 against the Stage 4 ordering)

**Status: DEFINITION durable, VALUES not. Do not cite the values.**

Surfaced 2026-08-12, when it was proposed as a companion to the gauge
comparison and described as "the strongest mechanism clue in the record."
It is not in the record. A repo-wide search across every `FINDINGS.md` and
every `.py` returns nothing; the only `rho` in Stage 2B is the encoder
gate's ratio. It originates in an external mechanism report whose own
closing line stated that the Bonsai repository remained unchanged — so the
numbers were verified at the time, but custody was never tracked.

The definition, recorded here so the quantity can be recomputed rather than
guessed at:

> Take the standardized (scaled, centred) full feature matrix `X`, decompose
> `X = U S V^T`, project the centred clean targets `Z = U^T Y_centred`. Then
>
>     accessible target energy = ||Z||^2 / ||Y_centred||^2
>
> — the fraction of centred clean-target energy lying in the feature span.

A derivation is its own provenance (principle 24 says so explicitly), which
is why the definition can be written down here and trusted. The reported
VALUES cannot: they were produced by a generator that does not exist in this
repository, which is the exact shape of `class0_constructions.pkl` and of
every other entry principle 24 was written for. They are therefore **not**
reproduced here, not as a table and not in passing, because a number quoted
once acquires the authority of the document quoting it.

The remedy is the one principle 24 names — promotion to committed code, not
citation of the capture. It is a minutes-scale CPU recompute from the cached
features whenever anyone wants it in-repo. Until then the honest statement
is: this quantity is defined, plausibly informative, and unmeasured HERE.

It was deliberately NOT added to the gauge comparison mid-flight. That run
was already launched against a committed pre-registration, and adding a
companion after the fact would have made the registration mean less.

## 7. sklearn's own achieved gradient norms, re-derived by GPU sweep

**Recorded:** `JAX_CLASSIFIER_PORT_FINDINGS.md`, "Follow-up:
convergence-criterion recalibration (evolved_T)", reproducible from
`diagnose_classifier_jax_grad_norm_calibration.py`.

**What it says:** a nine-row table of `||grad||` at sklearn's own converged
solution for every C in the locked grid, and the normalised
`||grad||/(C*n_train)`, which is tight over **[1.338e-3, 2.771e-3]**.
`GRAD_NORM_REL = 6e-3` was set at ~2.2x the top of that range. The same
document logs the consequence as an open item: *the large-C validation-loss
curve still diverges substantially from sklearn's -- the sole remaining open
item for this port.*

**The question that needed it (2026-08-12):** the GPU gauge replication found
JAX and sklearn apart by up to 1.2e-02 at high C, and a tolerance sweep was
designed to find the `GRAD_NORM_REL` at which they agree.

**What reading it first would have changed:** the sweep grid. It was set to
`6e-3, 1e-3, 1e-4, 1e-5, 1e-6` on the guess that matching sklearn needed a
much tighter threshold. The table gives the answer directly -- sklearn sits at
~2e-3 relative, so the crossing is between the first two grid points and
everything below 1e-3 is measuring convergence DEEPER than sklearn ever
reaches, not JAX catching up. The divergence itself was already logged as a
known open item, not a discovery.

**What the sweep still adds, and why it was not wasted:** the platform
disambiguation (JAX-CPU vs JAX-GPU vs sklearn, ruling out TF32 and GPU
numerics entirely), the dose-response of the gap against selected C across
ten arms, and the consequence for a published verdict -- the class C call in
`GAUGE_COMPARISON_2A_RESULT.md` does not survive the optimiser change. None of
that is in the record.

**The specific miss:** this document's own closing line is "before measuring
something, grep for it." The measurement was designed, a GPU was provisioned
and the job submitted before anyone grepped. The table surfaced only because
a code-intelligence tool volunteered the neighbouring diagnostic's name in an
unrelated notification.

## What this is not

Not a proposal to build anything. A detector for "this record answers that
question" is a harder problem than the pattern it would catch, and the
retrospective's own finding is that machinery justified by principle rather
than by a caught defect is the category that scored zero.

The useful response is cheaper: before measuring something, grep for it.
