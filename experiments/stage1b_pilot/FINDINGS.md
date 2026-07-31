# Stage 1B Pilot: Complete, With Two Corrections That Change the Interpretation

## What was run

72 trials: 1 class (KMNIST class 0), 2 initial conditions, 3 nodes
(low/median/high weighted-degree in T), both perturbation signs, 6
amplitudes (0.025 to 0.8), T alone (per the capability-first design).
Continuous-integration classification (primary window + extension,
single solve_ivp call, avoiding the restart-seam artifact found and
fixed before this run began).

## Correction 1: the reproducibility question was framed too broadly

The original write-up treated failure to reproduce across two *arbitrary*
initial conditions as evidence against structured capability generally.
That overstates what was tested. Reproducibility across arbitrary states
is necessary only for a *fixed* transformation (node, sign, amplitude) ->
response, independent of system state. Many real computational systems
are state-dependent -- memory, attention, gain control, attractor
selection. A multistable network producing different transformations
from different initial conditions is not necessarily failing to compute;
its current state may be part of the input. The defensible claim from
the original 72-trial comparison is narrower and different in kind:

**Nonlinear response is not a stable function of node degree, perturbation
sign, and amplitude alone. It is not yet established, or ruled out,
whether response is a stable function of dynamical state instead.**

## Correction 2a: an implementation bug in E(t), now fixed, plus corrected terminology

The previously defined tangent-departure metric,
E_eps(t) = ||P*Delta_theta_eps(t) - eps*P*delta(t)|| / (|eps|*||P*delta(t)|| + eta),
is a ratio of two norms and therefore mathematically non-negative by
construction. The originally reported values (-0.693, -1.255, -1.093)
could not be this quantity -- and were not: the implementation used
`epsilon` rather than `abs(epsilon)` in the denominator, so for the
negative-sign trials tested, the entire ratio flipped sign spuriously.
This has been fixed (denominator now uses |epsilon|), and this is
properly named as **vector-relative departure error**, not simply "E(t),"
to avoid ambiguity with the signed-magnitude alternative the review
proposed. Recomputed on the same case:

| Epsilon | Peak S | Vector-relative departure error (0,5,10,20,30,40,50) | Directional cosine similarity C(t) (same steps) |
|---|---|---|---|
| -0.025 | 865.68 | 0, 0.005, 0.005, 0.010, 0.028, 0.120, 0.693 | 1.000, 1.000, 1.000, 1.000, 1.000, 0.999, 0.991 |
| -0.8 | 18.08 | 0, 0.176, 0.291, 0.419, 0.481, 1.255, 1.093 | 1.000, 0.987, 0.958, 0.989, 0.961, **-0.789**, **-0.614** |

The magnitudes are unchanged from the earlier (buggy) report -- only the
erroneous sign is corrected -- so the substantive conclusion holds:

**At small epsilon, the actual trajectory closely tracks the tangent-
linear prediction throughout** (C stays at 0.9997-1.0000 for nearly the
entire window, departure error stays small until the very end) --
confirming the review's hypothesis directly: the striking peak of 865.68
is predominantly tangent-linear transient growth, not evidence of
finite-amplitude nonlinear capability. **At large epsilon, genuine
nonlinear reorganization appears**: C(t) goes negative late in the
trajectory -- the actual displacement direction has reversed relative to
the tangent-linear prediction -- and the departure error exceeds 1 (the
actual displacement's mismatch with the tangent prediction is now larger
than the tangent-predicted displacement itself). This is real nonlinear
behavior, but it shows up specifically where peak S is *smaller* (18.08,
not 866), because the system saturates and reorganizes rather than
continuing to amplify along the tangent direction. The directional-
reversal conclusion rests on C(t) < 0 specifically, which was never
affected by the sign bug -- the departure-error correction changes the
reported numbers but not which claim they support.

## Correction 2b: outcome breakdown prose error (the underlying counts were correct)

The stated fractions ("22 of 24," "12 of 24") used the wrong denominator
in prose -- there are 36 trials per initial condition (3 nodes x 2 signs
x 6 amplitudes), not 24, and the underlying counts were always internally
consistent: IC=2000 gives persistent-transient=22, decayed=12,
baseline-only-converged=2, summing correctly to 36; IC=2001 gives
decayed=32, persistent-transient=3, baseline-only-converged=1, also
summing to 36. The error was in how this was described, not in the
classification itself. Corrected statement: **IC=2000 shows 22 of 36
trials persistent-transient and 12 of 36 decayed; IC=2001 shows the
reverse concentration, 32 of 36 decayed and only 3 of 36
persistent-transient** -- consistent with the state-dependence framing
throughout this document, not a new finding.

A further improvement the review correctly identifies: "persistent
transient" as currently defined risks conflating a transient phenotype
(did the trajectory show large transient separation) with a terminal
classification (did it end at the same attractor). A trajectory can
transiently amplify substantially and still terminate at the same fixed
point as its baseline -- which is exactly what the outcome taxonomy
already shows happened in every persistent-transient case (all 25 such
trials across both ICs terminated at the same attractor; none are
recorded as different-equilibria or non-convergent). Separating these
into two explicit fields (transient phenotype; terminal outcome) would
be a real improvement to the taxonomy and is noted as a design change for
the next stage, not implemented retroactively on this pilot's data.

## The outcome taxonomy, computed directly (previously missing)

| Outcome | Count (of 72) |
|---|---|
| Decayed to same attractor | 44 |
| Persistent transient, same attractor | 25 |
| Baseline-only converged (asymmetric) | 3 |
| Different equilibria | **0** |
| No equilibrium recovered within horizon | **0** |

**No trial produced two recovered but distinct phase-locked
equilibria.** In 69 of 72 trials, both trajectories converged to the same
recovered attractor. In three trials, only the baseline trajectory
converged within the prespecified horizon -- the perturbed trajectory did
not. No trial had both trajectories fail to converge. This is a decisive,
direct answer to the question the original write-up left open: this
pilot found only transient gain variation and, in three cases, asymmetric
convergence -- no durable attractor-level difference was established
between baseline and perturbed trajectories, anywhere in the tested
amplitude range. This stronger claim ("never") is supported only for
the 69 jointly converged trials; the three asymmetric cases' eventual
attractors remain unresolved within the observation horizon, not shown
to be the same or different. Broken down by
IC (36 trials each, mutually exclusive categories): IC=2000 shows 22 of
36 persistent-transient and 12 of 36 decayed (plus 2 baseline-only-
converged); IC=2001 shows the reverse concentration, 32 of 36 decayed
and only 3 of 36 persistent-transient (plus 1 baseline-only-converged)
-- consistent with the
state-dependence framing above, not adding a new phenomenon.

## Revised capability hierarchy

Three distinguishable levels, following the review's framing directly:

1. **Fixed transformation** -- (node, sign, amplitude) -> response,
   independent of state. **Rejected** by this pilot's cross-IC comparison.
2. **State-conditional transformation** -- (state, node, sign, amplitude)
   -> response, with state as a genuine variable. **The leading
   hypothesis** -- consistent with both the cross-IC inconsistency and
   Stage 0's multistability finding, but not yet directly tested.
3. **Unstructured sensitivity** -- nearby or similar states produce
   unrelated outcomes with no stable conditional mapping. **Not yet
   distinguished from (2)** -- this is the critical open question, and
   only a state-conditioned redesign (grouping initial conditions by
   dynamical descriptors before comparing response maps) can resolve it.

## Revised primary conclusion

**The Stage 1B pilot rejects a state-independent amplitude-response law
for the learned topology. Small-amplitude extreme amplification can be
explained predominantly by tangent-linear growth, while at least one
large-amplitude condition produces genuine finite-amplitude directional
reorganization. No jointly converged trial reached a different
phase-locked equilibrium, so the observed nonlinear effects are
principally transient. Three perturbed trajectories did not meet the
convergence criterion within the prespecified horizon despite
convergence of their paired baselines. The pilot therefore establishes
state-dependent nonlinear transient behavior, but not yet structured
computational capability. Stage 1B.2 will test whether these transient
responses reproduce from controlled nearby states and whether they
implement consistent spatial information redistribution.**

The reported peak-amplification values combine tangent-linear transient
growth with finite-amplitude effects, confirmed directly above; nonlinear
capability claims depend on departure from the tangent prediction,
directional reorganization, and terminal-attractor outcomes -- reported
separately in this document, not collapsed into the single peak-S number
the original write-up leaned on. Response magnitude and its relationship
to node degree vary sharply between the two sampled initial conditions --
because the network is multistable, this heterogeneity may represent
state-conditional nonlinear *behavior* (not yet established as
*computation* -- see the three-way distinction below) rather than
irreproducibility in the ordinary sense.

## The amplitude-response map

| IC | Node | Sign | Peak amplification across amplitude grid (0.025→0.8) | Dominant outcome |
|---|---|---|---|---|
| 2000 | low | + | 11.9, 11.9, 11.9, 12.1, 12.3, 7.9 | persistent transient |
| 2000 | low | - | 11.9, 11.9, 11.9, 11.7, 10.1, 7.9 | persistent transient |
| 2000 | median | + | 242.9, 162.8, 86.3, 34.5, 19.0, 15.8 | persistent transient |
| 2000 | median | - | 865.7, 2860.4, 1025.1, 266.8, 70.6, 18.1 | persistent transient |
| 2000 | high | + | 1.0, 1.0, 1.0, 1.0, 1.1, 1.3 | decayed |
| 2000 | high | - | 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 | decayed |
| 2001 | low | + | 1.0, 1.0, 1.0, 1.0, 1.0, 6.6 | decayed (mostly) |
| 2001 | low | - | 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 | decayed |
| 2001 | median | + | 7.8, 7.6, 7.2, 6.5, 5.3, 3.5 | decayed |
| 2001 | median | - | 8.2, 8.4, 8.8, 9.6, 11.0, 20.6 | decayed (mostly) |
| 2001 | high | + | 7.9, 7.6, 7.1, 6.1, 4.6, 2.8 | decayed |
| 2001 | high | - | 8.5, 8.9, 9.7, 11.3, 15.5, 36.7 | decayed (mostly) |

## The reproducibility check, applied directly

The original state-independent response hypothesis predicts that
comparable node, sign, and amplitude conditions should produce similar
response maps across initial conditions. The two sampled initial
conditions provide a minimal direct test of that specific hypothesis --
not a general claim that arbitrary-state reproducibility is the
universal capability criterion (per Correction 1 above). Checking this
directly:

**Low node**: IC=2000 shows sustained, substantial amplification (peak
~8-12) across nearly every condition. IC=2001 shows almost none (peak
pinned at 1.0) across nearly every condition, with one exception at the
largest tested amplitude. This does not reproduce.

**Median node**: both ICs show real deviation from peak=1.0, but the
*magnitude* differs by roughly two orders of magnitude (IC=2000 peaks
at 2860; IC=2001 peaks at 20.6). Even granting both as "real transient
effects," the scale is not reproducible.

**High node**: this is the sharpest failure of reproducibility. IC=2000's
high node shows essentially no amplification anywhere in the grid
(peak 1.0-1.3). IC=2001's high node shows the *opposite* -- substantial,
amplitude-increasing amplification (up to 36.7), the same qualitative
shape (growing with |epsilon|) that IC=2000's median node showed. The
node-degree correlation that looked clean and orderly within IC=2000
(low/median amplify, high decays immediately) does not merely fail to
replicate in IC=2001 -- it partially inverts.

## What this means, stated precisely

**Capability is unresolved, not disproved.** The original framing --
"reproducible structured transformation is not established" -- was too
close to treating cross-IC inconsistency itself as the negative result.
The corrected framing: a *fixed*, state-independent (node, sign,
amplitude) -> response mapping is rejected by this pilot -- the same
node's behavior changes too much between IC=2000 and IC=2001 to support
that simple model. But a *state-conditional* mapping, where the network's
current position in its multistable landscape is itself part of what
determines the response, remains fully consistent with everything
observed, including the sharp differences between ICs and the partial
inversion at the high-degree node. That hypothesis has not been tested
directly -- doing so requires characterizing each initial condition's
dynamical state before comparing response maps, not simply noting that
two arbitrary states gave different answers.

The peak-amplification numbers used to describe IC=2000's "striking
pattern" also needed the tangent-departure check applied above: the
median node's dramatic small-amplitude peaks (up to 2860) are shown
directly to be predominantly tangent-linear, not evidence of finite-
amplitude capability at all. The genuine nonlinear signal in this
dataset -- directional reversal, confirmed via negative C(t) -- appears
at large amplitude, where peak S is actually smaller. Any future
capability claim needs to be built on E(t)/C(t)/attractor-outcome
evidence like this, not on peak S alone.

This is consistent with, and a direct behavioral consequence of, Stage
0's multistability finding: five distinct stable equilibria were found
from five initializations of this same class's topology. If finite-time
nonlinear response depends on which region of a richly multistable phase
landscape a trajectory currently occupies, then two different initial
conditions producing different amplitude-response behavior is exactly
what a state-conditional system should do, not a failure to reproduce in
the ordinary sense.

## What this does not mean

This is not evidence that T (or the oscillator dynamics generally) is
incapable of structured nonlinear transformation -- it is evidence that
*this* two-initial-condition pilot did not find a pattern that holds
independent of state. A pattern real within one trajectory but not the
next could still reflect genuine, interesting structure (e.g., basin-
dependent response character) rather than mere noise -- but establishing
that would need a design built around characterizing basin-dependence
directly, not the current amplitude-response comparison across only two
shared initial conditions.

## Honest limitations

- Two initial conditions is a minimal test of reproducibility, not a
  thorough one -- it can detect gross non-reproducibility (which it did)
  but cannot characterize how common the IC=2000-like pattern is, or
  whether some third IC would look like neither.
- One class only (KMNIST class 0) -- nothing here speaks to whether other
  classes' topologies behave similarly or differently.
- T only, no graph controls yet -- per the capability-first design, this
  question was correctly deferred, but it means nothing here yet
  addresses whether learned topology specifically shapes this behavior;
  that question doesn't arise until capability itself is established.
- The "decayed" classification uses the prespecified S<0.05 and RMS<0.01
  criteria over the last 20% of a 25-unit extension horizon -- this is a
  meaningful operational definition, but a different horizon or threshold
  could in principle classify some of these borderline cases differently.

## What the corrected pilot establishes, precisely

**Established:**
- Response is not determined by node degree, sign, and amplitude
  independently of state.
- Small-amplitude extreme amplification can be predominantly
  tangent-linear.
- At least one large-amplitude case shows strong directional departure
  from tangent prediction (negative C(t)).
- No distinct phase-locked equilibrium was recovered in any of the 72
  trials.
- Three trials produced asymmetric convergence within the prespecified
  horizon.
- The tested nonlinear behavior is therefore principally transient, not
  attractor-switching -- this narrows the scientific question rather than
  closing it.

**Not established:**
- That directional reorganization is reproducible within similar
  dynamical states.
- That it implements an information transformation rather than generic
  state sensitivity.
- That recovered-attractor identity is sufficient to predict response.
- That T differs from controls (deferred correctly, per the
  capability-first design -- this question doesn't arise until
  capability itself is resolved).

## A sharper capability criterion for the next stage

A future capability claim should require all three, not any one alone:

1. **Nonlinear departure** -- finite response departs materially from
   tangent prediction (the departure-error and directional-cosine
   diagnostics above).
2. **Conditional reproducibility** -- similar, controlled dynamical
   states produce similar response maps.
3. **Structured transformation** -- the response has identifiable
   organization (consistent directional redistribution, selective node
   recruitment, thresholded transient routing), not merely detectable
   nonlinearity.

Attractor switching is not required for a future capability claim. A
transient transformation could be computationally meaningful even if the
system later returns to the same attractor, provided nonlinear
departure, conditional reproducibility, and structured information
redistribution are all demonstrated -- not yet the case here.

This distinction is central to the project and the three levels cannot
be collapsed into each other: (1) nonlinear behavior, (2) structured
transformation, (3) useful computation. This pilot establishes the
first, offers preliminary evidence relevant to the second (the
directional-reversal signature), and establishes neither of the latter
two. Even a perfectly reproducible response map -- a stable
(state, impulse) -> response mapping -- would still only be a statement
about sensitivity, not about information transformation, without a
prespecified measure of what the response does spatially (see Stage
1B.2's added spatial-redistribution endpoint below).

## Immediate next steps: Stage 1B.2, a controlled state-conditioning design

The previous plan (sample many initial conditions, group post hoc by
dynamical descriptors) is at risk of becoming exploratory and
underdetermined -- with many candidate descriptors and relatively few
initial conditions, almost any grouping could appear to explain some
response variation. A controlled design is preferable, and is adopted
here in its place:

**A. Sample baseline trajectories.** Generate several unperturbed
trajectories for class 0; for each, recover terminal attractor identity,
force norm, coherence, and tangent amplification profile.

**B. Perturb at multiple controlled times along the same trajectory**,
rather than only at t=0: apply the identical node, sign, and amplitude
perturbation at prespecified times t in {0, t1, t2, t3} along one
baseline. This directly tests whether response changes with the
network's current dynamical state while holding the graph, original
trajectory, node, sign, and amplitude fixed -- a substantially cleaner
test of state-dependence than comparing unrelated random initial
conditions.

**C. Add nearby-state replication**: for selected baseline states,
create small random gauge-corrected perturbations around the state
before applying the experimental impulse, then check whether nearby
states produce similar departure curves, directional-cosine curves, and
terminal classifications. This is what actually distinguishes stable
state-conditional transformation from hypersensitive, unstructured
response.

**D. One primary conditioning variable, prespecified**: location along a
baseline trajectory (perturbation time) as primary. Recovered attractor
identity is **not** used to define dynamical-state equivalence -- the
pilot found principally transient effects, and all jointly converged
pairs reached the same attractor, so two baseline states headed toward
the same eventual equilibrium could still occupy very different regions
of phase space with different local response operators. "Same
destination" is not "same computational state." Attractor identity is
instead retained as a **cross-trajectory stratification variable** (for
grouping baselines by which equilibrium they eventually reach, when
comparing across different baseline trajectories), while nearby-state
replication (step C) is what actually tests local robustness at a given
point along one trajectory. Other descriptors (force norm, coherence,
Jacobian spectrum, etc.) remain secondary, explanatory measurements, not
additional primary axes to search across.

**E. A reduced response grid**: for state-conditioned replication (as
opposed to the apparatus-testing role the six-amplitude grid served in
this pilot), one tangent-consistent amplitude, one intermediate
amplitude, and one amplitude already shown to produce directional
reorganization -- concentrating compute on replication rather than
re-confirming smooth amplitude trends already established here.

**F. A prespecified spatial endpoint -- the missing piece for a genuine
computation claim, now fully specified rather than left as a list of
options.** Steps A-E, however well controlled, can only establish a
reproducible dynamical mapping (state, impulse) -> response. That is not
yet an information-processing claim: even a perfectly reproducible
departure-error and directional-cosine curve remains a statement about
sensitivity, not transformation, without a measure of what the response
does spatially. Four details are locked here, not left to be decided
after seeing results:

*Energy discards sign -- keep both representations.* Normalized nodewise
energy q_j(t) = x_eps,j(t)^2 / sum_k(x_eps,k(t)^2) measures where
displacement is concentrated but not its direction -- two responses with
opposite sign at every node produce identical q(t), which would erase
exactly the directional-reversal signature that was this pilot's
strongest nonlinear finding. Retain a signed companion
r(t) = x_eps(t) / ||x_eps(t)||_2, and use signed cosine similarity
between r(t) vectors to track direction, alongside q(t) for energy
redistribution.

*Evaluation time is prespecified, not chosen after inspection -- and
event-aligned timing needs a fixed-time check.* "Terminal or peak
redistribution pattern" left a researcher degree of freedom, since peak
separation, peak tangent departure, and peak spatial divergence can occur
at different times. Locked rule: evaluate spatial output at the time of
maximum vector-relative departure from tangent prediction within the
primary response horizon, t*_eps = argmax_t E_eps(t) for t in [0, T]. The
primary spatial output is q_eps(t*_eps) (and its signed companion
r_eps(t*_eps)). **But t*_eps is input-dependent -- different inputs may
have their strongest departure at different physical times, so comparing
across inputs purely event-aligned risks reflecting timing differences
rather than distinct spatial transformations.** One fixed-time robustness
endpoint is added alongside it: q_eps(T), evaluated at the end of the
prespecified primary horizon (a common elapsed time across every input).
The event-aligned result (t*_eps) remains primary; the fixed-time result
checks that input separation isn't created solely by comparing different
moments. Other secondary summaries -- time-integrated redistribution,
maximum JSD from tangent propagation -- may also be reported, but neither
substitutes for these two prespecified time points.

*t* needs protection against negligible departure, since argmax always
returns some time even when nothing nonlinear happened.* For the
tangent-consistent amplitude specifically, E_eps(t) may stay extremely
small throughout the horizon -- an unrestricted argmax will still return
a time, even when the maximum is numerical drift rather than a
scientifically real deviation. A prespecified departure threshold is
added: E_eps(t*_eps) >= E_min. Where the maximum does not exceed E_min,
the response is classified as tangent-consistent; t*_eps is not
interpreted as a nonlinear event; the event-aligned nonlinear spatial
endpoint is reported as undefined for that trial (not silently computed
anyway); and only the fixed-time spatial output and ordinary
finite-response pattern are retained. E_min is calibrated from repeated
solver tolerances or duplicated integrations under identical conditions
(i.e., the scale of purely numerical variation), not chosen from the
Stage 1B.2 result distribution itself -- calibrating it after seeing
results would let the threshold be tuned to produce a preferred
classification.


*Two distinct JSD-based quantities, named separately so "primary JSD"
never means both at once.* This design uses Jensen-Shannon divergence in
two related but scientifically different roles, which must be named and
reported separately in both code and text:

- **Tangent-relative nonlinearity, J_tan(t) = JSD(q_finite(t),
  q_tangent(t))**: how much has the finite response redistributed
  spatially relative to first-order propagation? This is a
  nonlinear-departure endpoint -- it establishes that the response is
  nonlinear, but says nothing by itself about whether the mapping is
  input-sensitive.
- **Output-map distance, d_q(a,b) = sqrt(JSD(q_a, q_b))**: how different
  are the spatial outputs produced by two separate trials? This supplies
  W, B, and Delta_map (below) -- the actual capability test.

The central Stage 1B.2 capability test is Delta_map > 0, using d_q, not
J_tan. J_tan is a necessary diagnostic (confirming genuine nonlinear
departure exists at all) but is not a substitute for the input-sensitivity
test itself. Signed cosine similarity (direction) and top-k node overlap
(recruitment of node subsets) remain secondary diagnostics alongside
both. Where the displacement norm falls below a fixed numerical
threshold, q(t) is undefined and must be reported as such, not stabilized
into an apparently meaningful distribution via an arbitrary denominator
constant.

*Conditional reproducibility is quantitative, not "similar by eye" --
now with an exact distance, exact definitions, and an exact permutation
scheme.* For the primary energy-distribution output, use the square root
of Jensen-Shannon divergence as the output-space distance,
d_q(a,b) = sqrt(JSD(q_a, q_b)) -- the square root is a metric, whereas
raw JSD is a divergence and does not satisfy the triangle inequality.
Within each controlled state neighborhood, define:

- W = (1/|P_same|) * sum over (a,b) in P_same of d_q(a,b), for replica
  pairs receiving the *same* input;
- **B, balanced across factors, not naively pooled**: "different input"
  combines differences in node, sign, and amplitude, which have unequal
  numbers of possible pairs -- pooling them directly would let whichever
  factor has the most pairs dominate B through combinatorics alone,
  rather than through a genuinely larger effect. The primary omnibus
  between-input distance is instead
  B = (1/3) * (B_node + B_sign + B_amplitude), where each component
  averages comparisons differing in that one factor while matching the
  others wherever possible (equivalently, sampling equal numbers of
  pairwise contrasts from each factor family is an acceptable
  alternative). The three factor-specific separations (B_node, B_sign,
  B_amplitude) are also reported secondarily -- they may show the system
  distinguishes perturbation location but not sign, or amplitude but not
  node, which matters for characterizing what kind of transformation has
  emerged, even though the balanced omnibus B is what feeds the primary
  test;
- effect size Delta_map = B - W.

For permutation inference, shuffle input labels **only within** the same
baseline state, perturbation time, and nearby-state replica block -- never
across unrelated state neighborhoods, which would destroy the
conditioning structure the experiment is designed to test. The primary
capability test is then exact: H0: Delta_map <= 0, versus H1:
Delta_map > 0. A structured, input-sensitive state-conditional mapping
requires rejecting H0 -- outputs must be more similar across repeated
presentations of the same input from nearby states than across distinct
inputs from the same state neighborhood. This is a stronger and different
claim than showing each response is merely reproducible in isolation: it
establishes input sensitivity alongside robustness, which is what
distinguishes a genuine mapping from noise that happens to look stable.

*Nearby-state perturbation scale is calibrated, not arbitrary.* Choose
the largest gauge-corrected nearby-state perturbation whose *unforced*
trajectory remains within a prespecified RMS distance of the reference
baseline over a short pre-impulse validation interval -- this makes
"nearby" operational (genuinely distinct above numerical precision, still
local to the baseline, unlikely to cross into a different dynamical
regime before the experimental impulse is applied) rather than a
geometric guess. The nearby-state perturbations themselves must be
zero-mean, unit-normalized before scaling, generated from a fixed seed,
and identical across different experimental inputs for a given replica --
otherwise input comparisons could be confounded by different neighborhood
samples rather than by the inputs actually being compared.

This is the first genuine capability test this project will have run:
does the dynamical system map distinguishable local inputs into
reproducible, structured response patterns conditional on its current
state -- not merely does it respond differently depending on where it
starts.

This tests the actual leading hypothesis directly and controllably:
**does a controlled dynamical state transform local perturbations into
reproducible, input-sensitive spatial response patterns, rather than
merely exhibiting nonlinear sensitivity?** That distinction, not a larger
uncontrolled sample, is what the next stage needs to resolve.

## Final pilot status

Stage 1B established state-dependent finite-amplitude nonlinear
transient behavior, but not structured internal transformation or useful
computation. Extreme normalized amplification at small amplitude may
remain primarily tangent-linear, while sufficiently large perturbations
can reorganize the response direction. No jointly converged pair reached
distinct phase-locked equilibria, and three perturbed trajectories
remained unconverged within the prespecified horizon. Stage 1B.2 will
determine whether controlled local dynamical states map distinguishable
perturbation inputs into robust, reproducible, and spatially structured
outputs.

That preserves the research hierarchy:

**nonlinear response != structured internal transformation != useful computation**

(a hierarchy of non-equivalence, not a sentence to be read continuously)
-- and Stage 1B.2 is the first direct test of the middle term.

**Stage 1B.2 design locks (conceptual):**

1. Perturbation times along the baseline trajectory.
2. Three amplitudes (one tangent-consistent, one intermediate, one
   already shown to produce directional reorganization).
3. Nearby-state perturbation scale and replication count.
4. The time-selection rule (t*_eps = argmax E_eps(t)), protected by a
   departure threshold E_min below which t*_eps is not interpreted as a
   nonlinear event.
5. Two separately-named JSD-based quantities: J_tan(t) for tangent-
   relative nonlinearity (diagnostic), d_q(a,b) = sqrt(JSD) for
   output-map distance (feeds the primary test).
6. Signed directional similarity (r(t)) as secondary, preserving the
   directional-reversal signal that energy-only q(t) would erase.
7. A quantitative within-state-versus-between-input reproducibility
   criterion (Delta_map = B - W, with B balanced across node/sign/
   amplitude factors rather than naively pooled; H0: Delta_map <= 0 vs.
   H1: Delta_map > 0; block-restricted permutation).
8. A fixed-time robustness endpoint (q_eps(T)) alongside the
   event-aligned primary time t*_eps.
9. An operational calibration rule for nearby-state perturbation scale.

**Exact numerical values still to record before execution** (parameter
choices within this already-settled design, not further conceptual
decisions):

- The perturbation times along the baseline trajectory.
- The three actual amplitudes.
- The primary horizon T.
- The fixed-time endpoint.
- The numerical norm threshold below which q is undefined.
- The nonlinear departure threshold E_min (calibrated from solver
  tolerance / duplicated integrations, not from Stage 1B.2 results).
- The nearby-state pre-impulse validation horizon.
- The maximum allowed unforced RMS divergence (defining "nearby").
- The candidate nearby-state scale grid.
- The number of nearby replicas.
- The permutation count or exact-enumeration rule.
- The top-k value for the secondary recruitment diagnostic.

At that point the experiment will test **structured internal
transformation**, not merely catalogue nonlinear sensitivity -- this is
level 2 of the durable three-way distinction (nonlinear response;
structured internal transformation; useful computation), and it is much
closer to a computation claim than anything tested so far in this
project. It does not yet establish level 3: demonstrating usefulness will
still require linking the resulting response patterns to an externally
defined task or information-processing objective, which is a separate
question from whether the transformation is structured and reproducible
in the first place.

## DOCUMENT STATUS: FROZEN

This document is frozen as the final Stage 1B pilot record and the
conceptual pre-registration for Stage 1B.2. No further conceptual
redesign is needed before implementation. The only remaining work is
numerical calibration -- the table below -- which should be the entire
content of the next review, filled in without examining any Stage 1B.2
output differences.

## Stage 1B.2 numerical calibration table -- AMENDED AND LOCKED

All values below were calibrated using only (a) the already-closed Stage
1B pilot's own validated results, (b) Stage 0's prior diagnostics, and
(c) dedicated numerical-repeatability and unforced-trajectory checks run
for this calibration alone -- no Stage 1B.2 experimental trial (baseline
perturbed at t_p, compared across inputs) was run or inspected before
every row was fixed.

**Blocking issue, resolved: Option A locked.** Node was previously locked
to a single value while the design requires B_node as one of three
balanced factors -- with only one node, B_node cannot be computed and the
experiment cannot test whether outputs distinguish perturbation location.
Locking the median node specifically (because it showed the clearest
pilot signal) also risked carrying result-guided selection into Stage
1B.2, which the design is built to avoid elsewhere.

**Option A is locked: low, median, and high weighted-degree nodes in T.**
3 nodes x 2 signs x 3 amplitudes = 18 inputs x 4 t_p x 6 replicas = **432
finite-response trials**. Node location is not incidental to this
experiment -- it is the spatial identity of the perturbation, and testing
whether that identity maps into distinguishable spatial outputs is
central to the structured-transformation hypothesis this stage exists to
test. Removing it (Option B) would answer a materially weaker question
for a smaller compute saving than the science is worth. The additional
cost is justified because this is the first direct test of structured
internal transformation in this project, and cutting the input space down
to avoid it would undercut the reason for running Stage 1B.2 at all.

| Parameter | Candidate values tested | Selection rule | Locked value |
|---|---|---|---|
| Perturbation node(s) | -- | Per the resolved node contradiction above (Option A) | low, median, high weighted-degree nodes in T -- **not** the single median node previously locked, which risked result-guided selection |
| Perturbation times | -- | Evenly-spaced fractions of the primary horizon, fixed independent of any response data | t_p in {0, 0.833, 1.667, 2.5}, measured on the **baseline trajectory's own clock** |
| Response horizon | -- | Stage 1B response coverage (already shown to capture both tangent-consistent and nonlinear regimes) | T = 2.5, measured as **tau, elapsed time since each t_p** -- i.e. for perturbation at t_p, response is observed over absolute baseline time [t_p, t_p+2.5], not a fixed absolute endpoint. The t_p=2.5 condition therefore has a genuine response window ([2.5, 5.0]), not zero. |
| Fixed-time spatial endpoint | -- | Common elapsed time across every input, on the tau clock | q_eps(tau=T), not q_eps(T) on the baseline clock |
| Event-aligned time | -- | argmax over the post-impulse response window | tau*_eps = argmax over tau in [0,T] of E_eps(tau) |
| Amplitudes | 0.025, 0.05, 0.1, 0.2, 0.4, 0.8 (Stage 1B pilot grid) | Tangent-consistent (pilot-confirmed C~1.0 throughout), intermediate, nonlinear (pilot-confirmed C<0) | 0.025 (tangent), 0.2 (intermediate), 0.8 (nonlinear) |
| q-norm threshold | -- | Displacement-norm noise floor, standard vs. tight solver tolerance (1e-6/1e-8 vs. 1e-10/1e-12), with safety margin | 1e-6 (~4 orders of magnitude above the measured 7.89e-11 floor) |
| E_min | -- | Duplicate-solve error envelope, same tolerance comparison applied to E(t) directly | 1e-4 (~30x the measured 3.66e-6 floor, comfortably below observed tangent-consistent E values of 0.002-0.004) |
| Nearby-state scale | 0.001, 0.005, 0.01, 0.05, 0.1, 0.2 -- **now tested at all four perturbation-time states with the same six fixed replica directions**, not just one state | Largest scale whose unforced-trajectory RMS divergence stays below the locked locality bound, at every t_p | 0.1 -- **verified**: max RMS across all 4 perturbation-time states and all 6 fixed replicas = 0.0085 (at t_p=0; 0.00445 at the other three), comfortably under the 0.01 bound at every state, not just the one originally tested |
| Validation horizon | -- | Fixed before scale testing, short relative to the primary horizon | 0.5 (time units, i.e. 1/5 of T) |
| RMS locality bound | -- | Principled, round, interpretable fraction of the full circular range (2*pi), fixed independent of the scale-testing results | 0.01 (~0.16% of the full circular range) |
| Replica count | -- | Adequate same-input pair count for permutation inference (C(R,2) pairs per state neighborhood) | R = 6 **perturbed replicas**, generated from a fixed seed, applied identically across all inputs for a given neighborhood. The unperturbed reference state is retained separately as a diagnostic and does **not** count as a seventh replica -- C(6,2)=15 same-input pairs uses the six perturbed replicas only. |
| Permutations | -- | Resolution appropriate to alpha=0.05, generous given permutation itself is cheap (reshuffling precomputed outputs, not re-integrating) | 10,000 (resolution to p=0.0001) |
| Top-k | -- | Fixed node fraction, scale-invariant across class topologies with different active-node counts | Top 5% of active nodes by response energy |

**Permutation algorithm -- the previous version was degenerate, now
fixed.** The prior fix ("apply one common random permutation of input
labels within that neighborhood, consistently across all six replicas")
does not generate a valid null distribution. A common relabeling across
every replica preserves which outputs share the same input label -- if
input A is renamed input F, it stays grouped with the same outputs across
every replica, just under a new name. The same-input pairs feeding W
remain the same actual pairs; the different-input contrasts feeding B
remain the same actual contrasts; Delta_map is unchanged by the
relabeling. The permutation distribution would be a degenerate point
mass at the observed value, incapable of testing whether the observed
input-output correspondence exceeds chance. This would have gone
completely unnoticed until 432 trials of data came back and the "test"
returned the same answer under every permutation -- exactly the kind of
statistical error that needs to be caught in calibration, not discovered
after the compute is spent.

**Corrected procedure**, testing the actual null hypothesis (output
patterns are not consistently associated with nominal input identity
across nearby replicas): for each perturbation-time neighborhood, (1)
keep the six replica identities fixed; (2) **within each replica
separately**, randomly permute the 18 input labels across its 18 outputs,
preserving the one-to-one label assignment inside that replica; (3)
recompute W, B_node, B_sign, B_amplitude, B, and Delta_map under this
independent relabeling; (4) repeat 10,000 times; (5) never exchange
outputs across replicas, perturbation times, or baseline trajectories.
Permuting independently per replica destroys consistent input identity
across replicas under the null (an output actually produced by input A in
replica 1 might be compared against an output actually produced by input
C in replica 2, as if they were "the same input") while preserving each
replica's complete output geometry, the repeated-measures block
structure, the observation count, and the balanced factorial input set.

**Exact one-sided Monte Carlo p-value**:
p = (1 + sum_{b=1}^{M} 1[Delta_map^(b) >= Delta_map^obs]) / (M + 1).
With M = 10,000, the smallest attainable p-value is 1/10,001 ~ 0.0001.

**Factor-specific permutations (secondary analyses)**: the omnibus,
independently-shuffled 18-label test is valid for the overall mapping
hypothesis. For factor-specific effects, use restricted permutations that
preserve the other two factors while testing the selected one: for
B_node, permute node labels within each matched sign-amplitude cell,
independently per replica; for B_sign, permute sign labels within each
matched node-amplitude cell; for B_amplitude, permute amplitude labels
within each matched node-sign cell. These are secondary analyses --
correct for the three factor-specific tests (e.g. Bonferroni, alpha/3)
if inferential p-values are reported for them, consistent with the
multiplicity discipline applied throughout this project.

**Balanced B definitions, clarified.** For each component, compare
outputs differing in exactly one factor while matching the others:
B_node = mean d_q[(n1,s,a),(n2,s,a)]; B_sign = mean d_q[(n,+,a),(n,-,a)];
B_amplitude = mean d_q[(n,s,a1),(n,s,a2)]. **Pairs are formed across
nearby replicas, not within the same replica** -- so B and W operate at
the same replication level. Otherwise W would measure cross-replica
variation while B partly measured within-replica variation, making
Delta_map = B - W harder to interpret cleanly (comparing apples to a
mix of apples and oranges). Concretely: W pairs the same input across
different replicas; each B_f pairs matched inputs differing only in
factor f, also across different replicas. Self-pairs are excluded; pairs
are unordered (d_q is symmetric, so (a,b) and (b,a) are the same pair,
counted once).

Before running on real data, this algorithm should be unit-tested on
synthetic data where identical input maps produce Delta_map approximately
0, and highly separated stable maps produce a clearly positive result --
confirming the test statistic and permutation scheme behave correctly
before trusting either on the actual experiment.

**Scope of inference, stated explicitly.** A fresh seed (3000) gives
*one* baseline trajectory and four controlled states along it -- adequate
for a local capability demonstration, but the four perturbation times are
repeated states along one trajectory, not four independent baseline
trajectories, and the inference is conditional on this specific
trajectory. A positive result would support: *along this prespecified
class-0 trajectory, nearby states implement an input-sensitive and
locally reproducible spatial mapping.* It would **not** yet support: *the
class-0 topology generally implements such a mapping* -- that would
require multiple independent baseline trajectories, not yet run. Delta_map
will be reported **by perturbation time as well as pooled** -- a pooled
positive result must not be allowed to conceal one highly informative
time point sitting alongside three null ones.

**Note on the nearby-state scale/locality-bound ordering**: the locality
bound (0.01) was chosen for its own principled interpretability (a small,
round fraction of the circular range) rather than derived from the
scale-testing results -- but both computations were run in the same
session before this table was written. This is disclosed rather than
presented as a fully blinded pre-registration; the bound was not tuned
to make any specific candidate scale pass.

**Class and baseline seed**: KMNIST class 0 (consistent with the closed
pilot); a fresh initial-condition seed (3000) not used in Stage 1B or
its calibration, reserved for the Stage 1B.2 experimental run itself.

Every row above is now locked: Option A (three nodes, 432 trials), and a
corrected permutation scheme that actually generates a valid null
distribution. Stage 1B.2 can execute immediately -- no other numerical or
conceptual decisions remain, and no Stage 1B.2 experimental trial
(perturbing the baseline at
t_p and comparing across inputs) was run to produce any value in this
table.

## Reproducing these results

`run_stage1b_pilot.py` (checkpointed, resumable), `stage1b_taxonomy.py`
(classification machinery, integration-structure fix applied). Full
results in `stage1b_pilot_results.pkl`, per-trial log in
`stage1b_pilot_progress.log`.
