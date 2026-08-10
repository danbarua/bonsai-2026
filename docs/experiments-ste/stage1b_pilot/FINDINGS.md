Simplified Technical English version of `experiments/stage1b_pilot/FINDINGS.md`.

# Stage 1B Pilot: Complete. Two Corrections Change How We Read It

## What The Team Ran

The team ran 72 trials. Each trial used:

- 1 class: KMNIST class 0.
- 2 starting conditions (IC=2000 and IC=2001). Each starting condition
  sets one baseline trajectory. A trajectory is one full run of the
  oscillator network over time, from one starting point.
- 3 nodes, picked by node_label. Node_label means which of three nodes
  gets the push. The team picked the nodes by how strongly connected
  they are in the learned graph (T): low, median, or high.
- Both signs of the push (positive and negative).
- 6 amplitude values, from 0.025 to 0.8. Amplitude means the size of
  the push.
- Only the learned topology (T). The trials used no control graphs
  yet. This follows the capability-first design.

The team classified each trial using one continuous run of the
integrator (`solve_ivp`). This run covered the main time window plus
an extension, in a single call. This method avoids a known artifact
from restarting the integration partway through. The team found and
fixed this artifact before this run began.

## Correction 1: The Repeatability Question Was Too Broad

The first write-up made a claim that was too broad. It said the
results did not repeat across two different starting conditions. It
used this as evidence against structured capability in general. This
claim goes further than the test actually supports.

Repeatable results across any starting condition are necessary only
for one specific case. That case is a fixed transformation: node,
sign, and amplitude decide the response, with no dependence on the
network's current state.

Many real computing systems depend on their current state. Examples
include memory, attention, gain control, and attractor selection (the
process of settling into one of several possible stable states). A
network with several possible stable states, called a multistable
network, can give different transformations from different starting
conditions. This does not mean the network fails to compute. The
network's current state may be part of the input.

The correct claim from the original 72-trial comparison is narrower.
It is different in kind:

**The nonlinear response is not a fixed function of node degree, push
sign, and push size (amplitude) alone. The team has not yet shown, or
ruled out, that the response is instead a fixed function that also
depends on the network's current dynamical state.**

## Correction 2a: A Bug in E(t), Now Fixed, and a Clearer Name

The team had defined a metric to measure the gap between the real,
nonlinear result and the linear guess. The real, nonlinear result is
called the finite response: what actually happened after the push, the
real difference from the unperturbed run. The linear guess is called
the tangent response: the best straight-line prediction of what would
happen, cheaper to compute than the real result.

This gap metric is called E_eps(t). The formula is:

E_eps(t) = ||P*Delta_theta_eps(t) - eps*P*delta(t)|| /
(|eps|*||P*delta(t)|| + eta)

E_eps(t) divides one distance by another distance. Because of this,
E_eps(t) can never be less than zero, by the way it is built.

The team first reported values of -0.693, -1.255, and -1.093 for this
metric. These values cannot be correct, because E_eps(t) can never be
negative. The values were wrong because of a bug. The code used
`epsilon` in the denominator (the bottom of the fraction), instead of
`abs(epsilon)`, the size of epsilon with the sign removed. For trials
where the push had a negative sign, this bug flipped the sign of the
whole result by mistake.

The team fixed this bug. The denominator now uses `abs(epsilon)`. The
team also gives this metric a clearer name: **vector-relative
departure error**. This name avoids confusion with a different,
signed-magnitude version of the metric that a reviewer had proposed.

Recomputed on the same case:

| Epsilon | Peak S | Vector-relative departure error (0,5,10,20,30,40,50) | Directional cosine similarity C(t) (same steps) |
|---|---|---|---|
| -0.025 | 865.68 | 0, 0.005, 0.005, 0.010, 0.028, 0.120, 0.693 | 1.000, 1.000, 1.000, 1.000, 1.000, 0.999, 0.991 |
| -0.8 | 18.08 | 0, 0.176, 0.291, 0.419, 0.481, 1.255, 1.093 | 1.000, 0.987, 0.958, 0.989, 0.961, **-0.789**, **-0.614** |

The size of each corrected value is the same as before. Only the wrong
sign is now fixed. So the main conclusion still holds:

**At small epsilon (a small push), the real trajectory stays close to
the linear guess for the whole run.** C(t), the directional cosine
similarity, stays between 0.9997 and 1.0000 for nearly the whole
window. C(t) measures how closely the direction of the real change
matches the direction of the linear guess. A value of 1.0 means the
two directions match exactly. The vector-relative departure error also
stays small until close to the end.

This confirms what the reviewer expected. The striking peak value of
865.68 is mostly tangent-linear transient growth. It is not evidence
of true finite-amplitude nonlinear capability.

**At large epsilon (a large push), real nonlinear reorganization does
appear.** Late in the trajectory, C(t) becomes negative. This means
the real direction of change has reversed compared to the linear
guess. The departure error also goes above 1. This means the mismatch
between the real change and the linear guess is now larger than the
linear guess itself.

This is real nonlinear behavior. But it appears specifically where the
peak amplification (S) is *smaller* (18.08, not 866). This happens
because the system reaches a limit and reorganizes, instead of
continuing to grow along the direction the linear guess predicts.

The conclusion about directional reversal rests only on C(t) being
negative. The sign bug never affected this part of the result. The fix
to the departure error changes the reported numbers. It does not
change which claim the numbers support.

## Correction 2b: A Wrong Number In The Text (The Counts Were Correct)

The first write-up used the wrong total when it stated fractions. It
said "22 of 24" and "12 of 24." The correct total is 36 trials per
starting condition, not 24. This comes from 3 nodes x 2 signs x 6
amplitudes.

The actual counts were always correct and consistent:

- IC=2000: 22 persistent-transient, 12 decayed, 2
  baseline-only-converged. Total: 36.
- IC=2001: 32 decayed, 3 persistent-transient, 1
  baseline-only-converged. Total: 36.

The error was only in how the team described these numbers in the
text. It was not in the classification itself.

Corrected statement: **IC=2000 shows 22 of 36 trials as
persistent-transient and 12 of 36 as decayed. IC=2001 shows the
opposite pattern: 32 of 36 decayed and only 3 of 36
persistent-transient.** This matches the state-dependence idea used
throughout this document. It is not a new finding.

A reviewer identified a further improvement. The current definition of
"persistent transient" risks mixing two different things together:

1. A transient phenotype: did the trajectory show a large, temporary
   separation from the baseline?
2. A terminal classification: did the trajectory end at the same
   attractor, or stable state, as the baseline?

A trajectory can amplify substantially for a while and still end at
the same fixed point as its baseline. This is exactly what happened in
every persistent-transient case. All 25 persistent-transient trials,
across both starting conditions, ended at the same attractor as their
baseline. None of them are recorded as "different equilibria" or
"non-convergent."

Splitting these into two separate fields, transient phenotype and
terminal outcome, would improve the outcome classification system
(the taxonomy). The team notes this as a design change for the next
stage. The team has not applied this change to this pilot's existing
data.

## The Outcome Categories, Calculated Directly (Missing Before)

| Outcome | Count (of 72) |
|---|---|
| Decayed to same attractor | 44 |
| Persistent transient, same attractor | 25 |
| Baseline-only converged (asymmetric) | 3 |
| Different equilibria | **0** |
| No equilibrium recovered within horizon | **0** |

**No trial produced two recovered but distinct stable states
(phase-locked equilibria).** In 69 of 72 trials, both the baseline
trajectory and the perturbed trajectory reached the same stable state.
In 3 trials, only the baseline trajectory reached a stable state
within the fixed time window. The perturbed trajectory did not. In no
trial did both trajectories fail to reach a stable state.

This gives a clear, direct answer to a question the first write-up
left open. This pilot found only temporary changes in the size of the
response (gain), plus, in 3 cases, uneven convergence, where one
trajectory settled and the other did not. No trial showed a durable,
attractor-level difference between the baseline and perturbed
trajectories, at any tested push size.

This stronger claim, "never," is supported only for the 69 trials
where both trajectories converged. For the 3 uneven cases, the
eventual attractor is not known within the observation window. The
team has not shown these to be the same as, or different from, each
other.

Broken down by starting condition (36 trials each, and each trial
falls into exactly one category):

- IC=2000: 22 of 36 persistent-transient, 12 of 36 decayed, plus 2
  baseline-only-converged.
- IC=2001: 32 of 36 decayed, only 3 of 36 persistent-transient, plus
  1 baseline-only-converged.

This matches the state-dependence idea described above. It is not a
new finding.

## A Revised List of Capability Levels

There are three levels the team can tell apart, following the
reviewer's framing:

1. **Fixed transformation.** The response depends only on (node, sign,
   amplitude), not on the network's state. This pilot's comparison
   across starting conditions **rejects** this level.
2. **State-conditional transformation.** The response depends on
   (state, node, sign, amplitude), with the network's state as a real
   variable. This is **the leading idea**. It fits both the mismatch
   between starting conditions and Stage 0's multistability finding,
   the earlier finding that the network has several stable states. It
   has not yet been directly tested.
3. **Unstructured sensitivity.** Nearby or similar states produce
   unrelated results, with no stable, predictable link between input
   and output. The team **has not yet told this apart from level 2**.
   This is the key open question. Only a redesign that groups starting
   conditions by their dynamical state, before comparing response
   patterns, can settle it.

## Revised Main Conclusion

**The Stage 1B pilot rejects the idea of one fixed amplitude-response
rule for the learned topology (T) that holds regardless of network
state.** At small push sizes, extreme normalized amplification can be
explained mostly by tangent-linear growth, growth predicted by the
linear guess. At least one large-push condition produces real,
finite-amplitude reorganization of the response direction.

No trial where both trajectories converged reached a different stable
state. So the nonlinear effects the team observed are mostly temporary
(transient), not switches between stable states. Three perturbed
trajectories did not meet the convergence rule within the fixed time
window, even though their paired baseline trajectories did converge.

The pilot therefore shows state-dependent nonlinear, temporary
behavior. It does not yet show structured internal transformation or
useful computation. Stage 1B.2 will test whether these temporary
responses repeat from controlled nearby states. It will also test
whether they carry out a consistent redistribution of information
across space.

The reported peak-amplification values mix tangent-linear temporary
growth with true finite-amplitude effects, as confirmed above. Claims
about nonlinear capability depend on three separate things, reported
separately in this document rather than folded into one peak
amplification number:

- Departure from the tangent-linear prediction.
- Directional reorganization.
- The final attractor outcome.

Response size, and how it relates to node degree, how connected a node
is, varies sharply between the two starting conditions the team
tested. Because the network has several possible stable states, it is
multistable, this variation may represent state-conditional nonlinear
*behavior*. This is not yet established as *computation*, see the
three-level distinction above. It is not simply a failure to repeat
results in the ordinary sense.

## The Map From Push Size To Response

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

## Checking Repeatability Directly

The original idea, a fixed, state-independent response rule, predicts
that matching node, sign, and amplitude conditions should give similar
response patterns across different starting conditions. The two
sampled starting conditions give a minimal, direct test of this
specific idea. This is not a general claim that repeatability across
any arbitrary state is the only valid test of capability, see
Correction 1 above. Checking this directly:

**Low node**: IC=2000 shows sustained, large amplification, peak about
8 to 12, in nearly every condition. IC=2001 shows almost none, peak
fixed at 1.0, in nearly every condition, except at the largest tested
amplitude. This does not repeat.

**Median node**: both starting conditions show real deviation from a
peak of 1.0. But the *size* of the effect differs by about two orders
of magnitude. IC=2000 peaks at 2860; IC=2001 peaks at 20.6. Even if
both are treated as real temporary effects, the scale does not repeat.

**High node**: this shows the clearest failure to repeat. IC=2000's
high node shows almost no amplification anywhere in the tested range,
peak 1.0 to 1.3. IC=2001's high node shows the *opposite*: large
amplification that grows with push size, up to 36.7. This is the same
pattern shape that IC=2000's median node showed. In IC=2000, the
pattern by node degree looked clean and orderly: low and median nodes
amplify, high node decays right away. In IC=2001, this pattern does
not just fail to repeat. It partly reverses.

## What This Means, Stated Precisely

**Capability is unresolved, not disproved.** The original framing said
"repeatable structured transformation is not established." This
framing came too close to treating the mismatch between starting
conditions as itself a negative result.

The corrected framing: a *fixed*, state-independent rule linking
(node, sign, amplitude) to response is rejected by this pilot. The
same node's behavior changes too much between IC=2000 and IC=2001 to
support that simple model.

But a *state-conditional* rule, where the network's current position
among its several stable states is itself part of what decides the
response, remains fully consistent with everything observed. This
includes the sharp differences between starting conditions and the
partial reversal at the high-degree node. The team has not directly
tested this idea yet. Testing it requires describing each starting
condition's dynamical state before comparing response patterns. It is
not enough to simply note that two arbitrary states gave different
answers.

The peak-amplification numbers used to describe IC=2000's "striking
pattern" also needed the tangent-departure check applied above. The
median node's dramatic small-push peaks, up to 2860, are shown
directly to be mostly tangent-linear. They are not evidence of
finite-amplitude capability at all. The real nonlinear signal in this
data set, directional reversal, confirmed by C(t) going negative,
appears at large push size, where the peak amplification (S) is
actually smaller. Any future capability claim needs to rest on
evidence like the departure error, C(t), and the attractor outcome,
not on the peak amplification (S) alone.

This result matches, and follows directly from, Stage 0's
multistability finding: five distinct stable states were found from
five different starting points, all using the same class's topology.
If finite-time nonlinear response depends on which part of a richly
multistable landscape a trajectory currently occupies, then two
different starting conditions producing different amplitude-response
behavior is exactly what a state-conditional system should do. It is
not a failure to repeat in the ordinary sense.

## What This Does Not Mean

This is not evidence that T, the learned topology, or oscillator
dynamics in general, cannot carry out structured nonlinear
transformation. It is evidence that *this* two-starting-condition
pilot did not find a pattern that holds regardless of state.

A pattern that is real within one trajectory but not the next could
still reflect genuine, interesting structure. For example, the
response could depend on which basin of attraction, the region of the
state space near a stable state, the trajectory is in. This would not
be mere noise. But confirming that would need a design built
specifically around describing basin-dependence. The current
amplitude-response comparison, across only two shared starting
conditions, is not built for that.

## Honest Limitations

- Two starting conditions give only a minimal test of repeatability,
  not a thorough one. This test can detect a large failure to repeat,
  which it did. It cannot show how common the IC=2000-like pattern is,
  or whether a third starting condition would look like neither of the
  two tested.
- The team tested only one class, KMNIST class 0. Nothing here shows
  whether other classes' topologies behave the same way or
  differently.
- The team tested only T, with no graph controls yet. This is correct
  under the capability-first design. But it means nothing here yet
  shows whether the learned topology specifically shapes this
  behavior. That question does not apply until capability itself is
  established.
- The "decayed" classification uses this rule, fixed in advance: peak
  amplification (S) below 0.05, and RMS below 0.01, measured over the
  last 20% of a 25-unit extension time window. This is a meaningful,
  usable rule. But a different time window or threshold could, in
  principle, classify some borderline cases differently.

## What The Corrected Pilot Establishes, Precisely

**Established:**

- The response does not depend only on node degree, sign, and
  amplitude, independent of state.
- Extreme amplification at small push size can be mostly
  tangent-linear.
- At least one large-push case shows a strong directional departure
  from the tangent-linear prediction, C(t) goes negative.
- No trial, out of 72, recovered a distinct stable state (phase-locked
  equilibrium).
- Three trials showed uneven convergence within the fixed time window.
- The tested nonlinear behavior is therefore mostly temporary
  (transient), not a switch between attractors. This narrows the
  scientific question. It does not close it.

**Not established:**

- That directional reorganization repeats within similar dynamical
  states.
- That the network carries out an information transformation, rather
  than showing generic sensitivity to state.
- That which attractor a trajectory ends at is enough to predict the
  response.
- That T differs from control graphs. This question is correctly
  deferred, under the capability-first design. It does not apply until
  capability itself is resolved.

## A Sharper Capability Test For The Next Stage

A future capability claim should require all three of the following
together, not just one alone:

1. **Nonlinear departure.** The finite response, the real, nonlinear
   result, departs clearly from the tangent-linear prediction, the
   linear guess. This uses the departure-error and C(t) checks
   described above.
2. **Conditional repeatability.** Similar, controlled dynamical states
   produce similar response patterns.
3. **Structured transformation.** The response has identifiable
   organization. Examples: consistent directional redistribution,
   selective recruitment of certain nodes, or thresholded temporary
   routing of the response. Simply detecting nonlinearity is not
   enough.

A future capability claim does not require attractor switching. A
temporary transformation could still be computationally meaningful,
even if the system later returns to the same attractor. This requires
showing nonlinear departure, conditional repeatability, and structured
redistribution of information together. This document does not yet
show all three together.

This distinction is central to the project. The three levels must not
be collapsed into each other:

1. Nonlinear behavior.
2. Structured transformation.
3. Useful computation.

This pilot establishes the first level. It offers early evidence
relevant to the second level, the directional-reversal signal. It
establishes neither of the last two levels.

Even a perfectly repeatable response map, a stable mapping from
(state, push) to response, would still only be a statement about
sensitivity. It would not be a statement about information
transformation. That would require a pre-set measure of what the
response does spatially, see Stage 1B.2's spatial-redistribution
measure below.

## Immediate Next Steps: Stage 1B.2, A Controlled State-Conditioning Design

The previous plan was to sample many starting conditions, then group
them after the fact by dynamical descriptors, measured properties of
the state. This plan risks becoming exploratory and poorly defined.
With many candidate descriptors and relatively few starting conditions,
almost any grouping could appear to explain some of the variation in
response. A controlled design is better. The team adopts the following
plan instead:

**A. Sample baseline trajectories.** Generate several trajectories
without any push (baseline trajectories), for class 0. For each one,
record its final attractor identity, force norm, a measure of how far
the system is from equilibrium, coherence, a measure of
synchronization, and its tangent amplification profile.

**B. Push the network at several controlled times along the same
trajectory**, not only at t=0, the very start. Apply the same node,
sign, and amplitude push at fixed times t in {0, t1, t2, t3} along one
baseline trajectory. This directly tests whether the response changes
with the network's current dynamical state, while holding the graph,
the original trajectory, the node, the sign, and the amplitude fixed.
This is a much cleaner test of state-dependence than comparing
unrelated, random starting conditions.

**C. Add nearby-state replication.** For selected baseline states,
create small, randomly generated, gauge-corrected changes around the
state, called replicas, before applying the experimental push. A
replica is a slightly changed version of the same moment on the same
trajectory. The team uses replicas to check whether a result still
happens from an almost identical starting point. A replica is not a
repeat of the same trial and is not used to average out noise. Then
check whether nearby states produce similar departure-error curves,
C(t) curves, and final classifications. This step is what actually
tells apart a stable, state-conditional transformation from an
unstructured, oversensitive response.

**D. One main conditioning variable, fixed in advance**: the location
along a baseline trajectory, called the perturbation time, or t_p. t_p
is the point on the baseline trajectory, measured in absolute time,
where the push is applied.

The final attractor a trajectory reaches is **not** used to define
whether two dynamical states count as equivalent. This pilot found
mostly temporary effects, and every trial where both trajectories
converged reached the same attractor. So two baseline states heading
toward the same final attractor could still sit in very different
regions of the state space, with different local response behavior.
"Same destination" does not mean "same computational state."

The final attractor is instead kept as a **cross-trajectory grouping
variable**. It is used to group baseline trajectories by which
attractor they eventually reach, when comparing across different
baseline trajectories. Nearby-state replication, step C, is what
actually tests local robustness, stability of the result under small
state changes, at one point along one trajectory. Other descriptors,
such as force norm, coherence, and the Jacobian spectrum, remain
secondary, explanatory measurements. They are not additional main
variables to search across.

**E. A smaller response grid.** For state-conditioned replication,
unlike the apparatus-testing role the six-amplitude grid played in
this pilot, use three amplitude values: one that is tangent-consistent,
matching the linear guess, one intermediate value, and one value
already shown to produce directional reorganization. This focuses
compute on replication, rather than re-confirming amplitude trends
already established here.

**F. A pre-set spatial measure. This is the missing piece for a
genuine computation claim, and it is now fully specified, not left as
a list of options.**

Steps A through E, however well controlled, can only establish a
repeatable dynamical mapping from (state, push) to response. That is
not yet an information-processing claim. Even a perfectly repeatable
departure-error and C(t) curve is still only a statement about
sensitivity, not transformation, without a measure of what the
response does spatially. Four details are fixed here, not left to be
decided after seeing results:

**Detail 1: Energy alone loses direction. Keep both a direction-free
and a direction-aware measure.**

The normalized nodewise energy is q_j(t) = x_eps,j(t)^2 /
sum_k(x_eps,k(t)^2). This is the fraction of the total response
sitting at node j, at a given time. It always adds up to 1, or 100%,
across all nodes at any one time. It measures where the displacement,
the change in state, is concentrated. It does not measure its
direction.

Two responses with the opposite sign at every node would produce the
identical q(t). This would erase exactly the directional-reversal
signal that was this pilot's strongest nonlinear finding. So the team
also keeps a signed version: r(t) = x_eps(t) / ||x_eps(t)||_2. The
team uses signed cosine similarity between r(t) vectors to track
direction. This is used alongside q(t), which tracks how energy is
redistributed across nodes.

**Detail 2: The evaluation time is fixed in advance, not chosen after
looking at the results. A fixed-time check is added alongside the
event-based time.**

The earlier phrase "terminal or peak redistribution pattern" left too
much choice to the researcher. Peak separation, peak tangent
departure, and peak spatial divergence can each happen at a different
time.

The team locks this rule: evaluate the spatial output at the time of
maximum vector-relative departure error, within the main response
window. This time is called t*_eps: t*_eps = argmax_t E_eps(t), for
t in [0, T]. The main spatial output is q_eps(t*_eps), together with
its signed version r_eps(t*_eps).

**But t*_eps depends on the input.** Different inputs may reach their
strongest departure at different actual times. So comparing inputs
purely at each one's own t*_eps risks showing timing differences,
rather than genuinely different spatial transformations.

To guard against this, the team adds one fixed-time check alongside
t*_eps: q_eps(T), measured at the end of the fixed main time window.
This uses the same elapsed time for every input. The event-based
result at t*_eps remains the main result. The fixed-time result checks
that any separation between inputs is not created only by comparing
different moments in time.

The team may also report other secondary summaries, such as
time-integrated redistribution and the maximum JSD, Jensen-Shannon
divergence, a way of comparing two distributions, from the
tangent-linear prediction. Neither of these secondary summaries
replaces the two fixed time points above.

**Detail 3: t*_eps needs a guard against near-zero departure.** The
argmax calculation always returns some time, even when nothing
nonlinear actually happened.

For the tangent-consistent amplitude specifically, E_eps(t) may stay
extremely small for the whole window. An unguarded argmax will still
return a time, even when the maximum value found is just numerical
drift, noise from the calculation, not a real scientific deviation.

The team adds a fixed departure threshold: E_eps(t*_eps) is greater
than or equal to E_min. If the maximum value does not reach E_min:

- The team classifies the response as tangent-consistent, matching
  the linear guess.
- t*_eps is not treated as a nonlinear event.
- The event-based nonlinear spatial result is reported as undefined
  for that trial. The team does not compute it anyway.
- Only the fixed-time spatial output and the ordinary finite-response
  pattern are kept.

E_min is calibrated from repeated solver tolerances, or duplicated
integrations run under identical conditions. This measures the scale
of pure numerical variation, noise from computation, not from the
science. E_min is not chosen by looking at the Stage 1B.2 result
distribution itself. Calibrating it after seeing results would let the
team tune the threshold to produce a preferred classification. The
team avoids this.

**Detail 4: Two separate JSD-based measures, with separate names, so
"primary JSD" never means both at once.**

This design uses Jensen-Shannon divergence, JSD, a way of comparing
two distributions, in two related but scientifically different roles.
Code and text must name and report these separately:

- **Tangent-relative nonlinearity, J_tan(t) = JSD(q_finite(t),
  q_tangent(t))**: how much has the finite response, the real result,
  moved apart from the tangent response, the linear guess, in terms
  of spatial pattern? This checks for nonlinear departure. It shows
  that the response is nonlinear. On its own, it says nothing about
  whether the mapping responds differently to different inputs.
- **Output-map distance, d_q(a,b) = sqrt(JSD(q_a, q_b))**: how
  different are the spatial outputs of two separate trials? This
  measure feeds W, B, and Delta_map, defined below. This is the actual
  capability test.

The central Stage 1B.2 capability test is Delta_map greater than 0,
using d_q, not J_tan. Delta_map is the project's main score for
whether the network turns different inputs into reliably different
output patterns; it is defined in full below. J_tan is a necessary
check. It confirms that genuine nonlinear departure exists at all. But
J_tan does not replace the input-sensitivity test itself.

Signed cosine similarity, which tracks direction, and top-k node
overlap, which tracks which set of nodes gets recruited, remain
secondary checks, alongside both J_tan and d_q.

If the size of the displacement falls below a fixed numerical
threshold, q(t) is undefined. The team must report it as undefined.
The team must not force it into an apparently meaningful distribution
using an arbitrary denominator constant.

**Conditional repeatability is measured with numbers, not judged
"similar by eye."** The team now defines an exact distance, exact
definitions, and an exact shuffling, or permutation, scheme.

For the main energy-distribution output, the team uses the square root
of the Jensen-Shannon divergence as the output-space distance:
d_q(a,b) = sqrt(JSD(q_a, q_b)). The square root version is a true
metric, a distance measure. Raw JSD is a divergence. A divergence does
not obey the triangle inequality, so it is not a true metric.

Within each controlled state neighborhood, a group of nearby states,
the team defines:

- **W**: the average distance between outputs that received the
  *same* input. W = (1/|P_same|) times the sum of d_q(a,b), over all
  replica pairs (a,b) that got the same input.
- **B, balanced across factors, not simply pooled.** "Different
  input" can mean a different node, a different sign, or a different
  amplitude. These three factors do not have the same number of
  possible pairs. If the team simply pooled all "different input"
  pairs together, whichever factor has the most pairs would dominate
  B, just because of counting, not because of a genuinely larger
  effect.

  Instead, the main between-input distance is: B = (1/3) times
  (B_node + B_sign + B_amplitude). Each of these three components
  averages comparisons that differ in exactly one factor, while
  matching the other two factors wherever possible. An equally valid
  alternative is to sample equal numbers of pairwise comparisons from
  each factor.

  The team also reports the three factor-specific separations,
  B_node, B_sign, and B_amplitude, as secondary results. These may
  show, for example, that the system tells apart push location but
  not push sign, or push size but not node. This matters for
  understanding what kind of transformation the network is carrying
  out. Even so, it is the balanced, combined B that feeds the primary
  test.

- **Effect size: Delta_map = B - W.** Delta_map is the project's main
  score for whether the network turns different kicks (pushes) into
  reliably different spatial output patterns. A higher Delta_map means
  a clearer structured transformation.

For the shuffling test, permutation inference, the team shuffles input
labels **only within** the same baseline state, the same perturbation
time (t_p), and the same nearby-state replica block. The team never
shuffles across unrelated state neighborhoods. Shuffling across
neighborhoods would destroy the very state-conditioning structure the
experiment is designed to test.

The main capability test is then exact: the null hypothesis is H0:
Delta_map is less than or equal to 0; the alternative is H1: Delta_map
greater than 0. To show a structured, input-sensitive, state-
conditional mapping, the team must reject H0. This means outputs must
be more similar across repeated presentations of the same input from
nearby states, than across different inputs from the same state
neighborhood.

This is a stronger, different claim than simply showing each response
repeats in isolation. It establishes input sensitivity together with
robustness, stability under small changes. This combination is what
tells apart a genuine mapping from noise that happens to look stable.

**The scale of the nearby-state changes is calibrated, not guessed.**
The team chooses the largest gauge-corrected nearby-state change whose
*unforced* trajectory, the trajectory with no experimental push, only
the small nearby-state change, stays within a fixed RMS, root-mean-
square, distance of the reference baseline, over a short check period
before the push. This makes "nearby" a precise, working definition:

- Genuinely different from the baseline, above the level of numerical
  noise.
- Still local to the baseline.
- Unlikely to cross into a different dynamical regime before the
  experimental push is applied.

This is a working rule, not a rough geometric guess.

The nearby-state changes themselves must be:

- Zero-mean: they average to zero across directions.
- Unit-normalized before scaling: their length is set to 1 before
  being scaled to the chosen size.
- Generated from a fixed random seed.
- Identical across different experimental inputs, for a given
  replica.

Without these rules, comparisons between inputs could be distorted by
different neighborhood samples, rather than by genuine differences
between the inputs being compared.

This will be the first genuine capability test this project has run.
It asks: does the dynamical system map distinguishable local inputs
into repeatable, structured response patterns, conditional on its
current state? This is a different, stronger question than simply
asking whether the system responds differently depending on where it
starts.

This tests the leading idea directly, under controlled conditions:
**does a controlled dynamical state transform local pushes into
repeatable, input-sensitive spatial response patterns, rather than
merely showing nonlinear sensitivity?** Answering this distinction, not
gathering a larger, uncontrolled sample, is what the next stage needs
to do.

## Final Pilot Status

Stage 1B established state-dependent, finite-amplitude, nonlinear,
temporary (transient) behavior. It did not establish structured
internal transformation or useful computation.

Extreme normalized amplification at small push size may remain mostly
tangent-linear. Large enough pushes can reorganize the direction of
the response. No pair of trajectories that both converged reached
distinct stable states, phase-locked equilibria. Three perturbed
trajectories did not converge within the fixed time window.

Stage 1B.2 will determine whether controlled local dynamical states
map distinguishable pushes into outputs that are robust, repeatable,
and spatially structured.

This keeps the research hierarchy intact:

**nonlinear response is not the same as structured internal
transformation, which is not the same as useful computation**

This is a chain of three separate, non-equivalent ideas, not one
sentence to read straight through. Stage 1B.2 is the first direct test
of the middle idea.

**Stage 1B.2 design decisions, fixed (conceptual):**

1. Perturbation times (t_p values) along the baseline trajectory.
2. Three amplitudes: one tangent-consistent, one intermediate, one
   already shown to produce directional reorganization.
3. The scale of nearby-state changes, and how many replicas to use.
4. The time-selection rule, t*_eps = argmax E_eps(t), guarded by a
   departure threshold E_min. Below E_min, t*_eps is not treated as a
   nonlinear event.
5. Two separately named JSD-based measures: J_tan(t) for
   tangent-relative nonlinearity, a diagnostic check, and
   d_q(a,b) = sqrt(JSD) for output-map distance, which feeds the main
   test.
6. Signed directional similarity, r(t), as a secondary measure. This
   keeps the directional-reversal signal that an energy-only measure,
   q(t), would erase.
7. A quantitative test comparing within-state to between-input
   variation: Delta_map = B - W, with B balanced across the node,
   sign, and amplitude factors rather than simply pooled. The test is
   H0: Delta_map is less than or equal to 0, versus H1: Delta_map
   greater than 0, using a block-restricted shuffling scheme.
8. A fixed-time robustness check, q_eps(T), alongside the event-based
   main time, t*_eps.
9. A working calibration rule for the scale of nearby-state changes.

**Exact numbers still to be set before running the experiment.** These
are specific parameter choices within the design above, not further
conceptual decisions:

- The perturbation times along the baseline trajectory.
- The three actual amplitude values.
- The main time window, T.
- The fixed-time endpoint.
- The numerical size threshold below which q is undefined.
- The nonlinear departure threshold, E_min. This must be calibrated
  from solver tolerance or duplicated integrations, not from Stage
  1B.2 results.
- The check period before the push, for validating nearby states.
- The largest allowed unforced RMS difference. This defines "nearby."
- The candidate list of nearby-state scale values.
- The number of nearby replicas.
- The number of shuffles (permutations), or an exact-enumeration rule.
- The top-k value for the secondary node-recruitment check.

Once these values are set, the experiment will test **structured
internal transformation**. It will not merely list examples of
nonlinear sensitivity. This is level 2 of the three-level distinction,
nonlinear response; structured internal transformation; useful
computation. It is much closer to a computation claim than anything
tested so far in this project.

It does not yet establish level 3. Showing usefulness will still
require linking the response patterns to a task or information-
processing goal defined outside the experiment. This is a separate
question from whether the transformation is structured and repeatable
in the first place.

## Document Status: Frozen

This document is frozen. It is the final record of the Stage 1B pilot
and the conceptual pre-registration, a plan fixed in advance, for
Stage 1B.2. No further conceptual redesign is needed before building
Stage 1B.2. The only work left is numerical calibration, shown in the
table below. This table should be the entire content of the next
review. The team fills it in without looking at any Stage 1B.2 output.

## Stage 1B.2 Numerical Calibration Table — Amended And Locked

The team calibrated every value below using only three sources: (a)
the already-closed Stage 1B pilot's own validated results, (b) Stage
0's earlier diagnostic checks, and (c) dedicated numerical-
repeatability and unforced-trajectory checks, run only for this
calibration.

No Stage 1B.2 experimental trial, a baseline pushed at t_p and compared
across inputs, was run or looked at before every row of the table
below was fixed.

**A blocking issue, now resolved: Option A is locked.** An earlier
version of the design fixed the perturbation node to a single value.
But the design requires B_node as one of three balanced factors. With
only one node, B_node cannot be calculated, and the experiment cannot
test whether outputs tell apart different push locations.

Fixing the node to the median value specifically, because it showed
the clearest signal in the pilot, also risked letting result-guided
selection into Stage 1B.2. The rest of the design is built to avoid
exactly this kind of selection.

**Option A is locked: use the low, median, and high weighted-degree
nodes in T.** This gives 3 nodes x 2 signs x 3 amplitudes = 18 inputs.
With 4 perturbation times (t_p) and 6 replicas, this gives **432
finite-response trials**.

The node location is not a side detail in this experiment. It is the
spatial identity of the push. Testing whether that identity maps into
distinguishable spatial outputs is central to the structured-
transformation idea this stage exists to test. Removing this factor,
Option B, would answer a clearly weaker question, for a smaller saving
in compute time than the science is worth. The extra cost is
justified: this is the first direct test of structured internal
transformation in this project. Cutting down the input space to avoid
this cost would undercut the reason for running Stage 1B.2 at all.

| Parameter | Candidate values tested | Selection rule | Locked value |
|---|---|---|---|
| Perturbation node(s) | -- | Follows the resolved node issue above (Option A) | low, median, and high weighted-degree nodes in T. Not the single median node fixed earlier, which risked result-guided selection. |
| Perturbation times | -- | Evenly spaced fractions of the main time window, fixed without looking at any response data | t_p in {0, 0.833, 1.667, 2.5}, measured on the baseline trajectory's own clock |
| Response time window | -- | Matches Stage 1B's response coverage, already shown to capture both tangent-consistent and nonlinear behavior | T = 2.5, measured as tau, elapsed time since each t_p, not absolute time. For a push at t_p, the response is observed over absolute baseline time [t_p, t_p+2.5], not a single fixed end time. So the t_p=2.5 condition has a genuine response window of [2.5, 5.0], not a window of zero length. |
| Fixed-time spatial endpoint | -- | Same elapsed time (on the tau clock) across every input | q_eps(tau=T), not q_eps(T) measured on the baseline's absolute clock |
| Event-based time | -- | argmax over the response window after the push | tau*_eps = argmax over tau in [0,T] of E_eps(tau) |
| Amplitudes | 0.025, 0.05, 0.1, 0.2, 0.4, 0.8 (the Stage 1B pilot grid) | One tangent-consistent value (pilot confirmed C(t) close to 1.0 throughout), one intermediate value, one nonlinear value (pilot confirmed C(t) below 0) | 0.025 (tangent), 0.2 (intermediate), 0.8 (nonlinear) |
| q-norm threshold | -- | The noise floor of the displacement size, comparing standard solver tolerance (1e-6/1e-8) against tight solver tolerance (1e-10/1e-12), with a safety margin | 1e-6 (about 4 orders of magnitude, roughly 10,000 times, above the measured noise floor of 7.89e-11) |
| E_min | -- | The error range from duplicate solves, using the same tolerance comparison applied directly to E(t) | 1e-4 (about 30 times the measured noise floor of 3.66e-6, and comfortably below the observed tangent-consistent E values of 0.002 to 0.004) |
| Nearby-state scale | 0.001, 0.005, 0.01, 0.05, 0.1, 0.2 -- now tested at all four perturbation-time states, using the same six fixed replica directions, not just one state | The largest scale whose unforced-trajectory RMS difference stays below the locked locality bound, at every t_p | 0.1. Verified: the maximum RMS across all 4 perturbation-time states and all 6 fixed replicas is 0.0085 (at t_p=0) and 0.00445 (at the other three). Both values are comfortably under the 0.01 bound, at every state, not just the one tested originally. |
| Validation time window | -- | Fixed before scale testing, short compared to the main time window | 0.5 time units (1/5 of T) |
| RMS locality bound | -- | A round, easy-to-interpret fraction of the full circular range (2*pi), fixed without looking at the scale-testing results | 0.01 (about 0.16% of the full circular range) |
| Replica count | -- | Enough same-input pairs for the shuffling test (C(R,2) pairs per state neighborhood) | R = 6 perturbed replicas, generated from a fixed random seed, applied identically across all inputs for a given neighborhood. The unperturbed reference state is kept separately, as a diagnostic check. It does not count as a seventh replica. C(6,2) = 15 same-input pairs uses the six perturbed replicas only. |
| Number of shuffles | -- | Fine enough resolution for alpha=0.05, and generous, because shuffling pre-computed outputs is cheap and does not require re-running the integration | 10,000 shuffles (resolution down to p=0.0001) |
| Top-k | -- | A fixed fraction of nodes, so the measure works the same way across class topologies with different numbers of active nodes | The top 5% of active nodes, ranked by response energy |

**The shuffling (permutation) method: the earlier version did not
work. It is now fixed.**

An earlier fix said: "apply one common random shuffle of input labels
within that neighborhood, the same shuffle across all six replicas."
This does not generate a valid null distribution, a valid picture of
what would happen by chance.

A single, shared relabeling across every replica keeps the same
outputs grouped under the same input label. If input A is renamed
input F, it stays grouped with the same outputs in every replica. Only
the name changes.

Because of this:

- The same-input pairs that feed W are still the same actual pairs.
- The different-input comparisons that feed B are still the same
  actual comparisons.
- Delta_map does not change under this kind of relabeling.

The shuffled distribution would collapse to a single point, exactly at
the observed value. It could not test whether the observed link
between input and output is stronger than chance.

This error would have gone completely unnoticed until 432 trials of
data came back, and the "test" gave the same answer under every
shuffle. This is exactly the kind of statistical error that needs to
be caught during calibration, not discovered after the compute has
already been spent.

**Corrected procedure.** This tests the actual null hypothesis: that
output patterns are not consistently linked to a given input's
identity, across nearby replicas.

For each perturbation-time neighborhood, the team follows these steps:

1. Keep the six replica identities fixed.
2. **Within each replica separately**, randomly shuffle the 18 input
   labels across its 18 outputs. Keep a one-to-one label assignment
   inside that replica.
3. Recompute W, B_node, B_sign, B_amplitude, B, and Delta_map, using
   this independent relabeling.
4. Repeat this 10,000 times.
5. Never exchange outputs across replicas, perturbation times, or
   baseline trajectories.

Shuffling independently within each replica destroys the consistent
input identity across replicas, under the null hypothesis. For
example, an output actually produced by input A in replica 1 might now
be compared against an output actually produced by input C in replica
2, as if they were "the same input." At the same time, this method
keeps each replica's complete output geometry, the repeated-measures
block structure, the number of observations, and the balanced
factorial input set unchanged.

**Exact one-sided Monte Carlo p-value:**

p = (1 + sum_{b=1}^{M} 1[Delta_map^(b) >= Delta_map^obs]) / (M + 1)

With M = 10,000, the smallest possible p-value is 1/10,001, about
0.0001.

**Factor-specific shuffling (secondary analyses).** The overall test,
shuffling all 18 labels independently, is valid for testing the
overall mapping idea. For testing one factor at a time, the team uses
a restricted shuffle that keeps the other two factors fixed:

- For B_node: shuffle node labels within each matched sign-amplitude
  group, independently per replica.
- For B_sign: shuffle sign labels within each matched node-amplitude
  group.
- For B_amplitude: shuffle amplitude labels within each matched
  node-sign group.

These are secondary analyses. If the team reports p-values for these
three factor-specific tests, the team must correct for testing three
things at once, for example using the Bonferroni method, alpha/3. This
follows the multiplicity discipline used throughout this project:
running several statistical tests raises the chance of a false
positive, so the threshold must be adjusted.

**Balanced B definitions, made clear.** For each component of B, the
team compares outputs that differ in exactly one factor, while the
other factors stay matched:

- B_node = mean of d_q[(n1,s,a),(n2,s,a)] -- same sign and amplitude,
  different node.
- B_sign = mean of d_q[(n,+,a),(n,-,a)] -- same node and amplitude,
  different sign.
- B_amplitude = mean of d_q[(n,s,a1),(n,s,a2)] -- same node and sign,
  different amplitude.

**The team forms these pairs across different replicas, not within
the same replica.** This keeps B and W measured at the same
replication level. Otherwise, W would measure variation across
replicas, while B partly measured variation within one replica. This
would make Delta_map = B - W harder to interpret cleanly, like
comparing apples to a mix of apples and oranges.

In practice: W pairs the same input across different replicas. Each
B_f pairs matched inputs that differ only in factor f, also across
different replicas. The team excludes pairs of an item with itself.
Pairs are unordered: since d_q is symmetric, (a,b) and (b,a) are the
same pair, counted once.

Before running this on real data, the team should test this algorithm
on synthetic, made-up and controlled, data. Identical input maps
should produce a Delta_map close to 0. Highly separated, stable maps
should produce a clearly positive result. This confirms that the test
statistic and the shuffling scheme both behave correctly, before
trusting either one on the actual experiment.

**What the result will and will not cover, stated clearly.** A fresh
seed (3000) gives *one* baseline trajectory, with four controlled
states along it. This is enough for a local demonstration of
capability. But the four perturbation times are repeated states along
one single trajectory. They are not four independent baseline
trajectories. The result will be conditional on this one specific
trajectory.

A positive result would support this claim: *along this pre-set
class-0 trajectory, nearby states carry out an input-sensitive,
locally repeatable spatial mapping.*

A positive result would **not** yet support this claim: *the class-0
topology in general carries out such a mapping.* That would require
multiple independent baseline trajectories, which have not yet been
run.

The team will report Delta_map **both by perturbation time and pooled
together (combined)**. A positive pooled result must not be allowed to
hide one highly informative time point sitting alongside three time
points with no effect.

**A note on the order of two calculations: the nearby-state scale and
the locality bound.** The team chose the locality bound (0.01) for its
own clear, principled reason: it is a small, round fraction of the
circular range. The team did not derive it from the scale-testing
results. But the team ran both calculations in the same working
session, before writing this table.

The team discloses this openly. The team does not present this as a
fully blinded pre-registration, a plan set before seeing any related
data. The team did not tune the bound to make any specific candidate
scale pass.

**Class and baseline seed.** KMNIST class 0, matching the closed
pilot. A fresh starting-condition seed (3000), not used in Stage 1B or
its calibration, reserved for the Stage 1B.2 experimental run itself.

Every row above is now locked: Option A (three nodes, 432 trials), and
a corrected shuffling scheme that actually generates a valid null
distribution. Stage 1B.2 can start immediately. No other numerical or
conceptual decisions remain. No Stage 1B.2 experimental trial, pushing
the baseline at t_p and comparing across inputs, was run to produce
any value in this table.

## Reproducing These Results

- `run_stage1b_pilot.py`: the main script. It saves checkpoints and can
  resume if stopped.
- `stage1b_taxonomy.py`: the classification code, with the
  integration-structure fix applied.
- Full results: `stage1b_pilot_results.pkl`.
- Per-trial log: `stage1b_pilot_progress.log`.

## Reproducibility Note (Independent Re-run)

A separate team member independently re-ran this pilot, using a
separate checkout, or copy, of this codebase. The original script used
file paths specific to one sandbox environment
(`/home/claude/oscillator_field`, from Claude's temporary development
environment). These paths do not exist outside that sandbox. The team
replaced them with paths relative to the script's own location. This
is a portability fix. It does not change the classification logic or
the trial grid.

**Topology input.** The original file `stage1a_all_classes.pkl`,
produced in that sandbox, is not present in the re-run's checkout. The
team substituted
`experiments/stage1b2_structured_transformation/results/class0_constructions.pkl`.
The team confirmed this file is structurally identical for class 0,
checking `data['n_active']` and `data['constructions']['T']`, using the
same access pattern already used by `run_stage1b2.py`. This pilot only
ever uses class 0 (`CLASS = 0`). So this substitute file is a complete
replacement for this run, not a partial one.

**Continuous dynamics: fully reproduced.** All 72 peak-amplification
values in the amplitude-response map above matched the reported
figures, to the same decimal place, across every (IC, node, sign,
amplitude) combination. This includes the most extreme case,
IC=2000/median/-eps. The originally reported values 865.7, 2860.4,
1025.1, 266.8, 70.6, 18.1 reproduced as 865.68, 2860.44, 1025.11,
266.76, 70.63, 18.08.

**Discrete outcome taxonomy: did not fully reproduce.**

| Outcome | This document | Independent re-run |
|---|---|---|
| Decayed to same attractor | 44 | 39 |
| Persistent transient, same attractor | 25 | 23 |
| Baseline-only converged (asymmetric) | 3 | 10 |
| Different equilibria | 0 | 0 |
| No equilibrium recovered within horizon | 0 | 0 |

The cause of this mismatch is precisely identifiable. It is not just
an unexplained difference. Every one of the re-run's 10
`baseline_only_converged` trials has a perturbed-trajectory force norm
between 1.0x and 1.6x the `FORCE_CONVERGED_THRESHOLD` value (1e-5).
This is exactly the boundary that the code's own documentation, in
`classify_terminal_state`, already flags as marginal, or uncertain.
That documentation states the threshold was "relaxed from 1e-6 after
direct diagnostic: L-BFGS reports genuine convergence... at
force~1.5e-6 for this system's small-spectral-gap graphs -- 1e-6 was
stricter than the optimizer's own achievable precision here."

This is environment-sensitive boundary fragility. It comes from small,
platform-dependent differences, from scipy or BLAS, underlying
numerical libraries, in the force norm that the L-BFGS optimizer
achieves, on these flat, slow-converging landscapes. This is not a bug
in either the original run or this reproduction. The underlying
trajectories are the same trajectories in both runs. The
exactly-reproduced peak-amplification values confirm this. Only the
pass/fail decision against a threshold that sits inside the solver's
own noise floor differs between the two runs.

**What this mismatch does not affect.** The finding that this
document's conclusions depend on is not affected. Both runs agree
exactly on two counts: 0 for `different_equilibria` and 0 for
`no_equilibrium_recovered_within_horizon`. In neither run did any
trial produce two recovered but distinct stable states (phase-locked
equilibria). In neither run did any trial fail to converge on both
sides.

The specific count of asymmetric convergence, 3 in this document
versus 10 in the re-run, and the per-IC breakdowns built on it
(22/12/2 and 32/3/1), should be read as sensitive to solver and
environment precision at this one specific threshold. They should not
be read as a stable, environment-independent property of the pilot.
