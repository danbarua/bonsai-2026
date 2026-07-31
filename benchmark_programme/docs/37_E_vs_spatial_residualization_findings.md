# E Is Largely Recoverable From Generic Spatial Structure, but Retains a Thin Unique Residual on Kuzushiji-MNIST

## What this test asked

Does E contain any useful information beyond generic spatial structure
(G_spatial), and does spatial structure contain anything beyond E? Both
directions tested via ridge residualization, with tuned regularization
(`LogisticRegressionCV`) throughout.

## E is almost entirely linearly predictable from spatial structure

| Dataset | R² of E from G_spatial | R² of G_spatial from E |
|---|---|---|
| Fashion-MNIST | 0.955-0.995 | 0.498-0.995 |
| notMNIST | 0.985-0.990 | 0.381-0.989 |
| Kuzushiji-MNIST | 0.878-0.966 | 0.247-0.970 |

**Individual E coordinates have R² values between 0.878 and 0.995 when
predicted from the spatial block** -- these are coordinate-wise values,
not automatically a block-level statement. Computed directly: the
block-level, variance-weighted R² (1 - sum of squared residuals / sum of
squared deviations, aggregated across all 10 dimensions at once) is
**0.9845 (Fashion-MNIST), 0.9880 (notMNIST), 0.9361 (Kuzushiji-MNIST)**.
This does properly support a block-level claim: E is largely linearly
recoverable from spatial structure -- a much higher overlap than T-E's
overlap ever showed (which topped out around 80%). "Largely recoverable"
is the correct framing, not "redundant" -- redundant would imply removing
E loses no useful information, and the residual results below show that
is not strictly true, at least on Kuzushiji-MNIST. The reverse direction
(spatial from E) is more variable -- some spatial dimensions (plausibly
the energy-related ones) are nearly perfectly predictable from E, others
(plausibly centroid or covariance) much less so.

## The decisive test: does E's residual against spatial still help?

| Dataset | T+G_spatial | T+G_spatial+residual-E | Gain | McNemar p |
|---|---|---|---|---|
| Fashion-MNIST | 78.00% | 79.00% | +1.00pp | 0.576 (n.s.) |
| notMNIST | 86.80% | 88.60% | +1.80pp | 0.0784 (n.s., borderline) |
| **Kuzushiji-MNIST** | 62.80% | 66.80% | +4.00pp | **0.0286** |

## The symmetric test: does spatial's residual against E still help?

| Dataset | T+E | T+E+residual-spatial | Gain | McNemar p |
|---|---|---|---|---|
| Fashion-MNIST | 76.00% | 78.20% | +2.20pp | 0.117 (n.s.) |
| notMNIST | 86.40% | 88.80% | +2.40pp | 0.0501 (n.s., borderline) |
| **Kuzushiji-MNIST** | 61.00% | 67.40% | +6.40pp | **1.66e-04** |

## Neither representation linearly subsumes the other on Kuzushiji-MNIST specifically -- this should not be generalized across datasets

**On Kuzushiji-MNIST, the only dataset where either residual reaches
clear significance: residual E significantly improves T+G_spatial
(p=0.0286), and residual spatial significantly improves T+E (p=1.66e-04).**
Both residuals carry real information there -- this is not a case of one
representation fully containing the other, on this dataset. The
comparison that should NOT be made is treating the smaller p-value as
proof of a larger effect: residual spatial produced a larger observed
accuracy gain (+6.4pp vs. +4.0pp) and stronger evidence against the null
than residual E, but p-values reflect both effect size and the pattern of
discordant predictions -- they do not by themselves establish that one
effect is statistically larger than the other. A direct paired
comparison, repeated splits, or a bootstrap over the difference would be
needed to support that stronger claim, which this document does not make.

**On Fashion-MNIST and notMNIST, neither residual reaches conventional
significance -- the experiment did not establish unique incremental value
in either direction on these two datasets.** notMNIST's symmetric-test
p-value (0.0501) is numerically close to the conventional threshold, but
"suggestive" should stay clearly distinguished from "confirmed." This is
consistent with the extremely high R² values there (up to 0.995), leaving
very little room for either residual to carry detectable independent
signal, and consistent with these datasets' established pattern of
weaker, more ambiguous evidence throughout this project.

## What this means for the representation -- and what it does not mean

**The strongest supported revision is narrower than "E did not discover a
new class-conditioned mechanism."** The experiments establish that most of
the information E makes accessible in these classification experiments is
generic spatial organization that T omitted, and that a simpler
class-agnostic image-domain representation captures that information at
least as effectively. They do not establish that the class-conditioned
oscillator representation implements no distinct mechanism -- mechanism
and the information it exposes are different questions. E may arise
through a genuinely class-conditioned relational process while still
making spatial geometry accessible in a form that is replaceable, for
this specific downstream linear classifier, by direct Cartesian
summaries. The spatial control has direct access to the pixel lattice
I(x,y); E does not -- it is obtained from image energy evaluated against
class-conditioned supports induced by oscillator-derived topology. The
finding is therefore not only "a simpler feature block can replace E." It
is also: a low-dimensional statistic extracted through the oscillator-
derived relational structure is highly predictable from conventional
image geometry -- which says something non-trivial about what the
dynamics preserve or re-express, not nothing about the dynamics at all.
This establishes informational overlap and downstream substitutability
for this benchmark. It does not establish computational redundancy in a
broader sense.

## The reverse block-level R², quantifying the asymmetry directly

| Dataset | R²_block(E ← G_spatial) | R²_block(G_spatial ← E) | Asymmetry |
|---|---|---|---|
| Fashion-MNIST | 0.9845 | 0.7840 | +0.2005 |
| notMNIST | 0.9880 | 0.7161 | +0.2718 |
| Kuzushiji-MNIST | 0.9361 | 0.6440 | +0.2921 |

(Arrow notation used deliberately over conditional notation, since this is
regression reconstruction -- R²_block(A ← B) reads as "how well B
reconstructs A" -- not a conditional-variance statement.)

**The central empirical result is the reconstruction asymmetry itself: a
ten-dimensional generic spatial block reconstructs 93.6-98.8% of E's
aggregate variance, while E reconstructs only 64.4-78.4% of the spatial
block's aggregate variance.** This is not a "strict superset" relationship
-- that would require all of E's information to be contained in
G_spatial plus more, which the Kuzushiji-MNIST residual result directly
contradicts. The precise statement: **generic spatial structure provides a
substantially more complete linear reconstruction of E than E provides of
generic spatial structure -- strongly asymmetric, though neither
representation fully contains the other on Kuzushiji-MNIST.** Schematically,
this looks like G_spatial → most of E, with small residual components on
both sides, rather than two equivalent representations or one strictly
containing the other. The asymmetry is largest on Kuzushiji-MNIST, where
E is also least recoverable from spatial structure (R²=0.936) -- a
structural correspondence with that dataset also being the only one
where residual E remains detectably useful. E appears to encode a narrow
projection of generic geometry; the spatial block contains considerably
more variation E does not represent, some of it label-relevant,
especially on Kuzushiji-MNIST.

## Multiple-testing status, acknowledged directly

This document contains six directional residual comparisons (two
directions x three datasets). A Bonferroni threshold across all six is
0.05/6 ≈ 0.0083. Kuzushiji-MNIST's residual-spatial result
(p=1.66x10^-4) is robust to this correction. **Kuzushiji-MNIST's
residual-E result (p=0.0286) is not** -- it does not survive a
family-wise correction across the six comparisons tested. Whether a
correction is formally required depends on whether this was a
prespecified hypothesis or part of an exploratory family; regardless, the
document should not describe p=0.0286 as unqualifiedly established
without this caveat. The corrected status: residual E shows *nominally*
significant incremental value on Kuzushiji-MNIST, weaker than the
residual-spatial result and not robust to correction across all six
comparisons. This does not erase the observation -- it calibrates it.

## Updated evidence hierarchy

| Claim | Status |
|---|---|
| E improves linear classification beyond T | Established |
| E contains label-relevant information not linearly recoverable from T | Established |
| Most E variance is linearly recoverable from compact generic spatial statistics | Established |
| Linear recoverability is strongly asymmetric in favor of the spatial block | Established |
| E has predictive privilege over generic spatial structure | Not established |
| E retains value beyond spatial structure | Nominal evidence on Kuzushiji-MNIST; weak under family-wise correction |
| Spatial structure retains value beyond E | Strongly supported on Kuzushiji-MNIST |
| Either representation strictly contains the other | Not established |
| E and spatial structure are computationally equivalent | Untested |
| Class conditioning contributes nothing to how E is formed | Not established |
| Spatial structure is the parsimonious classifier default | Strongly supported for this protocol |
| The full E construction re-expresses image geometry relationally | Supported |
| Oscillator dynamics specifically (rather than regional energy summaries generally) cause that re-expression | **Not supported -- a non-oscillator, class-template control reproduces the identical reconstruction asymmetry** |
| Oscillator-derived E significantly outperforms simple-E | Not supported anywhere tested |
| Simple-E significantly outperforms E | Nominal evidence on Fashion-MNIST; not robust across six comparisons |
| E and simple-E are statistically equivalent | Not established -- absence of evidence, not evidence of absence |
| E's residual beyond spatial structure is oscillator-specific | Not supported |
| G_spatial is the parsimonious benchmark replacement for E | Strongly supported, scoped to this software classification pipeline |
| E has no justified role in any architecture | **Not supported by anything tested here -- see scoping note below** |

## Two layers to this narrative -- both now resolved for this document

**The machine-learning result**: for these datasets and linear readouts, E
is largely substitutable by cheaper generic spatial statistics -- this
demotes E as an engineered classifier feature, and the practical
candidate representation moves from (T, E, R) to (T, G_spatial, R) for
this benchmark, unless E later demonstrates a practical advantage in
out-of-distribution transfer, low-data learning, noise robustness,
calibration, class-incremental learning, or interpretability specifically
tied to learned supports.

**The control**: a "simple-E" feature was built using the exact same
construction as E (sum of squared pixel energy within a class-specific
support, normalized identically) but with the support defined trivially
-- the top-N pixels by class-conditional mean intensity ("class-template support") -- prevalence-based, not necessarily contiguous or spatially localized, N matched exactly to the real oscillator-derived active
support size, with zero oscillator dynamics involved anywhere in its
construction.

| Dataset | R²_block(E ← G_spatial) | R²_block(simple-E ← G_spatial) | R²_block(G_spatial ← E) | R²_block(G_spatial ← simple-E) |
|---|---|---|---|---|
| Fashion-MNIST | 0.9845 | 0.9877 | 0.7840 | 0.8000 |
| notMNIST | 0.9880 | 0.9966 | 0.7161 | 0.7482 |
| Kuzushiji-MNIST | 0.9361 | 0.9547 | 0.6440 | 0.6455 |

**The simple, non-oscillator support produces numerically equal or
higher block-level R² in every reported comparison** -- the exact
differences (simple-E minus E): Fashion-MNIST forward +0.0032, notMNIST
forward +0.0086, Kuzushiji-MNIST forward +0.0186; Fashion-MNIST reverse
+0.0160, notMNIST reverse +0.0321, Kuzushiji-MNIST reverse +0.0015.
Without uncertainty estimates over these reconstruction scores,
"marginally stronger" is a fair descriptive summary, but "identical"
would overstate it -- the pattern is reproduced, the numbers are not
exactly matched. This directly answers the question this control was
designed to ask, precisely scoped: **does oscillator dynamics
specifically explain why E is highly reconstructible from generic
spatial statistics? No evidence that it does.** A trivial, non-dynamical
class-template support (top-N pixels by class-conditional mean intensity
-- not spatial-frequency or Fourier-domain, a term deliberately avoided
here to prevent confusion with the upcoming R frequency-band work)
reproduces the same strongly asymmetric pattern. The likely structural
explanation: any class-conditioned feature of the form
E_c(x) = sum over i in S_c of x_i^2 is a weighted regional energy
projection, and if the supports S_c reflect common class shapes and
locations at all, their outputs will naturally covary with total energy,
quadrant energy, centroid, spread, and orientation -- regardless of
whether S_c came from oscillator dynamics or a class-template mean
image. Note also that top-N pixels by mean intensity need not form one
contiguous region -- they may be spread across several characteristic
stroke locations -- so this is better described as projecting image
energy onto class-specific spatial supports than as a "spatially
localized" summary, which would wrongly imply contiguity.

**This closes one specific mechanism-isolation question, not the whole
thread**: it closes whether oscillator dynamics specifically explain the
*reconstruction asymmetry* (no evidence that they do). It does **not**
close whether oscillator-derived E contains any label-relevant residual
structure that simple-E lacks -- particularly the thin, Kuzushiji-MNIST-
specific predictive residual noted earlier. That is a different,
narrower, still-open question, addressed directly below.

## The narrow remaining question, closed: does oscillator-derived E predict anything simple-E doesn't?

Two direct tests, both under tuned regularization: (1) T+simple-E vs.
T+E, the direct predictive comparison; (2) T+G_spatial+residual-simple-E
vs. T+G_spatial+residual-E, the conditional-residual comparison --
specifically targeting whether oscillator-derived E's thin
Kuzushiji-MNIST residual (nominally significant against G_spatial alone,
though not surviving family-wise correction) survives when compared
directly against the equivalent residual from a trivial, non-dynamical
construction.

| Dataset | T+simple-E | T+E | McNemar p (direct) | T+G+residual-simple-E | T+G+residual-E | McNemar p (conditional) |
|---|---|---|---|---|---|---|
| Fashion-MNIST | **78.60%** | 76.00% | **0.0241** (favors simple-E) | 80.40% | 79.00% | 0.296 (n.s.) |
| notMNIST | 85.80% | 86.40% | 0.508 (n.s.) | 88.20% | 88.60% | 0.774 (n.s.) |
| Kuzushiji-MNIST | 61.20% | 61.00% | 1.000 (n.s.) | 65.20% | 66.80% | 0.302 (n.s.) |

**Oscillator-derived E significantly outperforms simple-E nowhere.** Five
of the six comparisons show no significant difference. The sixth --
Fashion-MNIST's direct comparison -- nominally favors simple-E
(p=0.0241), not E. Treating the six simple-E comparisons as one
exploratory family, a Bonferroni threshold is 0.05/6 ≈ 0.0083, which the
Fashion-MNIST result does not survive either -- so its correct status is
*nominal evidence favoring simple-E on Fashion-MNIST, not robust to
correction across the six comparisons*, exactly symmetric with how
Kuzushiji-MNIST's residual-E nominal result was treated earlier in this
document. On the Kuzushiji-MNIST conditional-residual comparison
specifically -- the one place oscillator-derived E had shown any
(already nominal) unique value against G_spatial alone -- residual-E
(66.80%) is numerically higher than residual-simple-E (65.20%), but
p=0.302 means the experiment did not detect a difference. **This does not
establish that the two are equivalent, or that the observed 1.6-point gap
is zero or practically irrelevant** -- establishing that would require a
prespecified equivalence margin, a confidence interval on the paired
difference, or a formal non-inferiority test, none of which were run
here. The defensible claim is narrower and still sufficient for the
practical conclusion: *across the tested datasets and comparisons, there
is no statistically supported evidence that oscillator-derived E
outperforms simple-E.* That is enough to remove oscillator-derived E from
the default benchmark representation; it does not require proving exact
equivalence.

**This closes the narrow question as far as the present evidence
permits, without overstating what "closed" means.** Across every test run
in this investigation -- reconstruction asymmetry, direct classification,
and conditional residual, on all three datasets -- no tested comparison
provides statistically supported evidence that oscillator-derived
supports contribute additional E-based classification value beyond
class-template supports. The full sequence: E improves on T; E is
largely linearly recoverable from generic spatial structure; that
recoverability is not shown to require oscillator dynamics (a trivial
control reproduces it); and the predictive behavior built on top of that
recoverability does not show oscillator-derived E outperforming the
trivial control either, anywhere it was tested. The observed
Kuzushiji-MNIST residual difference remains a numerical pattern, not an
established effect -- it is not load-bearing for any claim in this
document.

## Scoping note: "retire E from the benchmark" is not "retire E from the model family"

Every conclusion in this document concerns one specific use of E: as an
engineered feature exported to a linear classifier running on cached
digital features. Within that scope, the evidence is now complete and the
conclusion is firm -- there is no statistically supported reason to prefer
oscillator-derived E over G_spatial or over a trivial class-template
construction. That conclusion does not extend to a different use of E:
as an intrinsic state variable or native readout of a physical dynamical
system.

The comparison this document ran was, implicitly, "compute E versus
compute G_spatial on a conventional digital processor." In a biologically
constrained, analog, or neuromorphic implementation, that is not
necessarily the relevant comparison. The more relevant one may be "E
emerges locally from ongoing dynamics" versus "G_spatial requires
explicit Cartesian coordinate access, global moments, multiplication,
summation, and normalization" -- under those constraints, the apparently
simpler Cartesian representation may be the computationally less natural
one, not the more efficient one. E could remain valuable there for
reasons this benchmark cannot see: local or distributed computation,
compatibility with event-driven signals, operation without explicit
coordinate access, implementation through oscillator coupling or phase
relationships directly, low-power analog accumulation, graceful
degradation under noisy or missing units, or direct composability with
graph-native downstream processes. None of those properties are tested
by logistic regression on cached features, and nothing in this document
speaks to them either way.

Two uses of E should therefore be kept explicitly distinct going forward:
**E as an engineered feature** exported to a classifier -- the present
evidence gives little reason to prefer it over G_spatial or simple-E, and
it should be retired from the default benchmark feature vector.
**E as an intrinsic state variable or readout of a physical dynamical
system** -- its value remains open, because it may be cheap, native, and
operationally useful even where its information content is reproducible
elsewhere in a digital pipeline. The correct scope of this document's
conclusion is: *retire E from the default benchmark feature vector*, not
*retire E from the model family*.

## Honest limitations

- Ridge residualization at a single alpha (1.0), consistent with (and
  subject to the same caveats as) every other residualization in this
  project -- multiple ridge strengths were not re-tested here.
- This test used tuned regularization throughout, so there is no
  fixed-vs-tuned comparison to report, unlike the E-against-T and
  R-against-(T,E) tests.
- The very high R² values (up to 0.995) mean the residual components in
  several dimensions are extremely small in absolute terms -- the
  nominal Kuzushiji-MNIST residual-E effect rests on a genuinely thin
  slice of E's total variance and does not survive correction across the
  six directional comparisons tested in this document.

## Closing this thread before moving on

This document set out to establish three things, progressively narrower:
whether E is predictively privileged over generic spatial structure (no);
whether its high overlap with spatial structure reflects something
specific to oscillator dynamics (no, per the class-template control);
and whether oscillator-derived E retains any predictive edge over that
same trivial control, even in its own strongest remaining case -- the
Kuzushiji-MNIST residual (no, per the direct comparison just above,
where the residual gap is not significant and the direct comparison
elsewhere sometimes favors the trivial control outright). All three are
now answered as precisely as this evidence supports, and none of them
leave a load-bearing thread still open. What remains genuinely open --
and was never addressed by any test in this document, since nothing here
touches a task other than linear classification -- is whether the
oscillator-derived graph structure makes some *other* computation
(diffusion, segmentation, path planning, or the other candidates raised)
easier or more natural than the equivalent operation on raw pixels. That
is a different question from anything tested here, and answering it will
need different experiments, not a re-reading of this one.

## Immediate next steps

1. Given this result, the frequency-band ablation for R should be
   redesigned from the start around a dimensionality-matched,
   class-agnostic *image-domain* low-frequency control (DCT/Fourier
   energy, coarse spatial pooling, or low-order wavelet energy), per the
   review's explicit warning -- not run first and corrected after, the
   way this E investigation proceeded.
2. If E is to be retained in the final representation at all, its
   justification should now rest on practical grounds (if any exist --
   robustness, transfer, sample efficiency, calibration) rather than
   informational uniqueness, which this test does not support.

## Reproducing these results

Ridge regression (alpha=1.0) fit in both directions between E and
G_spatial on training data for each dataset; residuals evaluated with
`LogisticRegressionCV` throughout. No new per-image computation required
-- all features reused from existing caches.
