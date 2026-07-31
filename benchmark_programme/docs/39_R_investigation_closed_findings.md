# The R Investigation, Closed: Bandwise Ablation, Image-Domain Control, and Graph-Genericity Control

## Scope and discipline

This document answers the four prespecified questions in the order
specified, using one primary control per question (DCT for image-domain
frequency, a regular pixel-neighbor lattice graph for graph-genericity),
not a search across multiple candidate controls for the strongest story.
Tuned regularization (`LogisticRegressionCV`) throughout. All feature
computation reused cached oscillator phases from earlier in this session
-- no new dynamics simulation was required for any control in this
document.

## Question 1: what part of R carries the signal?

Bandwise ablation (low/mid/high, matched dimensionality) from both T+E
and the cleaner T+G_spatial baseline:

| Dataset | Baseline | R_low | R_mid | R_high |
|---|---|---|---|---|
| Fashion-MNIST | T+E=76.00%, T+G=78.00% | T+E+R: 77.40% (p=0.360) / T+G+R: 79.20% (p=0.430) | 76.00% (p=1.000) / 76.20% (p=0.163) | **80.20% (p=0.0031)** / 80.40% (p=0.065) |
| notMNIST | T+E=86.40%, T+G=86.80% | 87.00% (p=0.678) / 88.00% (p=0.345) | 85.40% (p=0.180) / 85.60% (p=0.180) | 86.40% (p=1.000) / 86.40% (p=0.774) |
| Kuzushiji-MNIST | T+E=61.00%, T+G=62.80% | 64.20% (p=0.052) / 64.40% (p=0.403) | 61.00% (p=1.000) / 61.60% (p=0.497) | 62.00% (p=0.500) / 64.40% (p=0.366) |

**No band reaches significance against the cleaner T+G_spatial baseline,
on any dataset.** The one nominal hit -- Fashion-MNIST's R_high against
T+E (p=0.0031) -- is conventionally significant and fairly small on its
own, but this document's bandwise family spans 18 comparisons (3 bands x
2 baselines x 3 datasets); a Bonferroni threshold across that family is
0.05/18 ≈ 0.00278, which p=0.0031 narrowly fails. The precise
description: **one isolated high-band effect was detected against T+E on
Fashion-MNIST (p=0.0031); it narrowly fails Bonferroni correction across
the 18 band-by-baseline comparisons, does not survive substitution of the
stronger T+G baseline, and does not replicate across datasets.** It is
also the *high*-frequency band, not the low-frequency band the original
spectral hypothesis was built around. **The sharper, defensible
conclusion: the prespecified low-frequency hypothesis failed. No R band
adds significantly beyond T+G_spatial, and the only isolated bandwise
effect occurs in the high-frequency band against the weaker T+E
baseline** -- this is a genuine negative mechanistic result, not merely an
absence of a positive one.

## Question 2: can class-agnostic image-domain frequency features replace R?

DCT chosen as the single prespecified primary control (cleanest choice
for fixed 28x28 images, per the explicit instruction not to search across
DCT/Fourier/wavelet variants for the best story). 10 lowest-order 2D DCT
coefficients, dimensionality-matched to R.

| Dataset | T+G+R | T+G+F_dct | McNemar p (R vs. F) | R²(R←F_dct) | R²(F_dct←R) |
|---|---|---|---|---|---|
| Fashion-MNIST | 79.20% | 79.20% | 1.000 | 0.545 | 0.504 |
| notMNIST | 88.00% | 89.00% | 0.487 | 0.364 | 0.341 |
| Kuzushiji-MNIST | 64.40% | 65.80% | 0.494 | 0.208 | 0.291 |

F_dct matches or numerically exceeds R directly on every dataset, never
significantly behind -- **but "replaceable by image frequency" is too
broad a summary for Kuzushiji-MNIST**, where the residual test below
shows R is not shown to be fully linearly subsumed by DCT, and retains a
nominal residual beyond it, even though R shows no direct classifier
advantage over DCT there. Overlap is
moderate (23-55% depending on direction and dataset), roughly symmetric
on Fashion-MNIST and notMNIST, with somewhat more independence on
Kuzushiji-MNIST.

**Bidirectional residualization, with multiplicity calibration.** Six
directional residual tests appear across this document's Question 2 (2
directions x 3 datasets); a Bonferroni threshold across them is
0.05/6 ≈ 0.0083.

| Dataset | T+G+F+residual-R (does R add beyond F_dct?) | T+G+R+residual-F (does F_dct add beyond R?) |
|---|---|---|
| Fashion-MNIST | 80.20% (p=0.473, n.s.) | 80.40% (p=0.362, n.s.) |
| notMNIST | 89.60% (p=0.549, n.s.) | 90.40% (p=0.0118 -- nominal only, does not survive correction) |
| Kuzushiji-MNIST | 69.60% (p=0.0183 -- nominal only, does not survive correction) | **69.60% (p=0.00041 -- survives correction)** |

Fashion-MNIST: neither residual matters -- R's value is adequately
explained by ordinary image-frequency structure. notMNIST: F_dct shows
nominal incremental value beyond R, but this does not survive correction
across the six directional comparisons; R shows no incremental value
beyond F_dct at all. Kuzushiji-MNIST: both directions are nominally
significant, but **only DCT-beyond-R survives Bonferroni correction** --
the robust result is that DCT retains information beyond R; the reverse
(R beyond DCT) is the weaker, nominal-only direction. Even after this
correction, R retains a nominal (uncorrected) residual beyond DCT
(p=0.0183), so "superset" overstates it -- a superset claim implies
containment, and the evidence instead shows asymmetric conditional value.
**DCT is the broader and more strongly supported representation on
Kuzushiji-MNIST: its residual beyond R survives correction, whereas R's
residual beyond DCT does not** -- not a tied pair of mutually-exclusive
residuals, but not strict containment either.

## Question 3: is any surviving R signal specific to oscillator-derived topology?

Primary control: a regular 4-connectivity pixel lattice graph (unit edge
weights, zero phase correlations anywhere), restricted to the same
active-node set as the real topology for a fair comparison, spectrally
decomposed identically to R.

| Dataset | T+G+R | T+G+R_lattice_control | McNemar p | T+G+F+residual-R | T+G+F+residual-R_control | McNemar p |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 79.20% | 79.20% | 1.000 | 80.20% | 79.60% | 0.701 |
| notMNIST | 88.00% | 87.00% | 0.424 | 89.60% | 89.80% | 1.000 |
| Kuzushiji-MNIST | 64.40% | **67.00%** | 0.148 | 69.60% | 69.00% | 0.795 |

**No significant difference detected between oscillator-derived R and
the lattice control, on any dataset, on either the direct or the
conditional-residual comparison.** Precisely, not "the lattice control
matches or exceeds R everywhere": oscillator-derived R is numerically
ahead in three of the six comparisons (Fashion-MNIST direct: tied;
notMNIST direct: R ahead 88.0 vs. 87.0; Kuzushiji-MNIST direct: lattice
ahead 67.0 vs. 64.4; Fashion-MNIST residual: R ahead 80.2 vs. 79.6;
notMNIST residual: lattice ahead 89.8 vs. 89.6; Kuzushiji-MNIST residual:
R ahead 69.6 vs. 69.0). No pattern favors either construction
consistently. **The supported conclusion is absence of detected oscillator
privilege, not equivalence or literal matching**: no statistically
supported evidence that oscillator-derived topology contributes anything
R-specific beyond what a trivial regular pixel-adjacency graph provides,
while stopping short of claiming the two constructions are proven
interchangeable -- non-significance means the experiment did not detect a
difference, not that no true difference exists.

**One additional test, run to address whether the lattice residual
itself carries information, not merely whether it matches R's residual**:
does T+G+F_dct+residual(R_lattice) significantly beat T+G+F_dct alone on
Kuzushiji-MNIST? Yes -- 65.80% to 69.00%, McNemar p=0.0441. This is a
positive result for graph-genericity specifically (not just an absence of
evidence for oscillator-specificity): the lattice-derived residual itself
carries real incremental information beyond DCT, strengthening (though,
given the growing family of comparisons in this document, not
conclusively proving) the case that Kuzushiji-MNIST's surviving signal is
graph-generic rather than oscillator-specific.

## Question 4: recorded, not addressed here

Per the explicit instruction, this question is recorded and not allowed
to contaminate the closure of the present investigation: whether graph
spectral organization is useful as a native mode of activity propagation,
a substrate for diffusion or denoising, a mechanism for multiscale
integration, or a graph-native route to segmentation -- none of that is
addressed by anything in this document, which tests R only as an
exported static feature for a linear classifier. That question belongs to
the model-family programme, not this feature-level enquiry.

## Applying the outcome framework, revised classification

- **Fashion-MNIST**: R has no demonstrated advantage beyond generic
  spatial and DCT features. Neither residual direction is supported, and
  lattice-derived spectra behave comparably. The isolated high-band
  effect (below) is baseline-dependent and non-robust across the
  comparison family.
- **notMNIST**: DCT directly matches or exceeds R. DCT shows nominal
  residual value beyond R (not surviving correction), while R shows none
  beyond DCT. Neither oscillator R nor lattice R is privileged. The
  evidence leans toward ordinary image-frequency structure but is weak
  after multiplicity correction.
- **Kuzushiji-MNIST**: DCT contains robust information beyond R on
  Kuzushiji-MNIST, while R shows weaker, nominal residual information
  beyond DCT. A
  lattice graph produces comparable direct and residual performance, and
  its own residual beyond DCT is nominally significant in isolation
  (p=0.044) -- an exploratory addition to an already substantial
  comparison family, not incorporated into a prespecified correction
  family. This gives nominal positive support toward a graph-generic
  interpretation, without establishing it outright: the R residual here
  provides no evidence of oscillator specificity, and is compatible with
  graph-generic spectral structure, which is the weaker and
  well-supported claim this document actually needs.

**No dataset reaches the outcome where oscillator-derived R demonstrates
a privilege surviving both the image-domain control and the
graph-genericity control.** This is the defensible primary conclusion;
the finer distinctions above (which residual directions are nominal vs.
robust, where graph-genericity has positive rather than merely
by-elimination support) matter for precision but do not change this
overall verdict.

## Closure table

| Question | Required evidence | Status |
|---|---|---|
| Which spectral bands matter? | Bandwise ablation | Answered -- no band robust against the cleaner baseline; the one nominal hit is high-frequency, unreplicated, and does not survive the stronger baseline |
| Is R replaceable by image frequency? | Direct matched-control comparison | Directly matched or exceeded by DCT on all datasets; not shown to be fully linearly subsumed on Kuzushiji-MNIST, where R retains a nominal residual beyond DCT |
| Is there unique information in either direction? | Bidirectional reconstruction and residualization | Answered -- no on Fashion-MNIST; F_dct-favoring on notMNIST; bidirectional but F_dct-leaning on Kuzushiji-MNIST |
| Is any surviving signal graph-generic? | Non-oscillator graph control | Compatible with a graph-generic explanation: the lattice construction produces comparable direct and residual behavior, and its Kuzushiji-MNIST residual nominally improves beyond DCT; no comparison supports oscillator privilege |
| Is any surviving signal oscillator-specific? | Direct conditional comparison against graph control | Not supported on any dataset: no significant direct or conditional advantage of oscillator-derived R over the lattice construction was detected |
| Are effects robust? | Repeated splits or confidence intervals, multiplicity treatment | Partially addressed: regularization controlled consistently and multiplicity can be calibrated (see corrections above); split robustness and uncertainty intervals remain untested -- this closes the current protocol, not population-level invariance |
| Is the conclusion limited to exported features? | Explicit scope statement | Yes -- see Question 4 and the scoping note below |

## Scoping note, identical in structure to the E investigation's

Every conclusion here concerns R as an engineered feature exported to a
linear classifier. It does not address R, or graph spectral structure
more generally, as an intrinsic property of a physical dynamical system.
The same distinction established for E applies here: **retire R from the
default benchmark feature vector** (the evidence supports this cleanly)
is a different claim from **R has no role in any architecture** (nothing
here tests that, and the same neuromorphic/analog considerations raised
for E apply with at least equal force to spectral graph structure, which
is a natural candidate for native computation in coupled-oscillator
hardware).

## Revised evidence hierarchy

| Claim | Status |
|---|---|
| The original low-frequency R hypothesis is supported | No |
| Any R band robustly improves beyond T+G_spatial | Not supported |
| DCT directly matches or numerically exceeds R | Established descriptively |
| R significantly outperforms DCT | Not supported, anywhere |
| DCT contains information beyond R | Strongly supported on Kuzushiji-MNIST (survives correction); nominal only on notMNIST |
| R contains information beyond DCT | Nominal evidence on Kuzushiji-MNIST; does not survive correction |
| Oscillator-derived R outperforms lattice-derived R | Not supported anywhere tested |
| Oscillator and lattice R are equivalent | Not established (absence of evidence, not evidence of absence) |
| Kuzushiji-MNIST's residual is oscillator-specific | Not supported |
| Kuzushiji-MNIST's residual is compatible with graph-generic structure | Supported; nominal positive evidence from the lattice residual (p=0.0441, exploratory, not part of a prespecified correction family) |
| R should remain in the benchmark feature vector | Not supported |
| Graph spectral modes may matter as native dynamical variables | Untested and open |

R does not support its original interpretation as a privileged
low-frequency readout of oscillator-derived topology. No spectral band
adds robustly beyond the generic spatial baseline, and a
dimensionality-matched DCT representation directly matches or exceeds R
on every dataset. On Kuzushiji-MNIST, DCT's residual beyond R survives
correction while R's residual beyond DCT does not -- DCT is the broader,
more strongly supported representation there, though not a strict
superset given R's nominal residual. No statistically significant direct
or conditional difference was detected between the regular-lattice and
oscillator-derived constructions, leaving no supported evidence that the
surviving signal depends specifically on oscillator topology. The
Kuzushiji-MNIST lattice-residual result (p=0.0441) gives nominal
positive support toward a graph-generic interpretation of the surviving
signal, but this was an exploratory addition to an already substantial
comparison family and should be read as nominal, not as establishing
graph-genericity outright -- the core conclusion only requires the
weaker, well-supported claim that oscillator specificity is unsupported.
These results do not establish equivalence between
the constructions, but they remove the empirical basis for retaining R
as a preferred exported classifier feature. The result is not merely "R
failed." The original low-frequency oscillator hypothesis failed;
ordinary image-frequency structure explains much of the exported
readout, while a generic lattice spectrum provides a viable non-oscillator
account of the residual behavior. Nothing presently observed requires
oscillator-derived topology. Graph spectral organization remains open as
an intrinsic computational property of the model family -- oscillator
specificity is unnecessary to explain the observations, which is a
different and weaker claim than the complete graph-generic mechanism
having been positively identified.

## Honest limitations

- One primary control per question, as instructed -- DCT was not compared
  against Fourier or wavelet alternatives, and the lattice graph is one
  reasonable non-oscillator construction among several the review
  suggested (class-template similarity graphs and degree-matched random
  graphs were not run).
- Ridge residualization at a single alpha (1.0), consistent with every
  other residualization in this project.
- No repeated splits or bootstrap confidence intervals were computed for
  any comparison -- consistent with the rest of this project's practice,
  and flagged consistently rather than newly here.
- Multiplicity was calibrated within the prespecified bandwise (18
  comparisons) and bidirectional-residual (6 comparisons) families,
  rather than across every statistical test in the entire document as
  one combined family. The additional Kuzushiji-MNIST lattice-residual
  comparison was exploratory and not incorporated into either
  prespecified correction family, so its p=0.0441 result should be
  treated as nominal, consistent with how it is described above. The
  qualitative conclusion (no dataset reaches a result establishing
  oscillator privilege) does not depend on any single comparison's exact
  significance level.

## Reproducing these results

`spectral_readout.py` extended with `build_spectral_basis_band`,
`normalized_band_score`, and `build_lattice_graph_basis`;
`global_ink_stats.py` extended with `dct_low_order_10d` (via
`scipy.fft.dctn`). All feature computation reused cached oscillator
phases (`{dataset}_phases_cache.npz`) from earlier in this session -- zero
new oscillator dynamics simulation was required for this entire
investigation.
