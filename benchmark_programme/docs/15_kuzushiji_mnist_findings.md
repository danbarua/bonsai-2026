# Second Transfer Target: Kuzushiji-MNIST -- Testing the Boundary of the Representational-Validity Claim

## Context and motivation

The Fashion-MNIST transfer established a striking result: Bonsai's
topology-overlap matrix, computed before any classifier exists, predicts
both Bonsai's own confusion pattern and an independently-trained MLP's
confusion pattern. Kuzushiji-MNIST (10 hiragana characters, same 28x28
format, same class count) was chosen as the next transfer target
specifically because it occupies a different position in the space of
possible outcomes: still sparse, stroke-based imagery like MNIST, but with
substantially higher within-class variation (different historical
calligraphic styles) and, unlike Fashion-MNIST, no obvious compositional
visual hierarchy between classes (ten distinct phonetic characters, not
categories like "upper-body garment" or "footwear" that share large
silhouette regions). The three anticipated outcomes, stated in advance:
MNIST-like sparse topologies with different overlap geometry; Fashion-like
correspondence, generalizing the phenomenon further; or a weak/failed
correlation, usefully bounding the claim. **The result obtained is
closest to the third, and is reported as such.**

**Data note**: the canonical host (codh.rois.ac.jp) is not reachable from
this project's sandboxed network environment; the four IDX files used
here were provided directly by the user after that constraint was
identified and disclosed.

## Result 1: classification accuracy is real but weaker than Fashion-MNIST

**52.4% at 200 images/class (5.24x chance for 10 classes)**, identical
frozen architecture, evaluated on a stratified 500-image test sample.
Notably weaker than Fashion-MNIST's 71.4%, though still clearly and
substantially above chance. This is consistent with KMNIST's known
difficulty in the broader literature -- even strong CNNs in the original
KMNIST paper score lower on this dataset than on MNIST, attributed to
higher within-class calligraphic variation.

## Result 2: structural statistics confirm density tracks visual form, independent of the specific dataset

| Statistic (200/class) | MNIST | Fashion-MNIST | Kuzushiji-MNIST |
|---|---|---|---|
| Edges | 418-725 | 3,388-39,617 | 685-1,672 |
| Mean node degree | 3.6-5.1 | 14.0-126.6 | 3.6-6.2 |
| Max node degree | 11-22 | 77-306 | 11-25 |
| Mean edge length | 1.30-3.68px | 4.03-8.42px | 1.15-1.50px |
| Largest component size | 175-341 | 336-688 | 376-616 |

**Kuzushiji-MNIST's density sits clearly in the same regime as MNIST**
(mean degree 3.6-6.2 vs MNIST's 3.6-5.1), both roughly two orders of
magnitude sparser than Fashion-MNIST (14.0-126.6) -- confirming the
hypothesis from the Fashion-MNIST work: **topology density tracks visual
form (thin strokes vs. spatially extensive regions), not the specific
dataset**, since two unrelated stroke-based character sets (digits and
hiragana) land in the same density regime despite having nothing else in
common. At the same time, Kuzushiji-MNIST is not identical to MNIST --
its edges are even more local on average (1.15-1.50px vs. 1.30-3.68px)
and its connected components are noticeably larger and less fragmented
(376-616 vs. 175-341), consistent with Kuzushiji characters typically
having more interconnected, curved strokes than simple digit shapes.

## Result 3: the topology-overlap-predicts-confusion phenomenon does not replicate with the same strength here

| Comparison | Spearman rho | Mantel p-value | Fashion-MNIST equivalent |
|---|---|---|---|
| Raw overlap vs. Bonsai's own confusion | 0.166 | 0.368 (not significant) | 0.424 (p=0.024) |
| Jaccard-normalized overlap vs. Bonsai's own confusion | 0.320 | 0.053 (borderline) | 0.578 (p=0.0024) |
| Raw overlap vs. MLP confusion | 0.279 | 0.108 (not significant) | 0.529 (p=0.0011) |
| Jaccard-normalized overlap vs. MLP confusion | 0.268 | 0.104 (not significant) | 0.636 (p=0.0021) |
| **MLP confusion vs. Bonsai confusion (direct)** | **0.456** | **0.0029 (significant)** | 0.794 (p<0.0001) |

**This is a precise, informative boundary, not a simple failure to
replicate.** The direct agreement between Bonsai's confusion matrix and an
independently-trained MLP's confusion matrix remains statistically
significant (p=0.0029) -- the two systems still agree, more than chance
would predict, on which characters are hard to tell apart, suggesting real
shared difficulty structure exists in this dataset. But **neither
classifier's confusion pattern correlates significantly with the topology
overlap matrix specifically** (all overlap-related comparisons, p>0.10).
The mechanism that successfully predicted confusion before training on
Fashion-MNIST does not transfer its predictive power to Kuzushiji-MNIST,
even though some form of genuine, cross-classifier-detectable dataset
structure still appears to be present.

**A plausible, disclosed-as-interpretation-not-fact explanation**: Fashion-
MNIST's classes share large, compositional silhouette regions (upper-body
garments occupy similar large areas of the frame; footwear occupies a
different characteristic region) that a population-correlation-based
topology can capture cleanly as overlap. Individual phonetic characters,
even visually similar ones, may share local stroke-level similarities that
this specific topology-construction procedure (built around a fixed
intensity threshold and background/ink partition, not stroke-level shape
matching) is not well-suited to detect as directly. This has not been
verified further and should be treated as a hypothesis for future
investigation, not an established mechanism.

## The MLP baseline (for reference)

`sklearn.MLPClassifier`, identical architecture and training-image budget
as the Fashion-MNIST comparison (hidden layers (128, 64), 500
images/class, same 500-image test sample). Reached **75.0% accuracy** --
a substantially larger gap over Bonsai's 52.4% (22.6 points) than was seen
on Fashion-MNIST (81.0% vs. 71.4%, a 9.6-point gap), consistent with
Kuzushiji-MNIST's higher difficulty disproportionately affecting Bonsai's
simpler, shallower mechanism relative to a fully-supervised, deeper
alternative.

## What this means for the representational-validity claim

The claim established on Fashion-MNIST cannot be stated as a general
property of the topology-as-representation approach across arbitrary
domains -- it should be scoped to what has actually been shown twice now:
**Bonsai's learned topology reliably transfers as a classifier (real,
above-chance accuracy on both additional domains tested), and on at least
one domain with strong compositional visual-category structure, its
overlap geometry predicts confusion before training occurs. Whether the
overlap-geometry-predicts-confusion phenomenon specifically requires that
kind of compositional category structure, or is a more general property
that simply needs a better-suited topology construction to detect on
domains like Kuzushiji-MNIST, is an open question this experiment
distinguishes but does not resolve.**

## Honest limitations

- Single run, no repeated seeds, matching every other unreplicated result
  in this project's history -- exact percentages and correlation values
  should be read with that caution.
- The 500-image test sample (not the full 10,000) applies here too, for
  the same disclosed compute-cost reasons as Fashion-MNIST.
- The explanation offered for *why* the correlation weakens (compositional
  silhouette structure vs. local stroke similarity) is a hypothesis, not
  a tested claim -- no experiment here isolates that specific mechanism.
- Only two transfer targets have been tested. Whether Fashion-MNIST or
  Kuzushiji-MNIST is the more "typical" case for arbitrary future domains
  is unknown.

## Reproducing these results

Kuzushiji-MNIST IDX files are hosted only at `codh.rois.ac.jp` (not
reachable from this project's sandbox); obtain via the official
`rois-codh/kmnist` repository's instructions, or via `torchvision.datasets.KMNIST`
/ `tensorflow_datasets` if available in your environment. Pipeline code
is identical to the frozen MNIST and Fashion-MNIST configurations.
