# Third Transfer Target: notMNIST -- A New Density Regime and a New Pattern in the Overlap-Confusion Relationship

## Context and data note

Kannada-MNIST was found, on closer inspection, not to be directly
available in the `vinayprabhu/Kannada_MNIST` GitHub repository in a
readily downloadable form (the repository is predominantly notebooks and
code, and the specific data subdirectory structure could not be resolved
within this session's tool constraints) -- it was dropped from this round
of transfer testing rather than substituted with an unverified guess.
notMNIST (`davidflanagan/notMNIST-to-MNIST`, letters A-J as 10 classes,
font glyphs rather than handwriting) was confirmed accessible (gzipped IDX
files committed directly to the repository, same pattern as Fashion-MNIST)
and is the subject of this document.

## Result 1: classification accuracy -- the strongest transfer result yet

**80.4% at 200 images/class (8.04x chance), identical frozen
architecture**, evaluated on the same stratified 500-image test sample
convention as every other transfer target. This exceeds both prior
transfer results (Fashion-MNIST 71.4%, Kuzushiji-MNIST 52.4%).

## Result 2: a third, intermediate density regime

| Statistic (200/class) | MNIST | Kuzushiji-MNIST | notMNIST | Fashion-MNIST |
|---|---|---|---|---|
| Edges | 418-725 | 685-1,672 | 2,085-6,985 | 3,388-39,617 |
| Mean node degree | 3.6-5.1 | 3.6-6.2 | 5.5-17.8 | 14.0-126.6 |
| Largest component size | 175-341 | 376-616 | 603-784 | 336-688 |
| Mean edge length | 1.30-3.68px | 1.15-1.50px | 1.55-3.01px | 4.03-8.42px |

notMNIST occupies a genuinely new position, not simply "closer to one end
or the other": its edge count and mean degree sit clearly **between** the
sparse handwritten-script regime (MNIST, Kuzushiji-MNIST) and the dense
clothing regime (Fashion-MNIST) -- consistent with font glyphs being
bolder and more solid than handwritten strokes, but not as spatially
extensive as full garment silhouettes. At the same time, its connected
components (603-784, frequently spanning nearly the entire 784-pixel
image) are the **least fragmented of any dataset tested**, even less than
Fashion-MNIST's -- a property not predicted by density alone. A plausible
explanation: computer-rendered font glyphs are more internally consistent
across samples within a class than either handwriting or photographed
clothing, which could produce more uniformly-connected topologies even at
moderate per-pixel degree. This is offered as interpretation, not
verified further here.

**One likely data-quality artifact, disclosed rather than smoothed over**:
class 8 ("I") is an outlier on every structural measure (6,985 edges,
784-of-784 ink pixels, mean degree 17.8 -- reaching into Fashion-MNIST's
range). The source repository's own README describes notMNIST as "harder
and less clean than MNIST," and this specific class's near-total ink
coverage is consistent with known rendering/corruption issues in the
dataset, not necessarily a deep structural finding about the letter "I"
itself.

## Result 3: the confusion pattern shows real, sensible visual structure

E confused with F (7 misclassifications) and F confused with E (8) --
both have prominent horizontal-bar letterforms in many fonts. I confused
with J (5) and J confused with I (3) -- J is often visually just an I with
a small hook, a well-known near-ambiguity in many typefaces. These are
exactly the kind of confusions a human would find unsurprising, unlike
Kuzushiji-MNIST's more diffuse pattern.

## Result 4: the overlap-confusion relationship shows a third, distinct pattern

| Comparison | Spearman rho | Mantel p-value |
|---|---|---|
| Bonsai confusion vs. raw overlap | 0.270 | 0.116 (not significant) |
| Bonsai confusion vs. Jaccard-normalized overlap | 0.140 | 0.402 (not significant) |
| MLP confusion vs. raw overlap | 0.397 | 0.0024 (significant) |
| MLP confusion vs. Jaccard-normalized overlap | 0.308 | 0.033 (significant) |
| Bonsai confusion vs. MLP confusion (direct) | 0.291 | 0.051 (borderline) |

Three genuinely different patterns have now been observed across three
transfer targets:

| Dataset | Overlap predicts Bonsai's confusion? | Overlap predicts MLP's confusion? | Bonsai/MLP agree directly? |
|---|---|---|---|
| Fashion-MNIST | Yes, strongly (p=0.0024) | Yes, strongly (p=0.0021) | Yes, strongly (p<0.0001) |
| Kuzushiji-MNIST | No (p=0.053, borderline) | No (p=0.104) | Yes (p=0.0029) |
| notMNIST | No (p=0.116-0.402) | **Yes (p=0.0024-0.033)** | Borderline (p=0.051) |

**notMNIST is the first case where the topology overlap matrix correlates
significantly with the MLP's confusion but not with Bonsai's own.** This
is a different failure mode than Kuzushiji-MNIST's (where overlap failed
to predict *either* classifier's confusion, though the two classifiers
still agreed with each other). Here, the overlap structure appears to
contain real signal -- confirmed independently through its correlation
with a completely different, fully-supervised model -- that Bonsai's own
specific shallow readout does not fully exploit. That is a different,
more precise claim than "the phenomenon doesn't replicate": it points at a
possible gap between what information is *present* in the topology and
what Bonsai's particular scoring mechanism *uses*, rather than an absence
of real structure.

**Also notable, and a caution against over-interpreting any single normalization
choice as universally correct**: Jaccard normalization *strengthened* the
correlation on both Fashion-MNIST and Kuzushiji-MNIST, but *weakened* it
here (raw 0.270 to Jaccard 0.140 for Bonsai; 0.397 to 0.308 for the MLP).
Plausibly related to the extreme edge-count outlier (class 8, "I") --
dividing by a very large union value for pairs involving that class may
compress real signal rather than remove a pure scale confound in this
particular case. No single normalization choice should be assumed correct
across all datasets without checking.

## Testing the reviewer's refined hypothesis

The hypothesis proposed after Kuzushiji-MNIST was: *overlap predicts
confusion when the dominant discriminative structure exists at the
spatial scale captured by the topology statistic.* notMNIST's confusions
(E/F sharing a horizontal bar; I/J sharing a vertical stem, differing by a
small hook) are, like Kuzushiji-MNIST's, plausibly driven by fairly local,
specific stroke-level features rather than broad regional similarity --
yet here the overlap statistic *did* correlate with the MLP's confusion,
just not Bonsai's own. This is not a clean confirmation or refutation of
the hypothesis; it suggests the picture may need a further distinction
between "does the discriminative structure exist at the right spatial
scale" and "does Bonsai's specific 20-feature summary of the topology
capture that scale's information as effectively as a richer functional of
the graph would" -- exactly the direction the reviewer's suggested next
step (testing degree distributions, spectral properties, community
structure as alternative summaries) would help resolve.

## Honest limitations

- Single run, no repeated seeds, same caveat as every dataset in this
  project's transfer-testing history.
- The 500-image test sample convention (not the full 10,000) applies here
  too, for the same disclosed compute-cost reasons.
- Kannada-MNIST was not tested in this round; the access difficulty was
  disclosed rather than resolved with a substituted or unverified dataset.
- The explanation for notMNIST's unusually low fragmentation (consistent
  font rendering) and for the Jaccard-normalization reversal (the "I"
  outlier) are both offered as interpretation, not independently verified.

## Where this leaves the emerging picture, across four datasets now

Density tracking visual form, not dataset identity, is now supported
across three independent domains (two sparse: MNIST, Kuzushiji-MNIST; one
dense: Fashion-MNIST; one intermediate: notMNIST) -- a genuinely
strengthened pattern. The overlap-predicts-confusion phenomenon, by
contrast, has now shown three qualitatively different behaviors in three
attempts (strong/clean on Fashion-MNIST, absent-but-cross-classifier-
agreement-intact on Kuzushiji-MNIST, present-for-MLP-but-not-Bonsai on
notMNIST) -- this is best described as a real, still-only-partially-
understood phenomenon under active characterization, not a settled
property with a single clean boundary condition.

## Reproducing these results

notMNIST IDX files: `https://raw.githubusercontent.com/davidflanagan/notMNIST-to-MNIST/master/`
(gzipped IDX files committed directly to the repository). Pipeline code
identical to every other frozen-configuration transfer experiment in this
project.
