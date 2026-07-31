# Capacity Experiment II: Degree-Aware / Graph-Smoothness Readout

## The question, and why this functional specifically

Capacity Experiment I ruled out one family of explanation: a second
pruning threshold on the *same* pairwise-correlation summary does not
detectably recover information beyond generic dimensionality effects.
That narrowed the hypothesis from "richer readout" to "richer readout
based on a genuinely different graph functional, not just another view of
the same one." This experiment tests the top-ranked such functional:
**graph smoothness**, x^T L x, computed against each class's Laplacian.

This is qualitatively different information from the original 20D
readout. The pairwise-correlation score (simple/cosine) asks: *does this
image's own pairwise correlation structure agree with the class's
diagnostic connections?* Smoothness asks a different question entirely:
*does this image's own per-pixel signal vary smoothly or roughly across
the edges the class considers important?* One compares structure to
structure; the other compares a signal to a structure. All topology edge
weights were confirmed positive (0.9 to ~0.999) before proceeding, so the
standard graph Laplacian quadratic form is well-defined without requiring
a signed-Laplacian variant.

## Methodology

Same frozen configuration and test set as every experiment in this
series: Fashion-MNIST, 200 images/class reference topologies (reused
directly from cache, no rebuilding), same 300/class hybrid-head training
range, same frozen 500-image test sample. The per-pixel phase signal used
for smoothness is extracted from the same oscillator dynamics run
(`get_local_converged_phases`) already used to build the pairwise
correlation matrix -- computing smoothness alongside the original score
where both are needed, rather than re-running the dynamics twice.

## Result 1: smoothness alone is a real, working signal

**10D smoothness-only: 70.20% accuracy** -- close to, though slightly below,
the original 20D pairwise-correlation approach (71.40%), and far above
chance (10%). This confirms smoothness is not merely noise riding on the
classifier's capacity; it is real, independently-useful information
derived from a genuinely different computation on the same topologies.

## Result 2: combining smoothness with the original 20D shows the same pattern as Capacity Experiment I

| Condition | Accuracy | Change from 20D |
|---|---|---|
| 20D baseline | 71.40% | -- |
| 10D smoothness-only | 70.20% | -1.20pp |
| **30D combined (20D + smoothness)** | **71.80%** | **+0.40pp** |
| 30D duplicated-10 control (10 redundant copied dims) | 71.60% | +0.20pp |
| 30 random-feature control | 72.00% | +0.60pp |

**The same pattern as the multi-threshold experiment recurs**: the real
combined representation does not exceed either control. This is a second,
independent negative result for the "combining functionals helps the
shallow linear readout" hypothesis -- now tested with a functional that is
genuinely different (confirmed by its standalone 70.2% accuracy), not
merely a second view of the same pairwise-correlation summary. That
strengthens, rather than weakens, the negative finding: it is not simply
that a second pruning threshold happened to be too similar to the first --
a functional that is clearly different on its own terms still does not
combine to beat generic dimensionality effects in this shallow classifier.

## Result 3: a genuinely new observation -- smoothness alone tracks the MLP's confusion more closely than the original readout does

| Comparison | Spearman rho | Mantel p-value |
|---|---|---|
| 10D smoothness-only confusion vs. topology overlap | 0.382 | 0.050 (borderline) |
| **10D smoothness-only confusion vs. MLP confusion** | **0.829** | **<0.0001** |
| 30D combined confusion vs. topology overlap | 0.420 | 0.027 |
| 30D combined confusion vs. MLP confusion | 0.800 | <0.0001 |
| *(20D baseline, for reference)* | *0.424 / 0.794* | *0.024 / <0.0001* |

Smoothness alone shows the highest MLP-confusion correlation of any single
condition tested so far in this series (0.829, versus 20D's 0.794) --
descriptively, not as a statistically compared difference; no test was
run between these two rho values specifically, matching the same caveat
applied throughout this series. This did not carry through to the 30D
combined condition (0.800, essentially matching the 20D baseline), which
is itself informative: whatever smoothness contributes on its own does
not straightforwardly persist when concatenated with the original 20D and
handed to the same shallow classifier.

## Interpreting the pattern across both capacity experiments

Two structurally different functionals (a second pruning threshold;
graph smoothness) have now both failed to improve on generic
dimensionality effects when simply concatenated with the original 20D
readout and handed to the same shallow `LogisticRegression`. Taken
together with smoothness's strong standalone performance and its notably
high standalone correlation with the MLP's confusion pattern, the most
defensible interpretation is that **the shallow linear combination step,
not the underlying graph information, may be the current bottleneck**.
Both functionals appear to carry real, partially non-redundant
information (their standalone confusion patterns are not identical, and
smoothness's MLP-correlation profile looks meaningfully different in
character from the original readout's), but naively concatenating raw
features and fitting a linear head does not appear to be an effective way
to combine them. This reframes the open question again: not "which
functional contains more information" but "what combination mechanism
would actually expose the complementary information multiple functionals
seem to carry individually."

## Honest limitations

- Single run throughout, same caveat as every experiment in this series.
- The 30D duplicated-10 control pads with 10 copied columns from the
  original 20D (specifically the "simple" scores) rather than a
  different padding scheme -- a different choice of which 10 columns to
  duplicate was not tested and could plausibly shift the control's exact
  value somewhat, though not the qualitative conclusion given how close
  all three 30D conditions are to each other.
- Why smoothness's standalone MLP-correlation is higher than the original
  readout's, and why this does not persist in combination, is not
  investigated further here -- offered as an observation motivating
  future work, not an explained mechanism.
- Only Fashion-MNIST was tested in this experiment; whether this pattern
  (real standalone signal, no benefit from naive combination) holds on
  the other transfer datasets is untested.

## Reproducing these results

`graph_smoothness.py` (new this experiment) plus the existing
`developmental_pruning.py`, `topology_matching_classifier.py`, and the
cached Fashion-MNIST artifacts (`fmnist_class_topologies_200.pkl`,
`fmnist_hybrid_train_batch{1,2}.pkl`, `fmnist_test_features.pkl`,
`fmnist_baselines.pkl`).
