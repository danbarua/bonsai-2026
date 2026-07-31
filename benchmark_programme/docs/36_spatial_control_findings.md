# The Spatial-Structure Control: A Significant Revision to What "E Beats Generic Ink" Actually Showed

## What changed, and why it matters

The prior global-ink control used only aggregate, non-spatial statistics
(total sum, total squared energy, threshold counts) -- none of which
encode *where* in the image energy is located. E, by construction,
carries substantial implicit spatial information, since it is computed
from a topology built over specific pixel locations. The review correctly
identified that beating a non-spatial control does not yet distinguish
"E captures class-specific support alignment" from the much weaker "E
captures spatial information at all, which a merely spatially-aware
control would also provide." This test closes that gap with a new
10-dimensional, still entirely class-agnostic control: quadrant energies,
center of mass, second moments (row/column variance and covariance), and
total energy -- genuine spatial structure, but no class-specific topology
or support of any kind. Tuned regularization (`LogisticRegressionCV`) was
used throughout, from the start, given the lesson from the prior round.

## Result: the spatial control matches or exceeds E everywhere

| Dataset | T+spatial(10D) | T+E | McNemar p | T+R+spatial | T+E+R | McNemar p |
|---|---|---|---|---|---|---|
| Fashion-MNIST | **78.00%** | 76.00% | 0.229 (n.s.) | **78.80%** | 77.40% | 0.382 (n.s.) |
| notMNIST | **86.80%** | 86.40% | 0.860 (n.s.) | **88.00%** | 87.00% | 0.442 (n.s.) |
| Kuzushiji-MNIST | **62.80%** | 61.00% | 0.362 (n.s.) | 64.40% | 64.20% | 1.000 (n.s.) |

**On every dataset, the class-agnostic spatial control performs
numerically as well as or better than E** -- never significantly, but
never in E's favor either. This includes Kuzushiji-MNIST, where E's
advantage over the simpler, non-spatial ink control was one of the most
robust results in this entire project (p=1.77x10^-6). Against a
control with comparable spatial richness but zero class-conditioning,
that advantage is gone.

## The honest, revised interpretation

**The earlier conclusion needs to be narrowed substantially.** E beats
crude aggregate ink statistics because those statistics lack spatial
information entirely -- not because E's *class-conditioning* specifically
is what makes it valuable. Once a class-agnostic control is given
comparable spatial structure (quadrant energies, centroid, moments), it
performs at least as well as E on every dataset tested. This does not
mean E is worthless or redundant with T -- the earlier T-vs-T+E
comparisons (E adding real, significant value over T alone) still stand,
unaffected by this test. What it means is narrower and more precise: **the
specific claim that E's value comes from *class-specific* support
alignment, as opposed to generic spatial organization that any
spatially-aware feature would capture, is not supported by this
comparison.** The simpler global-ink control was the wrong null model for
that specific claim; this is the right one, and it changes the answer.

## Updated evidence hierarchy

| Claim | Status |
|---|---|
| E adds real value over T alone | Established (unaffected by this test) |
| E captures more than non-spatial global ink statistics | Established on notMNIST and Kuzushiji-MNIST (prior result, unaffected) |
| **E captures more than class-agnostic spatial structure** | **Not established -- the spatial control matches or exceeds E on every dataset** |
| E's value is specifically about class-conditioned support alignment (rather than spatial information generally) | **Not supported by current evidence** |

## What this does and doesn't undermine

This does not overturn E's status as the most robust additional channel
beyond T -- that rests on the T-vs-T+E comparisons and the E-against-T
residualization test, both untouched by this result. What it overturns is
the more specific *mechanistic* story about *why* E works: not because it
is class-conditioned, but possibly just because it is spatially organized
in a way the simpler ink control wasn't. Whether class-conditioning adds
anything on top of generic spatial structure remains an open question
this comparison was designed to answer, and the answer, honestly, is no
evidence that it does.

## Honest limitations

- The spatial control's specific construction (quadrants, centroid,
  moments) is one reasonable choice; a different or richer spatial
  representation might behave differently, though it was designed
  directly from the review's suggested quantities.
- This result is a negative/null finding for the class-conditioning
  hypothesis specifically -- it does not distinguish whether E's spatial
  information and the generic control's spatial information are actually
  capturing the *same* structure (a further dependence measurement, as
  done earlier for T/E/R, would clarify this) or merely comparably useful
  but different structure.
- Given this changes the interpretation of a previously load-bearing
  result, all comparisons here used tuned regularization from the outset;
  no fixed-regularization version of this specific test was run, so there
  is no fixed-vs-tuned comparison to report for this control.

## Immediate next steps

1. A dependence measurement (CCA / distance correlation, matching the
   earlier T/E/R protocol) between E and this spatial control would
   clarify whether E is redundant with generic spatial structure or
   merely comparably informative via a different mechanism.
2. This result should be weighed carefully before continuing to the
   frequency-band ablation, since that experiment's motivating question
   (is R's value about low-frequency organization specifically) is a
   direct analogue of the question just answered negatively for E's
   class-conditioning -- the same class-agnostic-but-spatially-matched
   control logic likely applies there too.

## Reproducing these results

`global_ink_stats.py`, `spatial_structure_stats_10d`: quadrant energies,
center of mass, second moments, and total energy, computed directly from
raw pixel values with no class-specific reference. Calibrated with the
same fixed SEED=42 protocol; all comparisons used tuned regularization
(`LogisticRegressionCV`, Cs=10, cv=5) throughout.
