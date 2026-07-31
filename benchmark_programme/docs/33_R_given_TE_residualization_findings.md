# Extending the Residualization Test: Is R Still Informative Given Both T and E?

## The question, and why it's sharper than residualizing against T alone

The final representation under consideration is (T, E, R), not just
(T, E). The natural extension of the T-E residualization test asks: after
topology matching *and* active-support overlap are both already
available, does R's remaining information still matter? R_perp =
R - ridge-predicted-R(T, E), compared against T+E, T+E+predicted-R,
T+E+residual-R, and T+E+full-R.

## Results

| Dataset | R² of R from (T,E) | T+E | T+E+predicted-R | T+E+residual-R | T+E+full-R | % of full-R gain recovered by residual |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.556-0.844 | 74.40% | 75.80% | 77.00% | 77.40% | 86.7% |
| notMNIST | 0.425-0.754 | 83.60% | 86.20% | 85.60% | 86.80% | 62.5% |
| Kuzushiji-MNIST | 0.284-0.630 | 59.60% | 60.00% | 64.00% | 63.60% | 110.0% (see below) |

## The predicted-R movement is methodological, not mysterious

In the T-E test, predicted-E never meaningfully beat T alone anywhere.
Here, **predicted-R shows non-trivial gains over T+E on two of three
datasets** (Fashion-MNIST: +1.4pp, notMNIST: +2.6pp) -- and this has a
precise, understood mechanism, not just a plausible guess. A regularized
classifier minimizes L(y, Xw) + lambda*||w||^2. Appending a deterministic
linear transform of existing features, R_hat = XA, does not enlarge the
linear span -- Xw + R_hat*v = X(w + Av) for any v -- but it changes the
*available parameterization*: the same decision boundary can now be
represented with total penalty ||w||^2 + ||v||^2 instead of just ||w||^2
alone. This changes the effective regularization geometry and can shift
test accuracy even though no new information was added. The predicted-R
condition therefore demonstrates that the classifier is not invariant to
redundant reparameterization -- it should not be read as evidence about
R's informational content, in either direction. This effect is plausibly
more visible here than in the E-against-T case because predicting a
10-dimensional R from a 30-dimensional (T,E) input, then appending 10
more correlated columns to a regularized classifier, gives the
reparameterization effect more room to operate.

**This is exactly why the decisive comparisons are the other two, not
this one.**

## The two comparisons that actually matter

**T+E vs. T+E+residual-R** -- does R's non-(T,E)-predictable component add
real value:

| Dataset | McNemar p |
|---|---|
| Fashion-MNIST | 0.0725 (borderline, n.s.) |
| notMNIST | 0.0755 (borderline, n.s.) |
| **Kuzushiji-MNIST** | **0.0103** |

**T+E+full-R vs. T+E+residual-R** -- does the residual alone capture
essentially all of full R's value:

| Dataset | McNemar p |
|---|---|
| Fashion-MNIST | 0.824 (n.s.) |
| notMNIST | 0.210 (n.s.) |
| Kuzushiji-MNIST | 0.832 (n.s.) |

**A correction to the framing here: the full-vs-residual comparison is
not the decisive one, and calling it "clean and decisive" overstated
what a non-significant result can show.** A non-significant McNemar test
means the experiment did not detect a difference -- it does not establish
that residual R contains all the information full R supplies. The
genuinely decisive comparison for unique incremental information is
**T+E vs. T+E+residual-R**, not full-vs-residual. The safer statement:
*full R did not significantly outperform residual R on any dataset, and
their observed accuracies were close -- consistent with most of R's
useful contribution residing in the residual component, but not a formal
equivalence claim.*

**Raw percentage-point gains, alongside the fraction-recovered statistic**
(which is unstable when the denominator is small, exactly what produced
Kuzushiji-MNIST's unstable 110% figure):

| Dataset | Full-R gain (over T+E) | Residual-R gain (over T+E) |
|---|---|---|
| Fashion-MNIST | +3.0pp | +2.6pp |
| notMNIST | +3.2pp | +2.0pp |
| Kuzushiji-MNIST | +4.0pp | +4.4pp |

These communicate the result more transparently than the percentage
alone -- Kuzushiji-MNIST's residual gain (+4.4pp) is not literally larger
than full R's contribution in any meaningful sense; the two are close,
and the ratio between them is what becomes unstable near a small
denominator, not the underlying accuracies themselves.

The T+E-vs-residual-R comparison (the actually decisive one for unique
information) reaches clear significance only on Kuzushiji-MNIST, with
Fashion-MNIST and notMNIST directionally positive but individually
non-significant -- weaker evidence than the equivalent E-against-T
result, but pointing the same direction throughout.

## What this establishes, precisely

The updated, appropriately qualified statement: **conditional on T and E,
R contains clearly significant linearly non-recoverable information on
Kuzushiji-MNIST. Fashion-MNIST and notMNIST show positive but
individually non-significant residual gains. Across all datasets,
residual R performs close to full R, consistent with the shared
component contributing limited additional value** -- not a formal
equivalence claim, but consistent with one. This still supports the
(T, E, R) representation; it simply avoids overstating the statistical
evidence the way the original framing did. The E-against-T result remains
the cleaner of the two: E contains significant label-relevant information
not linearly recoverable from T on all three datasets, a stronger and
more uniform statement than what the R-against-(T,E) test can currently
support.

## Honest limitations

- The predicted-R anomaly (real movement where none was mechanically
  expected) is reported and given a plausible explanation, not resolved.
  It does not affect the residual-vs-full comparison, which is the one
  this document leans on for its central claim.
- Fashion-MNIST and notMNIST's primary comparison (T+E vs. T+E+residual-R)
  does not reach conventional significance -- weaker support than
  Kuzushiji-MNIST's, and should be described as directionally consistent,
  not confirmed, on those two datasets.
- As with the T-E test, this residual is about linear recoverability
  specifically; nonlinear dependence between R and (T,E) is not removed
  and may still be substantial, consistent with the distance-correlation
  measurements.
- Ridge regularization strength (alpha=1.0) was not tuned, consistent
  with every other residualization in this project.

## Immediate next steps, per the review's recommended order

1. Global ink and class-independent energy controls (still the cheapest
   remaining test, still owed).
2. Frequency-band comparison with E held fixed.
3. If pursued further: residualizing E against both T and R jointly, to
   quantify each channel's unique linear contribution conditional on the
   other two -- the review's suggested complete conditional-residualization
   picture, of which this document completes the second of three possible
   pairings (E-given-T done previously, R-given-T,E done here, E-given-T,R
   remains).

## Reproducing these results

Ridge regression fit from concatenated (T,E) to R on training data for
each dataset; residual formed and evaluated identically to the T-E
residualization test. No new per-image computation required.
