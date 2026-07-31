# The Regularization-Hygiene Check: E's Result Is Robust, R's Was More Fragile Than Stated

## What this test was for

The review identified a precise mechanism by which appending a
deterministic linear transform of existing features (predicted-E or
predicted-R) can shift a regularized classifier's test accuracy without
adding real information: it changes the available parameterization
(Xw + R_hat*v = X(w+Av) for any v), letting the same decision boundary be
represented with a different total penalty (||w||^2 + ||v||^2 instead of
||w||^2 alone). This test reruns both the E-against-T and R-against-(T,E)
residualization comparisons with per-condition cross-validated
regularization tuning (`LogisticRegressionCV`, 10-value C grid, 5-fold)
and three ridge residualization strengths (alpha = 0.1, 1.0, 10.0), on
Kuzushiji-MNIST -- the dataset carrying the strongest claims in both prior
documents.

## E-against-T: completely robust, and the gain gets larger under tuning

| Ridge alpha | T+residual-E accuracy | Gain over T | McNemar p |
|---|---|---|---|
| 0.1 | 61.20% | +11.60pp | 4.39x10^-10 |
| 1.0 | 61.20% | +11.60pp | 4.39x10^-10 |
| 10.0 | 61.20% | +11.60pp | 4.39x10^-10 |

Identical across all three ridge strengths, and with tuned regularization
the gain is *larger* than the original fixed-C result (+11.60pp vs. the
earlier +9.00pp), not smaller. This is exactly the outcome that makes a
conclusion hard to attribute to parameterization effects: it survives
independently-tuned regularization and multiple residualization
strengths without any softening.

## R-against-(T,E): the predicted-R movement persists even under tuning, and the residual's significance weakens

| Ridge alpha | T+E+predicted-R gain over T+E | T+E+residual-R gain over T+E | McNemar T+E vs. residual-R | McNemar full-R vs. residual-R |
|---|---|---|---|---|
| 0.1 | +1.20pp | +3.20pp | p=0.0519 | p=1.000 |
| 1.0 | +1.00pp | +3.20pp | p=0.0519 | p=1.000 |
| 10.0 | +0.80pp | +3.20pp | p=0.0519 | p=1.000 |

**Two things this reveals, both important:**

1. **The predicted-R movement is not fully eliminated by regularization
   tuning** -- it persists (+0.8 to +1.2pp) across all three ridge
   strengths even when the classifier's regularization is independently
   optimized for that specific condition. This confirms the review's
   mechanism is real and not simply an artifact of an arbitrarily-fixed
   C: regularization tuning selects the best single penalty for the whole
   feature set, but does not fully neutralize the geometric
   reparameterization effect, which operates through the *shape* of the
   penalty surface, not just its overall strength.

2. **Kuzushiji-MNIST's previously significant T+E-vs-residual-R result
   (p=0.0103 under fixed regularization) drops to borderline
   non-significance (p=0.0519) under tuned regularization.** This is the
   humbling correction this check was designed to surface: the one clear
   significant result in the R-against-(T,E) line of work was itself
   partly dependent on the specific fixed regularization strength used
   previously, not fully robust to this hygiene check. The full-vs-residual
   comparison remains completely flat (p=1.000, identical predictions
   across all three alphas) -- unchanged from before.

## The asymmetry is the finding

**E's residual value against T is robust** -- confirmed under tuned
regularization and multiple ridge strengths, with the effect if anything
strengthening. **R's residual value against (T,E) is comparatively
fragile** -- its one significant result does not survive the same hygiene
check. This is not a reason to discard the R-against-(T,E) result
entirely (it remains directionally consistent, and the underlying
predicted-R anomaly confirms real reparameterization sensitivity exists
in this specific comparison, which is itself informative about why the
result might be less stable) -- but it is a real downgrade from "clearly
significant on Kuzushiji-MNIST" to "borderline, and shown to be sensitive
to regularization choice in a way E's result is not."

## Updated evidence hierarchy

| Claim | Status |
|---|---|
| E contains linearly non-recoverable, label-relevant information beyond T | Established, robust to regularization tuning and ridge strength, on Kuzushiji-MNIST (and by the earlier untuned test, on all three datasets) |
| R contains linearly non-recoverable, label-relevant information beyond (T,E) | Weaker than previously stated -- borderline on Kuzushiji-MNIST under proper hygiene, not yet independently confirmed under tuning on the other two datasets |

## Honest limitations

- This hygiene check was run on Kuzushiji-MNIST only, given it carried
  the strongest claims in both prior documents -- Fashion-MNIST and
  notMNIST's already-weaker R-against-(T,E) results have not been
  re-verified under tuned regularization, and per the review's own
  logic, they should be expected to weaken further, not strengthen.
- OLS/pseudoinverse residualization (the review's suggested cleanest
  comparison, since ridge residuals are not strictly orthogonal to the
  predictor on training data) was not run in this pass -- ridge was
  retained across three alpha values as a partial check in its place.
- Repeated train/calibration splits (versus the single fixed split used
  throughout this project) remain untested -- the review's fourth
  suggested hygiene element.
- `LogisticRegressionCV`'s default C grid and 5-fold CV were used without
  further tuning of the tuning procedure itself.

## Immediate next steps

1. The global ink and class-independent energy controls remain the
   cheapest next mechanistic test, still owed.
2. If the R-against-(T,E) claim is to be pursued further, it should be
   re-verified on Fashion-MNIST and notMNIST under this same tuned-
   regularization protocol before being stated at any confidence level.
3. Frequency-band ablation with E held fixed, per the standing order --
   now with the appropriate expectation that any R-band comparisons
   should also be checked under tuned regularization before their
   significance is trusted.

## Reproducing these results

`sklearn.linear_model.LogisticRegressionCV` (Cs=10, cv=5) in place of the
fixed-C `LogisticRegression` used throughout the rest of this project,
applied identically to all conditions within each comparison; ridge
residualization repeated at alpha in {0.1, 1.0, 10.0} rather than the
single alpha=1.0 used previously.
