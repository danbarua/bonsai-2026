# Global Ink Control: E Is Not Just Measuring Stroke Density

## The question

E measures how much image energy falls on pixels belonging to each
class's learned active support. Is that value genuinely about
class-specific support alignment, or would any class-agnostic ink
statistic of the same dimensionality capture just as much? Tested with a
10-dimensional, purely class-agnostic control (total pixel sum, total
squared energy, and ink-pixel counts at 8 thresholds -- no reference to
any class's topology or support), matched exactly to E's dimensionality.

## Result 1: T+E decisively beats T+global-ink on every dataset

| Dataset | T alone | T+global-ink(10D) | T+E (class-specific) | McNemar p |
|---|---|---|---|---|
| Fashion-MNIST | 70.00% | 70.40% (+0.40pp) | 74.40% (+4.40pp) | **2.89e-03** |
| notMNIST | 79.40% | 80.60% (+1.20pp) | 83.60% (+4.20pp) | **2.60e-03** |
| Kuzushiji-MNIST | 49.40% | 53.20% (+3.80pp) | 59.60% (+10.20pp) | **1.01e-04** |

**Generic ink statistics do carry some real signal** -- a small but genuine
bump over T alone on every dataset (+0.4 to +3.8pp), confirming that part
of what any energy-based feature captures is simply "how much ink is on
the page." **But class-specific E captures substantially and
significantly more**, on every dataset, with the gap widest on
Kuzushiji-MNIST (+10.20pp vs. +3.80pp) -- exactly the dataset where E's
importance has been most robustly established throughout this line of
work.

## Result 2: does class-specific E outperform generic ink once T and R are already available? (Corrected after hygiene check)

The comparisons below were re-run with per-condition tuned regularization
(`LogisticRegressionCV`) after being challenged on why this hadn't been
done already -- there was no good reason it hadn't. The results changed
the picture meaningfully, and are reported as corrected, not as a
footnote.

| Dataset | Comparison | Fixed regularization | Tuned regularization |
|---|---|---|---|
| Fashion-MNIST | T+E vs. T+global-ink | p=2.89e-03 | p=0.134 (n.s.) |
| notMNIST | T+E vs. T+global-ink | p=2.60e-03 | **p=2.53e-05** |
| Kuzushiji-MNIST | T+E vs. T+global-ink | p=1.01e-04 | **p=1.77e-06** |
| Fashion-MNIST | T+E+R vs. T+R+global-ink | 0.845 (n.s.) | 1.000 (n.s.) |
| notMNIST | T+E+R vs. T+R+global-ink | p=3.91e-03 | **p=0.263 (n.s.) -- did not survive** |
| Kuzushiji-MNIST | T+E+R vs. T+R+global-ink | p=5.52e-03 | p=0.0198 (weaker, still holds) |

**The primary comparison (does class-specific E beat generic ink of the
same dimensionality) is robust** -- it strengthens under tuning on
notMNIST and Kuzushiji-MNIST, and Fashion-MNIST's already-weak result
becomes clearly non-significant, consistent with its established pattern
rather than contradicting it.

**The secondary comparison (does E add unique value beyond R+generic-ink)
did not fully survive.** notMNIST's previously-significant result
(p=0.0039) drops to non-significance (p=0.263) under tuned
regularization -- the same failure mode that demoted R's conditional
result in the prior round, now appearing in a comparison this document
originally reported without the same scrutiny. Only Kuzushiji-MNIST's
secondary comparison holds up, and more weakly than the fixed-regularization
number suggested (p=0.0198 vs. the original p=0.0055).

## What this establishes, corrected

**E measures something beyond generic stroke density -- this is
confirmed robustly by the primary comparison on notMNIST and
Kuzushiji-MNIST, the same two datasets where E's importance survived
every prior test.** The secondary claim (that E adds value even once R is
already present) is weaker than originally stated: it holds on
Kuzushiji-MNIST, not on notMNIST once properly checked. This narrows,
but does not overturn, the core finding -- E is not a proxy for ink
amount, but its incremental contribution specifically over an
already-R-equipped representation is less broadly established than this
document first reported.

## Honest limitations

- The global-ink control's 10 dimensions (sum, squared energy, 8
  threshold counts) are one reasonable construction, not the only
  possible one -- a different choice of thresholds or statistics could in
  principle behave somewhat differently, though the review's suggested
  quantities (G1, G2, G3-type statistics) are represented directly.
- The regularization-hygiene check has now been run (see the corrected
  Result 2 above) -- this is no longer an open gap, and the correction it
  produced is folded into this document's conclusions rather than left as
  a caveat.
- Fashion-MNIST's result (global-ink+R tying E+R, and the primary
  comparison losing significance under tuning) is reported plainly rather
  than explained -- consistent with the dataset's established pattern,
  not independently investigated further here.

## Reproducing these results

`global_ink_stats.py` (new): 10 class-agnostic statistics computed
directly from raw pixel values, no oscillator dynamics or per-image
computation beyond simple arithmetic. Calibrated with the same fixed
SEED=42 protocol used throughout this project's rebuild.


## Updated evidence hierarchy

| Claim | Status |
|---|---|
| Global ink contains useful image information | Established |
| E captures more than global ink | Established on notMNIST and Kuzushiji-MNIST |
| E captures more than global ink on Fashion-MNIST | Not established under tuned regularization |
| E adds unique value beyond T, R and global ink | Established only on Kuzushiji-MNIST |
| E is the strongest robust additional channel beyond T | Still supported |
