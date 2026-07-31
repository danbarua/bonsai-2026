# Full Distributional Evidence: E/R Component Importance and the Proper 20D Random Ensemble

## Two things this round fixes: single-seed reporting, and single-draw random comparison

The prior round's shuffle-control conclusions relied on one representative
seed's McNemar test per comparison, and the random-dimensionality control
used a single 10D random draw rather than an ensemble matched to E+R's
actual 20-dimensional size. Both are corrected here with full
distributional evidence across 10 shuffle seeds and 50 random-projection
seeds respectively.

## Shuffle controls, using all 10 seeds' statistics (not one representative)

| Dataset | Δ_R (loss from shuffling R) | Seeds E+R wins / significant | Δ_E (loss from shuffling E) | Seeds E+R wins / significant |
|---|---|---|---|---|
| Fashion-MNIST | +3.74pp ± 1.03pp, range [+1.80,+5.20] | 10/10 win, 8/10 sig. (p: 1.3e-3 to 2.5e-1) | +1.68pp ± 0.56pp, range [+0.40,+2.60] | 10/10 win, 1/10 sig. (p: 1.1e-2 to 8.3e-1) |
| notMNIST | +3.72pp ± 0.29pp, range [+3.40,+4.20] | **10/10 win, 10/10 sig.** (p: 3.2e-4 to 6.4e-3) | +2.18pp ± 0.41pp, range [+1.40,+2.80] | 10/10 win, 8/10 sig. (p: 1.3e-3 to 1.4e-1) |
| Kuzushiji-MNIST | +4.36pp ± 0.79pp, range [+3.00,+5.60] | 10/10 win, 9/10 sig. (p: 1.5e-3 to 1.2e-1) | +6.06pp ± 0.61pp, range [+5.40,+7.20] | **10/10 win, 10/10 sig.** (p: 1.0e-7 to 4.2e-5) |

**Correct E+R beat all 10 shuffled-R and all 10 shuffled-E conditions on
every dataset** -- exactly the simplest form of the result the review
anticipated, now confirmed distributionally rather than from single
draws. No p-value in this table is reported as "0.00000"; Kuzushiji-
MNIST's E-shuffle significance level is correctly stated as ranging from
1.0x10^-7 to 4.2x10^-5 across the 10 seeds.

## Which component matters more, using the full distribution (D statistic)

D^(j) = Acc(shuffled-E + R) - Acc(E + shuffled-R) per seed; positive means
preserving R is more valuable, negative means preserving E is more
valuable.

| Dataset | D: mean ± std, range | Interpretation |
|---|---|---|
| Fashion-MNIST | +2.06pp ± 1.19pp, [+0.20, +3.20] | R-preservation more valuable, consistently (D>0 in all 10 seeds) |
| notMNIST | +1.54pp ± 0.61pp, [+0.80, +2.80] | R-preservation more valuable, consistently (D>0 in all 10 seeds) |
| Kuzushiji-MNIST | -1.70pp ± 1.11pp, [-4.20, -0.20] | E-preservation more valuable, consistently (D<0 in all 10 seeds) |

**This is a distributional answer, not dependent on any single
permutation**: D's sign is consistent across all 10 seeds on every
dataset. R-correspondence dominates on Fashion-MNIST and notMNIST;
E-correspondence dominates, and more strongly, on Kuzushiji-MNIST. The
defensible statement, exactly as the review framed it: *across the
shuffle ensemble, disrupting R causes the larger average loss on
Fashion-MNIST and notMNIST, while disrupting E causes the larger average
loss on Kuzushiji-MNIST.*

## The proper 20-dimensional random ensemble (50 seeds, matched to E+R's actual size)

The prior single-draw comparison (20D+S+random10D, one seed) was
correctly flagged as insufficient given this project's own repeated
finding that random-projection controls vary substantially by seed. This
round uses 50 seeds of a full 20-dimensional random projection, matching
E+R's actual additional dimensionality exactly (not 10+10 added to S, but
20 dimensions added directly to 20D, mirroring E+R's structure).

| Dataset | E+R accuracy | Random-20D ensemble (mean, std, range) | Percentile | Matched/exceeded | Empirical p |
|---|---|---|---|---|---|
| Fashion-MNIST | 77.40% | 76.46%, 0.99pp, [74.40%, 78.40%] | 78.0% | 11 of 50 | 0.2353 (n.s.) |
| **notMNIST** | **86.80%** | 84.37%, 1.00pp, [82.40%, 86.60%] | **100.0%** | **0 of 50** | **0.0196** |
| **Kuzushiji-MNIST** | **63.60%** | 56.70%, 1.51pp, [52.60%, 59.40%] | **100.0%** | **0 of 50** | **0.0196** |

**This resolves the outstanding concern decisively, and more strongly
than the single-draw version suggested.** On notMNIST and Kuzushiji-MNIST,
E+R beats every one of 50 properly-matched random 20-dimensional
additions -- not a near-miss or a single lucky comparison, but a clean
sweep with a substantial margin (Kuzushiji-MNIST: 63.60% vs. a random
ensemble that tops out at 59.40%, a gap of over 4 points to the *best*
random draw, let alone the mean). Fashion-MNIST's non-significant result
(p=0.235) is also now more solidly established than the earlier
single-draw comparison could show -- consistent, not a fluke of one
unlucky projection either.

## Updated defensible statements, per dataset

**notMNIST and Kuzushiji-MNIST**: E+R beats every dimensionality-matched
control, every shuffled-correspondence control, and now every one of 50
properly-matched random 20D projections. This is comprehensive,
distributionally-confirmed evidence that correctly-paired, image-specific
class-conditioned support energy and normalized low-frequency allocation
carry real information no tested alternative reproduces.

**Fashion-MNIST**: correct pairing matters relative to scrambled pairing
(confirmed distributionally, not just one seed), but E+R is not shown to
beat generic random 20D alternatives, now confirmed across 50 seeds, not
one. The precise, defensible statement, per the review's own framing:
*Fashion-MNIST contains useful sample-specific E/R information, but its
residual image information is not especially aligned with this graph
decomposition -- a generic projection can expose at least as much
additional discriminative content.*

## Honest limitations

- Shuffle seeds (10) and random-ensemble seeds (50) are both finite
  samples: the qualitative conclusions (consistent sign of D; E+R beating
  the full random-20D distribution on two datasets) are robust across
  every seed tested, but neither ensemble size guarantees no
  counterexample exists beyond what was sampled.
- The class-independent global ink-energy control and the redesigned
  frequency-band ablation (E+R_band comparisons) remain the next
  specified steps and were not run in this pass, which focused entirely
  on resolving the statistical-rigor gaps this round's review identified.

## Reproducing these results

All features reused from the existing cache; no new per-image computation
required. Shuffle analysis: `feature_provenance.py`-tracked E and R
features permuted independently per seed. Random-20D ensemble: raw pixel
data projected through 50 independent Gaussian matrices (seeds 42-91),
standardized using training-set statistics per seed, matching the
protocol established for the original 100-seed pixel-space comparison.
