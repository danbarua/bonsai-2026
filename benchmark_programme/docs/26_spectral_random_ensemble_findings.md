# Is Spectral Structure Privileged, or One Member of a Broad Class? The Random-Ensemble Verdict

## A full rebuild happened first, and it is itself informative

The working environment reset between tasks mid-session, wiping every
downloaded dataset, computed topology, and cached feature matrix from
this entire project -- documented, expected behavior, not a bug. Every
persisted findings document and packaged tarball in `/mnt/user-data/outputs/`
survived; the code within those tarballs was recovered and used to rebuild
the full pipeline from scratch for Fashion-MNIST, notMNIST, and
Kuzushiji-MNIST (the last re-uploaded by the user, since its host is
unreachable from this sandbox).

**This rebuild is itself a successful verification.** Population-derived
topology edge counts, rebuilt independently from re-downloaded raw data,
matched the original session's values exactly, class by class, on all
three datasets (e.g., Fashion-MNIST: 25382, 9320, 39617... identical to
the first build). The pipeline is genuinely deterministic and
reproducible from raw data plus code, not dependent on any hidden state.

**Per the user's request, every stochastic operation in this rebuild uses
a single fixed constant (`SEED=42` in `constants.py`)** -- calibration
image selection, frozen test-set sampling, and the random-projection
ensemble below -- rather than the assortment of ad hoc seeds (555, 999,
2024, 0, 1, 2...) scattered through this project's history.

## The core finding replicates cleanly with a completely different seed

| Dataset | 20D | 20D + spectral | Change |
|---|---|---|---|
| Fashion-MNIST | 70.0% | 76.6% | **+6.6pp** |
| notMNIST | 79.4% | 84.4% | **+5.0pp** |
| Kuzushiji-MNIST | 49.4% | 59.0% | **+9.6pp** |

Larger effect sizes than the original session's seed (+4.0, +5.0, +6.8pp
respectively). This is useful evidence against the effect depending on the
original split -- but it should not be overstated as a repeated-seed
estimate of effect-size stability, since there is still only one rebuilt
topology, calibration sample, and test split under SEED=42, not multiple
independent seeds tested against each other. The defensible claim: the
positive spectral-combination effect survived a complete reconstruction
from raw data under a newly unified seed and produced equal or larger
gains on all three datasets -- substantial reproducibility evidence,
without implying an effect-size distribution has been measured.

## The decisive test: is spectral privileged among alternative projections?

For each dataset, 100 random Gaussian projections (seeds 42-141, matching
spectral's exact protocol: 10 dimensions, identical training/test images,
identical standardization from training statistics, identical classifier)
were each combined with 20D and evaluated, both raw and residualized
against 20D via ridge regression -- placing spectral's actual gain within
this empirical null distribution.

| Dataset | Spectral's delta | Random ensemble (mean, std, range) | Spectral percentile | Empirical p |
|---|---|---|---|---|
| Fashion-MNIST | +6.60pp | +4.52pp, 1.42pp, [+0.80, +8.80] | 94.0% | 0.069 (n.s.) |
| notMNIST | +5.00pp | +3.20pp, 1.07pp, [0.00, +5.40] | 95.0% | 0.059 (n.s.) |
| **Kuzushiji-MNIST** | **+9.60pp** | +4.62pp, 1.39pp, [+1.20, +7.60] | **100.0%** | **0.0099** |

**Kuzushiji-MNIST decisively answers the question the review posed:
spectral beats every single one of 100 random projections** -- its delta
(+9.60pp) exceeds even the best-performing random seed's delta (+7.60pp).
The residualized comparison gives the identical verdict (100th percentile,
p=0.0099) -- spectral's advantage here is not explained by anything a
generic alternative linear projection of the same dimensionality could
achieve, with or without controlling for overlap with 20D.

**Fashion-MNIST and notMNIST do not reach the same conclusion, and the
correct description is precise about where they sit.** Spectral is at the
94th and 95th percentiles respectively -- near the top of the random-
projection distribution, not the median. Exactly 6 of 100 random
projections matched or exceeded spectral on Fashion-MNIST, and 5 of 100
on notMNIST (0 of 100 on Kuzushiji-MNIST, for contrast). The accurate
statement: spectral performs near the top of the random-projection
distribution on Fashion-MNIST and notMNIST, but does not exceed that
distribution strongly enough to meet the pre-specified empirical
significance threshold with 100 projections. This is suggestive of
privilege on those two datasets, not evidence against it -- a materially
different claim than "near the median" would have implied.

## Applying the decision rule, per dataset rather than as one verdict

This does not collapse to a single answer across all three datasets, and
should not be forced into one:

- **Kuzushiji-MNIST**: "Spectral clearly exceeds the random ensemble" --
  the specifically-spectral claim is strong here. Low-frequency graph
  structure exposes unusually useful complementary information on this
  dataset specifically.
- **Fashion-MNIST and notMNIST**: closer to "spectral sits near the random
  median" -- the broader capacity finding (20D is not capacity-complete,
  something added on top of it helps) remains positive and is not in
  question, but whether *spectral specifically* is the privileged
  mechanism, versus one adequate member of a broad class of informative
  projections, remains genuinely open on these two datasets.

This is consistent with, and sharpens, the review's own anticipated
"helps only one dataset -- inspect the associated graph regime" branch,
except here spectral does not uniquely help only Kuzushiji-MNIST overall
(20D+spectral beats 20D on all three) -- it is only the *privileged-versus-
generic* question that resolves differently across datasets.

## Multiple-comparisons check

Three primary dataset-level empirical tests were run. A Bonferroni
threshold across them is 0.05/3 ≈ 0.0167. Kuzushiji-MNIST's empirical
p=0.0099 remains significant under this correction; Fashion-MNIST
(p=0.069) and notMNIST (p=0.059) do not. This strengthens rather than
weakens the dataset-specific reading: Kuzushiji-MNIST provides
statistically defensible evidence that the graph-derived spectral
representation is more complementary than 100 matched-dimensional generic
random projections, even under a correction for testing three datasets.

## An important limitation: the random control is not information-source-matched

The random-Gaussian control is dimension-matched and sample-specific, but
not matched to spectral's actual information source. Spectral features
are class-conditioned (derived from labeled population topologies),
restricted to each class's specific active support, and projected onto
topology-derived eigenvector bases. The Gaussian control is class-agnostic,
label-free, and projected directly from the complete pixel space with no
restriction to any class-specific structure. **This experiment therefore
answers a narrower question than originally framed**: is class-conditioned
spectral projection better than a generic unsupervised projection? It does
not yet isolate whether the advantage comes specifically from the
eigenstructure itself, versus class conditioning, active-support
restriction, or supervised population information more generally. That
distinction is the sharper control problem the next round of experiments
needs to address.

## What might distinguish Kuzushiji-MNIST -- offered as a hypothesis, not established

Kuzushiji-MNIST is the dataset with the highest within-class visual
variation of the three (different historical calligraphic styles), the
lowest 20D baseline accuracy, and -- from the original Capacity Experiment
II smoothness work -- the dataset where the original 20D readout was
already known to leave the most oracle headroom against an alternative
functional. It is plausible that datasets with more within-class
structural diversity have more room specifically for a *low-frequency,
smooth* organizing signal to add value beyond what a generic linear
projection captures, while datasets already well-served by 20D (Fashion-
MNIST, notMNIST, both with less headroom historically) leave less specific
room for spectral's particular character to distinguish itself from
generic alternatives. This is speculative and not tested directly here.

## Honest limitations

- Single run per condition; the random ensemble itself provides internal
  replication (100 seeds), but the spectral result, 20D baseline, and
  residualization each reflect one draw.
- The normalized-energy mechanistic control (dividing spectral projection
  energy by total active-subgraph input energy, to check whether spectral
  is partly a proxy for support overlap) remains untested, per the
  review's own staged ordering -- this random-ensemble comparison was
  prioritized first, correctly, but the normalized control is still owed.
- The hypothesis offered for why Kuzushiji-MNIST differs (within-class
  variation, headroom) is post hoc and untested against a fourth dataset
  or a direct manipulation.
- MNIST remains deliberately not investigated, per the review's explicit
  instruction from the prior round.

## Immediate next steps, in the specified order

1. Per-image active-energy normalization, plus active-support-energy
   itself retained as a standalone 10D control (so the normalization's
   denominator doesn't obscure whether support overlap is independently
   useful) -- all three datasets.
2. Random orthonormal bases on each class's exact active subgraph -- same
   support, same dimensionality, same class conditioning as spectral, but
   no learned eigenstructure. This is the minimum decisive next test for
   the information-source-matching gap identified above, and cheaper than
   the frequency-band or degree-preserving alternatives.
3. Low-frequency vs. matched mid-frequency vs. random-eigenvector bands,
   specifically on Kuzushiji-MNIST -- tests whether the privileged result
   there is specifically about low-frequency structure or the graph basis
   more generally.
4. Degree-preserving graph nulls -- only if the first three leave a
   specifically topological effect unresolved.

Late fusion remains correctly out of scope throughout -- raw concatenation
already works, and the open question is attribution, not combination
performance.

## Reproducing these results

Full pipeline rebuilt under `constants.SEED=42`: `{dataset}_raw_stats_200.pkl`
→ `{dataset}_class_topologies_200.pkl` → `{dataset}_baselines.pkl` /
`{dataset}_hybrid_batch{1,2}.pkl` / `{dataset}_test_features.pkl` (20D) and
`{dataset}_spectral_{train,test}.pkl` / `{dataset}_baselines_spectral.pkl`
(spectral, with provenance metadata via `feature_provenance.py`), for
Fashion-MNIST, notMNIST, and Kuzushiji-MNIST.
