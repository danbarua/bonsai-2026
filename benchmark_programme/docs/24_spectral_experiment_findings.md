# Capacity Experiment III: Spectral Structure -- The First Positive Capacity Result

## The question, and why this result matters more than the prior two

Does the learned topology encode class information in its spectral
organization not already captured by pairwise topology matching or graph
smoothness? Capacity Experiments I and II (a second pruning threshold;
graph smoothness) both gave clean negative results -- real independent
signal, but no detectable combined benefit over the 20D baseline, on any
of four datasets, under any of four tested combination mechanisms.
**Spectral structure breaks that pattern.** This is the first capacity
experiment in the series with a genuine positive result.

## Engineering prerequisite, applied throughout

Two prior bugs in this project (unexplained notMNIST 60D artifacts;
MNIST's already-normalized cache file being renormalized) both traced to
the same root cause: no explicit record of a saved feature matrix's
normalization state. Every artifact in this experiment carries explicit
provenance metadata (dataset, feature_type, normalization_state,
topology_threshold, pipeline_version, source_files, created_at), and
loading enforces the expected state rather than assuming it
(`feature_provenance.py`). This caught nothing wrong in this experiment --
worth stating plainly since a null result from a safety check is still a
successful use of it, not a non-event.

## Implementation, and two subtleties a naive version would have missed

s_c(x) = sum over the first 5 non-trivial eigenvectors of |<x, u_{c,k}>|^2,
uniform weighting, symmetric normalized Laplacian, exactly as specified.
Two complications, resolved before any bulk computation:

1. Most pixels are isolated (zero degree) in these sparse topologies --
   D^(-1/2) is undefined there. The Laplacian is restricted to the active
   (non-isolated) node subgraph only.
2. "Exclude the constant eigenvector" is not as simple as dropping one
   eigenvector -- a graph with multiple connected components has one
   trivial zero eigenvalue *per component*. This is computed explicitly
   via `connected_components`, not assumed to be exactly one. In practice,
   verified directly before scaling to all four datasets: the *active*
   subgraph (after excluding isolated nodes) has exactly one connected
   component for every class, on every dataset tested -- the earlier
   "hundreds of components" figures reported in prior structural-
   statistics work were counting isolated single-pixel components, a
   different measure than what matters here.

Verified on Fashion-MNIST first (sane, small, positive, ascending
eigenvalues) before scaling to the other three datasets, consistent with
this project's established discipline of checking one dataset before
propagating a new computation across four.

## Result 1: spectral alone is a real but weaker standalone readout

| Dataset | 20D | Smoothness | Spectral |
|---|---|---|---|
| Fashion-MNIST | 71.4% | 70.2% | 67.0% |
| notMNIST | 80.4% | 76.4% | 72.4% |
| Kuzushiji-MNIST | 52.4% | 46.2% | 42.0% |
| MNIST | 89.6% | 87.6% | 66.2% |

Spectral alone is the weakest of the three readouts on every dataset --
notably so on MNIST (66.2%, a 23-point gap to 20D, much larger than
smoothness's 2-point gap there). This on its own would suggest a weaker
functional. It is the *combination* result that changes the picture
entirely.

## Result 2: combining 20D with spectral gives real, statistically significant gains on three of four datasets

| Dataset | 20D | 20D + spectral (30D) | Change | McNemar p |
|---|---|---|---|---|
| Fashion-MNIST | 71.4% | **75.4%** | **+4.0pp** | **0.0055** |
| notMNIST | 80.4% | **85.4%** | **+5.0pp** | **0.00017** |
| Kuzushiji-MNIST | 52.4% | **59.2%** | **+6.8pp** | **0.00022** |
| MNIST | 89.6% | 90.6% | +1.0pp | 0.332 (n.s.) |

Three of four datasets show a statistically significant improvement from
simple concatenation -- something neither the multi-threshold experiment
nor the smoothness experiment achieved on any dataset, under any
combination method, including three dedicated late-fusion mechanisms.
MNIST shows the same direction but does not reach significance, plausibly
related to spectral's much larger standalone weakness there (66.2% vs.
89.6%, a 23-point gap) diluting its contribution once concatenated.

## Result 3: every complementarity diagnostic agrees, and shows a different character of relationship than smoothness had

| Dataset | 20D/spectral agreement | Oracle headroom | Effective rank (30D) | Mean abs. cross-correlation |
|---|---|---|---|---|
| Fashion-MNIST | 70.2% | 7.6pp | 4.87 | 0.260 |
| notMNIST | 69.0% | 5.6pp | 6.98 | 0.143 |
| Kuzushiji-MNIST | **32.8%** | **15.4pp** | 6.66 | 0.301 |
| MNIST | 66.4% | 2.8pp | 7.15 | 0.251 |

Compare directly to smoothness's corresponding figures from Capacity
Experiment II: prediction agreement there was 70.6-93.2% (much higher --
spectral disagrees with 20D far more often, most dramatically on
Kuzushiji-MNIST at just 32.8%); oracle headroom there was 1.8-3.8pp
(spectral's is 2.8-15.4pp, uniformly larger, with Kuzushiji-MNIST's 15.4pp
nearly four times smoothness's largest headroom); effective rank there
was 2.9-4.6 of 30 nominal dimensions (spectral's combined space is
consistently less redundant, 4.9-7.2); cross-correlation there averaged
0.51-0.70 (spectral's is 0.14-0.30, roughly half). **Every single
diagnostic, independently, points the same direction: spectral structure
captures something substantially more independent of the original 20D
representation than smoothness did**, and unlike smoothness, this
independence actually translates into combined-classifier improvement.

## Result 4: the improvement is broadly distributed, not concentrated in one or two classes

Checked directly rather than assumed, given how much this matters for
interpretation: on Kuzushiji-MNIST, 8 of 10 classes improve (gains of
+0.04 to +0.14), only one shows a small decline (-0.02). Fashion-MNIST and
notMNIST show a similar broad pattern (7-8 of 10 classes improving,
several by double-digit percentage points). This is not a narrow rescue
of one confusable pair (contrast with Capacity Experiment II's
Kuzushiji-MNIST deep dive, where smoothness's rescues concentrated
disproportionately on one specific confusion). Spectral's contribution
looks like a genuinely general property of the representation, not a
narrow patch.

## Applying the decision rule

Per the pre-specified framework: **"spectral improves 20D on multiple
datasets"** is the branch supported by the evidence -- three of four
datasets show a statistically significant gain, the fourth shows the same
direction without reaching significance. **This is the first real evidence
in this project that representational capacity can be expanded through a
different graph functional.** Per the review's explicit condition ("do
not reopen late fusion unless spectral shows materially greater
complementarity than smoothness did") -- it does, decisively, on every
diagnostic measured. Late fusion (validation-weighted probability fusion,
calibrated logit fusion, stacked fusion) is now justified as a genuine
next step for this specific pair, not a repeat of Capacity Experiment II's
closed branch.

## Honest limitations

- Single run per condition per dataset, no repeated seeds -- the same
  caveat as every other result in this project, though the consistency
  across four independent diagnostics (not just accuracy) gives more
  confidence than a bare accuracy comparison would.
- MNIST does not reach significance for the primary 20D+spectral
  comparison -- the positive direction is consistent with the other three
  datasets but should not be over-claimed as confirmed there specifically.
- The low-end eigenvalue distributions themselves (requested as a
  diagnostic, to check whether some class graphs have near-identical
  spectra that would predict weak discrimination) were inspected during
  basis construction (all sane, small, distinct, ascending) but not
  formally compared class-to-class for spectral similarity -- a finer
  analysis of exactly which eigenvalue gaps drive the largest
  per-class gains remains for follow-up.
- Smoothness+spectral (20D, without the original topology-matching
  features) and the full 40D combination were computed and are broadly
  consistent with the 20D+spectral pattern, but the primary, most
  carefully diagnosed comparison in this document is 20D vs. 20D+spectral
  specifically, per the review's stated priority ordering.
- Duplicated-feature and random-feature controls (used throughout
  Capacity Experiments I and II) were not re-run here -- given the
  magnitude and statistical significance of the raw improvement, plus the
  qualitatively different diagnostic profile versus smoothness (which did
  require controls to reveal it was NOT a real effect), this is a gap
  worth closing before fully trusting the 30D result, not an oversight to
  ignore.

## Immediate next steps

1. Run duplicated-feature and random-feature controls for the 20D+spectral
   condition specifically -- the one piece of the established discipline
   not yet applied here, and given how large this result is relative to
   everything else in the series, the most important thing to confirm
   before treating it as fully established.
2. Late fusion (validation-weighted, calibrated logit, stacked) for the
   20D/spectral pair, now justified per the decision rule.
3. Investigate MNIST's non-significant result specifically -- likely
   related to spectral's unusually large standalone weakness there.

## Reproducing these results

`spectral_readout.py` (new), `feature_provenance.py` (new, engineering
prerequisite), plus existing cached 20D and smoothness artifacts across
all four datasets. Spectral features saved with full provenance metadata
at `{dataset}_spectral_train.pkl` / `{dataset}_spectral_test.pkl` /
`{dataset}_baselines_spectral.pkl`.
