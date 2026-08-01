# Stage 1A Re-Verification: Log-Scale Iteration (v2) -- Findings

Design in `DESIGN_v2_log_scale.md`. Cross-references v1's `FINDINGS.md`,
which this document builds on rather than revises.

## Friction log / data-provenance check (done first, per DESIGN_v2's instruction)

Before writing any analysis code, checked what v1's driver actually
persisted: `results/stage1a_reverification_results.pkl` contains all 770
individual `(class, construction, seed) -> raw AUC` entries -- e.g.
`(0, 'rewired', 5) -> 38.39...` -- not class-level aggregates. Confirmed
`class 0 rewired` alone has 25 distinct seed entries. **This means v2 is
pure re-analysis, exactly as DESIGN_v2 intends: no new
`joint_tangent_matrix_response` call, no new seed, no new graph
construction was run anywhere in this iteration.** All AUC values are
loaded directly from v1's committed pkl and only the aggregation
function applied to them changes.

## What was run

`analyze_stage1a_log_scale.py`, covering the 3 in-scope comparisons
(hist_random, curr_random, rewired -- lattice excluded per DESIGN_v2,
since it has no seed axis and v1's result for it already stands). Full
robustness cascade (log-mean primary, log-median sensitivity, exact
sign-flip, within-class log-MCSE, hierarchical bootstrap, gated tertiary
mixed model), Holm-corrected across the 3 comparisons in this family (a
different, smaller family than v1's 4).

## Primary test results (log-mean-aggregated, Holm-corrected across 3)

| Comparison | Median d (log) | Median, multiplicative | HL (log) | HL, multiplicative | Sign+ /10 | p_exact | p_holm |
|---|---|---|---|---|---|---|---|
| T vs. historical half-edge random | -0.615 | x0.541 | -0.572 | x0.564 | 3/10 | 0.322 | 0.322 |
| T vs. current edge-count-matched random | -0.633 | x0.531 | -0.882 | x0.414 | 3/10 | 0.160 | 0.320 |
| T vs. degree-preserving rewiring | -1.288 | x0.276 | -1.195 | x0.303 | 1/10 | 0.037 | 0.111 |

**None of the three comparisons reach Holm-corrected significance in log
scale.** Rewiring comes closest (p_holm=0.111), consistent with it also
being the one comparison where mean/median disagreement persists (below).

## Does the log transform resolve the v1 mean-vs-median disagreement?

| Comparison | v1 (raw) primary p / sign+ | v1 median p / sign+ | v2 (log) primary p / sign+ | v2 median p / sign+ | Resolved? |
|---|---|---|---|---|---|
| Historical random | 0.00195 / 0-10 | 0.922 / 5-10 | 0.322 / 3-10 | 0.922 / 5-10 | **Yes** |
| Current random | 0.02734 / 1-10 | 0.492 / 4-10 | 0.160 / 3-10 | 0.322 / 4-10 | **Yes** |
| Rewiring | 0.01367 / 1-10 | 0.193 / 3-10 | 0.037 / 1-10 | 0.084 / 3-10 | **No** |

**Historical random and current random: RESOLVED.** In log scale,
primary and median aggregation now agree with each other (both
non-significant) and with the sign-flip test -- the mean-vs-median split
that made v1's nominal significance untrustworthy for these two
comparisons is gone. Per DESIGN_v2's decision rule, this is the
"resolved, consistent" case: **the conclusion is no significant
difference between T and either random-control definition**, and this
conclusion is now trustworthy in a way v1's raw-scale result was not.
The gated tertiary mixed model was fit for both (since the consistency
gate passed) and its 95% CIs bracket 1.0 on the multiplicative scale
(historical random: x[0.280, 1.541]; current random: x[0.146, 1.250]),
reinforcing the same conclusion from an independent angle.

**Rewiring: NOT resolved.** Primary (p=0.037) and sign-flip (p=0.041)
still say significant; median aggregation (p=0.084) still says not
significant, at a similar magnitude of disagreement to v1 (three of ten
classes positive under both mean and median, but the log-mean statistic
still crosses the raw p<0.05 line where the log-median one doesn't). Per
DESIGN_v2's pre-commitment, this is reported as the final answer for
this comparison under this re-verification effort, not chased with a
further transformation: **the T-vs-rewiring comparison is genuinely
inconclusive under both tested aggregation schemes, at n=10 classes / 25
seeds.** Its Holm-corrected p (0.111, across the 3-comparison v2 family)
is not significant either way, so the practical bottom line -- no
Holm-significant T-vs-rewiring difference established -- is unchanged
from v1's headline conclusion, even though the primary-vs-median
disagreement that made v1's result untrustworthy has narrowed but not
closed.

## Honest caveat on the two "resolved" comparisons: MCSE is still not small

Even for historical random and current random, within-class log-MCSE is
not smaller than |d_log| in most classes (e.g. historical random: MCSE
exceeds |d_log| in classes 2, 3, and 4; is close to it in several
others). DESIGN_v2's decision rule frames "resolved, consistent" as
primary + median + sign-flip agreeing, which they now do for these two
comparisons -- but the large remaining MCSE means this is better read as
"log-scale aggregation is stable enough to say confidently that these 25
seeds show no significant difference," not "25 seeds is enough precision
to rule out a small real difference." The seed-count stability
diagnostic (in `results/stage1a_log_scale_analysis.pkl`) shows the
log-mean estimate is far better behaved than v1's raw-scale one -- no
single-seed multi-fold jumps of the kind that motivated this iteration
-- but it has also not fully flattened out by k=25 in every class.

## Decision-rule verdict, stated plainly

- **T vs. historical half-edge random, coupling-budget normalized:**
  resolved under log-scale aggregation -- no significant difference,
  primary/median/sign-flip/mixed-model all agree.
- **T vs. current edge-count-matched random:** resolved under log-scale
  aggregation -- no significant difference, same agreement across checks.
- **T vs. degree-preserving rewiring:** still inconsistent between mean
  and median aggregation even in log scale. Per DESIGN_v2's
  pre-commitment, no further transformation is attempted. This
  comparison is reported as genuinely inconclusive under both tested
  aggregation schemes -- a property of this specific metric/design/
  sample size, not a failure of analysis technique -- though its
  Holm-corrected p-value (0.111) does not itself claim significance.
- **T vs. lattice** (not re-analyzed here): stands as v1 reported it --
  no significant difference, clean under every check, no aggregation
  ambiguity possible.

## What this two-part effort establishes overall

Across v1 and v2 combined, none of the four original Stage 1A controls
(historical random, current random, rewiring, lattice) shows a
Holm-significant difference from T that survives this project's full
robustness battery. Two of the three stochastic comparisons (historical
and current random) now have a *trustworthy* null result, not merely an
untested one -- the log transform did what the original Stage 1A
FINDINGS.md's secondary log-AUC check anticipated it might. The third
(rewiring) remains a genuinely open question this specific design cannot
resolve, honestly reported as such rather than forced to either
conclusion.

## What this does not do

Per `DESIGN_v2_log_scale.md`'s explicit scope: no new simulation was run,
T-vs-lattice was not re-examined, the seed count was not extended beyond
25, and no transformation beyond natural log was attempted for the
rewiring comparison's persisting disagreement. Extending the rewiring
comparison's seed count, or any other design change, is a new,
separately-justified follow-up, not a silent extension of this iteration.
