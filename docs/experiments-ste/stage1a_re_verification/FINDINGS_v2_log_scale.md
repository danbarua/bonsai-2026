Simplified Technical English version of `experiments/stage1a_re_verification/FINDINGS_v2_log_scale.md`.

# Stage 1A Re-Verification: Log-Scale Iteration (v2) -- Findings

The design for this iteration is in `DESIGN_v2_log_scale.md`. This
document cross-references version 1's (v1's) `FINDINGS.md`. This
document builds on v1. It does not revise v1.

## Friction Log and Data-Provenance Check (Done First, Per DESIGN_v2's Instruction)

Before writing any analysis code, the team checked what v1's driver
script actually saved. The file
`results/stage1a_reverification_results.pkl` contains all 770
individual entries. Each entry maps a (class, construction, seed)
combination to a raw AUC value. For example: (0, 'rewired', 5) maps to
38.39. These are not class-level aggregates. The team confirmed that
"class 0, rewired" alone has 25 distinct seed entries.

**This confirms that v2 is pure re-analysis, exactly as DESIGN_v2
intends.** No new `joint_tangent_matrix_response` call ran anywhere in
this iteration. No new seed was used. No new graph construction was
built. All AUC values are loaded directly from v1's committed data
file. Only the aggregation method applied to them changes.

## What the Team Ran

The team ran `analyze_stage1a_log_scale.py`. This script covers the 3
comparisons in scope: hist_random, curr_random, and rewired. Lattice
is excluded, per `DESIGN_v2_log_scale.md`, because it has no seed
axis, and v1's result for it already stands. The team ran the full
robustness sequence: the log-mean primary test, the log-median
sensitivity check, the exact sign-flip test, the within-class log-MCSE
check, the hierarchical bootstrap, and the gated third-priority mixed
model. The team applied Holm correction across the 3 comparisons in
this family. This is a different, smaller family than v1's family of
4.

## Primary Test Results (Log-Mean-Aggregated, Holm-Corrected Across 3)

| Comparison | Median d (log) | Median, multiplicative | HL (log) | HL, multiplicative | Sign+ /10 | p_exact | p_holm |
|---|---|---|---|---|---|---|---|
| T vs. historical half-edge random | -0.615 | x0.541 | -0.572 | x0.564 | 3/10 | 0.322 | 0.322 |
| T vs. current edge-count-matched random | -0.633 | x0.531 | -0.882 | x0.414 | 3/10 | 0.160 | 0.320 |
| T vs. degree-preserving rewiring | -1.288 | x0.276 | -1.195 | x0.303 | 1/10 | 0.037 | 0.111 |

**None of the three comparisons reach Holm-corrected significance on
the log scale.** Rewiring comes closest, with p_holm = 0.111. This
matches the fact that rewiring is also the one comparison where the
mean-versus-median disagreement persists (see below).

## Does the Log Transform Resolve v1's Mean-vs-Median Disagreement?

| Comparison | v1 (raw) primary p / sign+ | v1 median p / sign+ | v2 (log) primary p / sign+ | v2 median p / sign+ | Resolved? |
|---|---|---|---|---|---|
| Historical random | 0.00195 / 0-10 | 0.922 / 5-10 | 0.322 / 3-10 | 0.922 / 5-10 | **Yes** |
| Current random | 0.02734 / 1-10 | 0.492 / 4-10 | 0.160 / 3-10 | 0.322 / 4-10 | **Yes** |
| Rewiring | 0.01367 / 1-10 | 0.193 / 3-10 | 0.037 / 1-10 | 0.084 / 3-10 | **No** |

**Historical random and current random: RESOLVED.** In log scale, the
primary test and the median-aggregation check now agree with each
other. Both are not significant. Both also agree with the sign-flip
test. In v1, the mean-versus-median split had made the nominal
significance untrustworthy for these two comparisons. That split is
now gone. Per DESIGN_v2's decision rule, this is the "resolved,
consistent" case. **The conclusion is: no significant difference
between T and either random-control definition.** This conclusion is
now trustworthy, in a way that v1's raw-scale result was not.

The team fit the gated third-priority mixed model for both of these
comparisons, since the consistency gate passed for them. The mixed
model's 95% confidence intervals bracket 1.0 on the multiplicative
scale. For historical random, the interval is x[0.280, 1.541]. For
current random, the interval is x[0.146, 1.250]. This reinforces the
same conclusion, from an independent angle.

**Rewiring: NOT resolved.** The primary test (p=0.037) and the
sign-flip test (p=0.041) still say the difference is significant.
Median aggregation (p=0.084) still says it is not significant. This is
a similar level of disagreement to v1: three of ten classes are
positive under both mean and median aggregation, but the log-mean
statistic still crosses the p<0.05 line where the log-median statistic
does not. Per DESIGN_v2's advance commitment, the team reports this as
the final answer for this comparison, for this re-verification effort.
The team does not chase it with a further transformation. **The
T-vs-rewiring comparison is genuinely inconclusive under both tested
aggregation methods, at 10 classes and 25 seeds.** Its Holm-corrected
p-value, 0.111, across the 3-comparison v2 family, is not significant
either way. So the practical bottom line is unchanged from v1's
headline conclusion: no Holm-significant T-vs-rewiring difference is
established. This holds even though the primary-versus-median
disagreement that made v1's result untrustworthy has narrowed. It has
not closed.

## Honest Caveat on the Two "Resolved" Comparisons: MCSE Is Still Not Small

Even for historical random and current random, the within-class
log-MCSE is not smaller than the absolute value of d_log, in most
classes. For example: for historical random, MCSE is larger than the
absolute value of d_log in classes 2, 3, and 4. It is close to it in
several other classes. DESIGN_v2's decision rule defines "resolved,
consistent" as the primary test, the median check, and the sign-flip
test agreeing with each other. They now do agree, for these two
comparisons. But the MCSE is still large. So this result is better
read as: "log-scale aggregation is stable enough to say confidently
that these 25 seeds show no significant difference." It should not be
read as: "25 seeds is enough precision to rule out a small real
difference."

The seed-count stability check (in
`results/stage1a_log_scale_analysis.pkl`) shows that the log-mean
estimate behaves far better than v1's raw-scale one. It shows no
single-seed jumps of the kind that motivated this iteration. But it
has also not fully flattened out by seed 25, in every class.

## Decision-Rule Verdict, Stated Plainly

- **T vs. historical half-edge random, coupling-budget normalized:**
  resolved under log-scale aggregation. No significant difference. The
  primary test, the median check, the sign-flip test, and the mixed
  model all agree.
- **T vs. current edge-count-matched random:** resolved under
  log-scale aggregation. No significant difference. The same checks
  agree.
- **T vs. degree-preserving rewiring:** still inconsistent between
  mean and median aggregation, even on the log scale. Per DESIGN_v2's
  advance commitment, the team attempts no further transformation.
  This comparison is reported as genuinely inconclusive, under both
  tested aggregation methods. This is a property of this specific
  metric, design, and sample size. It is not a failure of analysis
  technique. Its Holm-corrected p-value, 0.111, does not itself claim
  significance.
- **T vs. lattice** (not re-analyzed here): stands as v1 reported it.
  No significant difference. Clean under every check. No aggregation
  ambiguity is possible for this comparison.

## What This Two-Part Effort Establishes Overall

Across v1 and v2 combined, none of the four original Stage 1A controls
(historical random, current random, rewiring, and lattice) shows a
Holm-significant difference from T that survives this project's full
robustness battery. Two of the three stochastic comparisons
(historical random and current random) now have a *trustworthy* null
result, not merely an untested one. The log transform did what the
original Stage 1A `FINDINGS.md`'s secondary log-AUC check anticipated
it might do. The third comparison, rewiring, remains a genuinely open
question that this specific design cannot resolve. The team reports
this honestly, rather than forcing it toward either conclusion.

## What This Document Does Not Do

Per `DESIGN_v2_log_scale.md`'s explicit scope: the team ran no new
simulation. The team did not re-examine T vs. lattice. The team did
not extend the seed count beyond 25. The team attempted no
transformation beyond the natural logarithm, for the rewiring
comparison's persisting disagreement. Extending the rewiring
comparison's seed count, or making any other design change, would be a
new, separately-justified follow-up. It is not a silent extension of
this iteration.
