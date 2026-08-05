"""
Stage 2B's confirmatory statistical machinery: the primary paired
class-stratified bootstrap, the paired t-test, the studentized chunked
sign-flip robustness test, the two prespecified Holm families, the
hierarchical identity-baseline gate, and the branched "one graph wins"
rule -- implementing DESIGN.md's "Primary comparison and test",
"Secondary comparisons: two families", and "Identity baselines"
sections exactly.

REUSED UNMODIFIED from `stage2a_stats` (imported, not copied):

- `holm_bonferroni` -- step-down Holm, already unit-tested in
  `tests/test_stage2a_stats.py`. Both Stage 2B families use it.
- `paired_class_stratified_bootstrap` -- the CPU reference bootstrap.
  Stage 2B's primary test is the same 20,000-resample paired
  class-stratified procedure at `seed=42`, so this function is called
  directly rather than reimplemented, and remains the oracle.

NOT reused: `stage2a_stats.paired_sign_flip_p`. That function implements
a different test -- the raw `|mean(d)|` statistic at 20,000 permutations
with a fully materialized float sign matrix. DESIGN.md's Stage 2B
Family-2 robustness check locks a studentized statistic at 100,000
flips with int8 signs generated and consumed in chunks. The two are
separate tests with separate scopes; `paired_sign_flip_p` is left
untouched because Stage 2A's committed findings depend on it.

Scope note: this module is pure functions over arrays of per-image
values. It loads no dataset, reads no image, and knows nothing about
corruption, encoding, or splits. Everything it needs is a per-image
clipped MSE vector per condition plus the class labels the bootstrap
stratifies on.
"""
import itertools
import os
import sys

import numpy as np
from scipy.stats import ttest_1samp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE2A_DIR = os.path.join(_THIS_DIR, "..", "stage2a_dynamics_classification")
sys.path.insert(0, os.path.abspath(_STAGE2A_DIR))

from stage2a_stats import (  # noqa: E402
    holm_bonferroni,
    paired_class_stratified_bootstrap,
)

# ---- Locked constants (DESIGN.md) ----

# "Primary comparison and test": 20,000-resample paired class-stratified
# bootstrap, two-sided 95% percentile interval, seed=42.
N_BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 42

# "Secondary comparisons: two families", Family-2 robustness check:
# 100,000 flips, seed 42, studentized statistic, two-sided, +1 correction.
N_SIGN_FLIPS = 100000
SIGN_FLIP_SEED = 42

# "Computational strategy": "chunked 512-4096 resamples/batch ... host
# loop over chunks permitted, per-resample Python loop is not".
SIGN_FLIP_CHUNK_RANGE = (512, 4096)
SIGN_FLIP_CHUNK = 2048

ALPHA = 0.05

# Condition names. `EVOLVED_GRAPHS` is ordered, and that order defines
# Family 2's canonical pair keys -- do not reorder it without accepting
# that every pair key changes.
PRE_EVOLUTION = "pre_evolution"
PRIMARY_GRAPH = "T"
EVOLVED_GRAPHS = ("T", "lattice", "rewired", "curr_random")
CONTROL_GRAPHS = ("lattice", "rewired", "curr_random")

# How this module compares an adjusted p against alpha, in BOTH families.
#
# DESIGN.md words Family 1's qualification as "Family-1 Holm-adjusted
# p < 0.05" -- strict -- and words Family 2 only as "after Family-2 Holm
# correction", without naming the comparison. Read literally those two
# can differ, at the single measure-zero point where an adjusted p lands
# exactly on alpha. This module applies DESIGN.md's own explicit Family-1
# rule to both families: strict `<`, the standard convention for
# "rejects at level alpha". That resolves a wording inconsistency in the
# locked design by using one of its own stated rules; it is not a change
# to the design and does not alter any comparison whose adjusted p is not
# exactly equal to alpha.
ALPHA_COMPARISON_NOTE = (
    "Both families qualify on strict `Holm-adjusted p < alpha`. "
    "DESIGN.md states this explicitly for Family 1 and leaves Family 2's "
    "comparison unworded; the Family-1 rule is applied to both. The "
    "imported `holm_bonferroni`'s own step-down decision is recorded "
    "alongside as `holm_rejected` and is unmodified -- it differs only "
    "for an adjusted p exactly equal to alpha."
)

SIGN_EXCHANGEABILITY_CAVEAT = (
    "Sign-exchangeability assumption: exact sign-flip validity requires "
    "each d_i to be exchangeable with -d_i under H0, i.e. a symmetry "
    "condition on d_i's distribution about zero -- strictly stronger "
    "than E[d_i]=0. That condition is not verified here; at Stage 2B's "
    "test-corpus size this is a large-sample-justified approximation "
    "for the mean contrast, not a test that is exact by construction. "
    "DESIGN.md requires this assumption to be stated wherever this "
    "p-value is reported."
)

# The CLT justification DESIGN.md states once, so assumption scrutiny
# applies symmetrically to the t-tests and not only to the sign-flip.
T_TEST_CLT_NOTE = (
    "At the locked test-corpus size the paired t-test is justified by "
    "the CLT even for skewed per-image MSE differences (DESIGN.md, "
    "'Primary comparison and test'). At the small sample sizes used in "
    "unit tests that justification does not apply."
)


# ---- Family-2 canonical pairs ----

def family2_pairs(graphs=EVOLVED_GRAPHS):
    """The six ordered pairs of Family 2, in one canonical order.

    `itertools.combinations` over `EVOLVED_GRAPHS` fixes both which pairs
    exist and which graph is the first element of each. The first element
    is the minuend: the pair's difference is always
    `MSE(first) - MSE(second)`, so a negative mean favors the FIRST
    graph. `pairwise_outcome` below is the only supported way to read a
    direction for a graph that happens to sit second in its pair."""
    return tuple(itertools.combinations(graphs, 2))


def pair_key(graph_a, graph_b):
    """The dict key for a Family-2 pair, `"<a>_vs_<b>"`."""
    return f"{graph_a}_vs_{graph_b}"


# ---- Paired t-test ----

def paired_t_test(d):
    """One-sample t-test of the paired differences against zero -- the
    p-value source for both Holm families (DESIGN.md: "Paired t-tests
    produce p-values; Holm across the three" / "across the six").

    `d` is the per-image difference vector, sign convention fixed by the
    caller. `mean < 0` means the minuend condition has the LOWER
    reconstruction error, i.e. the favorable direction for it.

    `t` is `mean(d) / (SD(d, ddof=1) / sqrt(n))`. That is the exact form
    the Family-2 sign-flip statistic studentizes, so `t` here and
    `t_observed` from `studentized_sign_flip_p` are the same number on
    the same input.

    `p_underflow` flags a p-value that reached floating-point zero. A
    reported "p = 0" is never correct (CLAUDE.md principle 6); when this
    flag is set, the honest statement is that the two-sided tail is below
    float64's smallest representable positive value, not that it is
    zero."""
    d = np.asarray(d, dtype=np.float64).ravel()
    n = d.size
    if n < 2:
        raise ValueError("paired t-test needs at least 2 paired differences")
    if not np.all(np.isfinite(d)):
        raise ValueError("non-finite paired differences -- the t-test is undefined")
    result = ttest_1samp(d, 0.0)
    sd = float(np.std(d, ddof=1))
    p = float(result.pvalue)
    return {
        "n": int(n),
        "mean": float(np.mean(d)),
        "sd": sd,
        "se": sd / np.sqrt(n),
        "t": float(result.statistic),
        "df": int(n - 1),
        "p_value": p,
        "p_underflow": bool(p == 0.0),
        "assumption": T_TEST_CLT_NOTE,
    }


# ---- Studentized chunked sign-flip test ----

def _studentized_statistic(M):
    """`mean(row) / (SD(row, ddof=1) / sqrt(n))` for every row of `M`.

    `ddof=1` is what makes this "matching the t-test's form" (DESIGN.md);
    the sample SD is the one a paired t-test uses.

    Degenerate rows -- SD exactly zero -- are outside DESIGN.md's formula,
    which assumes the ratio is defined. The convention adopted here, and
    surfaced in the returned diagnostics rather than left implicit:

      - SD = 0 and mean = 0 (an identically-zero row): the statistic is
        defined as 0. Every flip of an identically-zero difference vector
        produces the same degenerate row, so the two-sided comparison ties
        for every flip and the test correctly returns p = 1 -- the null is
        exactly true and there is no evidence against it. Letting 0/0
        stand as NaN would instead make every comparison False and return
        the Monte Carlo floor, which is the opposite of correct.
      - SD = 0 and mean != 0 (a constant nonzero row): the statistic is
        +/- infinity, which is what the division already produces. Only an
        all-same-sign flip can tie it, so the test returns the Monte Carlo
        floor -- the correct answer for a maximally separated input.

    Returns `(t, n_zero_sd)`."""
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    n = M.shape[1]
    mean = M.mean(axis=1)
    sd = M.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / np.sqrt(n))
    # 0/0 -> NaN -> defined as 0 (see docstring); c/0 -> +/-inf is kept.
    t = np.where(np.isnan(t), 0.0, t)
    return t, int(np.count_nonzero(sd == 0.0))


def studentized_sign_flip_p(d, n_flips=N_SIGN_FLIPS, seed=SIGN_FLIP_SEED,
                            chunk_size=SIGN_FLIP_CHUNK):
    """DESIGN.md's Family-2 robustness check, implemented literally.

    From the locked spec: "Sign-flip retained as robustness only: 100,000
    flips, seed 42, studentized statistic
    `T_b = mean(s_b * d) / (SD(s_b * d) / sqrt(n))` matching the t-test's
    form, two-sided (`|T_b| >= |T_obs|`), +1 correction
    (`p = (1 + count) / 100001`), sign-exchangeability assumption stated
    wherever reported."

    This is a DIFFERENT test from `stage2a_stats.paired_sign_flip_p`
    (raw `|mean(d)|`, 20,000 permutations, materialized float signs).
    That function is Stage 2A's and is not reused here.

    Robustness only: the Family-2 decision rule is the Holm-corrected
    paired t-test. This p-value is reported alongside it, never in place
    of it, and never fed into `holm_bonferroni`.

    Chunking, and why the sign stream is drawn from float64 uniforms
    ---------------------------------------------------------------
    Signs are generated and consumed in chunks of `chunk_size` flips; the
    full (n_flips, n) matrix is never materialized. DESIGN.md permits a
    host-level loop over chunks and forbids a per-resample Python loop,
    so each chunk's statistic is computed vectorized over its rows.

    The chunk's sign matrix is int8, per DESIGN.md's dtype table. It is
    built by thresholding float64 uniforms at 0.5 rather than by
    `Generator.integers(0, 2, dtype=uint8)` for a specific, non-obvious
    reason: `Generator.random` at float64 width consumes exactly one
    64-bit draw per element, so a sequence drawn in chunks is bit-identical
    to the same sequence drawn in one call, for ANY chunk size. Bounded
    integer generation at sub-64-bit width buffers bits within a call and
    discards the remainder at the call boundary, so chunked draws do NOT
    reproduce an unchunked stream -- verified directly, not assumed. Using
    it would make the p-value silently depend on `chunk_size`. The
    `chunk_size`-invariance test in `tests/test_stage2b_stats.py` is what
    keeps this property from being refactored away.

    Memory and runtime, measured rather than extrapolated (CLAUDE.md
    principle 18). The transient per-chunk product `s_b * d` is float64
    and sized `chunk_size * n * 8` bytes; the int8 sign matrix is
    `chunk_size * n`. Neither scales with `n_flips`. At the locked
    100,000 flips with n=10,000 on this project's CPU, one contrast
    measured:

        chunk_size    peak traced memory    wall time
             512               128 MB          6.7 s
            2048               512 MB          6.6 s
            4096              1024 MB          6.7 s

    -- identical p-value and identical exceedance count at all three, and
    no speed penalty for the smallest. Wall time is dominated by the
    float64 uniform draws, not by chunk bookkeeping, so `chunk_size` is a
    pure memory knob here; the default sits mid-range and a caller under
    memory pressure can drop to 512 at no cost. Six Family-2 contrasts
    come to roughly 40 s total.

    Returns a dict. `assumption` carries
    `SIGN_EXCHANGEABILITY_CAVEAT` so the number cannot be reported
    without the caveat travelling with it."""
    d = np.asarray(d, dtype=np.float64).ravel()
    n = d.size
    if n < 2:
        raise ValueError("sign-flip test needs at least 2 paired differences")
    if not np.all(np.isfinite(d)):
        raise ValueError("non-finite paired differences -- the statistic is undefined")
    if n_flips < 1:
        raise ValueError("n_flips must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    t_obs_arr, n_zero_sd_obs = _studentized_statistic(d[None, :])
    t_obs = float(t_obs_arr[0])
    abs_t_obs = abs(t_obs)

    rng = np.random.default_rng(seed)
    n_as_extreme = 0
    n_zero_sd_flips = 0
    n_chunks = 0
    remaining = n_flips
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        # float64 uniforms: chunk-stream-invariant (see docstring)
        u = rng.random((this_chunk, n))
        signs = np.where(u < 0.5, np.int8(-1), np.int8(1))
        products = signs * d                      # (this_chunk, n) float64, transient
        t_b, n_zero = _studentized_statistic(products)
        n_as_extreme += int(np.count_nonzero(np.abs(t_b) >= abs_t_obs))
        n_zero_sd_flips += n_zero
        n_chunks += 1
        remaining -= this_chunk

    p_value = (1 + n_as_extreme) / (n_flips + 1)
    lo, hi = SIGN_FLIP_CHUNK_RANGE
    return {
        "p_value": p_value,
        "p_monte_carlo_floor": 1.0 / (n_flips + 1),
        "at_monte_carlo_floor": bool(n_as_extreme == 0),
        "t_observed": t_obs,
        "n_as_extreme": int(n_as_extreme),
        "n_flips": int(n_flips),
        "n": int(n),
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "n_chunks": int(n_chunks),
        "chunk_size_in_design_range": bool(lo <= chunk_size <= hi),
        "sign_dtype": "int8",
        "degenerate_observed": bool(n_zero_sd_obs > 0),
        "n_zero_sd_flips": int(n_zero_sd_flips),
        "statistic": "studentized: mean(s*d) / (SD(s*d, ddof=1) / sqrt(n))",
        "comparison": "two-sided, |T_b| >= |T_obs|",
        "assumption": SIGN_EXCHANGEABILITY_CAVEAT,
    }


# ---- Primary test (outside both families, uncorrected) ----

def primary_bootstrap_test(mse_by_condition, y,
                           n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """DESIGN.md's primary comparison: `d_i = MSE_i(evolved_T) -
    MSE_i(pre_evolution)`, 20,000-resample paired class-stratified
    bootstrap, two-sided 95% percentile interval, seed=42.

    Decision rule, verbatim from the locked design: "Interval entirely
    below zero => evolution improves; entirely above => pre-evolution
    wins; straddling => null, stated plainly."

    This test sits OUTSIDE both Holm families and is uncorrected. That is
    enforced structurally, not by comment: the returned dict contains no
    p-value under any key, so it cannot be swept into a `holm_bonferroni`
    call by a caller assembling a family from p-values. The locked rule
    is CI-based; a p-value here would not merely be uncorrected, it would
    not be part of the rule at all.

    The bootstrap itself is `stage2a_stats.paired_class_stratified_bootstrap`,
    imported and called unmodified -- the same procedure Stage 2A's own
    primary result used, at the same resample count and seed."""
    d = _contrast(mse_by_condition, PRIMARY_GRAPH, PRE_EVOLUTION)
    y = np.asarray(y).ravel()
    if y.size != d.size:
        raise ValueError(f"y has {y.size} entries but the contrast has {d.size}")
    boot = paired_class_stratified_bootstrap(d, y, n_resamples=n_resamples, seed=seed)

    ci_low, ci_high = boot["ci_low"], boot["ci_high"]
    if ci_high < 0:
        verdict = "evolution_improves"
    elif ci_low > 0:
        verdict = "pre_evolution_wins"
    else:
        verdict = "null"
    return {
        "contrast": f"MSE({PRIMARY_GRAPH}) - MSE({PRE_EVOLUTION})",
        "observed_mean": boot["observed_mean"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "resampled_means": boot["resampled_means"],
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "verdict": verdict,
        "entirely_below_zero": bool(ci_high < 0),
        "multiplicity_family": None,
        "correction": "none -- the primary test sits outside both Holm families",
        "decision_rule": ("interval entirely below zero => evolution improves; "
                          "entirely above => pre-evolution wins; straddling => null"),
    }


# ---- Family 1: three controls vs. pre_evolution ----

def family1_vs_pre_evolution(mse_by_condition, y, alpha=ALPHA,
                             n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """DESIGN.md's Family 1, three tests: lattice / rewired / curr_random
    vs. `pre_evolution`. "Paired t-tests produce p-values; Holm across
    the three; bootstrap intervals alongside for effect size."

    Each contrast is `MSE(control) - MSE(pre_evolution)`, so `mean < 0`
    is the favorable direction for the control graph.

    Correction is within this family only, via the imported
    `stage2a_stats.holm_bonferroni`. `T` is deliberately absent: its
    comparison against `pre_evolution` is the primary test, which is
    uncorrected and outside both families.

    No sign-flip test here -- DESIGN.md attaches the sign-flip robustness
    check to Family 2 only."""
    y = np.asarray(y).ravel()
    raw_p = {}
    per_test = {}
    for graph in CONTROL_GRAPHS:
        d = _contrast(mse_by_condition, graph, PRE_EVOLUTION)
        if y.size != d.size:
            raise ValueError(f"y has {y.size} entries but the contrast has {d.size}")
        t_result = paired_t_test(d)
        boot = paired_class_stratified_bootstrap(d, y, n_resamples=n_resamples, seed=seed)
        raw_p[graph] = t_result["p_value"]
        per_test[graph] = {
            "contrast": f"MSE({graph}) - MSE({PRE_EVOLUTION})",
            "t_test": t_result,
            "bootstrap": {
                "observed_mean": boot["observed_mean"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "n_resamples": int(n_resamples),
                "seed": int(seed),
            },
            "favorable": bool(t_result["mean"] < 0),
        }
    adjusted, rejected = holm_bonferroni(raw_p, alpha=alpha)
    underflow = _underflow_summary(raw_p, per_test)
    return {
        "family": "family1",
        "description": "three controls vs. pre_evolution",
        "members": tuple(CONTROL_GRAPHS),
        "raw_p": raw_p,
        "holm_adjusted_p": adjusted,
        "holm_rejected": rejected,
        "per_test": per_test,
        "alpha": float(alpha),
        "n_tests": len(CONTROL_GRAPHS),
        **underflow,
    }


# ---- Family 2: six pairwise among the four evolved graphs ----

def family2_pairwise(mse_by_condition, alpha=ALPHA, run_sign_flip=True,
                     n_flips=N_SIGN_FLIPS, sign_flip_seed=SIGN_FLIP_SEED,
                     chunk_size=SIGN_FLIP_CHUNK):
    """DESIGN.md's Family 2, six tests: all pairwise comparisons among the
    four evolved graphs. "Paired t-tests, Holm across the six. Sign-flip
    retained as robustness only."

    Pairs and their direction are fixed by `family2_pairs()`, not by the
    caller: each contrast is `MSE(first) - MSE(second)` in
    `EVOLVED_GRAPHS` order, so a caller cannot invert a sign by supplying
    a difference vector the wrong way round. Use `pairwise_outcome` to
    read a direction for a graph that sits second in its pair.

    The Holm correction is applied to the six paired-t p-values only. The
    sign-flip p-values are reported per pair as robustness and are never
    passed to `holm_bonferroni` -- correcting them as a second family
    would create a family DESIGN.md does not prespecify."""
    raw_p = {}
    per_test = {}
    for graph_a, graph_b in family2_pairs():
        d = _contrast(mse_by_condition, graph_a, graph_b)
        t_result = paired_t_test(d)
        key = pair_key(graph_a, graph_b)
        raw_p[key] = t_result["p_value"]
        record = {
            "graph_a": graph_a,
            "graph_b": graph_b,
            "contrast": f"MSE({graph_a}) - MSE({graph_b})",
            "t_test": t_result,
            "a_favored": bool(t_result["mean"] < 0),
        }
        if run_sign_flip:
            record["sign_flip"] = studentized_sign_flip_p(
                d, n_flips=n_flips, seed=sign_flip_seed, chunk_size=chunk_size)
        per_test[key] = record
    adjusted, rejected = holm_bonferroni(raw_p, alpha=alpha)
    underflow = _underflow_summary(raw_p, per_test)
    return {
        "family": "family2",
        "description": "six pairwise comparisons among the four evolved graphs",
        "members": tuple(pair_key(a, b) for a, b in family2_pairs()),
        "raw_p": raw_p,
        "holm_adjusted_p": adjusted,
        "holm_rejected": rejected,
        "per_test": per_test,
        "alpha": float(alpha),
        "n_tests": len(family2_pairs()),
        **underflow,
        "sign_flip_role": ("robustness only -- the Family-2 decision rule is the "
                           "Holm-corrected paired t-test; sign-flip p-values are "
                           "not Holm-corrected and are not a second family"),
        "sign_flip_assumption": SIGN_EXCHANGEABILITY_CAVEAT if run_sign_flip else None,
    }


def _underflow_summary(raw_p, per_test):
    """Which of a family's t-test p-values reached floating-point zero.

    Surfaced at the FAMILY level, not only inside each `per_test` entry,
    because a family dict's `raw_p` / `holm_adjusted_p` are what a
    reporting caller reads -- and a raw p of `0.0` Holm-adjusts to `0.0`,
    so the underflow propagates into the corrected value with nothing at
    that layer to mark it. Reporting either as "p = 0" is never correct
    (CLAUDE.md principle 6); the supported statement is that the
    two-sided tail is below float64's smallest representable positive
    value.

    This is reachable at Stage 2B's real corpus size, not a contrived
    edge: a per-image MSE difference an order of magnitude above its own
    standard error over ~10,000 images already drives |t| past the point
    where the analytic tail underflows."""
    underflowed = tuple(k for k in raw_p if per_test[k]["t_test"]["p_underflow"])
    return {
        "p_underflowed": underflowed,
        "n_p_underflowed": len(underflowed),
        "underflow_note": (
            "a raw or Holm-adjusted p of 0.0 is a float64 underflow of the "
            "analytic t-tail, not an exact zero; report it as below the "
            "smallest representable positive double (CLAUDE.md principle 6)"
            if underflowed else None),
    }


def pairwise_outcome(family2, graph, other):
    """`graph`'s Family-2 result against `other`, read in `graph`'s own
    direction regardless of which side of the canonical pair it sits on.

    The canonical pair stores `MSE(first) - MSE(second)`. When `graph` is
    the SECOND element the stored mean is `MSE(other) - MSE(graph)`, so
    the direction must be negated to answer "does `graph` outperform
    `other`". The Holm-adjusted p-value is two-sided and symmetric in the
    pair, so it is read unchanged.

    Reading the stored mean without this negation inverts the direction
    for exactly half of the comparisons, and does so silently -- it is
    the specific error this helper exists to make impossible.

    Two significance fields, deliberately both present:

      - `holm_significant` is this module's own alpha comparison,
        `Holm-adjusted p < alpha` (strict), read from `family2["alpha"]`.
        This is the field the "one graph wins" rule consumes.
      - `holm_rejected` is the imported `holm_bonferroni`'s own step-down
        decision, recorded unchanged. `holm_bonferroni` is Stage 2A's and
        is not modified here.

    The two are equivalent except for an adjusted p exactly equal to
    alpha, where the step-down rule's `<=` rejects and the strict
    comparison does not. `ALPHA_COMPARISON_NOTE` records why strict `<`
    is the one used."""
    key = pair_key(graph, other)
    if key in family2["per_test"]:
        mean_diff = family2["per_test"][key]["t_test"]["mean"]
        graph_is_first = True
    else:
        key = pair_key(other, graph)
        if key not in family2["per_test"]:
            raise KeyError(f"no Family-2 pair for {graph!r} and {other!r}")
        mean_diff = -family2["per_test"][key]["t_test"]["mean"]
        graph_is_first = False
    alpha = family2["alpha"]
    return {
        "pair_key": key,
        "graph": graph,
        "other": other,
        "graph_is_first_in_pair": graph_is_first,
        "mean_diff": float(mean_diff),   # MSE(graph) - MSE(other), always
        "favorable": bool(mean_diff < 0),
        "holm_adjusted_p": family2["holm_adjusted_p"][key],
        "holm_significant": bool(family2["holm_adjusted_p"][key] < alpha),
        "holm_rejected": bool(family2["holm_rejected"][key]),
        "alpha": float(alpha),
    }


# ---- Descriptive ranking (NOT an inferential claim) ----

DESCRIPTIVE_RANKING_NOTE = (
    "DESCRIPTIVE ONLY. This is a plain readout of point estimates and "
    "directional pairwise outcomes, not a test and not a verdict. It "
    "applies no multiplicity correction of its own -- the "
    "Holm-surviving counts are read from Family 2's existing correction, "
    "and nothing further is corrected for producing this ordering. It "
    "qualifies NO graph as a winner: `unique_winner` in this same dict "
    "is the only supported inferential statement about which graph wins, "
    "and it is decided by the branched rule, not by this ranking. A "
    "graph named `best_point_estimate` here may well be a graph that "
    "`unique_winner` correctly declines to name."
)


def _descriptive_ranking(beats_others):
    """A descriptive companion to `unique_winner` -- never a substitute.

    Reports which graph has the best (lowest) mean per-image MSE, and for
    every graph how many of its three Family-2 pairwise comparisons it
    leads directionally, split into those that also survive Family-2 Holm
    and those that do not. That split is the point: it makes a "3 of 3
    directional, 2 of 3 Holm-surviving" near-miss legible at a glance
    instead of collapsing to a bare `unique_winner: None`.

    The model is Stage 2A's own reporting, which published the full
    descriptive ranking (`curr_random > rewired > T > lattice`) side by
    side with the rigorous test outcomes rather than merging the two into
    one number -- here applied to the pre-registered design rather than a
    post hoc one.

    Read entirely from `beats_others`' already-computed `per_opponent`
    records, so the direction and Holm reads behind these counts are the
    SAME reads `beats_all` used. Recomputing them from `family2` would
    reintroduce exactly the risk CLAUDE.md principle 16 names: a second,
    independently-written path around a verified helper, free to disagree
    with the first.

    `best_point_estimate` is always a graph, never `None`: some condition
    always has the lowest mean, even when nothing separates significantly.
    Because each pairwise mean is an exact difference of condition means
    (`mean(a - b) == mean(a) - mean(b)`), the directional outcomes induce
    a consistent total order, and the lowest-mean graph is exactly the one
    leading all three of its pairs. `order_is_strict` records that this
    held -- it is False only under an exact tie between two conditions'
    means, where the ordering degenerates and should not be read as one.
    """
    graphs = tuple(beats_others)
    per_graph = {}
    for graph in graphs:
        outcomes = beats_others[graph]["per_opponent"]
        directional = tuple(o for o in graphs
                            if o != graph and outcomes[o]["favorable"])
        surviving = tuple(o for o in directional
                          if outcomes[o]["holm_significant"])
        directional_only = tuple(o for o in directional if o not in surviving)
        n_opp = len(outcomes)
        per_graph[graph] = {
            "n_opponents": n_opp,
            "n_directional_wins": len(directional),
            "n_holm_surviving_wins": len(surviving),
            "directional_wins": directional,
            "holm_surviving_wins": surviving,
            "directional_only_wins": directional_only,
            "summary": (f"{len(directional)} of {n_opp} directional, "
                        f"{len(surviving)} of {n_opp} Holm-surviving"),
        }

    order = tuple(sorted(graphs,
                         key=lambda g: (-per_graph[g]["n_directional_wins"], g)))
    win_counts = [per_graph[g]["n_directional_wins"] for g in graphs]
    order_is_strict = len(set(win_counts)) == len(win_counts)
    best = order[0]
    return {
        "note": DESCRIPTIVE_RANKING_NOTE,
        "is_inferential": False,
        "best_point_estimate": best,
        "best_point_estimate_summary": per_graph[best]["summary"],
        "order": order,
        "order_is_strict": bool(order_is_strict),
        "per_graph": per_graph,
    }


# ---- "One graph wins", branched per candidate ----

def one_graph_wins(primary, family1, family2, alpha=ALPHA):
    """DESIGN.md's branched rule, verbatim: "T qualifies iff the primary
    bootstrap interval lies entirely below zero. A control graph
    qualifies iff its direction is favorable and its Family-1
    Holm-adjusted p < 0.05. A graph is the unique winner only if it
    qualifies by whichever rule applies to it AND outperforms each of the
    other three in the favorable direction after Family-2 Holm
    correction."

    The branch is the point: `T` is qualified by the PRIMARY bootstrap
    interval (its `pre_evolution` contrast is the primary test and has no
    p-value at all), while a control graph is qualified by its FAMILY-1
    Holm-adjusted p. Two different corrections at two different steps.
    Applying Family-1's rule to `T`, or the primary's rule to a control,
    would invert a headline claim.

    The winner step is separate again, and uses FAMILY-2 Holm --
    qualification and unique-winner status are never decided by the same
    correction.

    `unique_winner` is `None` whenever the leading graph fails Holm
    against even one rival. That is deliberate and is the whole point of
    the rule: naming a winner off a leader that cannot separate from one
    of its three rivals is the overclaim Stage 2A had to walk back once.
    `descriptive_ranking`, returned alongside, reports what the point
    estimates and directional outcomes actually looked like -- so a
    near-miss is legible instead of invisible -- and is descriptive only.
    It never qualifies a graph as a winner; see
    `DESCRIPTIVE_RANKING_NOTE`.

    `alpha` is compared strictly -- `Holm-adjusted p < alpha` -- in BOTH
    families, under the same field name `holm_significant` on each side.
    DESIGN.md words that rule explicitly for Family 1 and leaves Family
    2's comparison unworded; applying the Family-1 rule to both resolves
    the wording inconsistency without changing the locked design. See
    `ALPHA_COMPARISON_NOTE`. The imported `holm_bonferroni`'s own
    step-down decision is recorded alongside as `holm_rejected` and is
    unmodified; the two differ only for an adjusted p exactly equal to
    alpha."""
    qualification = {}

    qualification[PRIMARY_GRAPH] = {
        "rule": "primary bootstrap interval entirely below zero",
        "rule_source": "primary test (outside both families, uncorrected)",
        "ci_low": primary["ci_low"],
        "ci_high": primary["ci_high"],
        "qualifies": bool(primary["ci_high"] < 0),
    }
    for graph in CONTROL_GRAPHS:
        favorable = family1["per_test"][graph]["favorable"]
        adj_p = family1["holm_adjusted_p"][graph]
        holm_significant = bool(adj_p < alpha)
        qualification[graph] = {
            "rule": "favorable direction AND Family-1 Holm-adjusted p < alpha",
            "rule_source": "Family 1 (vs. pre_evolution, Holm across three)",
            "favorable": bool(favorable),
            "holm_adjusted_p": adj_p,
            "holm_significant": holm_significant,
            "holm_rejected": bool(family1["holm_rejected"][graph]),
            "qualifies": bool(favorable and holm_significant),
        }

    beats_others = {}
    for graph in EVOLVED_GRAPHS:
        outcomes = {other: pairwise_outcome(family2, graph, other)
                    for other in EVOLVED_GRAPHS if other != graph}
        beats_others[graph] = {
            "per_opponent": outcomes,
            "beats_all": bool(all(o["favorable"] and o["holm_significant"]
                                  for o in outcomes.values())),
        }

    winners = [g for g in EVOLVED_GRAPHS
               if qualification[g]["qualifies"] and beats_others[g]["beats_all"]]
    if len(winners) > 1:  # pragma: no cover -- arithmetically impossible
        raise RuntimeError(
            f"more than one graph beat all three others: {winners} -- "
            "mutually exclusive by construction, so this indicates a "
            "direction or key-lookup error in family2")

    return {
        "qualification": qualification,
        "beats_others": beats_others,
        "unique_winner": winners[0] if winners else None,
        "descriptive_ranking": _descriptive_ranking(beats_others),
        "alpha": float(alpha),
        "alpha_comparison": ALPHA_COMPARISON_NOTE,
        "rule": ("T qualifies iff the primary bootstrap interval lies entirely "
                 "below zero; a control graph qualifies iff its direction is "
                 "favorable and its Family-1 Holm-adjusted p < alpha; a graph is "
                 "the unique winner only if it qualifies by whichever rule "
                 "applies to it AND outperforms each of the other three in the "
                 "favorable direction after Family-2 Holm correction"),
    }


# ---- Identity baselines: the hierarchical gate ----

def identity_baseline_gate(primary, mse_by_condition, y, identity_key,
                           n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """DESIGN.md's "Identity baselines: hierarchical gate", levels 2 and 3.

    Level 2, the denoising gate: `evolved_T` vs. identity, evaluated ONLY
    if the primary succeeded. "Never rescues a failed primary" is enforced
    structurally -- when `primary["entirely_below_zero"]` is False the
    gate contrast is not computed at all and `gate_evaluated` is False, so
    there is no number for a later reader to quote out of order.

    Level 3, context: `pre_evolution` vs. identity, its own paired
    bootstrap interval, ALWAYS reported and explicitly outside the gate.
    It is computed whether or not the primary succeeded, because DESIGN.md
    reports it independently.

    The claim distinction this gate exists to preserve, from DESIGN.md's
    "Strongest supportable claim, stated in advance": the primary result
    alone supports "runtime graph evolution improves linear reconstruction
    of clean intensities on the fixed active support under one
    prespecified corruption level". "Actual denoising" is added only if
    this gate also succeeds. The two claims are distinct and are not
    conflated."""
    y = np.asarray(y).ravel()

    def _paired(minuend, subtrahend):
        d = _contrast(mse_by_condition, minuend, subtrahend)
        if y.size != d.size:
            raise ValueError(f"y has {y.size} entries but the contrast has {d.size}")
        boot = paired_class_stratified_bootstrap(d, y, n_resamples=n_resamples, seed=seed)
        return {
            "contrast": f"MSE({minuend}) - MSE({subtrahend})",
            "observed_mean": boot["observed_mean"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "entirely_below_zero": bool(boot["ci_high"] < 0),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
        }

    primary_succeeded = bool(primary["entirely_below_zero"])
    gate = _paired(PRIMARY_GRAPH, identity_key) if primary_succeeded else None
    context = _paired(PRE_EVOLUTION, identity_key)

    return {
        "identity_key": identity_key,
        "primary_succeeded": primary_succeeded,
        "gate_evaluated": primary_succeeded,
        "gate": gate,
        "gate_passed": bool(gate["entirely_below_zero"]) if gate else False,
        "pre_evolution_vs_identity": context,
        "context_note": ("pre_evolution vs. identity is reported independently and "
                         "sits outside the gate"),
        "claim_if_gate_passes": ("the 'actual denoising' claim may be added to the "
                                 "primary reconstruction claim"),
        "claim_if_gate_absent_or_failed": ("only the reconstruction claim is "
                                           "supported; the gate never rescues a "
                                           "failed primary"),
    }


# ---- Orchestrator ----

def run_stage2b_inference(mse_by_condition, y, alpha=ALPHA, identity_key=None,
                          run_sign_flip=True, n_flips=N_SIGN_FLIPS,
                          sign_flip_seed=SIGN_FLIP_SEED, chunk_size=SIGN_FLIP_CHUNK,
                          n_resamples=N_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """The full locked inference over per-image clipped MSE vectors.

    `mse_by_condition` maps each condition name to its per-image clipped
    MSE array on the active support: `pre_evolution` plus the four
    graphs in `EVOLVED_GRAPHS`, and optionally an identity baseline named
    by `identity_key`. `y` is the per-image class label the bootstrap
    stratifies on.

    The three levels stay structurally separate in the return value:
    `primary` carries no p-value and no family, `family1` and `family2`
    each carry their own Holm correction over their own prespecified
    membership, and `one_graph_wins` reads all three without recomputing
    any of them."""
    validate_conditions(mse_by_condition, y, identity_key=identity_key)
    primary = primary_bootstrap_test(mse_by_condition, y,
                                     n_resamples=n_resamples, seed=seed)
    family1 = family1_vs_pre_evolution(mse_by_condition, y, alpha=alpha,
                                       n_resamples=n_resamples, seed=seed)
    family2 = family2_pairwise(mse_by_condition, alpha=alpha,
                               run_sign_flip=run_sign_flip, n_flips=n_flips,
                               sign_flip_seed=sign_flip_seed, chunk_size=chunk_size)
    verdict = one_graph_wins(primary, family1, family2, alpha=alpha)
    out = {
        "primary": primary,
        "family1": family1,
        "family2": family2,
        "one_graph_wins": verdict,
        "n_images": int(np.asarray(y).size),
        "multiplicity_note": ("two prespecified families, each Holm-corrected within "
                              "itself; the primary test is outside both and "
                              "uncorrected"),
    }
    if identity_key is not None:
        out["identity_baselines"] = identity_baseline_gate(
            primary, mse_by_condition, y, identity_key,
            n_resamples=n_resamples, seed=seed)
    return out


# ---- Input validation ----

def validate_conditions(mse_by_condition, y, identity_key=None):
    """Every required condition present, same length as `y`, all finite.

    A missing or wrong-length condition is a caller error that would
    otherwise surface as a confusing broadcast or a silently truncated
    contrast."""
    required = [PRE_EVOLUTION, *EVOLVED_GRAPHS]
    if identity_key is not None:
        required.append(identity_key)
    missing = [k for k in required if k not in mse_by_condition]
    if missing:
        raise KeyError(f"missing conditions: {missing}")
    n = np.asarray(y).ravel().size
    if n < 2:
        raise ValueError("need at least 2 images")
    for key in required:
        arr = np.asarray(mse_by_condition[key], dtype=np.float64).ravel()
        if arr.size != n:
            raise ValueError(f"condition {key!r} has {arr.size} values but y has {n}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"condition {key!r} contains non-finite per-image MSE")
    return True


def _contrast(mse_by_condition, minuend, subtrahend):
    """`MSE(minuend) - MSE(subtrahend)`, per image. The single place a
    Stage 2B contrast's sign convention is fixed: a NEGATIVE mean always
    favors the minuend, because lower reconstruction error is better."""
    a = np.asarray(mse_by_condition[minuend], dtype=np.float64).ravel()
    b = np.asarray(mse_by_condition[subtrahend], dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"{minuend!r} has {a.size} values but {subtrahend!r} has {b.size}")
    return a - b
