"""Stage 2B's condition vocabulary, and the one mapping between its two
spellings.

Stage 2B names each condition twice. Statistics key their per-image MSE
vectors by graph name -- `"T"`, `"lattice"`, `"rewired"`,
`"curr_random"`, plus `"pre_evolution"` -- because that is how DESIGN.md
words the comparison families ("lattice/rewired/curr_random vs.
`pre_evolution`"). Object paths name the same conditions as a path
segment, and DESIGN.md's primary contrast is written
`d_i = MSE_i(evolved_T) - MSE_i(pre_evolution)`, so the segment for the
learned graph is `evolved_T`, not `T`.

Both spellings are in the design. Neither is wrong. What was missing is
anywhere that says they are the same condition: `stage2b_stats` owns one
list, `stage2b_gcs` validates path segments by shape rather than against
a vocabulary (deliberately -- a transport layer has no business owning
the design's condition names), and the `"T"` <-> `"evolved_T"`
correspondence therefore lived only in whichever driver happened to be
writing paths that day. This module is that correspondence, in one
place, imported by both sides of it. It depends on nothing: no numpy, no
scipy, no other Stage 2B module, so anything can import it.

## The decision this implements

DESIGN.md names two path segments literally, `evolved_T` and
`pre_evolution`. It does not name segments for the three control graphs;
Family 1 refers to them bare, as `lattice`, `rewired`, `curr_random`,
which is their statistics key, not a demonstrated path segment.

Decided: **the segment for an evolved graph is `evolved_{graph}`**, the
rule DESIGN.md's one worked example follows, applied uniformly --
`evolved_T`, `evolved_lattice`, `evolved_rewired`, `evolved_curr_random`,
alongside `pre_evolution`. The alternative, keeping the design's two
literal segments and spelling the other three bare, gives a path
vocabulary where the prefix marks evolution for one graph and nothing
for the rest, so `stage2b/train/stage2/lattice/` would not say on its
face whether it holds evolved or pre-evolution features.

This is an extension, not a transcription. It is recorded here, at the
call site of the choice, so a reader sees a decision rather than an
inherited fact -- and it changes no comparison, no statistic, and
nothing DESIGN.md locks.

## What this module deliberately does not do

It does not make `stage2b_gcs` validate against this vocabulary.
`stage2b_gcs` checks that a condition token cannot introduce a path
separator, a parent-directory hop, or a leading dot, and stops there;
that is correct, and pinning a condition list inside the transport layer
would be the design fact it refuses to own. This module is what a driver
imports to learn the right token; the transport layer's own check stays
a shape check.
"""
from types import MappingProxyType

# ---- The statistics vocabulary (DESIGN.md's comparison families) ----
#
# `EVOLVED_GRAPHS` is ordered, and that order defines Family 2's
# canonical pair keys -- do not reorder it without accepting that every
# pair key changes. `stage2b_stats` imports these rather than restating
# them.
PRE_EVOLUTION = "pre_evolution"
PRIMARY_GRAPH = "T"
EVOLVED_GRAPHS = ("T", "lattice", "rewired", "curr_random")
CONTROL_GRAPHS = ("lattice", "rewired", "curr_random")

# Every condition a per-image MSE vector can be keyed by, in the order
# results are reported.
ALL_CONDITIONS = (PRE_EVOLUTION, *EVOLVED_GRAPHS)

# ---- The object-path vocabulary ----
EVOLVED_PATH_PREFIX = "evolved_"

CONDITION_PATH_SEGMENT = MappingProxyType({
    PRE_EVOLUTION: PRE_EVOLUTION,
    **{graph: f"{EVOLVED_PATH_PREFIX}{graph}" for graph in EVOLVED_GRAPHS},
})

PATH_SEGMENT_CONDITION = MappingProxyType(
    {segment: condition for condition, segment in CONDITION_PATH_SEGMENT.items()})

PRIMARY_PATH_SEGMENT = CONDITION_PATH_SEGMENT[PRIMARY_GRAPH]   # "evolved_T"


def path_segment(condition):
    """The object-path segment for a statistics condition key.

    `path_segment("T")` is `"evolved_T"`. Raises rather than passing an
    unknown key through: an unrecognized condition silently becoming its
    own path segment is how a run writes half its artifacts under
    `.../T/` and the other half under `.../evolved_T/`, with both
    uploads succeeding."""
    try:
        return CONDITION_PATH_SEGMENT[condition]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown Stage 2B condition {condition!r}; expected one of "
            f"{tuple(CONDITION_PATH_SEGMENT)!r}") from None


def condition_for_path_segment(segment):
    """The statistics condition key for an object-path segment -- the
    inverse of `path_segment`. `condition_for_path_segment("evolved_T")`
    is `"T"`."""
    try:
        return PATH_SEGMENT_CONDITION[segment]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown Stage 2B path segment {segment!r}; expected one of "
            f"{tuple(PATH_SEGMENT_CONDITION)!r}") from None
