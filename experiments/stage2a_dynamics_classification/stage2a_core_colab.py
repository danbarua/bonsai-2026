"""The gauge functions, deployable to a remote session.

`stage2a_core` was written against the repo layout: it inserts sibling
experiment directories onto `sys.path` relative to its own location and
imports `build_and_verify_T`, `get_degree_stratified_nodes` and
`_local_converged_phases` at module scope. On a Colab VM those paths resolve
to nothing, so importing it fails before any function runs -- which is
exactly how a provisioned A100 died after a 242MB upload, on
`ModuleNotFoundError: No module named 'build_stage1d_constructions'`.

None of those imports are needed by the gauges. They serve the topology
construction and encoding paths, which a remote job that receives already
built topologies and already encoded phases never touches. So this is the
deployable subset: pure numpy, no repo layout, no sibling imports.

**This file duplicates code, which this project treats as a hazard rather
than a convenience** (principle 16 -- a reimplemented helper is a distinct
risk from a wrong one). The duplication is made safe by
`tests/test_stage2a_core_colab.py`, which asserts these functions agree
EXACTLY with `stage2a_core`'s originals on random inputs, in both directions,
so a change to either that the other does not follow fails locally. Edit the
original and this file together, or the test will say so.
"""
import numpy as np


def order_parameter(theta):
    """R(theta) = |mean(exp(i*theta))|. Identical to
    stage2a_core.order_parameter."""
    return float(np.abs(np.mean(np.exp(1j * theta))))


def reference_node_features(theta, ref_idx):
    """Locked primary gauge. Identical to
    stage2a_core.reference_node_features, including the assertions on the
    dropped pair -- they are cheap and they are the property
    tests/test_stage2a_core.py checks, so removing them here would make this
    copy weaker than the original rather than merely smaller."""
    shifted = theta - theta[ref_idx]
    cos_part = np.cos(shifted)
    sin_part = np.sin(shifted)
    assert abs(cos_part[ref_idx] - 1.0) < 1e-12
    assert abs(sin_part[ref_idx] - 0.0) < 1e-12
    cos_part = np.delete(cos_part, ref_idx)
    sin_part = np.delete(sin_part, ref_idx)
    return np.concatenate([cos_part, sin_part])


def circular_mean_features(theta):
    """Secondary robustness gauge. Identical to
    stage2a_core.circular_mean_features."""
    mu = np.angle(np.mean(np.exp(1j * theta)))
    shifted = theta - mu
    return np.concatenate([np.cos(shifted), np.sin(shifted)])
