"""The deployable gauge fork must not drift from the original.

`stage2a_core_colab` exists because `stage2a_core` cannot be imported on a
remote session: it inserts sibling experiment directories onto `sys.path`
relative to its own location, and imports three repo-local modules at module
scope, none of which the gauges need. The fork drops all of that.

What it buys in deployability it pays for in duplication, and duplication is
the specific hazard principle 16 names -- a reimplemented helper is a
different risk from a wrong one, and this project has been bitten by it. So
the copies are pinned to the originals here: same inputs, EXACTLY equal
outputs, in both directions, plus a check that the fork has not quietly
grown a repo-local import.

This runs locally, where both modules import fine. That is the point: the
environment that cannot check the equivalence is the one that needs it
guaranteed.
"""
import ast
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE2A = os.path.join(_REPO, "experiments", "stage2a_dynamics_classification")
sys.path.insert(0, _STAGE2A)

import stage2a_core as original          # noqa: E402
import stage2a_core_colab as deployable  # noqa: E402

# The functions the fork claims to carry. Named explicitly so that adding one
# to the fork without adding it here is caught by the coverage test below.
FORKED = ("order_parameter", "reference_node_features", "circular_mean_features")


def _phases(rng, n):
    return rng.uniform(0, 2 * np.pi, size=n)


@pytest.mark.parametrize("n", [4, 17, 60, 505])
def test_reference_node_features_is_bit_identical(n):
    rng = np.random.default_rng(n)
    for _ in range(20):
        theta = _phases(rng, n)
        ref_idx = int(rng.integers(0, n))
        a = original.reference_node_features(theta, ref_idx)
        b = deployable.reference_node_features(theta, ref_idx)
        assert np.array_equal(a, b), (
            f"n={n} ref_idx={ref_idx}: max|diff|={np.max(np.abs(a - b)):.3e}")


@pytest.mark.parametrize("n", [4, 17, 60, 505])
def test_circular_mean_features_is_bit_identical(n):
    rng = np.random.default_rng(n + 1000)
    for _ in range(20):
        theta = _phases(rng, n)
        a = original.circular_mean_features(theta)
        b = deployable.circular_mean_features(theta)
        assert np.array_equal(a, b), (
            f"n={n}: max|diff|={np.max(np.abs(a - b)):.3e}")


def test_order_parameter_is_bit_identical():
    rng = np.random.default_rng(7)
    for n in (4, 17, 60, 505):
        for _ in range(20):
            theta = _phases(rng, n)
            assert (original.order_parameter(theta)
                    == deployable.order_parameter(theta))


def test_edge_cases_agree_too():
    """Identical phases (R=1, circular mean degenerate-ish) and the ref_idx
    endpoints -- where a subtly different implementation is most likely to
    diverge."""
    for theta in (np.zeros(9), np.full(9, 1.234), np.linspace(0, 2 * np.pi, 9)):
        assert np.array_equal(original.circular_mean_features(theta),
                              deployable.circular_mean_features(theta))
        for ref_idx in (0, len(theta) - 1):
            assert np.array_equal(
                original.reference_node_features(theta, ref_idx),
                deployable.reference_node_features(theta, ref_idx))


def test_the_fork_carries_every_function_it_claims():
    for name in FORKED:
        assert hasattr(deployable, name), f"fork lost {name}"
        assert hasattr(original, name), f"original lost {name}"


def test_the_fork_exports_nothing_the_pin_does_not_cover():
    """A function added to the fork but not to FORKED would be unpinned --
    duplicated code with no equivalence check, which is the whole hazard."""
    src = open(os.path.join(_STAGE2A, "stage2a_core_colab.py")).read()
    defined = {n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    assert defined == set(FORKED), (
        f"fork defines {sorted(defined)} but only {sorted(FORKED)} are pinned "
        f"against the original -- add the new function to FORKED with a "
        f"bit-identity test, or it is unchecked duplication")


def test_the_fork_has_no_repo_local_imports():
    """The reason the fork exists. If someone reintroduces a sibling import
    or a sys.path insert, it stops being deployable and the next remote run
    dies after provisioning a GPU -- which is exactly how this file came to
    be written."""
    src = open(os.path.join(_STAGE2A, "stage2a_core_colab.py")).read()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    allowed = {"numpy"}
    assert imported <= allowed, (
        f"stage2a_core_colab imports {sorted(imported - allowed)}, which will "
        f"not exist on a remote session; keep it numpy-only")
    # Checked structurally, not by grepping the source: this file's own
    # docstring explains what it forked away from, and naming `sys.path`
    # there is not the same as touching it. A text match failed on the prose.
    touches_path = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "path"
        and isinstance(node.value, ast.Name) and node.value.id == "sys"]
    assert not touches_path, (
        "stage2a_core_colab manipulates sys.path -- that repo-layout "
        "assumption is precisely what it forked away from")
