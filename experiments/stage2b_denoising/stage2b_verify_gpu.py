"""Runs DESIGN.md's ridge equivalence gate on a real GPU.

Executed ON a Colab GPU runtime by `make stage2b-verify-gpu`, not
locally. It is a plain script, uploaded and run with
`mighty-colab exec -f`.

## What this establishes, and what it does not

DESIGN.md locks the equivalence gate -- JAX SVD ridge versus sklearn
`Ridge(solver="svd")`, max absolute difference in CLIPPED validation
predictions <= 1e-8 AND identical alpha selection -- at the 1,000- and
5,000-image ladder stages. Every run of that gate so far has been on
CPU, because that is the only device this project's dev machine has.

Whether the same code clears the same gate on a GPU is a genuinely
separate question, and not a rhetorical one:

- JAX must have `jax_enable_x64` actually in effect on the device. A
  silent fall back to float32 is exactly the failure Stage 1D already
  caught once (see CLAUDE.md's principle 16 and
  `experiments/stage1d_topology_specificity_gpu/FINDINGS.md`), and
  float32 against Stage 2A's measured ~2e6 condition numbers would blow
  the 1e-8 gate by orders of magnitude.
- GPU LAPACK/SVD implementations are not bit-identical to CPU ones.
  Agreement to 1e-8 is a claim about that specific pairing, not
  something inherited from the CPU result.

So this checks the JAX-on-GPU path against a sklearn-on-CPU oracle
running in the same process, at the two scales the design names.

What it does NOT establish: anything about real Stage 2B features. The
design matrices here are synthetic, shaped like the real ones (505
active-support targets, 1008 circular-embedding columns) and
deliberately ill-conditioned so the comparison is not trivially easy --
but they are not encoded phase states, because no ladder rung has run
yet and no real features exist. This is a device-numerics check, not a
pipeline check.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/content")

import stage2b_ridge as ridge  # noqa: E402  -- enables x64 at import
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

N_FEATURES = 1008   # circular embedding, reference-node columns dropped
N_TARGETS = 505     # the active support
N_CLASSES = 10
LADDER_SCALES = (1000, 5000)   # DESIGN.md's two equivalence-gate stages

OK = "GPU_VERIFY_OK"
FAIL = "GPU_VERIFY_FAIL"


def report(line):
    print(f"[verify-gpu] {line}", flush=True)


def device_preflight():
    """Refuse to report a pass that a CPU fallback could have produced."""
    devices = jax.devices()
    report(f"jax {jax.__version__}, devices: {devices}")
    kinds = {d.platform for d in devices}
    if not (kinds & {"gpu", "cuda", "rocm"}):
        raise SystemExit(
            f"{FAIL}: no GPU device visible ({devices}). This script exists to test GPU "
            f"numerics; passing on CPU would answer a question nobody asked.")

    # x64 must be in effect ON THE DEVICE, not merely requested in config.
    # Allocating and reading back the dtype is the only check that can
    # distinguish the two.
    probe = jnp.zeros(1, dtype=jnp.float64)
    report(f"x64 config flag: {jax.config.jax_enable_x64}, "
           f"realised device dtype: {probe.dtype} on {probe.devices()}")
    if probe.dtype != jnp.float64:
        raise SystemExit(
            f"{FAIL}: jnp.float64 resolved to {probe.dtype} on the device. The ridge SVD "
            f"would silently run in float32, which DESIGN.md's dtype table rules out.")


def synthetic_problem(n, seed):
    """A design matrix shaped like Stage 2B's, and deliberately hard.

    Real evolved-phase features are ill-conditioned (Stage 2A measured
    ~2e6). A well-conditioned random matrix would let both paths agree
    for reasons that say nothing about the real case, so a decaying
    spectrum plus duplicated columns is imposed to make the SVD work.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, N_FEATURES))
    # Decaying spectrum -> large condition number.
    scale = np.logspace(0, -5, N_FEATURES)
    X = base * scale
    # A few exactly-collinear columns: rank deficiency is where the two
    # ridge formulations are most free to disagree.
    X[:, -8:] = X[:, :8]
    W_true = rng.standard_normal((N_FEATURES, N_TARGETS))
    Y = X @ W_true + 0.01 * rng.standard_normal((n, N_TARGETS))
    y_strat = rng.integers(0, N_CLASSES, size=n)
    return X, Y, y_strat


def main():
    device_preflight()
    failures = []

    for n in LADDER_SCALES:
        X, Y, y_strat = synthetic_problem(n, seed=n)
        cond = float(np.linalg.cond(X))
        report(f"n={n}: X{X.shape} Y{Y.shape}, cond(X)={cond:.3e}")

        t0 = time.time()
        result = ridge.ridge_equivalence_check(X, Y, y_strat)
        elapsed = time.time() - t0

        max_pred = float(result["max_abs_clipped_pred_diff"])
        same_alpha = bool(result["alpha_agrees"])
        report(f"n={n}: max abs clipped-prediction diff = {max_pred:.3e} "
               f"(gate <= {result['tol']:.0e})")
        report(f"n={n}: alpha jax={result['alpha_jax']} sklearn={result['alpha_sklearn']} "
               f"identical={same_alpha}")
        report(f"n={n}: max abs coefficient diff = {float(result['max_abs_coef_diff']):.3e} "
               f"(diagnostic only, not a gate)")
        report(f"n={n}: {elapsed:.1f}s")

        if not result["passed"]:
            failures.append(
                f"n={n}: pred_diff={max_pred:.3e}, same_alpha={same_alpha}")

    if failures:
        for f in failures:
            report(f"FAILED {f}")
        raise SystemExit(f"{FAIL}: the equivalence gate did not hold on GPU at "
                         f"{len(failures)} of {len(LADDER_SCALES)} scales.")

    report(f"{OK}: gate held at every scale {LADDER_SCALES} on GPU")


if __name__ == "__main__":
    main()
