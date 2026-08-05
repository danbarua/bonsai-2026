"""Checks the CNN's float32 forward pass against CPU, on a real GPU.

Executed ON a Colab GPU runtime by `make stage2b-verify-cnn-gpu`. Plain
script, uploaded and run with `mighty-colab exec -f`.

## Why this is a separate question from the ridge GPU check

`stage2b_verify_gpu.py` already cleared DESIGN.md's ridge equivalence
gate on a GPU. That result does not transfer here, for a specific
reason: ridge is float64 end to end, which makes it immune to the entire
class of reduced-precision effects this script is about. The CNN is
`CNN_DTYPE = float32` and runs `lax.conv_general_dilated` through
equinox.

On NVIDIA hardware, XLA may compute float32 convolutions and matmuls at
reduced internal precision (TF32-class, ~10 explicit mantissa bits
rather than 24) unless precision is pinned. That is normally a good
trade. It is not obviously one here, because Stage 2B's early stopping
is `min_delta=0.0` with a strict `<` improvement test: `stage2b_cnn.py`
already argues in its own dtype section that float noise registering as
"improvement" resets patience and moves which epoch gets checkpointed.
A device silently computing the forward pass at ~10 mantissa bits moves
the validation metric far more than the float64 accumulation fix was
protecting against, and it would change `best_epoch`, change which of
seeds (0,1,2) `select_best_seed` returns, and change the reported CNN
MSE. Nothing announces any of that.

So this measures three things and reports them rather than guessing:

1. What `jax.default_matmul_precision` / the conv path actually resolves
   to on this device.
2. Max elementwise difference between the same model's forward pass on
   CPU and on GPU, on identical inputs, at default precision.
3. The same difference with precision explicitly pinned to `HIGHEST`.

If (2) is far larger than float32 epsilon and (3) is not, the remedy is
to pin precision and make that a locked implementation constant
alongside `CNN_DTYPE`. If (2) is already at float32 epsilon, the default
is fine on this device and the finding is that no pin is needed -- which
is worth knowing too, and worth knowing per-device.

This deliberately checks the FORWARD pass, not a full training run. The
forward pass is where the precision decision is made; a training-run
comparison would confound it with optimizer nondeterminism and give a
number nobody could interpret.
"""
import sys

import numpy as np

sys.path.insert(0, "/content")

import stage2b_cnn as cnn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

OK = "CNN_GPU_VERIFY_OK"
FAIL = "CNN_GPU_VERIFY_FAIL"
F32_EPS = float(np.finfo(np.float32).eps)   # 1.19e-07


def report(line):
    print(f"[verify-cnn-gpu] {line}", flush=True)


def forward_on(device, model, x, precision="module"):
    """One forward pass on a chosen device.

    `precision="module"` calls `cnn.forward` -- the module's OWN forward
    pass, carrying whatever pin it actually applies. That is the case
    that matters: an earlier version of this script re-implemented the
    vmap here and so measured a property of the device while never
    testing whether the module's pin was wired in at all (CLAUDE.md
    principle 16, in this script rather than in the code under test).

    The other two values bypass `cnn.forward` deliberately, to
    characterise the device itself: `None` for XLA's default and
    `"highest"` for an explicit pin.
    """
    xd = jax.device_put(x, device)
    md = jax.device_put(model, device)
    if precision == "module":
        return np.asarray(cnn.forward(md, xd))
    if precision is None:
        return np.asarray(jax.vmap(md)(xd))
    with jax.default_matmul_precision(precision):
        return np.asarray(jax.vmap(md)(xd))


def main():
    devices = jax.devices()
    report(f"jax {jax.__version__}, devices: {devices}")
    gpus = [d for d in devices if d.platform in ("gpu", "cuda", "rocm")]
    cpus = jax.devices("cpu")
    if not gpus:
        raise SystemExit(f"{FAIL}: no GPU visible ({devices}); this script exists to "
                         f"compare against one.")
    gpu, cpu = gpus[0], cpus[0]
    report(f"comparing cpu={cpu} vs gpu={gpu}")
    report(f"CNN_DTYPE = {cnn.CNN_DTYPE.__name__ if hasattr(cnn.CNN_DTYPE, '__name__') else cnn.CNN_DTYPE}, "
           f"x64 enabled = {jax.config.jax_enable_x64}")

    model = cnn.make_model(cnn.seed_keys(0)[0])
    rng = np.random.default_rng(0)
    x = cnn.as_image_batch(rng.uniform(0, 1, (16, cnn.IMAGE_SIDE, cnn.IMAGE_SIDE)))

    ref = forward_on(cpu, model, x, precision="module")
    report(f"cpu forward (via cnn.forward): dtype={ref.dtype}, shape={ref.shape}, "
           f"range=[{ref.min():.6f}, {ref.max():.6f}]")
    report(f"module pin: CNN_MATMUL_PRECISION={cnn.CNN_MATMUL_PRECISION!r}")

    results = {}
    for label, precision in (("default", None), ("HIGHEST", "highest"),
                             ("cnn.forward", "module")):
        got = forward_on(gpu, model, x, precision)
        absdiff = float(np.max(np.abs(got - ref)))
        scale = float(np.max(np.abs(ref))) or 1.0
        rel = absdiff / scale
        results[label] = rel
        verdict = "at float32 eps" if rel <= 10 * F32_EPS else "ABOVE float32 eps"
        report(f"gpu forward, precision={label:<8}: max abs diff={absdiff:.3e}, "
               f"max rel diff={rel:.3e}  ({verdict})")

    report(f"float32 eps = {F32_EPS:.3e}; treating 10x eps as the pass line")

    # The verdict is about the MODULE's forward pass, not the device's
    # default. Whether the device would have been sloppy left to itself is
    # reported for the record; whether the code we actually run is sloppy
    # is the gate.
    module_ok = results["cnn.forward"] <= 10 * F32_EPS
    device_default_clean = results["default"] <= 10 * F32_EPS

    if not module_ok:
        raise SystemExit(
            f"{FAIL}: cnn.forward disagrees with CPU by {results['cnn.forward']:.3e} "
            f"relative. The module's CNN_MATMUL_PRECISION pin is not taking effect on "
            f"this device. Given min_delta=0.0 with strict '<' early stopping, this "
            f"moves best_epoch, seed selection and the reported CNN MSE.")

    if device_default_clean:
        report(f"{OK}: cnn.forward agrees with CPU to float32 eps. Note this device's "
               f"DEFAULT path was already clean ({results['default']:.3e}), so it does "
               f"not exercise the pin -- T4 has no TF32 hardware and behaves this way. "
               f"A100 is the device that exhibits the effect.")
        return

    report(f"{OK}: cnn.forward agrees with CPU to float32 eps ({results['cnn.forward']:.3e}) "
           f"while this device's DEFAULT path does not ({results['default']:.3e}). "
           f"The pin is doing real work here -- this run genuinely exercises it.")


if __name__ == "__main__":
    main()
