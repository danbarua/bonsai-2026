"""
The exact remote GPU-session driver that produced the official-test-set
evolved states behind Stage 2A's locked confirmatory result
(FINDINGS.md's "Stage 2A: The Locked Confirmatory Result").

Committed here per the reproducibility gaps flagged in external review
(FINDINGS.md's "Reproducibility gaps" section) -- this project's usual
convention is that ephemeral GPU-session driver scripts stay
uncommitted (they're throwaway, regenerated per session), but this one
specific script is not exempt from that convention: it produced this
document's headline numbers, so it belongs in the reproducible record
even though it was originally written and run directly on a Colab
kernel via `mighty-colab exec`, not invoked locally.

Not runnable locally as-is -- this executes ON the remote GPU session
(uploaded via `mighty-colab upload`, run via `mighty-colab exec`), with
its inputs already staged at `/content/...` by the driving local script
(see `README.md`'s "Reproducing the confirmatory GPU evolution" section
for the exact upload/exec sequence). Reuses `evolve_on_graph_jax.py`
(uploaded alongside, unmodified) -- not a reimplementation.

Mirrors `stage3_gpu_evolve.py`'s chunked approach exactly (same
CHUNK_SIZE=1000 rationale: vmap materializes a (batch, n, n) diff tensor
per RHS evaluation, which does not fit in GPU memory for a single
60,000- or 10,000-image batch) -- the only difference from the
training-side script is the smaller (10,000-image) input.
"""
import sys
sys.path.insert(0, '/content')
import pickle
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from evolve_on_graph_jax import batched_evolve_on_graph_jax

print("JAX backend:", jax.default_backend(), "| x64:", jax.config.jax_enable_x64)

with open('/content/stage4_gpu_upload_topologies.pkl', 'rb') as f:
    pkg = pickle.load(f)
topologies = pkg['topologies']

theta0_batch_full = np.load('/content/stage4_theta0_test.npy')
n_images = theta0_batch_full.shape[0]
print(f"n_images={n_images}, theta0 dim={theta0_batch_full.shape[1]}, "
      f"topologies={list(topologies.keys())}")

CHUNK_SIZE = 1000
n_chunks = (n_images + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"CHUNK_SIZE={CHUNK_SIZE}, n_chunks={n_chunks}")

W0 = jnp.asarray(topologies['T'])
warmup_chunk = jnp.asarray(theta0_batch_full[:min(CHUNK_SIZE, n_images)])
_wu_theta, _wu_success = batched_evolve_on_graph_jax(warmup_chunk, W0)
jax.block_until_ready(_wu_theta)
print("warm-up compile done")

results = {}
total_elapsed = 0.0
for name, W in topologies.items():
    W_jax = jnp.asarray(W)
    theta_T_chunks = []
    success_chunks = []
    t0 = time.perf_counter()
    for c in range(n_chunks):
        lo = c * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, n_images)
        chunk = jnp.asarray(theta0_batch_full[lo:hi])
        theta_T_c, success_c = batched_evolve_on_graph_jax(chunk, W_jax)
        jax.block_until_ready(theta_T_c)
        theta_T_chunks.append(np.asarray(theta_T_c))
        success_chunks.append(np.asarray(success_c))
    elapsed = time.perf_counter() - t0
    total_elapsed += elapsed

    theta_T_full = np.concatenate(theta_T_chunks, axis=0)
    success_full = np.concatenate(success_chunks, axis=0)
    n_failed = int(np.sum(~success_full))
    print(f"[{name}] {elapsed:.3f}s for {n_images} images ({elapsed/n_images*1000:.3f} ms/image), "
          f"n_failed={n_failed}")
    results[name] = {
        'theta_T': theta_T_full,
        'success': success_full,
        'elapsed': elapsed,
    }

print(f"\nTotal GPU evolution time, all {len(topologies)} topologies, {n_images} images: "
      f"{total_elapsed:.3f}s ({total_elapsed/n_images/len(topologies)*1000:.3f} ms/image/topology)")

with open('/content/stage4_gpu_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'total_elapsed': total_elapsed, 'n_images': n_images,
                 'chunk_size': CHUNK_SIZE}, f)
print("Saved to /content/stage4_gpu_results.pkl")
