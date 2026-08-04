"""
The exact remote GPU-session driver that produced the training-set
evolved states behind Stage 2A's locked confirmatory result -- both the
feasibility stage 3 model-selection numbers and the final classifiers
`run_confirmatory_evaluation.py` refits and evaluates against the test
set trace back to `stage3_gpu_results.pkl`, which only this script
produces.

Committed alongside `stage4_gpu_evolve.py` for the same reason (see that
file's docstring): this project's usual convention exempts ephemeral
GPU-session driver scripts from being committed, but this one is not
exempt -- it produced data feeding this document's headline numbers.

Not runnable locally as-is -- executes ON the remote GPU session
(uploaded via `mighty-colab upload`, run via `mighty-colab exec`); see
`README.md`'s "GPU evolution" section for
the exact upload/exec sequence, including the 12-chunk `theta0` upload
workaround this script's own inline comment explains (the single
250MB-pickle upload hit the transfer endpoint's size limit). Reuses
`evolve_on_graph_jax.py` (uploaded alongside, unmodified) -- not a
reimplementation.
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

with open('/content/stage3_topologies.pkl', 'rb') as f:
    topologies = pickle.load(f)

# theta0 was uploaded as 12 separate .npy chunks (the single 250MB pickle
# hit the upload endpoint's size limit) -- reassembled here in chunk order.
N_UPLOAD_CHUNKS = 12
chunk_arrays = [np.load(f'/content/theta0_chunk_{i:02d}.npy') for i in range(N_UPLOAD_CHUNKS)]
theta0_batch_full = np.concatenate(chunk_arrays, axis=0)
n_images = theta0_batch_full.shape[0]
print(f"n_images={n_images}, theta0 dim={theta0_batch_full.shape[1]}, "
      f"topologies={list(topologies.keys())}")

# Chunked to stay within GPU memory: vmap materializes an (chunk, n, n)
# diff tensor per RHS eval, which does NOT fit at n_images=60000 all at
# once (would be ~120GB at float64). CHUNK_SIZE=1000 keeps the largest
# transient tensor around ~2GB, verified safe on the 100-image full-batch
# run (204MB) scaled up conservatively rather than assumed.
CHUNK_SIZE = 1000
n_chunks = (n_images + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"CHUNK_SIZE={CHUNK_SIZE}, n_chunks={n_chunks}")

# warm-up (compile), excluded from timing -- use a full-size chunk so the
# compiled shape matches every real chunk (all but possibly the last).
W0 = jnp.asarray(topologies['T'])
warmup_chunk = jnp.asarray(theta0_batch_full[:CHUNK_SIZE])
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

with open('/content/stage3_gpu_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'total_elapsed': total_elapsed, 'n_images': n_images,
                 'chunk_size': CHUNK_SIZE}, f)
print("Saved to /content/stage3_gpu_results.pkl")
