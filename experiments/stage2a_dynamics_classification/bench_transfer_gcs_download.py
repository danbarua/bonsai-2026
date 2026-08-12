"""Time the GCS->VM leg of the transfer benchmark, on the remote session.

Runs ON the VM (uploaded and exec'd), so every path is an explicit /content
path and nothing from the repository is assumed to be on disk. Pulls the
250MB Stage-3 input that `bench_transfer_upload.py` staged, times it, and
verifies the bytes arrived intact rather than merely quickly -- a fast
transfer of a truncated object is not a faster transfer.

The comparison this feeds: the same file reached a VM as twelve 20MB
`mighty-colab upload` calls, because a single 250MB push exceeded the
transfer endpoint's size limit. Both legs are timed over identical bytes.
"""
import os
import pickle
import time

import numpy as np
from google.cloud import storage

BUCKET = os.environ["BONSAI_GCS_BUCKET"]
OBJECT_NAME = os.environ["BENCH_OBJECT"]
DEST = "/content/bench_stage3_gpu_upload.pkl"
SENTINEL = "BENCH_DOWNLOAD_OK"

print(f"bucket={BUCKET}\nobject={OBJECT_NAME}", flush=True)

client = storage.Client()
blob = client.bucket(BUCKET).blob(OBJECT_NAME)
blob.reload()
n_bytes = blob.size
print(f"remote size: {n_bytes:,} bytes  crc32c={blob.crc32c}", flush=True)

t0 = time.perf_counter()
blob.download_to_filename(DEST)
elapsed = time.perf_counter() - t0

on_disk = os.path.getsize(DEST)
mb = on_disk / 1e6
print(f"\nGCS -> VM: {elapsed:.1f}s for {mb:.1f} MB = {mb / elapsed:.1f} MB/s",
      flush=True)

if on_disk != n_bytes:
    raise SystemExit(f"size mismatch: remote {n_bytes} vs local {on_disk}")

# Arrived intact, not merely arrived. Unpickling is the real acceptance
# test -- it is what the actual pipeline would do next.
t0 = time.perf_counter()
with open(DEST, "rb") as fh:
    payload = pickle.load(fh)
t_load = time.perf_counter() - t0
theta0 = np.asarray(payload["theta0_batch"])
print(f"unpickled in {t_load:.1f}s: theta0_batch {theta0.shape} {theta0.dtype}, "
      f"topologies={sorted(payload['topologies'])}", flush=True)
assert theta0.shape == (60000, 505), theta0.shape
assert np.isfinite(theta0).all(), "non-finite phases in the downloaded batch"

print(f"transfer_seconds={elapsed:.3f} bytes={on_disk} mb_per_s={mb / elapsed:.2f}")
print(SENTINEL, flush=True)
