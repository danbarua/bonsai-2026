"""Stage the 250MB Stage-3 GPU input on GCS, and time the local->GCS leg.

Half of a transfer benchmark. The question is whether pushing this file
through `mighty-colab upload` -- which needed splitting into twelve 20MB
chunks to get under the transfer endpoint's size limit -- is slower than
staging it once on GCS and pulling it down inside Google's own network.

This half runs LOCALLY and uploads once. `bench_transfer_gcs_download.py`
runs on the VM and times the pull. The CLI leg is timed by the Makefile
recipe around its upload loop, so all three numbers come from the same
250,561,166 bytes.
"""
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage2b_denoising"))
sys.path.insert(0, _THIS_DIR)

import stage2b_gcs as gcs
from stage2a_paths import train_scratch_dir

# Fixed, not generated per run: a fresh name each time would leave a 250MB
# object behind on every invocation, and the point is to measure transfer,
# not to accumulate garbage. Re-running overwrites the same object.
OBJECT_NAME = "benchmark/transfer/stage3-gpu-upload-nimble-otter-2f7c.pkl"
SOURCE = os.path.join(train_scratch_dir(), "stage3_gpu_upload.pkl")


def main():
    if not os.path.exists(SOURCE):
        raise SystemExit(f"missing {SOURCE} -- run `make stage2a-prepare-train` first")
    n_bytes = os.path.getsize(SOURCE)
    print(f"source: {SOURCE}\nbytes:  {n_bytes:,}")

    t0 = time.perf_counter()
    digest = gcs.crc32c_of_file(SOURCE)
    t_crc = time.perf_counter() - t0
    print(f"crc32c: {digest} (computed locally in {t_crc:.1f}s, "
          f"backend={gcs.checksum_backend()})")

    bucket = gcs.get_bucket()
    blob = bucket.blob(OBJECT_NAME)
    print(f"uploading to gs://{bucket.name}/{OBJECT_NAME} ...", flush=True)

    t0 = time.perf_counter()
    blob.upload_from_filename(SOURCE)
    elapsed = time.perf_counter() - t0

    mb = n_bytes / 1e6
    print(f"\nLOCAL -> GCS: {elapsed:.1f}s for {mb:.1f} MB "
          f"= {mb / elapsed:.1f} MB/s")
    print(f"object: gs://{bucket.name}/{OBJECT_NAME}")
    print("BENCH_UPLOAD_OK")


if __name__ == "__main__":
    main()
