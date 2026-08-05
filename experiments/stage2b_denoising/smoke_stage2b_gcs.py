"""Manual smoke check for Stage 2B's GCS transport, against the real
bucket. Run it by hand; it is not a test and pytest does not collect it.

    uv run python experiments/stage2b_denoising/smoke_stage2b_gcs.py

It needs `google-cloud-storage` installed (the `gpu` dependency group),
network, and the service-account key at `credentials_path()` -- none of
which the local development environment or `tests/test_stage2b_gcs.py`
has. That is the division: the tests cover every path, prefix, credential
and guard decision, offline; this script is the only thing that can
confirm a real write, read, existence check, and delete actually work.

What it does, on small probe objects under a `smoke_probe` condition
segment that no real artifact uses:

1. Resolves the credential path and reports whether it exists. The key's
   contents are never read or printed.
2. Round-trips a probe object on the training side: upload, exists,
   download, byte-compare, list.
3. Round-trips a probe object on the test side, which requires
   `allow_test_split=True` -- the same opt-in stage 4 uses.
4. Confirms the two delete refusals actually refuse (a training-side
   prefix without `force_non_test_prefix`, and a prefix outside
   `stage2b/` even when forced).
5. Deletes both probes and confirms they are gone. `--keep` skips this.

Exits non-zero on the first failure.
"""
import argparse
import filecmp
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stage2b_gcs as gcs  # noqa: E402

SMOKE_CONDITION = "smoke_probe"
PROBE_TEXT = b"stage2b gcs smoke probe\n"


def _say(message):
    print(f"  {message}", flush=True)


def _check(condition, message):
    if not condition:
        print(f"FAIL: {message}", flush=True)
        raise SystemExit(1)
    _say(f"ok: {message}")


def _probe_path(*, stage, split, allow_test_split=False):
    return gcs.object_path(stage=stage, condition=SMOKE_CONDITION, kind="probe",
                           ext="txt", split=split, allow_test_split=allow_test_split)


def _round_trip(bucket, workdir, *, stage, split, allow_test_split=False):
    """Upload, exists, download, byte-compare, list -- returns the probe's
    object path and the prefix holding it, both built by the module's own
    functions rather than assembled or parsed here."""
    name = _probe_path(stage=stage, split=split, allow_test_split=allow_test_split)
    prefix = gcs.condition_prefix(stage=stage, condition=SMOKE_CONDITION, split=split,
                                  allow_test_split=allow_test_split)

    local = os.path.join(workdir, f"probe_out_{split}.txt")
    with open(local, "wb") as handle:
        handle.write(PROBE_TEXT)

    gcs.upload_file(local, name, bucket=bucket, allow_test_split=allow_test_split)
    _check(gcs.object_exists(name, bucket=bucket, allow_test_split=allow_test_split),
           f"uploaded and exists: {name}")

    back = os.path.join(workdir, f"probe_back_{split}.txt")
    gcs.download_file(name, back, bucket=bucket, allow_test_split=allow_test_split)
    _check(filecmp.cmp(local, back, shallow=False), "downloaded bytes match what was sent")

    _check(name in gcs.list_objects(prefix, bucket=bucket,
                                    allow_test_split=allow_test_split),
           f"listed under {prefix}")
    return name, prefix


def _check_refusals(bucket):
    try:
        gcs.delete_prefix(gcs.TRAIN_ROOT, bucket=bucket)
    except PermissionError:
        _say("ok: deleting the training root without force_non_test_prefix refused")
    else:
        print("FAIL: deleting the training root was NOT refused", flush=True)
        raise SystemExit(1)

    try:
        gcs.delete_prefix("stage2a", bucket=bucket, force_non_test_prefix=True,
                          allow_test_split=True)
    except PermissionError:
        _say("ok: deleting outside stage2b/ refused even when forced")
    else:
        print("FAIL: deleting outside stage2b/ was NOT refused", flush=True)
        raise SystemExit(1)


def _chunked_resume(bucket, workdir):
    """The chunked upload, against the real bucket, resumed mid-transfer.

    Everything else here exercises paths a fake bucket already covers.
    This one does not: `upload_file_chunked` composes numbered part
    objects into the final object, and whether real GCS `compose`
    accepts this part naming -- and whether a composite object reads back
    byte-identical -- cannot be answered by a stand-in. The interruption
    is the point: an uninterrupted chunked upload would not exercise the
    checkpoint at all.
    """
    payload = bytes((i * 7 + 11) % 256 for i in range(300_000))
    local = os.path.join(workdir, "chunked_probe.bin")
    with open(local, "wb") as handle:
        handle.write(payload)

    name = gcs.object_path(stage=1, split="train", condition=SMOKE_CONDITION,
                           kind="chunked_probe", ext="bin")
    chunk = 64 * 1024

    class _Interrupt(RuntimeError):
        pass

    # Die after the second confirmed chunk, exactly as a killed process
    # would, leaving the checkpoint on disk.
    real_confirm = gcs._confirm_part
    seen = []

    def _dying_confirm(bkt, part, expected_size):
        real_confirm(bkt, part, expected_size)
        seen.append(part)
        if len(seen) == 2:
            raise _Interrupt("simulated process death after 2 confirmed chunks")

    gcs._confirm_part = _dying_confirm
    try:
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=chunk)
    except _Interrupt:
        pass
    finally:
        gcs._confirm_part = real_confirm

    checkpoint = gcs.checkpoint_path(local)
    _check(os.path.exists(checkpoint), "checkpoint survived the simulated death")

    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=chunk)
    _check(gcs.object_exists(name, bucket=bucket), f"resumed upload completed: {name}")

    readback = os.path.join(workdir, "chunked_readback.bin")
    gcs.download_file(name, readback, bucket=bucket)
    with open(readback, "rb") as handle:
        got = handle.read()
    _check(got == payload,
           f"composed object is byte-identical to the local file ({len(payload)} bytes)")
    _check(not os.path.exists(checkpoint), "checkpoint cleaned up after success")
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=gcs.GCS_BUCKET)
    parser.add_argument("--credentials", default=None,
                        help="service-account key path; defaults to credentials_path()")
    parser.add_argument("--keep", action="store_true",
                        help="leave the probe objects in place instead of deleting them")
    args = parser.parse_args()

    print(f"bucket:  {args.bucket}")
    print(f"project: {gcs.GCS_PROJECT}")
    key = args.credentials or gcs.credentials_path()
    print(f"key:     {key} (exists: {os.path.exists(key)}; contents never read)")

    bucket = gcs.get_bucket(bucket_name=args.bucket, credentials=args.credentials)

    with tempfile.TemporaryDirectory() as workdir:
        print("\ntraining-side round trip")
        train_name, train_prefix = _round_trip(bucket, workdir, stage=1, split="train")

        print("\ntest-side round trip (requires allow_test_split=True)")
        test_name, test_prefix = _round_trip(bucket, workdir, stage=gcs.TEST_SPLIT_STAGE,
                                             split="test", allow_test_split=True)

        print("\nchunked resumable upload (real compose, resumed mid-transfer)")
        chunked_name = _chunked_resume(bucket, workdir)

        print("\ndelete refusals")
        _check_refusals(bucket)

        if args.keep:
            print(f"\n--keep: leaving {train_name}, {test_name} and {chunked_name} in place")
            return

        print("\ncleanup")
        deleted = gcs.delete_prefix(test_prefix, bucket=bucket, allow_test_split=True)
        _check(test_name in deleted, f"deleted the test-side probe ({len(deleted)} object(s))")
        _check(not gcs.object_exists(test_name, bucket=bucket, allow_test_split=True),
               "test-side probe is gone")

        deleted = gcs.delete_prefix(train_prefix, bucket=bucket, force_non_test_prefix=True)
        _check(chunked_name in deleted, "deleted the chunked probe")
        _check(train_name in deleted,
               f"deleted the training-side probe ({len(deleted)} object(s))")
        _check(not gcs.object_exists(train_name, bucket=bucket),
               "training-side probe is gone")

    print("\nsmoke check passed")


if __name__ == "__main__":
    main()
