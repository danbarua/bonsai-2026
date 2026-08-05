"""Stage the four KMNIST IDX files into the Stage 2B bucket, once.

Run locally by `make stage2b-stage-inputs`. This is the only Stage 2B
upload that goes local -> GCS: `datasets/` is gitignored, so the ladder
driver's clone of the repo carries the pipeline but not its inputs, and the
files are far too large to push through a Colab session upload.

## Why all four, when the rung uses 1,000 training images

`bonsai.data.mnist_loader.load_mnist` opens train AND t10k unconditionally,
and topology construction goes through it. The t10k bytes are never bound to
a variable by the driver. Reading them so a graph can be rebuilt from class-0
TRAINING images is not a Stage 2B test-side result, and none of
`stage2b_gcs`'s test-split machinery is involved: every object written here
lives under the train root.

## Naming

The `split="train"` token in these object paths is the PIPELINE split --
these are inputs to a training-side rung -- not a claim about byte
provenance. The `kind` token carries provenance literally, matching the IDX
filename, which is why the t10k pair is `kmnist_t10k_*` and deliberately not
`kmnist_test_*`: the latter would read as a Stage 2B test-side artifact.

The filename -> kind mapping is imported from `run_ladder_stage1`, not
restated here. One source of truth means a renamed object cannot be staged
under one name and fetched under another.
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)

import stage2b_gcs as gcs                                          # noqa: E402
from run_ladder_stage1 import (KMNIST_EXT, KMNIST_FILES, KMNIST_SUBDIR,  # noqa: E402
                               LADDER_STAGE, SPLIT)

DEFAULT_KMNIST_DIR = os.path.join(_REPO_ROOT, KMNIST_SUBDIR)


def stage(kmnist_dir=DEFAULT_KMNIST_DIR, bucket_name=None, credentials=None,
          force=False):
    """Upload each IDX file, skipping any already present.

    Verification is the module default on every transfer: an object that
    does not match its local bytes is deleted rather than left in place
    claiming a step is done."""
    bucket = gcs.get_bucket(name=bucket_name, credentials=credentials)
    print(f"bucket: {bucket.name}")
    print(f"source: {kmnist_dir}")
    print(f"checksum backend: {gcs.checksum_backend()}")

    results = {}
    for filename, kind in sorted(KMNIST_FILES.items()):
        local = os.path.join(kmnist_dir, filename)
        if not os.path.isfile(local):
            raise FileNotFoundError(
                f"{local} is missing. All four IDX files are needed -- load_mnist "
                f"opens the t10k pair unconditionally. See this module's docstring.")
        name = gcs.object_path(stage=LADDER_STAGE, condition=None, kind=kind,
                               ext=KMNIST_EXT, split=SPLIT)

        def produce(path, local=local):
            # `ensure_artifact` wants a step that WRITES its local path.
            # Here the bytes already exist at exactly that path, so the step
            # is a no-op -- but it must still be expressed as one, or the
            # missing-artifact guard cannot tell "nothing to do" from "the
            # step silently produced nothing".
            if os.path.abspath(path) != os.path.abspath(local):
                raise RuntimeError(f"expected to stage {local}, asked for {path}")

        result = gcs.ensure_artifact(name, local, produce=produce, bucket=bucket,
                                     force=force)
        results[filename] = result
        print(f"  {filename:<28} -> {name}")
        print(f"     {result.summary()}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kmnist-dir", default=DEFAULT_KMNIST_DIR)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--credentials", default=None)
    parser.add_argument("--force", action="store_true",
                        help="re-upload even if the object already exists")
    args = parser.parse_args(argv)
    stage(kmnist_dir=args.kmnist_dir, bucket_name=args.bucket,
          credentials=args.credentials, force=args.force)
    print("staged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
