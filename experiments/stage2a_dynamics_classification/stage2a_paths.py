"""
Centralizes Stage 2A's scratch-directory resolution. The confirmatory
scripts previously hard-coded a private, machine-specific scratch path
(flagged in external review, FINDINGS.md's "Reproducibility gaps") --
every script that reads or writes the large intermediate artifacts
(encoded features, GPU-evolved states) now resolves its scratch
directory through this module instead.

Override via the STAGE2A_SCRATCH_ROOT environment variable; defaults to
a `scratch/` directory next to this file (gitignored, regenerable by
re-running the encode/evolve pipeline -- never committed, per this
project's convention for large cached artifacts).
"""
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def scratch_root():
    """The root scratch directory for this stage's large intermediate
    artifacts. STAGE2A_SCRATCH_ROOT overrides; defaults to a local,
    gitignored `scratch/` directory next to this file."""
    return os.environ.get("STAGE2A_SCRATCH_ROOT", os.path.join(_THIS_DIR, "scratch"))


def train_scratch_dir():
    """Where the 60,000-image official-training-set encode/evolve
    artifacts live (stage3_encode_local.pkl, stage3_gpu_results.pkl,
    stage3_gpu_upload.pkl, stage3_topologies.pkl, theta0 upload chunks)."""
    return os.path.join(scratch_root(), "stage3_train")


def test_scratch_dir():
    """Where the 10,000-image official-test-set encode/evolve artifacts
    live (stage4_encode_local.pkl, stage4_gpu_results.pkl,
    stage4_gpu_upload_topologies.pkl, stage4_theta0_test.npy) --
    separate from train_scratch_dir() so the one-time-touch test-set
    artifacts are never accidentally mixed with training-side data."""
    return os.path.join(scratch_root(), "stage4_test")
