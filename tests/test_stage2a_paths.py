"""
Tests for experiments/stage2a_dynamics_classification/stage2a_paths.py.

Tier 1 only (self-contained, always run, no data dependency): the
STAGE2A_SCRATCH_ROOT environment-variable override, and that
train_scratch_dir()/test_scratch_dir() are distinct paths nested under
scratch_root() -- this module exists specifically to replace hard-coded
private scratch paths flagged in external review, so its override
mechanism is worth testing directly, not just trusting by inspection.
"""
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2A_DIR = _REPO_ROOT / "experiments" / "stage2a_dynamics_classification"
sys.path.insert(0, str(_STAGE2A_DIR))

import stage2a_paths  # noqa: E402


def test_default_scratch_root_is_local_scratch_dir(monkeypatch):
    monkeypatch.delenv("STAGE2A_SCRATCH_ROOT", raising=False)
    root = stage2a_paths.scratch_root()
    assert Path(root) == _STAGE2A_DIR / "scratch"


def test_scratch_root_env_var_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STAGE2A_SCRATCH_ROOT", str(tmp_path))
    assert stage2a_paths.scratch_root() == str(tmp_path)


def test_train_and_test_scratch_dirs_distinct_and_nested(monkeypatch, tmp_path):
    monkeypatch.setenv("STAGE2A_SCRATCH_ROOT", str(tmp_path))
    train_dir = Path(stage2a_paths.train_scratch_dir())
    test_dir = Path(stage2a_paths.test_scratch_dir())
    assert train_dir != test_dir
    assert train_dir.parent == tmp_path
    assert test_dir.parent == tmp_path


def test_scratch_dirs_reflect_env_var_change_dynamically(monkeypatch, tmp_path):
    """scratch_root() reads the env var at call time, not at import time --
    confirms the module doesn't cache a stale value from first import."""
    other = tmp_path / "other_root"
    monkeypatch.setenv("STAGE2A_SCRATCH_ROOT", str(other))
    assert stage2a_paths.scratch_root() == str(other)
    monkeypatch.delenv("STAGE2A_SCRATCH_ROOT", raising=False)
    assert Path(stage2a_paths.scratch_root()) == _STAGE2A_DIR / "scratch"
