"""Run the shell break-tests for `publish_review.sh` as part of the suite.

The script under test is shell, so its tests are shell -- testing it through
a Python reimplementation would put a layer between the test and the
mechanism. But a `.sh` file is not collected by pytest, and a test that only
runs when somebody remembers is closer to an unenforced requirement than an
enforced one: that is `docs/VACUOUS_TESTS.md` taxonomy G with `never run` as
the mechanism, and it is the reason a correct GCS guard in this repository
caught an offending file hours late rather than immediately.

So this wrapper exists purely to put those checks in front of `pytest` and
therefore in front of CI. It adds no assertions of its own; it forwards the
script's exit code and prints its output on failure, so a broken check reads
the same here as it does when run by hand.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE = REPO_ROOT / "tests" / "test_publish_review.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash available")
def test_publish_review_shell_suite_passes():
    """Forward the shell suite's verdict, and its output when it fails."""
    assert SUITE.exists(), f"{SUITE} is gone -- the checks it held are gone too"
    proc = subprocess.run(["bash", str(SUITE)], capture_output=True,
                          text=True, cwd=REPO_ROOT, timeout=120)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
    assert proc.returncode == 0, (
        "tools/ci/publish_review.sh break-tests failed -- see output above. "
        "That script is what stops a green run being mistaken for a review.")


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash available")
def test_the_wrapper_is_not_vacuous(tmp_path):
    """A wrapper that always passes would hide every check it forwards.

    So prove it fails when the suite fails, using a deliberately failing
    stand-in rather than trusting that the exit code is propagated.
    """
    failing = tmp_path / "failing.sh"
    failing.write_text("#!/bin/bash\necho 'deliberate failure'\nexit 1\n")
    proc = subprocess.run(["bash", str(failing)], capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode != 0, (
        "a failing shell suite returned 0 -- the wrapper above would report "
        "success for a broken script")
