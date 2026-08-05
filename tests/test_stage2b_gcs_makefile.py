"""The Makefile and `stage2b_gcs` must agree about the bucket.

Moving the bucket name into a Makefile variable makes it overridable
without editing Python, which is what it is for -- and creates a second
place the name is written down, which is what this file is for. The two
declarations are independent strings in different languages, so nothing
but a test stops them drifting apart, and the failure mode is quiet: a
target exports one bucket, a script that was invoked some other way
defaults to the other, and artifacts land in two places with no error
anywhere.

That is the same shape as the bug that prompted this rename. The name
lived in `stage2b_gcs.py`, `README.md` and a test assertion; the test
pinned it, so it looked covered, but the pin simply restated the module
and would have gone on passing had any *other* copy been wrong.

CLAUDE.md principle 20: this is the executable form of a fact that would
otherwise live in a Makefile comment nobody re-checks. Tier 1 throughout
-- parses two files, touches no network and provisions nothing.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = REPO_ROOT / "Makefile"
sys.path.insert(0, str(REPO_ROOT / "experiments" / "stage2b_denoising"))

import stage2b_gcs as gcs  # noqa: E402


def _make_var(name):
    """The value of a `NAME ?= value` (or `:=`, or `=`) assignment.

    Backslash continuations are joined first: `GCS_ENV` spans two lines,
    and a pattern anchored to a single line silently returns half of it --
    which read as "GCS_ENV does not mention the bucket" when it does.

    Returns None when the variable is not declared at all, so a test can
    say which of "wrong value" and "missing entirely" happened."""
    text = re.sub(r"\\\n\s*", " ", MAKEFILE.read_text())
    pattern = rf"^{re.escape(name)}\s*[?:]?=\s*(.*?)\s*$"
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches[-1] if matches else None


def test_the_makefile_declares_the_same_default_bucket_as_the_module():
    declared = _make_var("BONSAI_GCS_BUCKET")
    print(f"\n[bucket] Makefile BONSAI_GCS_BUCKET = {declared!r}")
    print(f"[bucket] stage2b_gcs.DEFAULT_GCS_BUCKET = {gcs.DEFAULT_GCS_BUCKET!r}")
    assert declared is not None, (
        "the Makefile no longer declares BONSAI_GCS_BUCKET; the GCS targets would fall "
        "back to the module default silently")
    assert declared == gcs.DEFAULT_GCS_BUCKET, (
        f"the Makefile exports {declared!r} but the module defaults to "
        f"{gcs.DEFAULT_GCS_BUCKET!r}. A script run through a target and the same script "
        f"run directly would use different buckets.")


def test_the_declared_bucket_is_a_name_the_resolver_accepts():
    """Guards against a rename that is valid Make and invalid GCS -- which
    would only fail at the point of touching the real bucket."""
    declared = _make_var("BONSAI_GCS_BUCKET")
    assert gcs.bucket_name(env={gcs.BUCKET_ENV_VAR: declared}) == declared


def test_every_gcs_touching_target_exports_the_bucket():
    """`GCS_ENV` carries both the bucket and the credentials path. Any
    target that reaches GCS must use it rather than relying on ambient
    environment -- otherwise `make` and a bare `uv run` disagree."""
    text = MAKEFILE.read_text()
    gcs_env = _make_var("GCS_ENV")
    assert gcs_env is not None and "BONSAI_GCS_BUCKET" in gcs_env, (
        f"GCS_ENV should carry the bucket; got {gcs_env!r}")

    recipes = {}
    current, body = None, []
    for line in text.splitlines():
        if line.startswith("\t"):
            if current:
                body.append(line)
            continue
        if current:
            recipes[current] = "\n".join(body)
            current, body = None, []
        m = re.match(r"^([A-Za-z0-9_-]+):(?!=)", line)
        if m:
            current, body = m.group(1), []
    if current:
        recipes[current] = "\n".join(body)

    # Scripts and tests that construct a real GCS client. A target naming
    # any of these is reaching the bucket and must say which one.
    gcs_scripts = ("smoke_stage2b_gcs.py", "test_stage2b_gcs_roundtrip.py")
    touching = {name: body for name, body in recipes.items()
                if any(script in body for script in gcs_scripts)}

    assert touching, (
        "parsed no GCS-touching targets -- this test's parser has gone stale and the "
        "check below is vacuous")

    offenders = []
    for name, body in sorted(touching.items()):
        ok = "$(GCS_ENV)" in body
        print(f"[bucket] {name}: {'exports' if ok else 'DOES NOT export'} $(GCS_ENV)")
        if not ok:
            offenders.append(name)

    assert not offenders, (
        f"these targets reach GCS without exporting the bucket and credentials: "
        f"{offenders}. They would use whatever the ambient environment holds.")
