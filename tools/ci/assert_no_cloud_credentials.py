#!/usr/bin/env python3
"""Assert that this process cannot reach the science bucket.

CI runs credential-free on purpose. Several Stage 2B tests build a live
GCS client, and the two-tier convention skips them cleanly when the
credential or the library is missing -- so a credential-free build runs
Tier 1 and stays honest. The reverse is what this guards: an unattended
build that quietly gained write access to `bonsai-2026-stage2b-cache`
would be indistinguishable, from its logs, from one that never had it.

Two capabilities, both required for a write, both checked:

  * `google-cloud-storage`, which ships in the `gpu` dependency group and
    is therefore absent from a default `uv sync`. Without it there is no
    client to build at all.
  * a credential path in the environment.

An intention is not a control. This turns "CI must not be able to spend"
into something that fails on the day it stops being true.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# Every environment variable that would hand a process a usable credential.
# `BONSAI_GCS_CREDENTIALS` is this project's own (Makefile, stage2b_gcs);
# the other two are what any Google client library reads on its own.
CREDENTIAL_ENV_VARS = (
    "BONSAI_GCS_CREDENTIALS",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CLOUDSDK_AUTH_ACCESS_TOKEN",
)

CLIENT_LIBRARIES = ("google.cloud.storage",)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def problems(env: dict[str, str] | None = None,
             installed=None) -> list[str]:
    """Every reason this process could reach the bucket.

    `installed` is injectable so a test can drive both answers. The script
    itself always probes the real interpreter -- that is the whole point of
    it -- but a developer who has synced `--group gpu` for a GPU target
    should not get a red `make test` for it.
    """
    env = os.environ if env is None else env
    installed = _installed if installed is None else installed
    found = []
    for module in CLIENT_LIBRARIES:
        if installed(module):
            found.append(
                f"`{module}` is importable, so this build can construct a live GCS "
                f"client. It ships in the `gpu` dependency group, which CI must not "
                f"install -- check that nothing added `--group gpu` to the sync.")
    for var in CREDENTIAL_ENV_VARS:
        if env.get(var):
            found.append(
                f"`{var}` is set. CI runs credential-free by design; an unattended "
                f"identity with write access to the science bucket is not a trade "
                f"this project makes.")
    return found


def main() -> int:
    found = problems()
    for problem in found:
        print(f"[verify] FAIL: {problem}", file=sys.stderr)
    if found:
        return 1
    print("[verify] OK: no GCS client library, no credentials in the environment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
