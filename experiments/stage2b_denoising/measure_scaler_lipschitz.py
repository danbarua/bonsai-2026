"""Read the scaler's Lipschitz constant out of the frozen final ridge fit.

WHY THIS EXISTS. `run_abs_conv_eps_sensitivity.py`'s axis 4 asserts an
end-to-end bound `|Delta_g| <= 2B` on an encoder phase residual `B`,
justified as "Lip <= 2 on the prediction residual". That is a claim about
the WHOLE chain: ODE flow -> feature scaling -> ridge readout -> clipping ->
MSE -> Delta_g. This script measures ONE link of it, the one whose constant
this project already computes and stores and then never reads back.

The standardiser applied before the ridge divides each column by its
standard deviation, so its Lipschitz constant is `1 / min_col_std`. That
value is computed by `stage2b_ridge.scaler_centering_margin`, attached by
`fit_final`, and persisted into `ridge_final_g13_88edf9ac.npz`'s
`summary_json` by `run_ladder_stage3.py`. It was stored for a different
purpose -- guarding that the scaler is centred -- and answers this question
for free.

CLAUDE.md principle 24: the number this prints is about to anchor a decision
(whether the 2B bound stands), so it comes from committed code rather than
from a shell command in a transcript.

No GPU, no credentials, no fit. The object is public-read, so this is one
HTTPS GET and a dict lookup. It re-runs anywhere with a network connection.

    uv run python experiments/stage2b_denoising/measure_scaler_lipschitz.py
"""
import argparse
import io
import json
import urllib.request

import numpy as np

BUCKET = "bonsai-2026-stage2b-cache"
OBJECT = "stage2b/train/stage3/common/ridge_final_g13_88edf9ac.npz"
PUBLIC_URL = f"https://storage.googleapis.com/{BUCKET}/{OBJECT}"

# The bound axis 4 claims for the entire chain, for comparison. Not a
# tolerance -- the point of this script is that one link exceeds it.
CLAIMED_END_TO_END_LIPSCHITZ = 2.0


def fetch_summary(url=PUBLIC_URL):
    """`summary_json` from the frozen final-fit npz, over anonymous HTTPS.

    Deliberately not through `stage2b_gcs`: that module requires
    google-cloud-storage, which is a cloud-runtime dependency and absent
    from the local environment by design. The bucket is public-read
    (`stage2b_gcs.DEFAULT_GCS_BUCKET`, "public read (anonymous
    objectViewer)"), so a plain GET needs neither the library nor a key.
    """
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return json.loads(str(data["summary_json"]))


def scaler_lipschitz(summary):
    """{condition: (min_col_std, 1/min_col_std)}, ascending by std."""
    margins = summary["centering_margins"]
    rows = {name: float(entry["min_col_std"]) for name, entry in margins.items()}
    return {name: (std, 1.0 / std)
            for name, std in sorted(rows.items(), key=lambda kv: kv[1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=PUBLIC_URL)
    args = parser.parse_args(argv)

    table = scaler_lipschitz(fetch_summary(args.url))

    print(f"source: {args.url}\n")
    print(f"{'condition':18} {'min_col_std':>14} {'scaler Lipschitz':>18} "
          f"{'vs claimed 2':>14}")
    print("-" * 68)
    for name, (std, lip) in table.items():
        print(f"{name:18} {std:14.6e} {lip:18.1f} "
              f"{lip / CLAIMED_END_TO_END_LIPSCHITZ:13.0f}x")

    evolved = {n: lip for n, (_, lip) in table.items()
               if n in ("T", "lattice", "rewired", "curr_random")}
    worst = max(evolved.values())
    print(f"\nWorst evolved condition: {worst:.1f}, "
          f"{worst / CLAIMED_END_TO_END_LIPSCHITZ:.0f}x the bound axis 4 claims "
          f"for the ENTIRE chain.")
    print(f"T (the condition Stage 4 selected): {evolved['T']:.1f}, "
          f"{evolved['T'] / CLAIMED_END_TO_END_LIPSCHITZ:.0f}x.")
    print("\nOne link of five. The ODE flow and the ridge operator norm are "
          "unmeasured;\nDelta_g's two-arm subtraction contributes a further "
          "factor of 2 on top.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
