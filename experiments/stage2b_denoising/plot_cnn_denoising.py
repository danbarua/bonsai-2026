"""Render the CNN denoiser on ten held-out KMNIST characters.

One column per class, rows: the clean target, the corrupted input the model
actually sees, the model's output, and the residual it added. The residual
row is the interesting one -- `x_hat_0 = x_t_clip + f_psi(x_t_clip)`, so it
is literally what the network contributes.

Uses the KMNIST TEST split, on Dan's ruling (2026-08-11, "it's just for
visualisation"). Nothing here computes a metric: no number produced by this
file enters any record, and DESIGN.md's lock is on evaluating against that
corpus, not on looking at it. Held-out images are the honest choice for a
demonstration -- training-split characters would show in-sample behaviour.

Weights come from `cnn_weights.npz`, the object the stage-3 backfill wrote.
Fetched over plain HTTPS from the public-read bucket, so this runs from a
fresh clone with no credentials.
"""
import argparse
import io
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

import stage2b_cnn as cnn            # noqa: E402
import stage2b_corruption as corruption   # noqa: E402

BUCKET_URL = "https://storage.googleapis.com/bonsai-2026-stage2b-cache"
WEIGHTS_OBJECT = "stage2b/train/stage3/common/cnn_weights.npz"
TOPOLOGIES_OBJECT = "stage2b/train/stage1/common/topologies.npz"
SPLIT = "test"
N_CLASSES = 10
DEFAULT_OUT = "results/cnn_denoising_kmnist.png"


def fetch_npz(object_name, cache_dir):
    local = os.path.join(cache_dir, object_name.replace("/", "__"))
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(f"{BUCKET_URL}/{object_name}", local)
    with np.load(local, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def one_per_class(labels, n_classes=N_CLASSES):
    """The first test image of each class, in class order -- deterministic,
    so re-running this produces the same figure."""
    return np.array([int(np.flatnonzero(labels == c)[0]) for c in range(n_classes)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/kmnist")
    parser.add_argument("--cache-dir", default="results/_plot_cache")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed-key", default=None,
                        help="which weights_seed* to render; default is the "
                             "best seed recorded in weights_json")
    args = parser.parse_args()

    from bonsai.data.mnist_loader import load_mnist
    _, _, x_test, y_test = load_mnist(args.data_dir, gz=False)

    weights = fetch_npz(WEIGHTS_OBJECT, args.cache_dir)
    meta = json.loads(weights["weights_json"].item())
    key = args.seed_key or meta["best_weights_key"]
    model = cnn.deserialise_model(weights[key])
    topo = fetch_npz(TOPOLOGIES_OBJECT, args.cache_dir)
    active = np.asarray(topo["active_indices"])

    rows = one_per_class(np.asarray(y_test))
    # `load_mnist` yields (n, 28, 28); everything below works in the flat
    # 784-grid the active support is indexed in.
    clean = np.asarray(x_test)[rows].astype(np.float64).reshape(len(rows), 784) / 255.0
    # `allow_test_split` is passed deliberately: these are pixels for a
    # figure, not an evaluation. ALPHA_BAR is the frozen production value.
    _x_t, noisy = corruption.corrupt_corpus(
        clean, SPLIT, rows, alpha_bar=corruption.ALPHA_BAR, allow_test_split=True)

    out = np.asarray(cnn.forward(model, cnn.as_image_batch(noisy, "noisy")))
    out = out.reshape(len(rows), 784)
    residual = out - noisy

    # The 279 coordinates outside the active support get no training signal,
    # so shading them makes "where the model was actually asked to work"
    # visible rather than something the reader has to know.
    support = np.zeros(784, dtype=bool)
    support[active] = True

    panels = [
        ("clean $x_0$", clean, dict(cmap="gray", vmin=0, vmax=1)),
        ("corrupted $x_t$", noisy, dict(cmap="gray", vmin=0, vmax=1)),
        ("CNN $\\hat{x}_0$", out, dict(cmap="gray", vmin=0, vmax=1)),
        ("residual $f_\\psi(x_t)$", residual,
         dict(cmap="RdBu_r", vmin=-0.6, vmax=0.6)),
    ]
    fig, axes = plt.subplots(len(panels), N_CLASSES,
                             figsize=(N_CLASSES * 1.15, len(panels) * 1.32))
    for r, (label, data, kw) in enumerate(panels):
        for c in range(N_CLASSES):
            ax = axes[r, c]
            ax.imshow(data[c].reshape(28, 28), **kw)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#bbbbbb"); spine.set_linewidth(0.5)
            if r == 0:
                ax.set_title(f"class {c}", fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(
        f"Residual CNN denoiser on held-out KMNIST  |  {key}, "
        f"{cnn.count_trainable_parameters(model):,} params, "
        f"trained on {meta['platform']['gpu']}  |  "
        f"$\\bar\\alpha$={corruption.ALPHA_BAR}, {support.sum()} active coordinates",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")

    inside = np.abs(residual[:, support]).mean()
    outside = np.abs(residual[:, ~support]).mean()
    print(f"mean |residual| inside the active support:  {inside:.4e}")
    print(f"mean |residual| outside it (untrained):     {outside:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
