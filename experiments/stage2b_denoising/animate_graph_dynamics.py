"""Animate the graph dynamics on ten held-out KMNIST characters.

The counterpart to `plot_cnn_denoising.py`, on the SAME ten corrupted
inputs, so the two figures can be read side by side.

## What is being animated, and what is not

Not denoising, frame by frame. The graph pipeline is

    x  ->  theta_0^784  ->  theta_0^505  ->  theta_T^505  ->  features

and only the middle arrow has a time axis: `dtheta/dt = sum_j W_ij
sin(theta_j - theta_i)` integrated to `T_HORIZON = 2.5`. What moves in
these frames is the PHASE FIELD, painted back onto the 28x28 grid at the
505 active coordinates. The features Stage 2B actually reads are taken at
the final frame only.

That distinction is the whole reason the CNN cannot be animated
comparably: `x_hat_0 = x_t_clip + f_psi(x_t_clip)` is one feedforward
pass with no intermediate state to show. A four-frame CNN "animation"
would be an invention.

Shown in the locked reference-node gauge, `cos(theta_i - theta_ref)`,
which is what the primary feature vector is built from -- not raw
`theta`, which would show a global rotation the gauge removes and the
features never see.

All four topologies are evolved from the identical encoded state, so any
difference between rows is the topology and nothing else.

Reads `topologies.npz` from the public-read bucket over plain HTTPS, so
this runs from a fresh clone with no credentials. Nothing here computes a
metric; the test split is used for the same reason the CNN figure uses
it, on Dan's ruling of 2026-08-11.
"""
import argparse
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation   # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
from scipy.integrate import solve_ivp      # noqa: E402

import stage2b_corruption as corruption     # noqa: E402

BUCKET_URL = "https://storage.googleapis.com/bonsai-2026-stage2b-cache"
TOPOLOGIES_OBJECT = "stage2b/train/stage1/common/topologies.npz"

# The locked evolution constants, from stage2a_core: same ODE, same
# tolerances. `t_eval` only chooses where the solution is SAMPLED and does
# not change the trajectory, which is what makes frames legitimate here.
T_HORIZON = 2.5
RTOL, ATOL, MAX_STEP = 1e-6, 1e-8, 0.05
REF_IDX = 363          # T's median-degree node in 505-space (DESIGN.md)
ENCODER_SEED = 0
SPLIT = "test"
N_CLASSES = 10
CONDITIONS = [("W_T", "T (learned)"), ("W_lattice", "lattice"),
              ("W_rewired", "rewired"), ("W_curr_random", "random")]


def fetch_npz(object_name, cache_dir):
    local = os.path.join(cache_dir, object_name.replace("/", "__"))
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(f"{BUCKET_URL}/{object_name}", local)
    with np.load(local, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def evolve_saving_frames(theta0, W, n_frames, t_max=T_HORIZON):
    """The locked evolution, sampled at `n_frames` points. Returns
    (n_nodes, n_frames). Raises on solver failure rather than returning a
    partial trajectory -- DESIGN.md's 'zero silent solver failures'.

    `t_max` shortens the WINDOW, not the dynamics: the ODE, solver and
    tolerances are unchanged, so a run to 1.0 traces the same trajectory
    the run to 2.5 passes through. Useful because most of the visible
    motion is early -- the controls have largely collapsed by t~0.5, and
    at the full horizon the interesting part is four frames wide."""
    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return np.sum(W * np.sin(diff), axis=1)

    sol = solve_ivp(rhs, (0.0, t_max), theta0, method="RK45",
                    rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
                    t_eval=np.linspace(0.0, t_max, n_frames))
    if not sol.success:
        raise RuntimeError(f"graph evolution failed: {sol.message}")
    return sol.y


def paint(values, active, side=28, background=np.nan):
    """505 active values into the 28x28 grid; inactive coordinates stay
    `background` so 'where the dynamics live' is visible rather than
    implied."""
    grid = np.full(side * side, background, dtype=np.float64)
    grid[active] = values
    return grid.reshape(side, side)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/kmnist")
    parser.add_argument("--cache-dir", default="results/_plot_cache")
    parser.add_argument("--out", default="results/graph_dynamics_kmnist.gif")
    parser.add_argument("--filmstrip", default="results/graph_dynamics_filmstrip.png")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--t-max", type=float, default=T_HORIZON,
                        help="window to animate, in the ODE's time units; "
                             "shortens the view, not the dynamics")
    args = parser.parse_args()
    if not 0 < args.t_max <= T_HORIZON:
        parser.error(f"--t-max must lie in (0, {T_HORIZON}]; beyond the locked "
                     f"horizon is extrapolation, not this pipeline")

    from bonsai.data.mnist_loader import load_mnist
    from bonsai.dynamics.learned_topology_construction import _local_converged_phases

    _, _, x_test, y_test = load_mnist(args.data_dir, gz=False)
    y_test = np.asarray(y_test)
    rows = np.array([int(np.flatnonzero(y_test == c)[0]) for c in range(N_CLASSES)])
    clean = np.asarray(x_test)[rows].astype(np.float64).reshape(len(rows), 784) / 255.0
    _x_t, noisy = corruption.corrupt_corpus(
        clean, SPLIT, rows, alpha_bar=corruption.ALPHA_BAR, allow_test_split=True)

    topo = fetch_npz(TOPOLOGIES_OBJECT, args.cache_dir)
    active = np.asarray(topo["active_indices"])

    # One encode per image, shared by every topology -- so a row-to-row
    # difference cannot be an encoding difference.
    theta0 = np.stack([
        _local_converged_phases(noisy[i].reshape(28, 28), seed=ENCODER_SEED).flatten()[active]
        for i in range(len(rows))])

    traj, order = {}, {}
    for key, label in CONDITIONS:
        W = np.asarray(topo[key])
        stacked = np.stack([evolve_saving_frames(theta0[i], W, args.frames, args.t_max)
                            for i in range(len(rows))])          # (10, 505, frames)
        gauged = np.cos(stacked - stacked[:, REF_IDX, :][:, None, :])
        traj[key] = gauged
        order[key] = np.abs(np.mean(np.exp(1j * stacked), axis=1)).mean(axis=0)
        print(f"{label:14s} evolved; R(0)={order[key][0]:.4f} -> "
              f"R(T)={order[key][-1]:.4f}")

    times = np.linspace(0.0, args.t_max, args.frames)
    cmap = plt.get_cmap("twilight_shifted").copy()
    cmap.set_bad("#f2f2f2")     # the 279 inactive coordinates

    fig, axes = plt.subplots(len(CONDITIONS), N_CLASSES,
                             figsize=(N_CLASSES * 1.12, len(CONDITIONS) * 1.30))
    images = []
    for r, (key, label) in enumerate(CONDITIONS):
        for c in range(N_CLASSES):
            ax = axes[r, c]
            im = ax.imshow(paint(traj[key][c, :, 0], active),
                           cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.5)
            if r == 0:
                ax.set_title(f"class {c}", fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            images.append((im, key, c))
    title = fig.suptitle("", fontsize=9)

    def draw(frame):
        for im, key, c in images:
            im.set_data(paint(traj[key][c, :, frame], active))
        title.set_text(
            f"Graph dynamics on held-out KMNIST, $\\cos(\\theta_i-\\theta_{{ref}})$  |  "
            f"t = {times[frame]:.3f} / {args.t_max:g}"
            + (f" (locked horizon {T_HORIZON})" if args.t_max < T_HORIZON else "")
            + "  |  "
            + "   ".join(f"R$_{{{lab}}}$={order[k][frame]:.3f}"
                         for k, lab in CONDITIONS))
        return [im for im, _, _ in images] + [title]

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    anim = animation.FuncAnimation(fig, draw, frames=args.frames, blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    print(f"wrote {args.out}")

    # A still, for readers and renderers that will not play a GIF: one
    # character across time, all four topologies.
    keep = [0, args.frames // 5, 2 * args.frames // 5, 3 * args.frames // 5,
            4 * args.frames // 5, args.frames - 1]
    char = 1
    fig2, axes2 = plt.subplots(len(CONDITIONS), len(keep),
                               figsize=(len(keep) * 1.25, len(CONDITIONS) * 1.32))
    for r, (key, label) in enumerate(CONDITIONS):
        for c, frame in enumerate(keep):
            ax = axes2[r, c]
            ax.imshow(paint(traj[key][char, :, frame], active),
                      cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.5)
            if r == 0:
                ax.set_title(f"t={times[frame]:.2f}", fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig2.suptitle(f"One character (class {char}), four topologies, identical "
                  f"encoded state", fontsize=9)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    fig2.savefig(args.filmstrip, dpi=170)
    print(f"wrote {args.filmstrip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
