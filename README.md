# Bonsai 2026

Bonsai is an investigation into whether coupled-oscillator dynamics, evolved over topologies learned from image populations, produce useful representations of images—and whether the dynamics themselves perform structured computation.

This project transitioned from a benchmark-feature programme (investigating static features for classifiers) to a "Dynamics-as-Computation" focus, exploring how intrinsic properties of physical dynamical systems can represent and process information.

## Overview

The core of Bonsai involves:
- **Topology Learning**: Constructing graphs based on image data (MNIST, KMNIST, etc.).
- **Oscillator Dynamics**: Simulating coupled-oscillator evolution over these topologies.
- **Causal Ablation**: Using various controls (random, rewired, lattice) to verify that performance depends on specific learned connectivity and dynamics. This finding is specific to the closed benchmark-feature programme (`docs/PROJECT_MEMORY.md` Part 1); the later dynamics-as-computation work (Stage 1D/2A) found the *opposite* for oscillator dynamics itself -- learned topology is not distinguishable from matched generic controls, under either an internal structured-transformation statistic or an external classification task.
- **Dynamics-as-Computation**: Analyzing structured responses to perturbations as a form of computation. Established at Level 2 (structured internal transformation) across independent trajectories, and at Level 3 (useful computation) under a bounded classification design (Stage 2A) -- see `docs/PROJECT_MEMORY.md` Part 3 for scope and open follow-ons.
- **Denoising (Stage 2B)**: Asks the Stage 2A-shaped question -- does runtime graph evolution add value on top of the same encoded phase state -- for single-step active-support reconstruction under a fixed corruption process, instead of classification. Feasibility ladder in progress; see `docs/PROJECT_MEMORY.md` Part 3 for current status.

For detailed history and current status, see `docs/PROJECT_MEMORY.md`; for the methodological principles this project holds itself to, see `CLAUDE.md`. For definitions of project-specific terms (trajectory, baseline seed, replica, t_p, tau, finite/tangent/residual response, etc.), see `docs/GLOSSARY.md`.


## Requirements

- **Python**: >= 3.14
- **Core Dependencies**:
  - `numpy` >= 2.5.1
  - `scipy` >= 1.18.0
  - `scikit-learn` >= 1.9.0
  - `tqdm` >= 4.70.0
  - `typing-extensions` >= 4.16.0
- **Development Dependencies**:
  - `pytest` >= 9.1.1

## Setup

This project uses `hatchling` as a build backend and is compatible with `uv`.

### Using `uv` (Recommended)

1.  **Install dependencies**:
    ```bash
    uv sync
    ```
2.  **Run scripts (Mandatory)**:
    Use `uv run` to ensure all dependencies and the `bonsai` package are correctly loaded:
    ```bash
    uv run python <script_path>
    ```

### Manual Setup

1.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install dependencies**:
    ```bash
    pip install -e .
    ```

## Directory Structure

- `src/bonsai/`: Core package source code.
  - `data/`: Data loaders (MNIST, KMNIST, etc.).
  - `dynamics/`: Oscillator simulation and topology construction (lattice, rewired, matched-sparsity).
  - `stats/`: Statistical measures (permutation tests, tangent departure).
- `experiments/`: Active research stages and pilots (e.g., `stage1a_re_verification`, `stage1b_pilot`, `stage1b2_structured_transformation`, `stage1c_trajectory_generalization`, `stage2a_dynamics_classification`, `stage2b_denoising`).
- `benchmark_programme/`: Historical record of the initial phase, including findings and older test suites. **Note: This directory contains frozen snapshots and should not be modified.**
- `datasets/`: Local storage for MNIST-format datasets (MNIST, KMNIST, Fashion-MNIST, notMNIST).
- `docs/`: Durable project documentation and memory.
- `tests/`: Current quantitative verification suite.

## Scripts

All research scripts should be executed from the project root using `uv run python ...`.

- **`experiments/stage1b_pilot/run_stage1b_pilot.py`**: Runs a pilot batch for dynamics-as-computation trials.
- **`main.py`**: A sample entry point (placeholder).

The root `Makefile` wraps the Stage 2A reproduction workflow (local
encode steps, remote GPU evolution via `mighty-colab`, artifact
verification) behind `make stage2a-*` targets, so those commands don't
need to be run from any particular subdirectory -- run `make
stage2a-help` for the full list, or see
`experiments/stage2a_dynamics_classification/README.md` for the
workflow they belong to.

The same `Makefile` also wraps Stage 2B's feasibility-ladder drivers
behind `make stage2b-*` targets (`stage2b-test` for the fast local
suite, `stage2b-ladder-stage1` / `stage2b-ladder-stage2` for the
Colab-GPU ladder rungs, both billing while running) -- run `make help`
for the full list, or see `experiments/stage2b_denoising/README.md` for
the workflow.

## Environment Variables

No specific environment variables are required for basic operation. TODO: Document any future variables for remote execution or data paths.

## Tests

Tests are managed via `pytest`. Always run tests through `uv run` to ensure the environment is correct.

```bash
# Run all tests
uv run pytest

# Run tests excluding slow reproduction checks
uv run pytest -m "not slow"
```

## License

MIT -- see [`LICENSE.md`](LICENSE.md).
