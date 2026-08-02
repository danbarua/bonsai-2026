# Junie Guidelines for Bonsai 2026

## Python Execution
- **Mandatory Command**: Always use `uv run python SCRIPT.py` to run Python scripts.
- **Environment**: Do not attempt to use `python3` or `python` directly unless it's within the `uv run` context. The `uv`-managed virtual environment contains all necessary dependencies (`scipy`, `numpy`, `tqdm`, and the `bonsai` package).
- **Package Installation**: The core `bonsai` package is editable-installed. Imports like `from bonsai.dynamics import ...` will resolve correctly when using `uv run`.

## Research & Methodology
- **Project Memory**: Before starting any task, read `docs/PROJECT_MEMORY.md` to understand the current state, established findings, and active frontier.
- **Methodological Principles**: Follow the 17 principles in `CLAUDE.md`'s "Methodological discipline" section. Key highlights:
    - Verify before trusting.
    - Prefer narrower, precisely scoped claims.
    - Unit-test new statistical machinery on synthetic data first.
    - Use `SeedSequence.spawn` for parallelized Monte Carlo work to ensure independent random streams.
- **Dynamics-as-Computation**: Respect the three-way distinction:
    1. Nonlinear response.
    2. Structured internal transformation.
    3. Useful computation.

## Execution Discipline
- **Parallelism**: This environment has 10 CPU cores. Use `multiprocessing.Pool` with `max(1, cpu_count()-1)` for long-running analyses.
- **Process Management**: Launch long-running computations with `waitForExit=false` and monitor via console output.
- **Terminals**: Use fresh, non-reused terminal windows for foreground processes to avoid killing existing tasks.

## Directory Structure
- `src/bonsai/`: Shared library code.
- `experiments/`: Active research scripts (cumulative).
- `benchmark_programme/`: Frozen historical snapshots (do not modify).
- `datasets/`: Raw data.
- `tests/`: Quantitative claim verification.
