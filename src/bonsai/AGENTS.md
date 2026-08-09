# src/bonsai/

The only shared, cumulatively-developed code in the repository. Editable-
installed, so every `experiments/` stage and much of `tests/` resolves
`from bonsai... import ...` against these files -- an edit here changes
what a historical stage's driver would now produce. `benchmark_programme/`
is the opposite discipline (frozen per-milestone snapshots) and depends on
nothing here.

## Blast radius

44 files import `bonsai.*`, across `experiments/` (stages 0 through 2B),
`tests/`, `docs/report_visuals/`, and this package itself. Import sites
per module, highest first:

| Module | Sites | Notes |
|---|---|---|
| `data/mnist_loader.py` | 30 | `load_mnist(data_dir, gz=)` -- handles both compressed and plain IDX |
| `dynamics/learned_topology_construction.py` | 14 | `build_class_topology`, `population_developmental_stat` |
| `dynamics/graph_oscillator_field.py` | 6 | the simulator: `GraphOscillatorField`, `find_equilibrium_lbfgs`, `force_jacobian`, `rotation_projector`, the tangent/finite response family |
| `dynamics/matched_sparsity_ablation.py` | 4 | |
| `dynamics/degree_preserving_rewiring.py` | 4 | |
| `dynamics/construction_bundle.py` | 4 | composes the four-graph bundle from the modules above |
| `dynamics/lattice_construction.py` | 3 | |
| `dynamics/historical_matched_sparsity_random.py` | 3 | |
| `stats/permutation.py`, `stats/tangent_departure.py` | **0** | see below |

`stats/` has no import sites anywhere in the repository. The root
`AGENTS.md` directs new permutation-test work to reuse
`stats/permutation.py` rather than writing its own -- that direction
stands, but the module currently has no caller exercising it, so its
first real use is also its first integration test. Unit-test any new
permutation scheme on synthetic data before trusting a p-value from it
(root principle 10).

## Consolidation status is per file, not uniform

Stage 0 and Stage 1A code was consolidated into this package. The
`NOTE.md` files in those experiment directories once read as if the
whole consolidation had been behaviourally verified. Checked file by
file, it had not been:

- `graph_oscillator_field.py` -- verified (multistability, spectral gap,
  RK45/DOP853 cross-validation reproduced).
- `degree_preserving_rewiring.py` -- verified (byte-exact rewired
  construction reproduced).
- `matched_sparsity_ablation.py` -- **not verified**, and found to
  actively disagree with the historical cached result it had been
  assumed equivalent to.

A diff showing a consolidation lost nothing establishes completeness,
not behavioural equivalence (root principle 15). Treat any remaining
module here as behaviourally unverified against its historical origin
unless `tests/` pins it.

## Editing rules

- **Import these helpers; never reimplement them in caller code.** The
  paid instance: a GPU port verified correct to 1e-6 per field still
  produced a wrong result because the calling script redrew replica
  directions with raw `uniform(-1,1)` instead of calling the already-
  imported `generate_fixed_replica_directions()`. Same bug then turned
  up a second time, independently, in a nearby notebook. Verifying the
  helper does not clear the glue around it (root principle 16).
- **Changing a function's numerics invalidates the `tests/` claims that
  pin it**, and those tests pin published `FINDINGS.md` numbers. Run
  `make test` before and after; a Tier 2 test that *skips* on your
  machine has not cleared the change, it has declined to check it.
- Parallelised Monte Carlo work uses `SeedSequence(seed).spawn(n)` per
  worker. A shared or repeated seed silently correlates draws
  (principle 8); a within-worker chunking change silently alters the
  stream (principle 19). These are separate failures and correct
  seeding does not protect against the second.

## Layout

`data/` (loaders), `dynamics/` (simulator + graph construction and
controls), `stats/` (permutation framework, tangent-departure
diagnostics). Every `__init__.py` is empty -- import from the full
module path, there is no curated public surface to add to.
