This stage's original driver code (`degree_preserving_rewiring.py`,
`matched_sparsity_ablation.py`, `graph_oscillator_field.py`) has been
consolidated into `src/bonsai/dynamics/` -- the versions captured in
this stage's original tarball were superseded by later, strictly-more-
complete versions of the same files from Stage 1B.2's era (confirmed by
diff before consolidating, not assumed). See `FINDINGS.md` for this
stage's actual results; the code that produced them now lives in the
shared package.
