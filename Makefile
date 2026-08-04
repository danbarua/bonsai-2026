# Stage 2A workflow helper.
#
# Why this exists: external review found the Stage 2A README's GPU
# workflow commands used bare filenames and bare `scratch/...` paths
# that only resolve when the shell's cwd is
# experiments/stage2a_dynamics_classification/ -- contradicting the
# README's own "run from repository root" convention used everywhere
# else. Rather than documenting a fragile "cd into this exact
# subdirectory first" instruction, this Makefile is the single place
# that owns the actual commands; the README points here instead of
# duplicating them (per the convention: no duplicated filenames between
# README and Makefile).
#
# Anchoring: REPO_ROOT is resolved via `git rev-parse --show-toplevel`
# at parse time, not hardcoded -- this makes every target work
# identically whether `make` is invoked from the canonical checkout or
# from a `git worktree` copy elsewhere on disk (e.g. a scratch/tmp
# directory used for agent isolation). Do not replace this with a
# literal path.
#
# Make orchestrates only -- it does not contain scientific logic. All
# graph construction, feature generation, solver configuration,
# classifier fitting, statistical analysis, and artifact validation
# remain in the Python scripts under STAGE2A_DIR; these targets just
# wire together already-existing, already-documented command sequences.

REPO_ROOT := $(shell git rev-parse --show-toplevel)
STAGE2A_DIR := $(REPO_ROOT)/experiments/stage2a_dynamics_classification
PYTHON ?= uv run python
SESSION_TRAIN ?= stage3-evolve
SESSION_TEST ?= stage4-evolve

.PHONY: stage2a-help
stage2a-help:  ## List available stage2a-* targets
	@grep -E '^stage2a-[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk -F':|##' '{printf "  %-28s %s\n", $$1, $$3}'

.PHONY: stage2a-prepare-train
stage2a-prepare-train:  ## Encode 60k KMNIST training images + split for GPU upload (local, CPU, ~70s)
	$(PYTHON) $(STAGE2A_DIR)/run_feasibility_stage3_encode.py
	$(PYTHON) $(STAGE2A_DIR)/prepare_stage3_gpu_upload.py

.PHONY: stage2a-evolve-train-gpu
stage2a-evolve-train-gpu:  ## Upload + run Stage-3 (training set) GPU evolution via mighty-colab -- bills while running
	cd $(STAGE2A_DIR) && \
	mighty-colab sessions && \
	mighty-colab new -s $(SESSION_TRAIN) --gpu A100 && \
	mighty-colab reinstall -s $(SESSION_TRAIN) jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8 && \
	mighty-colab upload -s $(SESSION_TRAIN) evolve_on_graph_jax.py /content/evolve_on_graph_jax.py && \
	mighty-colab upload -s $(SESSION_TRAIN) scratch/stage3_train/stage3_topologies.pkl /content/stage3_topologies.pkl && \
	for i in 00 01 02 03 04 05 06 07 08 09 10 11; do \
		mighty-colab upload -s $(SESSION_TRAIN) scratch/stage3_train/theta0_chunk_$$i.npy /content/theta0_chunk_$$i.npy || exit 1; \
	done && \
	mighty-colab exec -s $(SESSION_TRAIN) -f stage3_gpu_evolve.py && \
	mighty-colab download -s $(SESSION_TRAIN) /content/stage3_gpu_results.pkl scratch/stage3_train/stage3_gpu_results.pkl && \
	mighty-colab stop -s $(SESSION_TRAIN)

.PHONY: stage2a-prepare-test
stage2a-prepare-test:  ## Encode 10k KMNIST official test images (local, CPU, ~25s) -- the ONE place test-set images/labels are touched
	$(PYTHON) $(STAGE2A_DIR)/run_official_test_encode.py

.PHONY: stage2a-evolve-test-gpu
stage2a-evolve-test-gpu:  ## Upload + run Stage-4 (official test set) GPU evolution via mighty-colab -- bills while running
	cd $(STAGE2A_DIR) && \
	mighty-colab sessions && \
	mighty-colab new -s $(SESSION_TEST) --gpu A100 && \
	mighty-colab reinstall -s $(SESSION_TEST) jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8 && \
	mighty-colab upload -s $(SESSION_TEST) evolve_on_graph_jax.py /content/evolve_on_graph_jax.py && \
	mighty-colab upload -s $(SESSION_TEST) scratch/stage4_test/stage4_gpu_upload_topologies.pkl /content/stage4_gpu_upload_topologies.pkl && \
	mighty-colab upload -s $(SESSION_TEST) scratch/stage4_test/stage4_theta0_test.npy /content/stage4_theta0_test.npy && \
	mighty-colab exec -s $(SESSION_TEST) -f stage4_gpu_evolve.py && \
	mighty-colab download -s $(SESSION_TEST) /content/stage4_gpu_results.pkl scratch/stage4_test/stage4_gpu_results.pkl && \
	mighty-colab stop -s $(SESSION_TEST)

.PHONY: stage2a-analyze
stage2a-analyze:  ## Feasibility stage-3 classifier CV model selection (~4hr on CPU sklearn -- see FINDINGS.md Result 3 first)
	$(PYTHON) $(STAGE2A_DIR)/analyze_stage3_results.py

.PHONY: stage2a-confirm
stage2a-confirm:  ## Run the locked confirmatory evaluation (final refits, primary/secondary bootstrap, McNemar, MLP baselines)
	$(PYTHON) $(STAGE2A_DIR)/run_confirmatory_evaluation.py

.PHONY: stage2a-posthoc
stage2a-posthoc:  ## Post hoc graph-to-graph pairwise comparison (seconds -- reuses saved per-image losses, no new GPU time)
	$(PYTHON) $(STAGE2A_DIR)/run_posthoc_graph_pairwise.py

.PHONY: stage2a-manifest
stage2a-manifest:  ## Regenerate results/ARTIFACT_MANIFEST.json (hashes, dimensions, selected C, environment metadata)
	$(PYTHON) $(STAGE2A_DIR)/generate_artifact_manifest.py

.PHONY: stage2a-verify
stage2a-verify: stage2a-manifest  ## Regenerate the manifest and diff it against the committed version
	@echo "Diffing regenerated manifest against the committed one."
	@echo "NOTE: environment.git_commit_sha will always differ on a later commit -- that field alone changing is NOT a reproduction failure. Compare the artifact/graph/array sha256 fields specifically."
	git diff --stat $(STAGE2A_DIR)/results/ARTIFACT_MANIFEST.json
