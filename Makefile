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
# mighty-colab is a pinned dependency-group (pyproject.toml's
# [dependency-groups].gpu, not a project dependency proper -- it's an
# ops/CLI tool, not imported by any Python code here), the official
# PyPI release, not a locally hand-installed `uv tool`. `uv run --group
# gpu` transparently syncs that group into .venv on first use, so a
# clean checkout needs no separate `uv tool install` step -- matching
# this project's existing "always uv run, never a bare global binary"
# convention (CLAUDE.md).
#
# To pick up a newly published release:
#
#     uv sync --group gpu --upgrade-package mighty-colab
#
# `--reinstall-package` is the wrong tool for that job -- it reinstalls
# whatever `uv.lock` already pins, so it repairs a broken install but
# leaves the version exactly where it was. Only `--upgrade-package` moves
# the lock entry. Both are quiet no-ops when nothing has changed, so the
# version reported afterwards is the check, not the command's output.
MIGHTY_COLAB ?= uv run --group gpu mighty-colab
SESSION_TRAIN ?= stage3-evolve
SESSION_TEST ?= stage4-evolve
SESSION_CLASS0 ?= class0-audit-gpu

# `mighty-colab exec --timeout` defaults to 30 SECONDS, and it bounds the
# gap between outputs, not the run: a remote script that goes quiet for
# longer than this dies with `TimeoutError: Timeout waiting for output`
# even though the kernel is working normally. Every long-running driver
# here goes quiet for far longer than 30s -- `stage3_gpu_evolve.py` prints
# once per topology, `class0_support_audit_classify_gpu.py` once per
# download and then not at all while cuML fits -- so each GPU target must
# pass this explicitly. A generous value costs nothing on a healthy run
# and bounds what a genuinely hung kernel can bill.
EXEC_TIMEOUT ?= 3600

# `mighty-colab stop` is the only thing between a failed run and an A100
# that bills until someone notices, so every recipe checks its exit status
# instead of discarding it. Two outcomes have to stay distinguishable, and
# they mean opposite things:
#
#   already absent -> nothing is billing. This is the GOAL, and it is the
#     normal case on any path where provisioning failed before a session
#     was ever created (which recipes below do reach -- the `;` after
#     `rc=0` ends the `&&` chain, so teardown runs even when `new` or an
#     upload failed). Verified against 0.2.1: `stop` on an unknown session
#     prints "not found." to stdout and exits 0.
#   could not stop -> something may still be billing. This is a LEAK, and
#     it is the one outcome worth failing an otherwise-successful target
#     for, because the cost keeps accruing while nobody is looking.
#
# STOP_ABSENT_RC is what "already absent" exits with. It is 0 today, which
# collapses the two cases into one check; if a future release gives absent
# its own code, set this to that code and the recipes keep their meaning
# without being rewritten. Pinned by tests/test_mighty_colab_contract.py.
STOP_ABSENT_RC ?= 0

# Evaluated after teardown, with $$src holding stop's status and $$rc the
# run's own verdict so far. A leak fails the target, but never overwrites
# a verdict that already failed -- the science's failure is the more
# useful headline, and the leak is reported on its own line regardless.
define check_teardown
if [ $$src -ne 0 ] && [ $$src -ne $(STOP_ABSENT_RC) ]; then echo "[make] LEAK WARNING: teardown of session '$(1)' exited $$src -- it may still be running and billing."; echo "[make]   check with: $(MIGHTY_COLAB) sessions"; echo "[make]   stop it with: $(MIGHTY_COLAB) stop -s $(1)"; if [ $$rc -eq 0 ]; then rc=$$src; fi; fi
endef

# GPU-target idempotency: `mighty-colab new -s <name>` provisions a fresh
# session unconditionally, so re-running a GPU target after a partial
# failure (a dropped upload, a flaky exec) would try to allocate a second
# session under the same name instead of resuming the one already up --
# and `mighty-colab status -s <name>` returns exit code 0 even when the
# session doesn't exist (prints "Session '<name>' not found." to stdout but
# does not fail) -- verified directly, not assumed, and re-verified against
# 0.2.1 after that release moved several other commands' error text to
# stderr. So the GPU targets below grep that message rather than trusting
# the exit status, redirect stderr into the grep so a future move of this
# message does not silently break the guard, and only call `new` when a
# session by that name genuinely isn't there yet.
# Pinned by tests/test_mighty_colab_contract.py.

.PHONY: stage2a-help
stage2a-help:  ## List every stage2a-* target, grouped by pipeline stage
	@awk 'BEGIN {FS = ":.*##"} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^stage2a-[a-zA-Z0-9_-]+:.*##/ {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Small-scale feasibility (mechanical validation only, not confirmatory -- see FINDINGS.md)

.PHONY: stage2a-feasibility1
stage2a-feasibility1:  ## Feasibility stage 1: 1,000 images, CPU, ~1 min
	$(PYTHON) $(STAGE2A_DIR)/run_feasibility_stage1.py

.PHONY: stage2a-feasibility2
stage2a-feasibility2:  ## Feasibility stage 2: 5,000 images, CPU, ~6 min
	$(PYTHON) $(STAGE2A_DIR)/run_feasibility_stage2.py

##@ Local data preparation (CPU, free)

.PHONY: stage2a-prepare-train
stage2a-prepare-train:  ## Encode 60k KMNIST training images + split for GPU upload (local, CPU, ~70s)
	$(PYTHON) $(STAGE2A_DIR)/run_feasibility_stage3_encode.py
	$(PYTHON) $(STAGE2A_DIR)/prepare_stage3_gpu_upload.py

.PHONY: stage2a-prepare-test
stage2a-prepare-test:  ## Encode 10k KMNIST official test images (local, CPU, ~25s) -- the ONE place test-set images/labels are touched
	$(PYTHON) $(STAGE2A_DIR)/run_official_test_encode.py

##@ GPU evolution (mighty-colab, bills while running)

.PHONY: stage2a-evolve-train-gpu
stage2a-evolve-train-gpu:  ## Upload + run Stage-3 (training set) GPU evolution via mighty-colab -- bills while running
	rc=0; src=0; \
	cd $(STAGE2A_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	if $(MIGHTY_COLAB) status -s $(SESSION_TRAIN) 2>&1 | grep -q "not found"; then \
		$(MIGHTY_COLAB) new -s $(SESSION_TRAIN) --gpu A100; \
	else \
		echo "[make] Reusing existing session $(SESSION_TRAIN) (resuming after a partial run, or you re-ran this target with a session still up)"; \
	fi && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_TRAIN) jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8 && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TRAIN) evolve_on_graph_jax.py /content/evolve_on_graph_jax.py && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TRAIN) scratch/stage3_train/stage3_topologies.pkl /content/stage3_topologies.pkl && \
	for i in 00 01 02 03 04 05 06 07 08 09 10 11; do \
		$(MIGHTY_COLAB) upload -s $(SESSION_TRAIN) scratch/stage3_train/theta0_chunk_$$i.npy /content/theta0_chunk_$$i.npy || exit 1; \
	done && \
	rc=0; \
	$(MIGHTY_COLAB) exec -s $(SESSION_TRAIN) -f stage3_gpu_evolve.py --timeout $(EXEC_TIMEOUT) || rc=$$?; \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(SESSION_TRAIN) /content/stage3_gpu_results.pkl scratch/stage3_train/stage3_gpu_results.pkl || rc=$$?; \
	fi; \
	src=0; $(MIGHTY_COLAB) stop -s $(SESSION_TRAIN) || src=$$?; \
	$(call check_teardown,$(SESSION_TRAIN)); \
	exit $$rc

.PHONY: stage2a-evolve-test-gpu
stage2a-evolve-test-gpu:  ## Upload + run Stage-4 (official test set) GPU evolution via mighty-colab -- bills while running
	rc=0; src=0; \
	cd $(STAGE2A_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	if $(MIGHTY_COLAB) status -s $(SESSION_TEST) 2>&1 | grep -q "not found"; then \
		$(MIGHTY_COLAB) new -s $(SESSION_TEST) --gpu A100; \
	else \
		echo "[make] Reusing existing session $(SESSION_TEST) (resuming after a partial run, or you re-ran this target with a session still up)"; \
	fi && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_TEST) jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8 && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TEST) evolve_on_graph_jax.py /content/evolve_on_graph_jax.py && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TEST) scratch/stage4_test/stage4_gpu_upload_topologies.pkl /content/stage4_gpu_upload_topologies.pkl && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TEST) scratch/stage4_test/stage4_theta0_test.npy /content/stage4_theta0_test.npy && \
	rc=0; \
	$(MIGHTY_COLAB) exec -s $(SESSION_TEST) -f stage4_gpu_evolve.py --timeout $(EXEC_TIMEOUT) || rc=$$?; \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(SESSION_TEST) /content/stage4_gpu_results.pkl scratch/stage4_test/stage4_gpu_results.pkl || rc=$$?; \
	fi; \
	src=0; $(MIGHTY_COLAB) stop -s $(SESSION_TEST) || src=$$?; \
	$(call check_teardown,$(SESSION_TEST)); \
	exit $$rc

##@ Analysis and confirmatory evaluation (CPU)

.PHONY: stage2a-analyze
stage2a-analyze:  ## Feasibility stage-3 classifier CV model selection (~4hr on CPU sklearn -- see FINDINGS.md Result 3 first)
	$(PYTHON) $(STAGE2A_DIR)/analyze_stage3_results.py

.PHONY: stage2a-confirm
stage2a-confirm:  ## Run the locked confirmatory evaluation (final refits, primary/secondary bootstrap, McNemar, MLP baselines)
	$(PYTHON) $(STAGE2A_DIR)/run_confirmatory_evaluation.py

.PHONY: stage2a-posthoc
stage2a-posthoc:  ## Post hoc graph-to-graph pairwise comparison (seconds -- reuses saved per-image losses, no new GPU time)
	$(PYTHON) $(STAGE2A_DIR)/run_posthoc_graph_pairwise.py

##@ Artifact verification

.PHONY: stage2a-manifest
stage2a-manifest:  ## Regenerate results/ARTIFACT_MANIFEST.json (hashes, dimensions, selected C, environment metadata)
	$(PYTHON) $(STAGE2A_DIR)/generate_artifact_manifest.py

.PHONY: stage2a-verify
stage2a-verify:  ## Regenerate a candidate manifest and fail if it mismatches the committed one on any load-bearing field
	$(PYTHON) $(STAGE2A_DIR)/generate_artifact_manifest.py --out $(STAGE2A_DIR)/scratch/ARTIFACT_MANIFEST.candidate.json
	$(PYTHON) $(STAGE2A_DIR)/compare_artifact_manifests.py $(STAGE2A_DIR)/results/ARTIFACT_MANIFEST.json $(STAGE2A_DIR)/scratch/ARTIFACT_MANIFEST.candidate.json

##@ Class-0 support audit (post hoc -- see FINDINGS.md's "class-0 confound" sections)

.PHONY: stage2a-class0-audit
stage2a-class0-audit:  ## Part 1: retained-ink statistics (local, free)
	$(PYTHON) $(STAGE2A_DIR)/run_class0_support_audit.py

.PHONY: stage2a-class0-classify
stage2a-class0-classify:  ## Part 2: the two baseline classifier fits (local sklearn; needs stage2a-prepare-train's artifacts)
	$(PYTHON) $(STAGE2A_DIR)/run_class0_support_audit_classify.py

# Part 2's GPU variant: a genuinely tested and verified upload/exec
# sequence (run for real on a mighty-colab A100 session named
# class0-audit-gpu; its output is the raw_pixels_505restricted /
# encoded_784_unrestricted numbers already reported in FINDINGS.md's
# "class-0-support audit" section) -- not the committed
# run_class0_support_audit_classify.py --cuml path (that one still reads
# through stage2a_paths.scratch_root() and has never actually been run
# remotely), but the dedicated remote driver
# class0_support_audit_classify_gpu.py, which downloads its inputs
# directly from the public GCS mirror rather than needing them uploaded.
# See CUML_ACCEL_FINDINGS.md and that script's own docstring for the
# distinction.
.PHONY: stage2a-class0-classify-gpu
stage2a-class0-classify-gpu:  ## Part 2's cuml.accel GPU variant via mighty-colab -- bills while running
	rc=0; src=0; \
	cd $(STAGE2A_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	if $(MIGHTY_COLAB) status -s $(SESSION_CLASS0) 2>&1 | grep -q "not found"; then \
		$(MIGHTY_COLAB) new -s $(SESSION_CLASS0) --gpu A100; \
	else \
		echo "[make] Reusing existing session $(SESSION_CLASS0) (resuming after a partial run, or you re-ran this target with a session still up)"; \
	fi && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_CLASS0) --requirement cuml_requirements.txt && \
	$(MIGHTY_COLAB) upload -s $(SESSION_CLASS0) stage2a_classifier.py /content/stage2a_classifier.py && \
	$(MIGHTY_COLAB) upload -s $(SESSION_CLASS0) stage2a_stats.py /content/stage2a_stats.py && \
	rc=0; \
	$(MIGHTY_COLAB) exec -s $(SESSION_CLASS0) -f class0_support_audit_classify_gpu.py --timeout $(EXEC_TIMEOUT) || rc=$$?; \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(SESSION_CLASS0) /content/class0_support_audit_classify_results.pkl results/class0_support_audit_classify_results.pkl || rc=$$?; \
	fi; \
	src=0; $(MIGHTY_COLAB) stop -s $(SESSION_CLASS0) || src=$$?; \
	$(call check_teardown,$(SESSION_CLASS0)); \
	exit $$rc

##@ Testing

TEST_FILES := tests/test_stage2a_core.py tests/test_stage2a_stats.py \
              tests/test_stage2a_classifier.py tests/test_stage2a_pipeline.py \
              tests/test_stage2a_topologies.py tests/test_stage2a_paths.py

.PHONY: stage2a-test
stage2a-test:  ## Run the Stage 2A test suite (Tier 2 cases skip cleanly without local cached artifacts)
	cd $(REPO_ROOT) && uv run pytest $(TEST_FILES) -v

STAGE2B_TEST_FILES := tests/test_stage2b_corruption.py tests/test_stage2b_encoder_gate.py \
                      tests/test_stage2b_ridge.py tests/test_stage2b_stats.py \
                      tests/test_stage2b_cnn.py tests/test_stage2b_partition.py \
                      tests/test_stage2b_gcs.py tests/test_stage2b_gcs_roundtrip.py

.PHONY: stage2b-test
stage2b-test:  ## Run the Stage 2B test suite (fast only; the Colab round trip is excluded)
	cd $(REPO_ROOT) && uv run pytest $(STAGE2B_TEST_FILES) -m "not slow" -q

# The round trip is the only Stage 2B test that leaves this machine: it
# provisions a real Colab CPU runtime, writes an object to GCS from it,
# and reads that object back here both with credentials and anonymously.
# It bills while running (seconds, on CPU) and needs the service-account
# key, so it is `slow`-marked and excluded from every other target.
# `-s` is deliberate, not a debugging leftover -- the step-by-step
# evidence is most of what this test is for (CLAUDE.md principle 20).
BONSAI_GCS_CREDENTIALS ?= $(HOME)/.config/colab-cli/bonsai-colab-storage-key.json

.PHONY: stage2b-test-roundtrip
stage2b-test-roundtrip:  ## Real Colab+GCS round trip -- provisions a CPU runtime, bills while running
	cd $(REPO_ROOT) && BONSAI_GCS_CREDENTIALS="$(BONSAI_GCS_CREDENTIALS)" \
		uv run --group gpu pytest tests/test_stage2b_gcs_roundtrip.py -m slow -s

.PHONY: test
test:  ## Run the whole default suite (every stage, slow reproduction checks excluded)
	cd $(REPO_ROOT) && uv run pytest tests/ -m "not slow" -q

##@ Stage 2B verification against real infrastructure

STAGE2B_DIR := $(REPO_ROOT)/experiments/stage2b_denoising
SESSION_2B_VERIFY ?= stage2b-verify
# T4 has no TF32 hardware (Ampere and later only), so a T4 pass cannot
# answer the reduced-precision question for the A100 the pipeline
# actually targets. Overridable so both can be checked.
VERIFY_GPU ?= A100

# DESIGN.md specifies the ridge equivalence gate (JAX SVD vs sklearn,
# max abs clipped-prediction difference <= 1e-8 and identical alpha
# selection) at the 1,000- and 5,000-image ladder stages -- but every run
# of it so far has been on CPU, because that is all this machine has.
# Whether JAX's float64 SVD on a GPU meets the same gate is a separate
# question from whether the code is right, and it is the question that
# matters before a ladder rung is ever driven on one.
# NOTE, load-bearing: the verdict comes from the script's own success
# sentinel, not from the exec exit status alone. Since 0.2.0 `mighty-colab
# exec` does propagate an uncaught remote exception as a non-zero exit (it
# always exited 0 before that, which is why the sentinel was introduced),
# but an exit code still cannot distinguish "ran and passed" from "exited
# cleanly without ever reaching its verdict" -- a truncated or short-
# circuited script exits 0 either way. So both GPU targets below capture
# the output, tear the session down unconditionally, and require BOTH a
# zero exit and the sentinel. Chaining `&& stop` on the exec's exit status
# is the trap that made `stage2a-verify` a no-op gate and, once exec could
# fail, would have left a billing A100 running on every failure.
.PHONY: stage2b-verify-gpu
stage2b-verify-gpu:  ## Run the ridge equivalence gate on a real GPU -- bills while running
	rc=0; src=0; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	if $(MIGHTY_COLAB) status -s $(SESSION_2B_VERIFY) 2>&1 | grep -q "not found"; then \
		$(MIGHTY_COLAB) new -s $(SESSION_2B_VERIFY) --gpu $(VERIFY_GPU); \
	else \
		echo "[make] Reusing existing session $(SESSION_2B_VERIFY)"; \
	fi && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_VERIFY) stage2b_ridge.py /content/stage2b_ridge.py && \
	rc=0; out=$$($(MIGHTY_COLAB) exec -s $(SESSION_2B_VERIFY) -f stage2b_verify_gpu.py --timeout $(EXEC_TIMEOUT) 2>&1) || rc=$$?; \
	echo "$$out"; \
	src=0; $(MIGHTY_COLAB) stop -s $(SESSION_2B_VERIFY) || src=$$?; \
	if [ $$rc -ne 0 ] || ! echo "$$out" | grep -q GPU_VERIFY_OK; then \
		echo "[make] FAILED: the GPU ridge gate did not report success (exec rc=$$rc)."; \
		if [ $$rc -eq 0 ]; then rc=1; fi; \
	fi; \
	$(call check_teardown,$(SESSION_2B_VERIFY)); \
	exit $$rc

# The ridge GPU check above says nothing about the CNN: ridge is float64
# end to end and therefore immune to reduced-precision effects, while the
# CNN is float32 and runs convolutions through XLA, which may select a
# TF32-class path by default. With min_delta=0.0 and strict `<` early
# stopping, that would silently move best_epoch, seed selection and the
# reported MSE -- so the forward pass is compared CPU-vs-GPU directly.
.PHONY: stage2b-verify-cnn-gpu
stage2b-verify-cnn-gpu:  ## Compare the CNN float32 forward pass CPU vs GPU -- bills while running
	rc=0; src=0; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	if $(MIGHTY_COLAB) status -s $(SESSION_2B_VERIFY) 2>&1 | grep -q "not found"; then \
		$(MIGHTY_COLAB) new -s $(SESSION_2B_VERIFY) --gpu $(VERIFY_GPU); \
	else \
		echo "[make] Reusing existing session $(SESSION_2B_VERIFY)"; \
	fi && \
	$(MIGHTY_COLAB) install -s $(SESSION_2B_VERIFY) equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_VERIFY) stage2b_cnn.py /content/stage2b_cnn.py && \
	rc=0; out=$$($(MIGHTY_COLAB) exec -s $(SESSION_2B_VERIFY) -f stage2b_verify_cnn_gpu.py --timeout $(EXEC_TIMEOUT) 2>&1) || rc=$$?; \
	echo "$$out"; \
	src=0; $(MIGHTY_COLAB) stop -s $(SESSION_2B_VERIFY) || src=$$?; \
	if [ $$rc -ne 0 ] || ! echo "$$out" | grep -q CNN_GPU_VERIFY_OK; then \
		echo "[make] FAILED: the CNN GPU check did not report success (exec rc=$$rc)."; \
		if [ $$rc -eq 0 ]; then rc=1; fi; \
	fi; \
	$(call check_teardown,$(SESSION_2B_VERIFY)); \
	exit $$rc

.PHONY: stage2b-smoke-gcs
stage2b-smoke-gcs:  ## Real-bucket GCS smoke check: transport, chunked resumable upload, both delete refusals
	cd $(REPO_ROOT) && uv run --group gpu python $(STAGE2B_DIR)/smoke_stage2b_gcs.py

.PHONY: help
help:  ## List every target in this file, grouped by section
	@awk 'BEGIN {FS = ":.*##"} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
