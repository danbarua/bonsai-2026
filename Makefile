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

# Overridable for the same reason MIGHTY_COLAB is. The ladder target's
# pre-flight refusals (dirty tree, unpushed HEAD) are behaviour worth
# testing, and testing them needs a git that can be made to report either
# answer -- otherwise those tests would pass or fail depending on the state
# of whoever's checkout runs them, and the leak-handling cases below could
# not run at all on a dirty tree. `tests/test_mighty_colab_contract.py`
# drives both directions through a stub. Note this is deliberately NOT an
# escape hatch in the recipe: there is no flag that skips the checks, only
# a different git to ask.
GIT ?= git

# The pre-flight refusal that gates every GPU target on the DRIVER'S OWN
# SOURCE CLOSURE being committed, replacing the whole-tree `git status
# --porcelain` check the targets used to carry.
#
# Why the coarse check went: its own refusal message defeated it. "The
# runtime fetches one pinned commit; uncommitted work would not be in it"
# is an argument about code that reaches the computation -- and the remote
# executes that pinned commit by construction, so uncommitted work
# ELSEWHERE in the tree cannot reach it. What can is a file in the
# driver's import closure differing from HEAD, which porcelain reports as
# one line among many with no way to tell it apart from an editor's
# leftovers or a second concurrent effort's scratch. The stage-3
# regeneration is the case that separated them: closure clean, tree dirty
# with four unrelated paths, and the run correct to proceed. Keeping both
# would have meant the coarse one fires first and the sharp one is dead
# code, while every GPU launch waits on a spotless tree.
#
# The check is a CLI entry into `stage2b_fingerprint`, not shell: one
# definition of "dirty", owned by the module that already implements it
# blob-by-blob against HEAD and has the tests for it. A shell
# reimplementation is the reimplemented-helper failure CLAUDE.md
# principle 16 names.
#
# Overridable for exactly the reason `GIT` is -- the refusal is behaviour
# worth testing in both directions, and `tests/test_mighty_colab_contract.py`
# stubs this to drive them. Not an escape hatch: there is no flag that
# skips the check, only a different checker to ask.
#
# Whole-tree state is still RECORDED -- the fingerprint captures it in
# every artifact's manifest -- and never enforced.
CLOSURE_CHECK ?= uv run python $(STAGE2B_DIR)/stage2b_fingerprint.py --check-closure
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

# Must list every tests/test_stage2b_*.py on disk. An explicit list is
# what lets `stage2b-test` stay fast and stable while `tests/` grows, but
# it also means a new file is covered by the whole-suite target and
# silently skipped by this one -- which is how `test_stage2b_contracts.py`
# and `test_stage2b_gcs_makefile.py` both came to be missing here.
# `test_stage2b_gcs_makefile.py` asserts this list is complete rather than
# leaving that to whoever adds the next file.
STAGE2B_TEST_FILES := tests/test_stage2b_corruption.py tests/test_stage2b_encoder_gate.py \
                      tests/test_stage2b_ridge.py tests/test_stage2b_stats.py \
                      tests/test_stage2b_cnn.py tests/test_stage2b_partition.py \
                      tests/test_stage2b_contracts.py tests/test_stage2b_gcs.py \
                      tests/test_stage2b_gcs_makefile.py \
                      tests/test_stage2b_gcs_roundtrip.py \
                      tests/test_stage2b_ladder_stage1.py \
                      tests/test_stage2b_ladder_stage2.py \
                      tests/test_stage2b_fingerprint.py \
                      tests/test_stage2b_negative_path_evidence.py \
                      tests/test_stage2b_encode_stage3_local.py \
                      tests/test_stage2b_compare_stage3.py

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

# The bucket every Stage 2B artifact lands in. `stage2b_gcs.bucket_name()`
# reads this from the environment and falls back to its own default, so
# the two must agree -- `tests/test_stage2b_gcs_makefile.py` asserts they
# do rather than trusting this comment. Override it to point a run at a
# scratch bucket without editing any Python:
#
#     make stage2b-smoke-gcs BONSAI_GCS_BUCKET=some-other-bucket
#
BONSAI_GCS_BUCKET ?= bonsai-2026-stage2b-cache

# Every target below that reaches GCS passes both of these explicitly.
# Exporting them from a single place is the point of the rename that
# created them: the bucket name lived in three files and a test pinned the
# wrong one of them.
GCS_ENV := BONSAI_GCS_CREDENTIALS="$(BONSAI_GCS_CREDENTIALS)" \
           BONSAI_GCS_BUCKET="$(BONSAI_GCS_BUCKET)"

# The same two settings for a script that runs on a Colab runtime instead of
# here. `GCS_ENV` sets them in the LOCAL make shell, which a remote kernel
# never sees, so a target that execs a GCS-touching script needs this form
# instead -- `mighty-colab exec --env` sets them in the remote kernel. The
# credentials value differs deliberately: it is the path the key was
# uploaded TO on the runtime, not the local key's path.
# `tests/test_stage2b_gcs_makefile.py` accepts this form only for recipes
# that actually exec, so a locally-run script cannot satisfy the
# bucket-export requirement with it.
REMOTE_KEY_PATH ?= /content/bonsai-colab-storage-key.json
GCS_EXEC_ENV := --env BONSAI_GCS_BUCKET="$(BONSAI_GCS_BUCKET)" \
                --env BONSAI_GCS_CREDENTIALS="$(REMOTE_KEY_PATH)"

.PHONY: stage2b-test-roundtrip
stage2b-test-roundtrip:  ## Real Colab+GCS round trip -- provisions a CPU runtime, bills while running
	cd $(REPO_ROOT) && $(GCS_ENV) \
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
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/smoke_stage2b_gcs.py

##@ Stage 2B feasibility ladder

SESSION_2B_LADDER ?= stage2b-ladder
# Stated rather than inherited from VERIFY_GPU: stage 1 runs no CNN and its
# ridge is float64 end to end, so the TF32 question VERIFY_GPU exists to
# pin down is immaterial here. That makes this a free choice, and a free
# choice should be visible.
LADDER_GPU ?= A100

# `datasets/` is gitignored, so the ladder driver's clone of the repo
# carries the pipeline but none of its inputs -- and 47MB of IDX files is
# far past the Colab session upload ceiling this project has already hit
# once. Stage them to the bucket from here instead, once; the driver
# downloads them on the runtime. All four files, because load_mnist opens
# the t10k pair unconditionally and topology construction goes through it.
.PHONY: stage2b-stage-inputs
stage2b-stage-inputs:  ## Upload the four KMNIST IDX files to the Stage 2B bucket (once; local -> GCS)
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/stage_kmnist_inputs.py

# Feasibility-ladder stage 1 (n=1,000): the first run that joins the Stage
# 2B modules together. Same verdict discipline as the two verify targets
# above -- capture the output, tear the session down unconditionally, and
# require BOTH a zero exit and the driver's own sentinel, because an exit
# code cannot distinguish "ran and passed" from "exited before reaching its
# verdict".
#
# Two refusals before any money is spent. The runtime fetches ONE pinned
# commit from the public repo, so an uncommitted file THIS DRIVER IMPORTS,
# or an unpushed HEAD, would run code that is not the code being tested --
# and the failure would look like a science result rather than a mistake.
# The first refusal is closure-keyed (see CLOSURE_CHECK): it asks whether
# the driver's own import closure is committed, not whether the repository
# is tidy. The driver hashes the clone's copy of itself against
# BONSAI_DRIVER_SHA256 computed here, which is what closes the gap that
# `exec --file` transmits code with no __file__ to check.
.PHONY: stage2b-ladder-stage1
stage2b-ladder-stage1:  ## Run Stage 2B ladder stage 1 (n=1,000) on a Colab GPU -- bills while running
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/run_ladder_stage1.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	driver_sha=$$(shasum -a 256 $(STAGE2B_DIR)/run_ladder_stage1.py | cut -d' ' -f1); \
	echo "[make] commit $$commit, driver sha256 $$driver_sha"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	if $(MIGHTY_COLAB) status -s $(SESSION_2B_LADDER) 2>&1 | grep -q "not found"; then \
		$(MIGHTY_COLAB) new -s $(SESSION_2B_LADDER) --gpu $(LADDER_GPU); \
	else \
		echo "[make] Reusing existing session $(SESSION_2B_LADDER)"; \
	fi && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_LADDER) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_LADDER) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB) exec -s $(SESSION_2B_LADDER) -f run_ladder_stage1.py --timeout $(EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1 2>&1) || rc=$$?; \
	echo "$$out"; \
	src=0; $(MIGHTY_COLAB) stop -s $(SESSION_2B_LADDER) || src=$$?; \
	if [ $$rc -ne 0 ] || ! echo "$$out" | grep -q STAGE1_OK; then \
		echo "[make] FAILED: ladder stage 1 did not report success (exec rc=$$rc)."; \
		if [ $$rc -eq 0 ]; then rc=1; fi; \
	fi; \
	$(call check_teardown,$(SESSION_2B_LADDER)); \
	exit $$rc

# Feasibility-ladder stage 2 (n=5,000): adds runtime/feature-validity
# measurement at scale, the production SVD's own condition-number
# diagnostic, ridge-grid behaviour, the ladder's second real-data ridge
# equivalence gate, and the first CNN training against real data. Own
# session, distinct from stage 1's -- this target's unconditional teardown
# must not be able to kill a session stage 1 still expects to be running.
# equinox/optax join the reinstall line here only: stage 1 has no CNN and
# does not need them, so its own target is left untouched.
SESSION_2B_LADDER2 ?= stage2b-ladder2

# Ladder stage 3, PHASE A: encode the 54,000-image fit side HERE, on CPU,
# and write only the encoded array to GCS. No Colab session is involved
# and nothing bills.
#
# Split out from the GPU phase deliberately. Encoding is the one CPU-bound
# step in the pipeline; evolution, ridge and the CNN are what actually use
# the A100. Running the encode inside a provisioned GPU session would leave
# a metered A100 idle for the majority of the run's wall-clock. Measured,
# not assumed: this machine encodes ~20x faster in wall-clock than the
# Colab CPU would single-worker (~3.4x per core, times 9 workers), so
# stage 3's fit side lands in ~10 minutes here.
#
# This does not contradict DESIGN.md's "generate in the cloud" convention
# -- that constraint's own stated reason is the Colab session UPLOAD
# limit, which a direct Mac->GCS write never touches. `stage2b-stage-inputs`
# already writes to the bucket from here on exactly the same transport.
.PHONY: stage2b-encode-stage3-local
stage2b-encode-stage3-local:  ## Stage 3 Phase A: encode all 60,000 training images locally on CPU, push to GCS (free, no session)
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/encode_stage3_local.py

# The acceptance test for the regeneration above, and a separate target
# because it is a separate claim: that the 54,000 images the previous
# Phase A run encoded come back bit-exact, joined by official index. Reads
# only -- it downloads two artifacts and writes nothing to the bucket.
.PHONY: stage2b-compare-stage3
stage2b-compare-stage3:  ## Verify the stage-3 regeneration against the 54,000-image baseline (read-only)
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/compare_stage3_regeneration.py \
			--json-out $(STAGE2B_DIR)/results/stage3_regeneration_acceptance.json

.PHONY: stage2b-ladder-stage2
stage2b-ladder-stage2:  ## Run Stage 2B ladder stage 2 (n=5,000, CNN development) on a Colab GPU -- bills while running
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/run_ladder_stage2.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	driver_sha=$$(shasum -a 256 $(STAGE2B_DIR)/run_ladder_stage2.py | cut -d' ' -f1); \
	echo "[make] commit $$commit, driver sha256 $$driver_sha"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	if $(MIGHTY_COLAB) status -s $(SESSION_2B_LADDER2) 2>&1 | grep -q "not found"; then \
		$(MIGHTY_COLAB) new -s $(SESSION_2B_LADDER2) --gpu $(LADDER_GPU); \
	else \
		echo "[make] Reusing existing session $(SESSION_2B_LADDER2)"; \
	fi && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_LADDER2) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_LADDER2) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB) exec -s $(SESSION_2B_LADDER2) -f run_ladder_stage2.py --timeout $(EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1 2>&1) || rc=$$?; \
	echo "$$out"; \
	src=0; $(MIGHTY_COLAB) stop -s $(SESSION_2B_LADDER2) || src=$$?; \
	if [ $$rc -ne 0 ] || ! echo "$$out" | grep -q STAGE2_OK; then \
		echo "[make] FAILED: ladder stage 2 did not report success (exec rc=$$rc)."; \
		if [ $$rc -eq 0 ]; then rc=1; fi; \
	fi; \
	$(call check_teardown,$(SESSION_2B_LADDER2)); \
	exit $$rc

.PHONY: help
help:  ## List every target in this file, grouped by section
	@awk 'BEGIN {FS = ":.*##"} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
