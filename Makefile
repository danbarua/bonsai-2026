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

# Global flags must precede the subcommand (`mighty-colab --json exec`, not
# `mighty-colab exec --json`), so the JSON form is a separate variable rather
# than a flag appended at each call site. Used only where a recipe INSPECTS
# the result; `sessions`/`upload`/`install`/`reinstall`/`download` stay plain,
# because `--json` implies `--logtostderr` and nothing here parses them.
#
# What `--json` buys, concretely -- each of these was a real defect in the
# text-parsing form it replaces:
#
#   1. The CLI's exit code now answers only "did the client complete its
#      transaction", and the remote job's own outcome moved into the
#      envelope's `status` (`ok` / `job_raised` / `error`). A script ending
#      in `SystemExit(0)` -- which IPython reports as an exception -- used to
#      come back rc=1 and be treated as a failed run. That cost this project
#      a completed 13-minute A100 result, discarded with the session.
#   2. `status -s <name>` on a missing session exits 1 with
#      `reason=session_not_found` instead of exiting 0 and printing "not
#      found." to stdout. The old guard grepped that prose, so an auth or
#      network failure ALSO read as "not found" and provisioned a second
#      session. The reason code distinguishes them; $(ensure_session) refuses
#      rather than guessing.
#   3. `stop` on an absent session returns `status=ok reason=already_stopped`,
#      which is what STOP_ABSENT_RC previously encoded as a bare exit code.
#
# Verified against 0.4.1 directly, no session provisioned:
#   status -s <missing> -> rc=1 status=error  reason=session_not_found
#   stop   -s <missing> -> rc=0 status=ok     reason=already_stopped
#   exec   -s <missing> -> rc=1 status=error  reason=session_not_found
# Pinned by tests/test_mighty_colab_contract.py.
MIGHTY_COLAB_JSON ?= $(MIGHTY_COLAB) --json

# Every envelope-inspecting recipe needs this. It is already installed in the
# CI image; tests/test_ci_image_dependencies.py derives that requirement from
# the Makefile as well as tools/ci/*.sh, so removing it from either fails
# there rather than at the point a GPU target is next driven.
JQ ?= jq

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
# `already_stopped` is the reason code 0.4.1 returns for the absent case,
# and it arrives as DATA rather than as an exit-code convention -- which is
# what STOP_ABSENT_RC used to encode, and why that variable is gone. A
# release that changed absent's exit code used to require re-pointing a
# Makefile variable; now the envelope says which case it is regardless.
#
# $(1) = session name. Sets $$src (stop's own CLI exit status) and
# $$sreason, both read by $(check_teardown) immediately below. Deliberately
# two steps rather than one: teardown must run unconditionally, and its
# verdict must be evaluated AFTER the run's own verdict is already in $$rc.
define stop_session
src=0; \
sout=$$($(MIGHTY_COLAB_JSON) stop -s $(1)) || src=$$?; \
sreason=$$(printf '%s' "$$sout" | $(JQ) -r '.status // "malformed"')
endef

# Evaluated after teardown, with $$src holding stop's CLI status, $$sreason
# the envelope's own verdict, and $$rc the run's verdict so far. A leak
# fails the target, but never overwrites a verdict that already failed --
# the science's failure is the more useful headline, and the leak is
# reported on its own line regardless.
#
# BOTH conditions are checked, and neither is redundant: a non-zero exit
# catches the CLI failing to complete its transaction at all (network drop,
# crash) where there is no envelope to read, and a non-`ok` status catches a
# transaction that completed while reporting the teardown itself failed.
# Treating either alone as sufficient is how a billing A100 goes unnoticed.
define check_teardown
if [ $$src -ne 0 ] || [ "$$sreason" != "ok" ]; then \
	echo "[make] LEAK WARNING: teardown of session '$(1)' exited $$src (status=$$sreason) -- it may still be running and billing."; \
	echo "[make]   check with: $(MIGHTY_COLAB) sessions"; \
	echo "[make]   stop it with: $(MIGHTY_COLAB) stop -s $(1)"; \
	if [ $$rc -eq 0 ]; then if [ $$src -ne 0 ]; then rc=$$src; else rc=1; fi; fi; \
fi
endef

# GPU-target idempotency: `mighty-colab new -s <name>` provisions a fresh
# session unconditionally, so re-running a GPU target after a partial
# failure (a dropped upload, a flaky exec) would try to allocate a second
# session under the same name instead of resuming the one already up. So
# `new` is called only when a session by that name genuinely isn't there.
#
# The THREE-way branch is the point, and it is what the prose-grepping
# version it replaces could not express. That guard asked whether "not
# found" appeared in `status`'s combined output, which conflates two
# opposite situations: the session is genuinely absent (provision one), and
# the question could not be answered at all -- expired credentials, a
# network failure, a backend 5xx (provisioning here is exactly wrong, and
# would allocate a second billable VM while the first is still up). The
# envelope separates them: `reason=session_not_found` is the absent case
# specifically, and anything else that isn't `ok` is refused loudly rather
# than guessed at.
#
# $(1) = session name, $(2) = flags passed to `new` (e.g. `--gpu A100`).
define ensure_session
st=$$($(MIGHTY_COLAB_JSON) status -s $(1) 2>/dev/null); \
sst=$$(printf '%s' "$$st" | $(JQ) -r '.status // "malformed"'); \
srsn=$$(printf '%s' "$$st" | $(JQ) -r '.reason // ""'); \
if [ "$$sst" = "ok" ]; then \
	echo "[make] Reusing existing session $(1)"; \
elif [ "$$srsn" = "session_not_found" ]; then \
	$(MIGHTY_COLAB) new -s $(1) $(2); \
else \
	echo "[make] REFUSING: cannot determine whether session '$(1)' exists (status=$$sst reason=$$srsn)."; \
	echo "[make]   Provisioning now could allocate a SECOND billable VM alongside one already running."; \
	echo "[make]   Check credentials and connectivity, then: $(MIGHTY_COLAB) sessions"; \
	exit 1; \
fi
endef

# The run's own verdict, from the envelope captured in $$out. Sets $$rc.
#
# $(1) = human-readable description, $(2) = the driver's success sentinel, or
# EMPTY for a driver that prints none (the three Stage 2A evolution targets).
# An empty sentinel checks the envelope status only, which is strictly more
# than those targets checked before -- they trusted `exec`'s exit code alone,
# and that is precisely the signal `--json` moves into the body. Migrating
# them WITHOUT this check would have made them worse, not better: under
# `--json` the CLI exits 0 whenever it completed its transaction, so a raised
# remote job would have read as success. Stage 2A is closed and locked, so
# its drivers are not being edited to add sentinels; the status check is the
# part obtainable without touching them.
#
# The sentinel is NOT made redundant by `--json`, and keeping both is
# deliberate. `status=ok` means the remote code did not raise; it cannot
# distinguish "ran to completion and passed its gate" from "exited cleanly
# without ever reaching its verdict" -- a truncated or short-circuited
# script satisfies `status=ok` either way. What `--json` retires is the
# EXIT-CODE half of the old check, which conflated a raising job with a CLI
# that never ran one, and which read a successful script ending in
# `SystemExit(0)` as a failure.
#
# The sentinel is looked for inside `.blocks[].outputs[]` rather than in raw
# stdout. Those are the remote cell's own outputs, so a sentinel-shaped
# string appearing in the CLI's chatter, in a log line, or in the submitted
# source cannot satisfy the check -- which grepping merged stdout+stderr
# could not rule out.
define check_run_verdict
if [ $$rc -ne 0 ]; then \
	echo "[make] FAILED: $(1) -- the mighty-colab CLI itself exited $$rc (no envelope to read)."; \
else \
	jstat=$$(printf '%s' "$$out" | $(JQ) -r '.status // "malformed"'); \
	jrsn=$$(printf '%s' "$$out" | $(JQ) -r '.reason // ""'); \
	if [ "$$jstat" != "ok" ]; then \
		echo "[make] FAILED: $(1) -- remote job status=$$jstat reason=$$jrsn."; \
		printf '%s' "$$out" | $(JQ) -r '.blocks[]?.outputs[]?.traceback // empty | if type=="array" then join("\n") else . end'; \
		rc=1; \
	elif [ -n "$(2)" ] && ! printf '%s' "$$out" | $(JQ) -e -r '.blocks[]?.outputs[]? | tostring' 2>/dev/null | grep -q '$(2)'; then \
		echo "[make] FAILED: $(1) -- the job completed without raising, but never printed its success sentinel $(2)."; \
		echo "[make]   That is the signature of a script that exited early or was truncated before reaching its verdict."; \
		rc=1; \
	fi; \
fi
endef

# Prints the remote job's own stdout/stderr for a human. Under `--json` the
# captured $$out is an envelope, so the readable text has to be extracted
# rather than echoed -- `echo "$$out"` would print a one-line blob. Both
# nbformat text shapes are handled: `text` is a string in some outputs and a
# list of lines in others, and joining only the list form silently drops the
# other. Falls back to printing the raw envelope if it cannot be parsed at
# all, so a malformed response is never swallowed.
define show_run_output
if printf '%s' "$$out" | $(JQ) -e . >/dev/null 2>&1; then \
	printf '%s' "$$out" | $(JQ) -r '.blocks[]?.outputs[]? | (.text // .traceback // empty) | if type=="array" then join("") else . end'; \
else \
	echo "[make] (unparseable envelope, printing raw)"; \
	echo "$$out"; \
fi
endef

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
	$(call ensure_session,$(SESSION_TRAIN),--gpu A100) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_TRAIN) jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8 && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TRAIN) evolve_on_graph_jax.py /content/evolve_on_graph_jax.py && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TRAIN) scratch/stage3_train/stage3_topologies.pkl /content/stage3_topologies.pkl && \
	for i in 00 01 02 03 04 05 06 07 08 09 10 11; do \
		$(MIGHTY_COLAB) upload -s $(SESSION_TRAIN) scratch/stage3_train/theta0_chunk_$$i.npy /content/theta0_chunk_$$i.npy || exit 1; \
	done && \
	rc=0; \
	out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_TRAIN) -f stage3_gpu_evolve.py --timeout $(EXEC_TIMEOUT)) || rc=$$?; \
	$(call show_run_output); \
	$(call check_run_verdict,stage3_gpu_evolve.py did not complete,); \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(SESSION_TRAIN) /content/stage3_gpu_results.pkl scratch/stage3_train/stage3_gpu_results.pkl || rc=$$?; \
	fi; \
	$(call stop_session,$(SESSION_TRAIN)); \
	$(call check_teardown,$(SESSION_TRAIN)); \
	exit $$rc

.PHONY: stage2a-evolve-test-gpu
stage2a-evolve-test-gpu:  ## Upload + run Stage-4 (official test set) GPU evolution via mighty-colab -- bills while running
	rc=0; src=0; \
	cd $(STAGE2A_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_TEST),--gpu A100) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_TEST) jax[cuda12]==0.11.0 diffrax==0.7.2 equinox==0.13.8 && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TEST) evolve_on_graph_jax.py /content/evolve_on_graph_jax.py && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TEST) scratch/stage4_test/stage4_gpu_upload_topologies.pkl /content/stage4_gpu_upload_topologies.pkl && \
	$(MIGHTY_COLAB) upload -s $(SESSION_TEST) scratch/stage4_test/stage4_theta0_test.npy /content/stage4_theta0_test.npy && \
	rc=0; \
	out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_TEST) -f stage4_gpu_evolve.py --timeout $(EXEC_TIMEOUT)) || rc=$$?; \
	$(call show_run_output); \
	$(call check_run_verdict,stage4_gpu_evolve.py did not complete,); \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(SESSION_TEST) /content/stage4_gpu_results.pkl scratch/stage4_test/stage4_gpu_results.pkl || rc=$$?; \
	fi; \
	$(call stop_session,$(SESSION_TEST)); \
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
	$(call ensure_session,$(SESSION_CLASS0),--gpu A100) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_CLASS0) --requirement cuml_requirements.txt && \
	$(MIGHTY_COLAB) upload -s $(SESSION_CLASS0) stage2a_classifier.py /content/stage2a_classifier.py && \
	$(MIGHTY_COLAB) upload -s $(SESSION_CLASS0) stage2a_stats.py /content/stage2a_stats.py && \
	rc=0; \
	out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_CLASS0) -f class0_support_audit_classify_gpu.py --timeout $(EXEC_TIMEOUT)) || rc=$$?; \
	$(call show_run_output); \
	$(call check_run_verdict,class0_support_audit_classify_gpu.py did not complete,); \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(SESSION_CLASS0) /content/class0_support_audit_classify_results.pkl results/class0_support_audit_classify_results.pkl || rc=$$?; \
	fi; \
	$(call stop_session,$(SESSION_CLASS0)); \
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
                      tests/test_stage2b_compare_stage3.py \
                      tests/test_stage2b_ladder_stage3.py \
                      tests/test_stage2b_ladder_stage4.py \
                      tests/test_stage2b_gate_corpus.py \
                      tests/test_stage2b_audit.py \
                      tests/test_stage2b_audit_driver.py \
                      tests/test_stage2b_abs_conv_eps_sensitivity.py \
                      tests/test_stage2b_arm_x86_propagation.py \
                      tests/test_stage2b_artifact_manifest.py

.PHONY: stage2b-test
stage2b-test:  ## Run the Stage 2B test suite (fast only; the Colab round trip is excluded)
	cd $(REPO_ROOT) && uv run pytest $(STAGE2B_TEST_FILES) -m "not slow" -rs -q

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

# The GCP project, declared here for the same reason the bucket is.
#
# It was a bare literal in `stage2b_gcs.py`, and the test that "pinned" it
# asserted `gcs.GCS_PROJECT == "bonsai-504422"` -- a literal against a copy
# of itself, which can only fail if someone edits the module and forgets the
# test. Anyone changing a project id would grep for the old value and fix
# both. That is exactly what `tools/gates/gate_inventory.py`'s
# `break_demonstrated` field forbids: evidence the test fails when the
# PRODUCTION value changes, "not merely when the constant literal is edited,
# which tests that the literal equals itself".
#
# Declaring it here makes the check a comparison of two INDEPENDENT sources,
# the same shape that has worked for the bucket, and gives `infra/` a third
# source to agree with rather than a fifth copy to drift from.
BONSAI_GCP_PROJECT ?= bonsai-504422

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

.PHONY: stage2b-test-audit-crosscheck
stage2b-test-audit-crosscheck:  ## The audit driver's stage-1/2 historical cross-check against the REAL bucket -- reads only, anonymous, no billing
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu pytest tests/test_stage2b_audit_driver.py -m slow -s

.PHONY: stage2b-generate-artifact-manifest
stage2b-generate-artifact-manifest:  ## Regenerate the committed ARTIFACT_MANIFEST.json from the real bucket -- anonymous read, no credentials, no billing
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/generate_stage2b_artifact_manifest.py

.PHONY: stage2b-test-artifact-manifest
stage2b-test-artifact-manifest:  ## The artifact-manifest generator's test against the REAL bucket -- anonymous, no billing
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu pytest tests/test_stage2b_artifact_manifest.py -m slow -s

# A TARGET MEANS THE SAME THING EVERYWHERE. `test` and `stage2b-test` run
# capability-free, locally and in CI alike, so "green here" and "green in
# CI" are the same claim. That is the whole value of running them locally,
# and a flag that widened one environment and not the other would quietly
# remove it.
#
# It is also what keeps `tools/ci/ci_skip_baseline.txt` meaningful: the
# baseline is a set of skips measured against ONE capability profile, and a
# second profile needs a second baseline nobody can state the environment
# for.
#
# The optional capabilities get their own target below, where the
# requirement is DECLARED by the name a person chose rather than acquired as
# a side effect of a flag on a target CI also invokes.
.PHONY: test
test:  ## Run the whole default suite (every stage, slow reproduction checks excluded)
	cd $(REPO_ROOT) && uv run pytest tests/ -m "not slow" -rs -q

# Everything `test` runs, plus the optional cloud capabilities installed.
#
# The one test this exists for is
# `test_crc32c_agrees_with_google_crc32c_where_it_is_installed`, which pins
# the pure-Python CRC32C fallback against the library GCS actually uses. It
# `importorskip`s `google_crc32c`, so it is SILENT when the group is absent
# -- and a skip is not a failure. Without a target that guarantees the
# group, that cross-check runs nowhere and the guard is dead code.
#
# CI does NOT invoke this, deliberately: `tools/ci/assert_no_cloud_credentials.py`
# fails the build when `google.cloud.storage` is importable, and installing
# the group would trip it. Read that as the two facts agreeing rather than
# conflicting -- CI is credential-free by design, and this target is for a
# machine that already has the capability.
.PHONY: test-capabilities
test-capabilities:  ## Run the default suite with the optional cloud group installed
	cd $(REPO_ROOT) && uv run --group gpu pytest tests/ -m "not slow" -rs -q

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
# NOTE, load-bearing: the verdict comes from BOTH the envelope's `status`
# and the script's own success sentinel, and neither is redundant.
#
# `status` covers what the exit code no longer can. Under `--json` the CLI
# exits 0 whenever it completed its transaction, so a remote job that
# RAISED still returns 0 at the process level -- the failure lives in
# `status=job_raised`. A recipe branching on the exit code alone would read
# a crashed run as a pass.
#
# The sentinel covers what `status` cannot: "exited cleanly without ever
# reaching its verdict". A truncated or short-circuited script satisfies
# `status=ok` exactly as a passing one does.
#
# So every GPU target below captures the envelope, tears the session down
# unconditionally, and requires both. Chaining `&& stop` on the exec's exit
# status is the trap that made `stage2a-verify` a no-op gate and would have
# left a billing A100 running on every failure.
.PHONY: stage2b-verify-gpu
stage2b-verify-gpu:  ## Run the ridge equivalence gate on a real GPU -- bills while running
	rc=0; src=0; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_2B_VERIFY),--gpu $(VERIFY_GPU)) && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_VERIFY) stage2b_ridge.py /content/stage2b_ridge.py && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_VERIFY) -f stage2b_verify_gpu.py --timeout $(EXEC_TIMEOUT)) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_VERIFY)); \
	$(call check_run_verdict,the GPU ridge gate did not report success,GPU_VERIFY_OK); \
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
	$(call ensure_session,$(SESSION_2B_VERIFY),--gpu $(VERIFY_GPU)) && \
	$(MIGHTY_COLAB) install -s $(SESSION_2B_VERIFY) equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_VERIFY) stage2b_cnn.py /content/stage2b_cnn.py && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_VERIFY) -f stage2b_verify_cnn_gpu.py --timeout $(EXEC_TIMEOUT)) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_VERIFY)); \
	$(call check_run_verdict,the CNN GPU check did not report success,CNN_GPU_VERIFY_OK); \
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

# Brings the four already-staged objects under the manifest contract.
# Sidecars only -- no payload bytes move, because the payloads are the same
# IDX files that were uploaded once and have not changed. Each object is
# verified against its local copy before being described, so a manifest
# cannot record a digest for bytes the bucket does not hold.
.PHONY: stage2b-publish-input-manifests
stage2b-publish-input-manifests:  ## Attach manifests to the staged KMNIST objects (no payload upload)
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/stage_kmnist_inputs.py \
			--publish-manifests

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
	$(call ensure_session,$(SESSION_2B_LADDER),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_LADDER) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_LADDER) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_LADDER) -f run_ladder_stage1.py --timeout $(EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_LADDER)); \
	$(call check_run_verdict,ladder stage 1 did not report success,STAGE1_OK); \
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

# Writes a preserved annotation beside a stage-3 run report -- the
# correction that stops a historical STAGE3_OK being read as a passed gate
# when the gate was never implemented. Additive and refuses to replace an
# existing annotation, so re-running is safe. RUN_ID is required; there is
# no default, because annotating the wrong run is worse than not annotating.
#
#     make stage2b-annotate-report RUN_ID=20260808T014606Z
#     make stage2b-annotate-report RUN_ID=... ANNOTATE_ARGS=--dry-run
.PHONY: stage2b-annotate-report
stage2b-annotate-report:  ## Annotate a stage-3 run report (RUN_ID=... required; --dry-run via ANNOTATE_ARGS)
	@test -n "$(RUN_ID)" || { echo "[make] RUN_ID is required, e.g. make stage2b-annotate-report RUN_ID=20260808T014606Z"; exit 1; }
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/annotate_run_report.py \
			--run-id $(RUN_ID) $(ANNOTATE_ARGS)

# Reconciles every binding clause in the frozen Stage 2B protocol corpus
# against the code that enforces it. Local, read-only, no bucket and no
# GPU. Exits non-zero while any candidate is undispositioned or any row is
# incomplete -- so a red exit here is the normal state until the inventory
# is finished, not a broken target.
#
# It goes through gate_corpus.py rather than calling the reconciler with a
# doc list written out here, because WHICH documents are scanned is itself
# load-bearing: 89 is a fact about four specific files, and a doc list in a
# recipe is a hand-maintained set nobody checks. gate_corpus.py asserts its
# corpus against the documents on disk, in both directions, before the
# reconciler runs.
.PHONY: stage2b-gate-inventory
stage2b-gate-inventory:  ## Reconcile the Stage 2B binding-clause inventory against enforcing code (local, read-only)
	cd $(REPO_ROOT) && \
		uv run python $(STAGE2B_DIR)/gate_corpus.py \
			--inventory $(STAGE2B_DIR)/gates.toml $(GATE_ARGS)

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
	$(call ensure_session,$(SESSION_2B_LADDER2),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_LADDER2) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_LADDER2) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_LADDER2) -f run_ladder_stage2.py --timeout $(EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_LADDER2)); \
	$(call check_run_verdict,ladder stage 2 did not report success,STAGE2_OK); \
	$(call check_teardown,$(SESSION_2B_LADDER2)); \
	exit $$rc

SESSION_2B_LADDER3 ?= stage2b-ladder3

# Ladder stage 3, PHASE B: the full 60,000-image corpus on an A100.
#
# `EXEC_TIMEOUT` is overridden here rather than left at the 3600s default,
# and the reason is measured rather than precautionary. Stage 2's recorded
# `8_ridge` is 305.53s at n=5,000, dominated not by the JAX SVD but by
# `sklearn_ridge_predict` -- `Ridge(solver="svd")` once per alpha on the
# CPU, 315 oracle fits against 35 production ones. The SVD of an (n x 1008)
# matrix is linear in n for n >> p, so ridge alone projects to ~3,700s at
# 12x scale. With evolution (~490s), features (~560s) and the CNN (~1,200s)
# the projected total is ~5,900s of compute before bootstrap, install and a
# 229MB download -- so the default would time out mid-ridge ON A HEALTHY
# RUN. The driver's own step-2b sizing probe is what actually gates the
# spend; this only stops the harness from killing a run that is fine.
STAGE3_EXEC_TIMEOUT ?= 10800

.PHONY: stage2b-ladder-stage3
stage2b-ladder-stage3:  ## Run Stage 2B ladder stage 3 Phase B (n=60,000) on a Colab GPU -- bills while running
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/run_ladder_stage3.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	driver_sha=$$(shasum -a 256 $(STAGE2B_DIR)/run_ladder_stage3.py | cut -d' ' -f1); \
	echo "[make] commit $$commit, driver sha256 $$driver_sha"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_2B_LADDER3),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_LADDER3) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_LADDER3) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_LADDER3) -f run_ladder_stage3.py --timeout $(STAGE3_EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_LADDER3)); \
	$(call check_run_verdict,ladder stage 3 did not report success,STAGE3_OK); \
	$(call check_teardown,$(SESSION_2B_LADDER3)); \
	exit $$rc

SESSION_2B_BACKFILL ?= stage2b-cnn-backfill
# Stage 3 measured the whole three-seed train at 639.6s on an A100
# (`wallclock_s_per_seed` in cnn_production.npz: 228.4 / 219.2 / 192.0).
# 3600 covers that plus the clone, the pip install and the corpus download,
# without inheriting stage 3's 10800 -- this trains and uploads one object,
# it does not run a ladder.
BACKFILL_EXEC_TIMEOUT ?= 3600

.PHONY: stage2b-backfill-cnn-weights
stage2b-backfill-cnn-weights:  ## Add the trained CNN weights to stage 3's cnn_production.npz on a Colab GPU -- bills while running
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/backfill_cnn_weights.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	driver_sha=$$(shasum -a 256 $(STAGE2B_DIR)/backfill_cnn_weights.py | cut -d' ' -f1); \
	echo "[make] commit $$commit, driver sha256 $$driver_sha"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_2B_BACKFILL),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_BACKFILL) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_BACKFILL) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_BACKFILL) -f backfill_cnn_weights.py --timeout $(BACKFILL_EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1 --env BONSAI_GPU="$(LADDER_GPU)" $(BACKFILL_EXTRA_ENV)) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_BACKFILL)); \
	$(call check_run_verdict,the CNN weights backfill did not report success,BACKFILL_OK); \
	$(call check_teardown,$(SESSION_2B_BACKFILL)); \
	exit $$rc

.PHONY: stage2b-backfill-cnn-weights-dry
stage2b-backfill-cnn-weights-dry:  ## The same run, stopping before the upload -- bills while running
	$(MAKE) stage2b-backfill-cnn-weights BACKFILL_EXTRA_ENV='--env BONSAI_BACKFILL_DRYRUN=1'

SESSION_2B_CNNARCH ?= stage2b-cnn-arch
# A forward pass over 512 images on a 9,857-parameter net. The clone, the
# pip install and two public-read fetches dominate; the compute is
# seconds.
CNNARCH_EXEC_TIMEOUT ?= 1800
CNNARCH_REMOTE_OUT ?= /content/cnn_forward_x86.npz
CNNARCH_LOCAL_OUT ?= $(STAGE2B_DIR)/results/cnn_forward_x86.npz

# The gauge-sensitivity comparison Stage 2A pre-registered and never ran,
# executed against Stage 2B. Interpretations are fixed in advance in
# GAUGE_COMPARISON_PREREGISTRATION.md; read that before reading a result.
#
# No GPU EVOLUTION happens here -- the evolved states are already persisted
# as `stage3/*/theta_T.npz` and are consumed read-only over public HTTPS.
# What justifies a remote run is the ridge cross-validation: 13 alphas x 5
# folds x 5 conditions x 2 gauges at n=60,000, which is the stage this
# project has already measured to dominate runtime (principle 18 -- the
# stage that was "a few seconds" at n=1,000 and 79x at full scale).
#
# The driver writes its own JSON summary; the target downloads it. Outputs
# are RUN_SCOPED under a name of this experiment's own, never near the
# lineage stage3/common paths.
GAUGE_SESSION ?= stage2b-gauge
GAUGE_EXEC_TIMEOUT ?= 5400
GAUGE_REMOTE_OUT ?= /content/gauge_comparison.json
GAUGE_LOCAL_OUT ?= $(STAGE2B_DIR)/results/gauge_comparison.json
GAUGE_STAGE ?= 3

.PHONY: stage2b-gauge-comparison
stage2b-gauge-comparison:  ## Run the pre-registered gauge comparison on Colab -- bills while running
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/run_gauge_comparison.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	echo "[make] commit $$commit, stage $(GAUGE_STAGE)"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(GAUGE_SESSION),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(GAUGE_SESSION) jax[cuda12]==0.11.0 scikit-learn && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(GAUGE_SESSION) -f run_gauge_comparison.py --timeout $(GAUGE_EXEC_TIMEOUT) --env BONSAI_COMMIT="$$commit" --env JAX_ENABLE_X64=1 --env GAUGE_STAGE=$(GAUGE_STAGE) --env GAUGE_OUT="$(GAUGE_REMOTE_OUT)" --env GAUGE_CACHE_DIR=/content/gauge_cache) || rc=$$?; \
	$(call show_run_output); \
	$(call check_run_verdict,the gauge comparison did not report success,GAUGE_OK); \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(GAUGE_SESSION) $(GAUGE_REMOTE_OUT) $(GAUGE_LOCAL_OUT) || rc=$$?; \
	fi; \
	$(call stop_session,$(GAUGE_SESSION)); \
	$(call check_teardown,$(GAUGE_SESSION)); \
	exit $$rc

.PHONY: stage2b-cnn-arch-x86
stage2b-cnn-arch-x86:  ## CNN forward pass on Colab x86, downloaded for comparison -- bills while running
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/measure_cnn_arch_agreement.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	echo "[make] commit $$commit"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_2B_CNNARCH),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_CNNARCH) jax[cuda12]==0.11.0 diffrax==0.7.2 equinox optax && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_CNNARCH) -f measure_cnn_arch_agreement.py --timeout $(CNNARCH_EXEC_TIMEOUT) --env BONSAI_COMMIT="$$commit" --env JAX_ENABLE_X64=1 --env CNN_ARCH_PHASE=run --env CNN_ARCH_OUT="$(CNNARCH_REMOTE_OUT)" --env CNN_ARCH_CACHE_DIR=/content/cnn_arch_cache) || rc=$$?; \
	$(call show_run_output); \
	$(call check_run_verdict,the x86 forward pass did not report success,CNN_ARCH_OK); \
	if [ $$rc -eq 0 ]; then \
		$(MIGHTY_COLAB) download -s $(SESSION_2B_CNNARCH) $(CNNARCH_REMOTE_OUT) $(CNNARCH_LOCAL_OUT) || rc=$$?; \
	fi; \
	$(call stop_session,$(SESSION_2B_CNNARCH)); \
	$(call check_teardown,$(SESSION_2B_CNNARCH)); \
	exit $$rc

.PHONY: stage2b-cnn-arch-compare
stage2b-cnn-arch-compare:  ## Compare the ARM and x86 forward passes (free, local)
	cd $(STAGE2B_DIR) && \
	uv run --group gpu python measure_cnn_arch_agreement.py --phase compare \
		--a results/cnn_forward_arm.npz --b results/cnn_forward_x86.npz \
		--json-out results/cnn_arch_agreement.json

SESSION_2B_LADDER4 ?= stage2b-ladder4

# `EXEC_TIMEOUT` override, by the same reasoning as stage 3's. No stage-4
# run has happened -- this is not itself a measurement -- but two of its
# legs ARE grounded in real numbers pulled from stage 3's own committed
# artifacts (`gsutil cat` against the public-read bucket, no session
# needed) rather than a fresh guess:
#   - CNN (3 seeds, 54,000/6,000 fit/validation, A100): stage 3's own
#     `stage3_report_20260807T155651Z.json` records total_wallclock_s =
#     639.6s (228.4/219.2/192.0 per seed) -- this driver retrains from
#     scratch a second time (see run_ladder_stage4.py's module docstring)
#     at the SAME scale, so that recorded total is the right anchor, not
#     the ~1,200s FINDINGS.md projection table itself flags as unvalidated
#     ("CNN cost scales with epochs x batches, not simply n").
#   - Ridge (7 conditions, refit on the full 60,000-row training scale):
#     the SAME report's "7_ridge" step (recomputing at production scale
#     under the amended thirteen-decade grid) took 1,716.3s -- this driver
#     performs the same operation once more, so budgeted at that figure
#     rather than a fresh estimate.
# Evolution and features are population-scaled from that report's own
# "5_evolution" (393.5s) and "6_features" (513.8s) legs, at 10,000/60,000
# of the population. Stage 3's TRAIN-side artifacts (corpus, five features
# arrays, ridge_final, cnn_production) downloaded fresh into this session
# are unmeasured but bounded by the ~3.4GB stage 3 itself uploaded. Summed
# and rounded up generously, 7200s leaves a wide margin; it is a harness
# safety net, not a scientific tolerance, exactly as stage 3's own comment
# states -- the driver's own halts are what actually gate correctness.
STAGE4_EXEC_TIMEOUT ?= 7200

# A second, EXPLICIT confirmation beyond typing the command, because this
# target is different in kind from stages 1-3: it is the one-shot official
# result (AUDIT_PROTOCOL.md: "Stage 4 stays blocked behind the package
# review and explicit release"), and `refuse_if_official_result_exists`
# only protects against a SECOND run, not a first one launched before the
# package review has actually happened. This is a structural speed bump,
# not a substitute for that review.
.PHONY: stage2b-ladder-stage4
stage2b-ladder-stage4:  ## Run Stage 2B ladder stage 4, the ONE locked evaluation on the official test corpus -- bills while running, and requires STAGE4_RELEASE_CONFIRMED=1
	@if [ "$(STAGE4_RELEASE_CONFIRMED)" != "1" ]; then \
		echo "[make] REFUSING: this is the one-shot official Stage 4 evaluation on the"; \
		echo "[make] official KMNIST test corpus. AUDIT_PROTOCOL.md requires the"; \
		echo "[make] pre-Stage-4 package review plus Dan's explicit release before this"; \
		echo "[make] runs. Re-invoke as: STAGE4_RELEASE_CONFIRMED=1 make stage2b-ladder-stage4"; \
		echo "[make] only once both of those have actually happened."; \
		exit 1; \
	fi
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/run_ladder_stage4.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	driver_sha=$$(shasum -a 256 $(STAGE2B_DIR)/run_ladder_stage4.py | cut -d' ' -f1); \
	echo "[make] commit $$commit, driver sha256 $$driver_sha"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_2B_LADDER4),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_LADDER4) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_LADDER4) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_LADDER4) -f run_ladder_stage4.py --timeout $(STAGE4_EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_LADDER4)); \
	$(call check_run_verdict,ladder stage 4 did not report success,STAGE4_OK); \
	$(call check_teardown,$(SESSION_2B_LADDER4)); \
	exit $$rc

SESSION_2B_AUDIT ?= stage2b-audit

# No measured timing exists for this driver yet (nothing has run) -- the
# budget matches the driver's OWN sizing-probe reasoning
# (`run_audit.py`'s `PROBE_RIDGE_BUDGET_S`/`PROBE_RUN_BUDGET_S` comments):
# a pure-JAX ridge step at ~2.86x stage 3's fold-level SVD count but with
# NO sklearn oracle leg (stage 3's own dominant cost, "315 oracle SVDs
# against 35 production ones"), so expected markedly cheaper wall-clock
# than stage 3's 1,716.3s ridge step despite the higher SVD count. A
# harness safety net, not a scientific tolerance -- the driver's own
# sizing probe and step halts are what actually gate correctness.
STAGE2B_AUDIT_EXEC_TIMEOUT ?= 5400

# Mirrors stage 4's speed bump exactly: a SECOND, EXPLICIT confirmation
# beyond typing the command. This audit is not the one-shot official
# result stage 4 is, but it is still real, metered GPU compute that
# nothing in this repository may launch without Dan's release.
.PHONY: stage2b-audit
stage2b-audit:  ## Run the Stage 2B amendment-impact audit -- bills while running, and requires STAGE2B_AUDIT_RELEASE_CONFIRMED=1
	@if [ "$(STAGE2B_AUDIT_RELEASE_CONFIRMED)" != "1" ]; then \
		echo "[make] REFUSING: this launches real, metered GPU compute (evolving the"; \
		echo "[make] 150-step budget and the 60,000-image OOF ridge in two alpha"; \
		echo "[make] regimes). Re-invoke as: STAGE2B_AUDIT_RELEASE_CONFIRMED=1 make stage2b-audit"; \
		echo "[make] only once Dan has explicitly released this run."; \
		exit 1; \
	fi
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/run_audit.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	driver_sha=$$(shasum -a 256 $(STAGE2B_DIR)/run_audit.py | cut -d' ' -f1); \
	echo "[make] commit $$commit, driver sha256 $$driver_sha"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_2B_AUDIT),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_AUDIT) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage equinox optax && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_AUDIT) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_AUDIT) -f run_audit.py --timeout $(STAGE2B_AUDIT_EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_AUDIT)); \
	$(call check_run_verdict,the amendment audit did not report success,AUDIT_OK); \
	$(call check_teardown,$(SESSION_2B_AUDIT)); \
	exit $$rc


##@ Stage 2B Companion Protocol 1 (ARM/x86 propagation)

SESSION_2B_PROTOCOL1 ?= stage2b-protocol1

.PHONY: stage2b-protocol1-arm-construct
stage2b-protocol1-arm-construct:  ## Protocol 1: build stress set + slice ARM encodings (local CPU, free)
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/run_arm_x86_propagation_stress.py --phase arm-construct

.PHONY: stage2b-protocol1-x86-encode
stage2b-protocol1-x86-encode:  ## Protocol 1: encode stress set on Colab x86 (bills while running)
	rc=0; src=0; \
	cd $(REPO_ROOT) && \
	if ! $(CLOSURE_CHECK) $(STAGE2B_DIR)/run_arm_x86_propagation_stress.py; then \
		exit 1; \
	fi; \
	commit=$$($(GIT) rev-parse HEAD); \
	if ! $(GIT) branch -r --contains $$commit 2>/dev/null | grep -q .; then \
		echo "[make] REFUSING: HEAD $$commit is not on any remote. Push before running -- the runtime can only fetch what origin has."; \
		exit 1; \
	fi; \
	driver_sha=$$(shasum -a 256 $(STAGE2B_DIR)/run_arm_x86_propagation_stress.py | cut -d' ' -f1); \
	echo "[make] commit $$commit, driver sha256 $$driver_sha"; \
	cd $(STAGE2B_DIR) && \
	$(MIGHTY_COLAB) sessions && \
	$(call ensure_session,$(SESSION_2B_PROTOCOL1),--gpu $(LADDER_GPU)) && \
	$(MIGHTY_COLAB) reinstall -s $(SESSION_2B_PROTOCOL1) jax[cuda12]==0.11.0 diffrax==0.7.2 google-cloud-storage && \
	$(MIGHTY_COLAB) upload -s $(SESSION_2B_PROTOCOL1) $(BONSAI_GCS_CREDENTIALS) $(REMOTE_KEY_PATH) && \
	rc=0; out=$$($(MIGHTY_COLAB_JSON) exec -s $(SESSION_2B_PROTOCOL1) -f run_arm_x86_propagation_stress.py --timeout $(EXEC_TIMEOUT) $(GCS_EXEC_ENV) --env BONSAI_COMMIT="$$commit" --env BONSAI_DRIVER_SHA256="$$driver_sha" --env JAX_ENABLE_X64=1 --env PROTOCOL1_PHASE=x86-encode) || rc=$$?; \
	$(call show_run_output); \
	$(call stop_session,$(SESSION_2B_PROTOCOL1)); \
	$(call check_run_verdict,protocol1 x86-encode did not report success,PROTOCOL1_X86_ENCODE_OK); \
	$(call check_teardown,$(SESSION_2B_PROTOCOL1)); \
	exit $$rc

.PHONY: stage2b-protocol1-propagate
stage2b-protocol1-propagate:  ## Protocol 1: evolve both arches, frozen ridge, five-stage report (local)
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run --group gpu python $(STAGE2B_DIR)/run_arm_x86_propagation_stress.py --phase propagate

.PHONY: stage2b-protocol1
stage2b-protocol1: stage2b-protocol1-arm-construct  ## Protocol 1 umbrella: arm-construct, then print next steps
	@echo "[make] Protocol 1 arm-construct done."
	@echo "[make] Next: push HEAD, then: make stage2b-protocol1-x86-encode"
	@echo "[make] Then: make stage2b-protocol1-propagate"


.PHONY: stage2b-protocol2
stage2b-protocol2:  ## Protocol 2: ABS_CONV_EPS sensitivity table (local CPU; GCS_ENV enables sidecar when configured)
	cd $(REPO_ROOT) && $(GCS_ENV) \
		uv run python $(STAGE2B_DIR)/run_abs_conv_eps_sensitivity.py
##@ Vacuous-test review (local preflight)
# Default Haiku: the Actions path has defaulted to Sonnet and cost $5 on a
# partial pass (PR #29). Local preflight is the cheap half; Actions stays the
# durable sticky. Override with MODEL=sonnet or REVIEW_MODEL.
REVIEW_PR ?=
MODEL ?= haiku

.PHONY: vacuous-review
vacuous-review:  ## Local vacuous-test review on Haiku. Usage: make vacuous-review PR=29
	@if [ -z "$(PR)$(REVIEW_PR)" ]; then \
		echo "[make] Usage: make vacuous-review PR=<n>   (optional: MODEL=haiku|sonnet)"; \
		exit 2; \
	fi
	cd $(REPO_ROOT) && \
		REVIEW_MODEL=$(MODEL) \
		bash tools/ci/vacuous_review_local.sh --pr $(or $(PR),$(REVIEW_PR)) --model $(MODEL)

.PHONY: vacuous-review-delta
vacuous-review-delta:  ## Print review_delta only (no model). Usage: make vacuous-review-delta PR=29
	@if [ -z "$(PR)$(REVIEW_PR)" ]; then \
		echo "[make] Usage: make vacuous-review-delta PR=<n>"; \
		exit 2; \
	fi
	cd $(REPO_ROOT) && bash tools/ci/vacuous_review_local.sh --pr $(or $(PR),$(REVIEW_PR)) --delta-only

.PHONY: help
help:  ## List every target in this file, grouped by section
	@awk 'BEGIN {FS = ":.*##"} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
