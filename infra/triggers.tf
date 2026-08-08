# WHERE AND WHEN A CLOUD RUN EARNS ITS COST.
#
# Agents run `make stage2b-test` habitually and locally. On the same commit,
# a cloud run of the same suite differs in exactly three ways:
#
#   1. Linux/x86 instead of macOS/ARM. The suite has never run there.
#   2. A clean checkout -- no datasets, no cached .pkl, no stale .venv,
#      nothing installed that uv.lock does not name.
#   3. A record nobody has to remember to keep.
#
# None of those change between one push and the next. So there is NO
# PER-PUSH TRIGGER, and that is a design conclusion rather than a budget
# concession -- re-running per push buys nothing an agent has not already
# had, minutes earlier, faster.
#
# The measurement that settled it: 368 commits over 7 days on `stage2b`,
# clustering to roughly 46 pushes/day. At any plausible per-build duration
# that is about twice the free tier, spent re-answering a question already
# answered locally.
#
# Path filtering was measured and rejected rather than assumed. The
# intuitive ignore set (`.claude`, `.github`, `docs`, `*.md`) would skip 39%
# of pushes -- but twelve test files read those paths, and `docs/` is
# directly tested by `test_doc_references.py` and `test_catalogue_counts.py`.
# The set no test reads at all covers 7 of 367 commits, 2%. A filter that
# saves nothing safely is not worth the comment explaining it.
#
# Every trigger is `location = "global"`. Regional triggers would put builds
# somewhere the un-regioned `gcloud builds list` in cloudbuild.yaml cannot
# see, and 2nd-generation connections are regional -- so adopting them means
# patching that query first.

locals {
  repo_uri       = "https://github.com/${var.github_owner}/${var.github_repo}"
  checkpoint_ref = "refs/heads/${var.checkpoint_branch}"
}

# --- checkpoint: the batch, declared rather than inferred ----------------
#
# Merging `stage2b` into `stage2b-ci` says "this is a coherent point, check
# it on the other platform, from clean". That is a declaration; the poll it
# replaces inferred the same thing from elapsed time.
#
# Reach it by PULL REQUEST, not a direct push. A direct merge skips the
# vacuous-test review and produces a green CI that nothing reviewed -- a
# green meaning less than it says, which is the failure this whole apparatus
# exists to avoid. Through a PR the review fires on the diff and this fires
# on the merge.
#
# Second keep, free: `git log stage2b-ci..stage2b` is exactly the work that
# has not had a full run. Derived, not tracked by hand.
resource "google_cloudbuild_trigger" "checkpoint" {
  name            = "bonsai-ci-checkpoint"
  description     = "Full suite on the ${var.checkpoint_branch} checkpoint. Linux/x86, clean checkout."
  location        = "global"
  filename        = "cloudbuild.yaml"
  service_account = google_service_account.ci_runner.id

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^${var.checkpoint_branch}$"
    }
  }

  # Set here rather than relying on cloudbuild.yaml's default: only TRIGGER
  # substitutions appear in `build.substitutions`, which is what
  # `gcloud builds list --filter=` reads. A YAML default is invisible to
  # build history.
  substitutions = {
    _TIER = "full"
  }

  depends_on = [google_project_service.ci]
}

# --- deps: the one thing ONLY a clean checkout can catch -----------------
#
# `CI_CLOUDBUILD.md` proposed a dependency tier for `mighty-colab` version
# bumps, and no such step was ever implemented -- so a deps trigger would
# have duplicated the fast tier for no signal. This one is justified
# differently and concretely.
#
# An undeclared dependency is invisible to every local run, because the
# developer's environment already has it. It is visible to a clean checkout
# and nowhere else. The repository has a live instance: `equinox` is
# hard-imported by `experiments/stage2b_denoising/stage2b_cnn.py` and
# `tests/test_stage2b_cnn.py`, and is declared in no dependency group --
# it arrives only as a transitive of `diffrax`. `uv sync --frozen` installs
# it today, so nothing fails; a re-lock or an upstream drop is the exposure.
#
# Measured at 2.0 changes/day to `uv.lock` or `pyproject.toml`, which is
# almost free and watches exactly the failure no local run can see.
resource "google_cloudbuild_trigger" "deps" {
  name            = "bonsai-ci-deps"
  description     = "Full suite when the dependency pins move. Catches undeclared deps a local run cannot."
  location        = "global"
  filename        = "cloudbuild.yaml"
  service_account = google_service_account.ci_runner.id

  included_files = ["uv.lock", "pyproject.toml"]

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = var.trigger_branch_regex
    }
  }

  substitutions = {
    _TIER = "full"
  }

  depends_on = [google_project_service.ci]
}

# --- manual: run it when you want it -------------------------------------
#
# No `github {}` block: that block describes an EVENT and a manual trigger
# has none, so there is nothing to derive the source from. `source_to_build`
# says what to check out; `git_file_source` says where the build config is.
resource "google_cloudbuild_trigger" "manual" {
  name            = "bonsai-ci-manual"
  description     = "Full suite, on demand, against ${var.checkpoint_branch}."
  location        = "global"
  service_account = google_service_account.ci_runner.id

  source_to_build {
    uri       = local.repo_uri
    ref       = local.checkpoint_ref
    repo_type = "GITHUB"
  }

  git_file_source {
    path      = "cloudbuild.yaml"
    uri       = local.repo_uri
    revision  = local.checkpoint_ref
    repo_type = "GITHUB"
  }

  substitutions = {
    _TIER = "full"
  }

  depends_on = [google_project_service.ci]
}
