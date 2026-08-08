# EVERY trigger is `location = "global"`, and that is forced by committed
# code rather than chosen.
#
# `cloudbuild.yaml`'s decide step reads build history with:
#
#     gcloud builds list --limit=1 --sort-by=~finishTime \
#         --filter="status=SUCCESS AND substitutions._TIER=full"
#
# with NO `--region` flag -- so it reads GLOBAL history. The dispatch on the
# next lines DOES pass `--region="${_TRIGGER_REGION}"`. Regional triggers
# would therefore produce builds the query can never see: `last_full` empty
# on every poll, `since_full_s=999999999`, and the deadline branch firing a
# full run EVERY FIFTEEN MINUTES, forever, reporting success each time.
#
# Silent, green, and expensive -- which is why this is written down here
# rather than left as a default nobody revisits. It is also the deciding
# argument for the 1st-generation GitHub App path: 2nd-generation
# connections are regional, so adopting them REQUIRES adding `--region` to
# that query first.

locals {
  repo_uri    = "https://github.com/${var.github_owner}/${var.github_repo}"
  default_ref = "refs/heads/${var.default_branch}"
}

# --- fast: every push to a shared branch ---------------------------------

resource "google_cloudbuild_trigger" "fast" {
  name            = "bonsai-ci-fast"
  description     = "Fast tier: make stage2b-test on every push to a shared branch."
  location        = "global"
  filename        = "cloudbuild.yaml"
  service_account = google_service_account.ci_runner.id

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = var.trigger_branch_regex
    }
  }

  # `_TIER` is set here even though cloudbuild.yaml already defaults it to
  # `fast`. Only TRIGGER substitutions appear in `build.substitutions`,
  # which is what `gcloud builds list --filter="substitutions._TIER=..."`
  # reads -- a YAML default is invisible to that query. Setting it here
  # keeps build history answerable.
  substitutions = {
    _TIER = "fast"
  }

  depends_on = [google_project_service.ci]
}

# --- full: manual, and dispatched by the poll ----------------------------
#
# No `github {}` block: that block describes an EVENT, and a manual trigger
# has none, so there is nothing to derive the source from. Two blocks stand
# in for it -- `source_to_build` says what to check out, `git_file_source`
# says where the build config lives.

resource "google_cloudbuild_trigger" "full" {
  name            = "bonsai-ci-full"
  description     = "Full tier: make test. Dispatched by the poll, or run by hand."
  location        = "global"
  service_account = google_service_account.ci_runner.id

  source_to_build {
    uri       = local.repo_uri
    ref       = local.default_ref
    repo_type = "GITHUB"
  }

  git_file_source {
    path      = "cloudbuild.yaml"
    uri       = local.repo_uri
    revision  = local.default_ref
    repo_type = "GITHUB"
  }

  substitutions = {
    _TIER = "full"
  }

  depends_on = [google_project_service.ci]
}

# --- poll: decides, and dispatches. Runs no tests ------------------------

resource "google_cloudbuild_trigger" "poll" {
  name            = "bonsai-ci-poll"
  description     = "Gated tier: decides whether to dispatch a full run. Runs no tests."
  location        = "global"
  service_account = google_service_account.ci_poll.id

  pubsub_config {
    topic = google_pubsub_topic.poll.id
  }

  source_to_build {
    uri       = local.repo_uri
    ref       = local.default_ref
    repo_type = "GITHUB"
  }

  git_file_source {
    path      = "cloudbuild.yaml"
    uri       = local.repo_uri
    revision  = local.default_ref
    repo_type = "GITHUB"
  }

  substitutions = {
    _TIER = "gated"

    # `BRANCH_NAME` on a Pub/Sub-invoked build is flagged unverified in
    # CI_CLOUDBUILD.md's soft spots -- if empty, the poll fails loudly every
    # fifteen minutes rather than dispatching against the wrong branch.
    # Setting `_BRANCH` is the documented fix and costs nothing if
    # BRANCH_NAME turns out to be populated after all.
    _BRANCH = var.default_branch

    # A REFERENCE, not a string. The self-dispatch loop that the decide step
    # refuses at runtime becomes structurally unexpressible here: this cannot
    # name the poll trigger without a dependency cycle Terraform would
    # reject. It also orders creation correctly for free.
    _FULL_TRIGGER = google_cloudbuild_trigger.full.name

    _TRIGGER_REGION = "global"
  }

  depends_on = [
    google_project_service.ci,
    google_project_iam_member.poll_builds_editor,
    google_service_account_iam_member.poll_may_act_as_runner,
  ]
}

# --- deps: NOT created by default ----------------------------------------
#
# Guarded by `enable_deps_trigger`, which defaults false. See variables.tf:
# the dependency-bump step this would fire does not exist in cloudbuild.yaml
# yet, so today it would only duplicate the fast tier on the same push.

resource "google_cloudbuild_trigger" "deps" {
  count = var.enable_deps_trigger ? 1 : 0

  name            = "bonsai-ci-deps"
  description     = "Fires when the dependency pins move. See CI_CLOUDBUILD.md tier 3."
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
    _TIER = "fast"
  }

  depends_on = [google_project_service.ci]
}
