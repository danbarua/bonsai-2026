# EVERY trigger names a service account explicitly, and that is a security
# requirement rather than tidiness. The reason was measured on this project,
# not assumed:
#
#   $ gcloud projects get-iam-policy bonsai-504422
#   roles/cloudbuild.builds.builder  545167512937@cloudbuild.gserviceaccount.com
#
#   $ gcloud iam roles describe roles/cloudbuild.builds.builder
#   ... storage.objects.create, storage.objects.delete,
#       storage.objects.update, storage.buckets.create,
#       cloudbuild.builds.create ...
#
# This is an OLD project, so the legacy default Cloud Build service account
# still exists and still holds that role AT PROJECT LEVEL -- which means on
# `bonsai-2026-stage2b-cache`. A trigger created without `service_account`
# runs as that identity: able to overwrite or delete irreplaceable research
# artifacts, and able to create arbitrary builds.
#
# `docs/proposals/CI_CLOUDBUILD.md` forbids exactly this, and the guard that
# exists to enforce it CANNOT SEE IT. `tools/ci/assert_no_cloud_credentials.py`
# checks three environment variables and whether `google.cloud.storage`
# imports. A Cloud Build service-account credential is served by the
# METADATA SERVER -- no environment variable, no library import. The
# capability-absence assertion passes while the capability is present.
#
# Which is today's recurring shape one more time: a check correct about what
# it measures, read as an answer to a different question. These two accounts
# are the second source that does not share its blind spot.

resource "google_service_account" "ci_runner" {
  account_id   = "bonsai-ci-runner"
  display_name = "Bonsai CI runner (fast / full)"
  description  = "Runs the pytest suite. Holds no role on any bucket and cannot create builds."
}

resource "google_service_account" "ci_poll" {
  account_id   = "bonsai-ci-poll"
  display_name = "Bonsai CI poll (gated dispatcher)"
  description  = "Runs no tests. Reads build history and dispatches bonsai-ci-full."
}

# The ONLY project role the runner gets.
#
# Required because a user-specified build service account cannot write the
# default Cloud Build logs bucket, which is why `cloudbuild.yaml` sets
# `logging: CLOUD_LOGGING_ONLY`. The repository was already shaped for this
# design before the design existed.
resource "google_project_iam_member" "runner_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.ci_runner.email}"
}

resource "google_project_iam_member" "poll_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.ci_poll.email}"
}

# `cloudbuild.builds.editor` goes to the POLL account and nowhere else.
#
# It covers both halves of the decide step: `gcloud builds list` for the
# deadline rule, and `gcloud builds triggers run` to dispatch. `builds.viewer`
# is not granted separately because editor already contains builds.list and
# builds.get.
#
# `CI_CLOUDBUILD.md` is candid that this role is broader than the job needs
# and that Cloud Build has no per-trigger IAM, so "may run trigger X" is not
# expressible. Two things bound the consequence: `tools/ci/ci_targets.py`
# constrains what any build of this config can invoke, and the decide step
# refuses to dispatch when `_FULL_TRIGGER` names its own trigger -- which
# triggers.tf makes structurally unexpressible by using a reference.
resource "google_project_iam_member" "poll_builds_editor" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = "serviceAccount:${google_service_account.ci_poll.email}"
}

# Dispatching `bonsai-ci-full` starts a build that runs AS the runner, so the
# poll needs actAs on it.
#
# Whether `builds.triggers.run` re-checks actAs on the target trigger's
# account is documented for direct build submission and could not be
# verified for the trigger-run path without applying. Granting it costs
# nothing -- `serviceAccountUser` on one account is not a spend path. Its
# ABSENCE, if required, surfaces as PERMISSION_DENIED inside an unattended
# poll build fifteen minutes after apply, in a build nobody is watching.
resource "google_service_account_iam_member" "poll_may_act_as_runner" {
  service_account_id = google_service_account.ci_runner.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.ci_poll.email}"
}

# DELIBERATELY ABSENT, and the absences are the design:
#
#   * No role on bonsai-2026-stage2b-cache, or any other bucket. If the
#     first build fails reading the Cloud Build staging bucket, scope the
#     fix to `${var.project_id}_cloudbuild` with roles/storage.objectViewer.
#     Never a project-level storage role, and never
#     roles/cloudbuild.builds.builder -- that is the role being designed
#     away from.
#   * No key material. Nothing here creates a service-account key; the
#     triggers use the identity directly.
#   * No role for either account on colab-gcs-sa. The science credential is
#     not CI's to touch.
