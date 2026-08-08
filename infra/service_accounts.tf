# ONE service account, with ONE role, and every trigger names it explicitly.
#
# The explicitness is a security requirement rather than tidiness, and the
# reason was measured on this project rather than assumed:
#
#   $ gcloud projects get-iam-policy bonsai-504422
#   roles/cloudbuild.builds.builder  545167512937@cloudbuild.gserviceaccount.com
#
#   $ gcloud iam roles describe roles/cloudbuild.builds.builder
#   ... storage.objects.create, storage.objects.delete,
#       storage.objects.update, storage.buckets.create,
#       cloudbuild.builds.create ...
#
# This is an old project, so the legacy default Cloud Build service account
# still exists and still holds that role AT PROJECT LEVEL -- which means on
# `bonsai-2026-stage2b-cache`. A trigger created without `service_account`
# runs as that identity.
#
# What that costs is not hypothetical. The science bucket has NO OBJECT
# VERSIONING; it has a soft-delete policy of 604800 seconds:
#
#   $ gcloud storage buckets describe gs://bonsai-2026-stage2b-cache
#   soft_delete_policy:
#     retentionDurationSeconds: '604800'
#
# Seven days of forensic recovery, not immutability. DESIGN.md's frozen
# claim that the tables "survive as history because the storage model
# refuses to destroy them" is not what the storage model does -- the refusal
# lives in `ensure_artifact`, and binds only writers that go through
# `stage2b_gcs.py`. A principal holding `storage.objects.delete` never
# enters that process.
#
# And `tools/ci/assert_no_cloud_credentials.py` CANNOT SEE ANY OF IT. It
# reads three environment variables and tries to import
# `google.cloud.storage`. A Cloud Build service-account credential is served
# by the METADATA SERVER -- no variable, no import -- and the build runs
# images that have `gcloud`. The capability-absence assertion passes while
# the capability is present: a check testing the observable it can reach,
# blind to the channel the credential actually arrives by.
#
# So this file is the second source that does not share that blind spot.

resource "google_service_account" "ci_runner" {
  account_id   = "bonsai-ci-runner"
  display_name = "Bonsai CI runner"
  description  = "Runs the test suite. Holds no role on any bucket and cannot create builds."
}

# The ONLY role, and it is required rather than chosen: a user-specified
# build service account cannot write the default Cloud Build logs bucket,
# which is why `cloudbuild.yaml` sets `logging: CLOUD_LOGGING_ONLY`. The
# repository was already shaped for this design before the design existed.
resource "google_project_iam_member" "runner_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.ci_runner.email}"
}

# DELIBERATELY ABSENT, and the absences are the design:
#
#   * `roles/cloudbuild.builds.editor`. It existed only so a poll could
#     dispatch a full run. Nothing dispatches anything now, so the one role
#     CI_CLOUDBUILD.md admits is broader than its job -- with no narrower
#     alternative expressible, since Cloud Build has no per-trigger IAM --
#     is not granted at all. Dropping the poll removed a spend path the
#     spend guard could not see.
#   * A second service account, and the `actAs` grant that let one act as
#     the other. Both were poll machinery.
#   * Any role on any bucket. If a build ever fails reading the Cloud Build
#     staging bucket, scope the fix to `${var.project_id}_cloudbuild` with
#     `roles/storage.objectViewer` -- never a project-level storage role,
#     and never `roles/cloudbuild.builds.builder`, which is the role being
#     designed away from.
#   * Key material. Nothing here creates a service-account key.
#   * Anything touching `colab-gcs-sa`. The science credential is not CI's.
