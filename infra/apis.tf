resource "google_project_service" "ci" {
  for_each = toset([
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com", # NOT currently enabled
    "pubsub.googleapis.com",
    "logging.googleapis.com",
    "iam.googleapis.com",
  ])

  project = var.project_id
  service = each.value

  # Both false, and this is the blast radius that hides inside a scope which
  # otherwise reads as safe.
  #
  # `terraform destroy` on a configuration described as "only new CI
  # resources" would, with the defaults, DISABLE Cloud Build, Pub/Sub and
  # Logging FOR THE WHOLE PROJECT -- including everything the science track
  # depends on. Disabling an API is not a CI-scoped act, however
  # CI-scoped the reason for enabling it was.
  disable_on_destroy         = false
  disable_dependent_services = false
}

# NOT managed here, deliberately: cloudresourcemanager.googleapis.com.
#
# The provider needs it to manage project IAM at all, so enabling it from
# within this configuration would mean relying on the provider to bootstrap
# the API it requires in order to function, on the same apply. It is one
# out-of-band command, listed in infra/README.md.
#
# Also not managed: secretmanager.googleapis.com. The 1st-generation GitHub
# App path needs no personal access token, so no secret exists to store.
# Keeping the API disabled keeps that true in a way a comment cannot.
