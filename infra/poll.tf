resource "google_pubsub_topic" "poll" {
  name       = "bonsai-ci-poll"
  depends_on = [google_project_service.ci]
}

# THE ONLY RESOURCE HERE THAT CAUSES SPENDING, and it does so unattended.
#
# It ships PAUSED. Everything else in this configuration is free to exist
# and costs nothing until something runs; this is the thing that makes
# something run, ninety-six times a day, with nobody watching.
#
# Unpausing is a separate, deliberate apply because it is the act that
# asserts CI_CLOUDBUILD.md's first-run protocol has been discharged: a full
# run watched to completion, and one deliberate failure watched to close.
# Neither has happened -- the suite has never run on Linux/x86 at all, and
# the document is explicit that a first green on a never-exercised path is
# the presence-shaped failure surface.
#
#     terraform apply -var=poll_paused=false
resource "google_cloud_scheduler_job" "poll" {
  name        = "bonsai-ci-poll"
  description = "Publishes to bonsai-ci-poll so the gated tier can decide."
  region      = var.region
  schedule    = var.poll_schedule
  time_zone   = "Etc/UTC"
  paused      = var.poll_paused

  pubsub_target {
    topic_name = google_pubsub_topic.poll.id
    data       = base64encode(jsonencode({ source = "bonsai-ci-scheduler" }))
  }

  depends_on = [google_project_service.ci]
}
