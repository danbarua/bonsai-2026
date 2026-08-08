output "ci_runner_email" {
  description = "Identity the fast and full tiers run as."
  value       = google_service_account.ci_runner.email
}

output "ci_poll_email" {
  description = "Identity the gated tier runs as. The only one holding cloudbuild.builds.editor."
  value       = google_service_account.ci_poll.email
}

output "triggers" {
  description = "Trigger ids, for gcloud."
  value = {
    fast = google_cloudbuild_trigger.fast.trigger_id
    full = google_cloudbuild_trigger.full.trigger_id
    poll = google_cloudbuild_trigger.poll.trigger_id
  }
}

output "poll_is_paused" {
  description = "False means CI is spending on a schedule with nobody watching."
  value       = google_cloud_scheduler_job.poll.paused
}

# The first-run sequence, emitted by apply so it survives without anyone
# re-reading a document. Steps 1 and 2 are CI_CLOUDBUILD.md's first-run
# protocol; step 3 is what asserts they were done.
output "next_steps" {
  description = "Run these in order. Do not skip to the last one."
  value       = <<-EOT

    1. Run the full tier by hand and WATCH IT:

         gcloud builds triggers run bonsai-ci-full \
             --region=global --branch=${var.default_branch} --project=${var.project_id}

       The suite has never run on Linux/x86. Expect the skip set to differ
       and the vacuity check to fail -- that is the correct direction, and a
       measurement rather than a defect. Regenerate the baseline from the
       CI JUnit report, never from a developer checkout.

    2. Confirm the deadline rule can see that build:

         gcloud builds list --project=${var.project_id} \
             --filter="status=SUCCESS AND substitutions._TIER=full" \
             --format='value(id,finishTime)'

       If this returns nothing, DO NOT unpause the poll. Filtering on a
       substitution key is flagged unverified in CI_CLOUDBUILD.md; an empty
       result means the deadline branch would fire on every single poll.

    3. Only then, and only after watching one deliberate failure close:

         terraform apply -var=poll_paused=false

  EOT
}
