output "ci_runner_email" {
  description = "The identity every trigger runs as. Holds roles/logging.logWriter and nothing else."
  value       = google_service_account.ci_runner.email
}

output "triggers" {
  description = "Trigger ids, for gcloud."
  value = {
    checkpoint = google_cloudbuild_trigger.checkpoint.trigger_id
    deps       = google_cloudbuild_trigger.deps.trigger_id
    manual     = google_cloudbuild_trigger.manual.trigger_id
  }
}

# The first-run sequence, emitted by apply so it survives without anyone
# re-reading a document.
output "next_steps" {
  description = "Run these in order. The first build is a measurement, not a formality."
  value       = <<-EOT

    1. Create the checkpoint branch, once:

         git branch ${var.checkpoint_branch} origin/stage2b
         git push -u origin ${var.checkpoint_branch}

       That first push fires `bonsai-ci-checkpoint`. WATCH IT.

       The suite has never run on Linux/x86. Expect the skip set to differ
       and the vacuity check to fail -- that is the correct direction and a
       measurement, not a defect. Regenerate the baseline from the CI JUnit
       report, never from a developer checkout. Read the build DURATION
       while you are there: every cost figure in this design is arithmetic
       on a constant nobody has measured.

    2. Thereafter, checkpoint by PULL REQUEST:

         gh pr create --base ${var.checkpoint_branch} --head stage2b

       Not a direct merge. A direct push skips the vacuous-test review and
       produces a green CI that nothing reviewed.

       What has not been checked yet is derived, not tracked:

         git log ${var.checkpoint_branch}..stage2b --oneline

    3. Watch one deliberate failure close before trusting a green. Break
       something small, push it to ${var.checkpoint_branch}, see the build go
       red, revert. `docs/proposals/CI_CLOUDBUILD.md`'s first-run protocol is
       four requirements and creating triggers discharges none of them.

    Run the full suite on demand at any time:

      gcloud builds triggers run bonsai-ci-manual \
          --region=global --project=${var.project_id}

  EOT
}
