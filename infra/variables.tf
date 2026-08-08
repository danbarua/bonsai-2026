variable "project_id" {
  description = <<-EOT
    The GCP project. Declared in the Makefile as BONSAI_GCP_PROJECT and
    asserted against experiments/stage2b_denoising/stage2b_gcs.py's
    GCS_PROJECT by tests/test_stage2b_gcs_makefile.py, so this value cannot
    drift from the science module's without a test failing.
  EOT
  type        = string
  default     = "bonsai-504422"
}

variable "project_number" {
  description = "Numeric project id. Only used to name Google-managed service agents."
  type        = string
  default     = "545167512937"
}

variable "region" {
  description = <<-EOT
    Region for regional resources (Cloud Scheduler). Matches the existing
    buckets, which are all US-CENTRAL1.

    NOTE this is NOT the trigger location -- see triggers.tf, where global is
    a correctness requirement rather than a preference.
  EOT
  type        = string
  default     = "us-central1"
}

variable "github_owner" {
  type    = string
  default = "danbarua"
}

variable "github_repo" {
  type    = string
  default = "bonsai-2026"
}

variable "checkpoint_branch" {
  description = <<-EOT
    The branch a full run is declared on. Merge `stage2b` into it -- via a
    PULL REQUEST, so the vacuous-test review sees the diff -- when the work
    has reached a point worth checking on Linux/x86 from a clean checkout.

    It also answers "what has not been checked", derived rather than
    tracked: `git log stage2b-ci..stage2b`.
  EOT
  type        = string
  default     = "stage2b-ci"
}

variable "trigger_branch_regex" {
  description = <<-EOT
    Branches where a dependency change fires a build. Note this is ONLY
    consulted by the deps trigger -- there is no per-push trigger, because a
    cloud re-run of the suite an agent just ran locally differs only in
    platform and checkout cleanliness, and neither changes push to push.
  EOT
  type        = string
  default     = "^(stage2b|main)$"
}
