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

variable "default_branch" {
  description = "Branch the manual and poll triggers check out."
  type        = string
  default     = "stage2b"
}

variable "trigger_branch_regex" {
  description = "Branches whose pushes fire the fast tier."
  type        = string
  default     = "^(stage2b|main)$"
}

variable "poll_schedule" {
  description = <<-EOT
    Cron for the idle-or-deadline poll.

    Every poll is a build, and a build that decides to do nothing still pays
    source fetch and container start. At 15 minutes that is 96 builds/day.
    Measure the real per-build overhead before tightening this -- the free
    tier is roughly 2,500 build-minutes/month and the poll alone can
    plausibly consume most of it. `docs/proposals/CI_CLOUDBUILD.md` records
    the arithmetic and is explicit that it IS arithmetic, not measurement.
  EOT
  type        = string
  default     = "*/15 * * * *"
}

variable "poll_paused" {
  description = <<-EOT
    Ships PAUSED, and this is the one default that must not be flipped
    casually.

    The scheduler is the only resource here that causes spending, and it
    does so unattended. `docs/proposals/CI_CLOUDBUILD.md`'s first-run
    protocol requires a human to watch a full run succeed AND watch one
    deliberate failure close before CI is trusted. Unpausing is the act that
    asserts both happened, so it is a separate, deliberate apply.
  EOT
  type        = bool
  default     = true
}

variable "enable_deps_trigger" {
  description = <<-EOT
    OFF, because the behaviour it would trigger does not exist yet.

    `docs/proposals/CI_CLOUDBUILD.md` describes a dependency tier that
    detects a `mighty-colab` version bump and FAILS WITH AN INSTRUCTION to
    run the billing round trip by hand. No such step is implemented in
    cloudbuild.yaml. A deps trigger today would fire `_TIER=fast` on a push
    that `bonsai-ci-fast` already handles: a duplicate build, double the
    minutes, and no signal that the fast tier did not already give.

    The variable exists so the wiring is one flag away on the day the step
    is written.
  EOT
  type        = bool
  default     = false
}
