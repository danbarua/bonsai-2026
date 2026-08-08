terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source = "hashicorp/google"
      # Pin the MAJOR. Breaking changes stay out; fixes come in. The exact
      # build is pinned by .terraform.lock.hcl, which is committed -- this
      # constraint is the policy, the lockfile is the fact.
      version = "~> 7.0"
    }
  }

  # State lives in its own bucket, and the choice is load-bearing rather
  # than conventional.
  #
  # NOT the science bucket: `gs://bonsai-2026-stage2b-cache` grants
  # `roles/storage.objectViewer` to `allUsers`. State written there is
  # WORLD-READABLE, and it enumerates every service account, every IAM
  # binding and the whole trigger topology.
  #
  # NOT local either: this repository is worked from several git worktrees
  # by concurrent sessions, and a local `terraform.tfstate` is one
  # `git clean` from a lost-state incident.
  #
  # This bucket is created OUT OF BAND -- see infra/README.md. It cannot be
  # created by the configuration that stores its state in it, and a resource
  # created once, never changed, and never to be destroyed is not one that
  # benefits from being in state.
  backend "gcs" {
    bucket = "bonsai-504422-tfstate"
    prefix = "ci"
  }
}
