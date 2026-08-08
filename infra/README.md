# CI infrastructure

Terraform for the Cloud Build triggers that run this repository's test
suite. `cloudbuild.yaml` at the repo root is what they execute;
`docs/proposals/CI_CLOUDBUILD.md` is why.

Nothing here has been applied yet.

## What this manages, and what it must never touch

**Managed:** two service accounts and their IAM, three Cloud Build triggers
(a fourth behind a flag), a Pub/Sub topic, a Cloud Scheduler job, and API
enablement.

**Not managed, deliberately:** the science buckets
(`bonsai-2026-*-cache`), `colab-gcs-sa`, and the state bucket. They are
invisible to `terraform destroy`, which is the point — a destroy on a
configuration scoped to CI must not be able to reach irreplaceable research
data.

`prevent_destroy` was considered and rejected as the mechanism. It protects
only while the lifecycle block exists; delete the block and the next apply
plans a delete with no warning. On a repository worked by concurrent
sessions, protection that lives in an editable line is not protection.

## Before the first apply

Four things, in order. The first is the only one that needs a browser.

**1. Authorise Cloud Build against the repository.** Terraform cannot do
this: a GitHub App installation requires a human to click Install on
github.com, and no API exists for it.

> Cloud Console → Cloud Build → Repositories → Connect repository →
> **1st gen** → GitHub (Cloud Build GitHub App) → authorise →
> select `danbarua/bonsai-2026`.

Check the **1st gen** tab is offered before going further. If Google has
retired it for new connections, stop: the 2nd-generation path is regional,
and regional triggers break the deadline rule (see `triggers.tf`). Adopting
it requires adding `--region` to the `gcloud builds list` call in
`cloudbuild.yaml` first.

**2. Enable the API the provider itself needs.**

```bash
gcloud services enable cloudresourcemanager.googleapis.com --project=bonsai-504422
```

Managing this from inside the configuration would mean relying on the
provider to bootstrap the API it needs in order to function, on the same
apply.

**3. Create the state bucket.** It cannot be created by the configuration
that stores its state in it, and a resource created once and never changed
does not benefit from being in state.

```bash
gcloud storage buckets create gs://bonsai-504422-tfstate \
    --project=bonsai-504422 --location=us-central1 \
    --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://bonsai-504422-tfstate --versioning
```

`--public-access-prevention` is not boilerplate here. The science bucket
grants `roles/storage.objectViewer` to `allUsers`; state written somewhere
similar would publish every service account, every IAM binding, and the
whole trigger topology.

**4. Plan, and read it.**

```bash
terraform -chdir=infra init
terraform -chdir=infra plan
```

## Reading the plan

`terraform plan` is the review gate, and it is most of why this exists
rather than a shell script. Three things to check by eye:

- **No resource grants any role on a `bonsai-2026-*-cache` bucket.**
- **`roles/cloudbuild.builds.editor` appears exactly once**, on
  `bonsai-ci-poll`.
- **`google_cloud_scheduler_job.poll` has `paused = true`.**

Then:

```bash
terraform -chdir=infra apply
```

Apply prints a `next_steps` output with the first-run sequence. Follow it in
order; the last step is the one that starts spending.

## Cost

Everything here is free to exist. The scheduler is the only resource that
causes spending, which is why it ships paused: every poll is a build, and a
build that decides to do nothing still pays source fetch and container
start.

At fifteen minutes that is 96 builds a day. `CI_CLOUDBUILD.md` estimates
30–60s of overhead each, which is 1,440–2,880 build-minutes a month against
a free tier of roughly 2,500 — so at the pessimistic end **the poll alone
can exhaust the free tier before a single test runs.** That estimate is
arithmetic on an assumed constant, not a measurement, and the document says
so. Measure a week of real builds before tightening the interval. If it
needs cutting, `"*/15 8-23 * * *"` removes a third at no practical cost —
nobody pushes at 04:00, and the deadline rule catches up on the first
morning poll.

## State

Local `terraform.tfstate` is gitignored but should not exist: the backend is
GCS. If you find one, the backend was not initialised.

`.terraform.lock.hcl` **is committed** — it is a lockfile, not state, and it
is what actually pins the provider build.
