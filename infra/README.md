# CI infrastructure

Terraform for the Cloud Build triggers that run this repository's test
suite. `cloudbuild.yaml` at the repo root is what they execute;
`docs/proposals/CI_CLOUDBUILD.md` is why.

Nothing here has been applied yet.

## Where and when CI runs, and why it is not every push

Agents run `make stage2b-test` habitually and locally. On the same commit, a
cloud run of the same suite differs in exactly three ways: **Linux/x86**
instead of macOS/ARM, a **clean checkout**, and a **record**. None of those
change between one push and the next, so a per-push trigger re-answers a
question an agent already answered, minutes earlier, faster.

Measured: 368 commits over 7 days on `stage2b`, ~46 pushes/day — roughly
twice the free tier, spent on duplication.

So there are three triggers and none of them is per-push:

| trigger | fires on | why |
|---|---|---|
| `bonsai-ci-checkpoint` | push to `stage2b-ci` | the batch, on the other platform, from clean |
| `bonsai-ci-deps` | `uv.lock` / `pyproject.toml` change | an undeclared dependency is invisible to every local run |
| `bonsai-ci-manual` | you | when you want it |

**Reach the checkpoint by pull request.** `gh pr create --base stage2b-ci
--head stage2b`. A direct merge skips the vacuous-test review and produces a
green CI that nothing reviewed. And `git log stage2b-ci..stage2b` is then
exactly the work that has not had a full run — derived, not tracked.

Path filtering was measured and rejected. The intuitive ignore set
(`.claude`, `.github`, `docs`, `*.md`) would skip 39% of pushes, but twelve
test files read those paths and `docs/` is directly tested. The set no test
reads covers 2%.

## What this manages, and what it must never touch

**Managed:** one service account with one role, three Cloud Build triggers,
and API enablement. No scheduler, no Pub/Sub, nothing that spends
unattended.

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

The **Google Cloud Build** GitHub App is already installed on the account
(`github.com/danbarua/bonsai-2026/settings/installations`), so the GitHub
half of this is done. What may still be missing is the GCP half — linking
that repository to this project — which is the same wizard and is what
`terraform plan` will fail on if it has not happened.

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
- **`roles/cloudbuild.builds.editor` appears NOWHERE.** It existed only so a
  poll could dispatch a full run; nothing dispatches anything now.
- **Every trigger names `service_account`.** A trigger without one runs as
  the legacy default Cloud Build account, which holds
  `roles/cloudbuild.builds.builder` at project level — and that contains
  `storage.objects.delete` on the science bucket.

Then:

```bash
terraform -chdir=infra apply
```

Apply prints a `next_steps` output with the first-run sequence.

## Cost

Everything here is free to exist, and **nothing spends unattended** — there
is no scheduler, so no build starts without a push or a person.

Measured rather than assumed: ~2 dependency changes a day, plus checkpoints
and manual runs at whatever rate they are wanted. A handful of builds a day
against a ~2,500 build-minute free tier.

For contrast, the two designs this replaced: the 15-minute poll was 96
builds/day, and per-push CI ~46/day. Either alone was roughly twice the free
tier before a single useful build ran.

The per-build DURATION on Cloud Build is still unmeasured, and every cost
statement above is arithmetic on it. Read it off the first few builds.

## State

Local `terraform.tfstate` is gitignored but should not exist: the backend is
GCS. If you find one, the backend was not initialised.

`.terraform.lock.hcl` **is committed** — it is a lockfile, not state, and it
is what actually pins the provider build.
