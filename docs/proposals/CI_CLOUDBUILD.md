# CI on Google Cloud Build

A proposal. `cloudbuild.yaml` and the guards under `tools/ci/` exist in the
tree; no trigger, service account, or GCP resource has been created, and
nothing here has run on Cloud Build. The trigger configuration a human must
create is in "What a human has to create", below.

## The limit, first, because a green build will otherwise be read as more

**CI verifies the tests that exist.** Five of the six incidents recorded on
this project on 2026-08-07 were *absent* gates -- a documented requirement
with no enforcement anywhere in the code. CI would have caught none of
them, because there was nothing to run. `tools/gates/gate_inventory.py` is
the mechanism aimed at that class, and it reconciles documented MUST/HALT
clauses against the code that enforces them; a test runner cannot.

This matters more once CI is green every day, not less. A project with
working CI is *more* prone to reading "our tests pass" as "we have tests
for what we promised", and those are different claims. The second one is
answered by `docs/VACUOUS_TESTS.md` and by the gate inventory, not by this
file.

## Why Cloud Build runs at all

Work on this repository runs in supervised loops across several agents.
A break that lands on `stage2b` between loops sits there until a human
notices -- which, on the night that motivated this, was after a sleep. The
design goal is that breaks surface *inside* the work loops, on a cadence
the loops can act on. It is not general hygiene, and it is not a gate on
merging.

Two GitHub Actions workflows already exist (`.github/workflows/claude.yml`,
`claude-code-review.yml`). Both invoke Claude Code -- review on pull
requests and `@claude` mentions. Neither runs pytest, and neither fires on
a push to a branch. They do not overlap with this.

## What runs

`cloudbuild.yaml` has four steps: `decide`, `spend-guard`, `verify`,
`no-vacuous-green`. `_TIER` selects how much of the suite runs.

| tier | what it does | when |
| --- | --- | --- |
| `fast` | `make stage2b-test` | every push to a shared branch |
| `gated` | decides, and dispatches to the `full` trigger | scheduled poll |
| `full` | `make test` | dispatched by a poll, or manually |

A `gated` build runs no tests. It decides and dispatches, which is what
keeps a poll and a full run separable in build history -- see the
idle-or-deadline section for why that separation is load-bearing rather
than tidy.

The build invokes `make`, not a restated pytest command line. The Makefile
already owns the invocation, its marker selection and its file list;
copying that into a YAML file is the reimplemented-helper failure CLAUDE.md
principle 16 names, and it would drift the first time a marker changed.
Reporting is added through `PYTEST_ADDOPTS` (`--junitxml`, `-rs`), which
pytest reads on its own, so no Makefile target had to change.

Environment: `ghcr.io/astral-sh/uv:python3.14-bookworm-slim`, with `git`
and `make` installed by apt in the step that needs them. Installation is
`uv sync --frozen` -- `uv.lock` exactly, nothing resolved. That single
command is the whole clean-checkout install: `uv sync` installs the project
itself in editable mode, so `from bonsai... import ...` resolves without a
separate `uv pip install -e .`, and it installs the `dev` group (pytest)
because uv treats `dev` as a default group. The `gpu` group -- `mighty-colab`
and `google-cloud-storage` -- is not a default group and is therefore
absent. That absence is load-bearing, not incidental, and is asserted
rather than assumed.

## Trigger branches

`stage2b` and `main`, at minimum.

`stage2b` is where the work actually lands and where the break that
motivated this sat. `main` is the branch everything eventually reaches and
the one a cold reader trusts. `stage2a` is effectively closed -- Stage 2A's
findings are frozen -- so a push to it is rare enough that a manual run
covers it; adding it costs nothing if that changes.

Agent worktree branches are deliberately excluded. They are short-lived,
frequently rebased, and their intermediate states are not states anyone
keeps.

## The two things this build must not do

### It must not be able to spend

Every GPU and GCS target in this project is a recipe in the same Makefile
the suite reads. A build able to invoke one is an unattended spend surface
with no human in the loop, and release has been human-gated throughout
Stage 2B.

Two layers, in this order.

**Capability absence, which needs no list.** No credential is mounted, no
CI service account holds a role on `bonsai-2026-stage2b-cache`, and
`google-cloud-storage` is not installed. A target that tried to provision
or write would fail at the point it tried.
`tools/ci/assert_no_cloud_credentials.py` runs before the suite and fails
the build if either capability appears -- the day someone adds the library
or mounts a key, rather than whenever it is next noticed.

**An allowlist, derived rather than hand-written.**
`tools/ci/ci_targets.py` names three invocable targets -- `test`,
`stage2a-test`, `stage2b-test` -- and derives the billable set from the
Makefile itself: any recipe whose text reaches `$(MIGHTY_COLAB)`,
`$(GCS_ENV)` or `$(GCS_EXEC_ENV)`. It currently derives 15 of the 31
targets as billable. A hand-written denylist is exactly the artifact
principle 21 says will silently under-cover; a derivation covers the next
GPU target on the day it is written. The parsing goes through
`tests/_makefile.py`, which is the one Makefile parser in this repository.

Four checks, and the last two are what bind `cloudbuild.yaml`: every
allowlisted name is a real target; the allowlist is disjoint from the
derived billable set; every allowlisted recipe still runs `uv run pytest`
and nothing else; and `cloudbuild.yaml` invokes only allowlisted targets
and does not name a billable one anywhere in its text. The last check reads
raw text rather than parsing YAML -- over-flagging is the safe direction
for a spend guard, and it is why `cloudbuild.yaml` refers readers here
instead of naming targets in its comments.

Both flavours of spend are in scope. Colab and GCS bill directly;
`stage2a-analyze` bills no GPU at all and takes about four hours of CPU,
which on Cloud Build is four hours of build minutes. It is excluded by
being absent from a three-entry allowlist.

`tests/test_cloudbuild_ci.py` exercises each check against synthetic
Makefile recipes -- a billing target added to the allowlist, a config
invoking an unlisted target, a billable target named in a comment, an
allowlisted target rewritten to call `$(PYTHON)`, an allowlist naming a
renamed target -- and asserts the specific complaint each time. Checking a
narrowing by running the real Makefile past it proves the code works and
says nothing about the narrowing; that is the half of principle 21 that hid
`STAGE2B_TEST_FILES`'s gap under `pytest tests/`.

### It must not pass vacuously

A CI machine has no `datasets/`, no gitignored `.pkl`/`.npz` caches, no GCS
credentials, no Colab session, no `claude` CLI and no `mighty-colab`. The
two-tier convention skips cleanly on every one of them, which is correct
behaviour and also the failure mode: from an exit code, a build where forty
more tests started skipping is indistinguishable from one where they all
ran. `docs/VACUOUS_TESTS.md` is fourteen incidents of that shape at smaller
scale.

`tools/ci/check_suite_not_vacuous.py` reads the JUnit report and compares
the **set** of skipped tests against a committed, measured baseline
(`tools/ci/ci_skip_baseline.txt`, 36 entries). Three failures, all fatal:

- a skip not in the baseline -- CI lost coverage it had
- a baseline entry that ran -- the baseline is stale, and a stale baseline
  stops detecting the skips it exists to detect
- a baseline entry that was not collected -- the test was renamed or
  deleted, and the entry now hides a gap rather than recording one

**Why a set and not counts.** A floor on `passed` plus a ceiling on
`skipped` is simpler and ratchets in the safe direction, but it is blind to
a swap: test A starts skipping while test B stops, both totals hold, and
the guard reports OK. That is principle 12's "the rounded statistic
matched" one layer up. `test_a_swap_that_leaves_the_counts_identical_still_fails`
is the case, and it is the reason for the choice. The accepted cost is that
the baseline is a file people have to regenerate deliberately; the command
is in its own header.

There is deliberately no minimum-test-count constant. A collapse in
collection shows up as every baseline entry going missing, so a count
threshold would be a second hand-maintained expression of something the set
already covers -- and the number it needs is the kind CLAUDE.md principle
24 calls a decision anchor, reproducible only from the script that measured
it. The set is regenerated by committed code with a documented flag.

Skip reasons print on **every** run, passing or failing, via `-rs` and the
checker's own unconditional report. An absence that has to be inferred from
a total is the same as no report at all -- the same argument as the
`session_open` marker in the provenance hooks.

## What CI does not cover

Stated plainly because a green build otherwise reads as more than it is.

**Excluded because they cost money or need a session:**

| test | mechanism | why |
| --- | --- | --- |
| `tests/test_stage2b_gcs_roundtrip.py::test_object_written_from_colab_is_readable_from_outside` | `@pytest.mark.slow`, plus `pytest.skip` when the credential, `mighty-colab` or `google-cloud-storage` is absent | provisions a real Colab runtime and writes to the science bucket; bills while running |
| `tests/test_provenance_live_registration.py` (6 tests) | `@pytest.mark.slow`, plus `skipif` on `shutil.which("claude") is None` and on `.claude/settings.json` being absent | spawns a real headless `claude -p` session |
| `tests/test_stage1b_pilot.py` (4 tests) | `@pytest.mark.slow` | full-grid reproduction, minutes not seconds, and needs `class0_constructions.pkl` |

All eleven are excluded twice over: `make test` and `make stage2b-test`
both pass `-m "not slow"`, and each would skip on a missing capability even
if the marker were dropped.

**Excluded because the data is not in the repository** (36 tests, the
committed baseline). `datasets/*/` is gitignored, as are every cached
`.pkl`/`.npz` under `experiments/*/results/` and the `tarballs/*_handoff/`
artifacts. The Tier-2 tests that verify a construction byte-exactly against
a historical artifact, and the Tier-2 tests that need raw KMNIST, therefore
skip. Every one is named in `tools/ci/ci_skip_baseline.txt` with the reason
pytest gave.

**Not covered at all, by anything CI runs:** GPU numerics (the TF32
question is a property of the device and a CPU pass says nothing about it),
GCS round trips, live hook wiring in a real Claude Code session, and every
scientific result -- no ladder rung, no evolution, no ridge fit, no
classifier. CI runs the fast test suite. It is not a reproduction of any
finding.

## The three tiers, and which half of each is a real Cloud Build feature

### Fast on every push -- entirely native

A push trigger with a branch regex. `make stage2b-test` is about 80 seconds
locally. Nothing custom.

### Medium/slow on idle-or-deadline -- half native, half implemented here

**Cloud Build has no inactivity trigger and no debounce.** There is no
native way to express "X minutes after the last push". What is native: push
triggers, scheduled invocation via Cloud Scheduler, manual triggers,
Pub/Sub triggers, `includedFiles`/`ignoredFiles` globs, and build history
that can be queried read-only.

The rule is implemented in the `decide` step, invoked on a poll:

```
Cloud Scheduler (every N minutes)
  -> Pub/Sub topic
    -> trigger bonsai-ci-poll, _TIER=gated
      -> decide step: skip, or dispatch
        -> trigger bonsai-ci-full, _TIER=full  -> make test
```

`decide` computes seconds since the last commit (`git log -1 --format=%ct`
on the checked-out source) and seconds since the last successful full run
(`gcloud builds list`, read-only). Then: nothing pushed since the last full
run, skip; branch quiet for `_IDLE_MINUTES`, dispatch; `_DEADLINE_MINUTES`
elapsed while the branch stayed busy, dispatch. The deadline is what bounds
the worst case -- `stage2b` moved under in-flight branches at least five
times in one night, and idle alone could starve indefinitely.

**Why the poll dispatches instead of running the suite itself.** A poll
that ran `make test` directly would record `_TIER=gated` in build history
whether it ran the suite or declined, and the two would be
indistinguishable to the very query the deadline depends on. Separating
them gives each build kind one meaning: a `gated` build is a decision, a
`full` build is a run, and "when did a full run last finish" is a one-line
filter. The cost is one extra trigger and the `cloudbuild.builds.editor`
role on the poll's service account.

The build history is the state, deliberately: no marker object, no bucket,
no new write surface. That matters given the section above.

**The cost is the poll, and it is not free.** Every poll is a build, and a
skipping build still pays source fetch and container start -- call it 30-60
seconds. At a 5-minute poll that is roughly 150-300 build-minutes a day
against a 2,500-minute monthly free tier, which the poll alone would exhaust.
A 15-minute poll is roughly 50-100 a day and leaves room for the runs
themselves, at the cost of 15-minute resolution on "idle". Start at 15
minutes and read the actual billing rather than this estimate -- these are
arithmetic from an assumed per-build overhead, not a measurement.

The alternative that avoids paying Cloud Build for the decision is a small
Cloud Run job or Cloud Function invoked by Scheduler, which queries build
history and calls `triggers.run` only when the rule fires. It is cheaper
per poll and it is a second GCP resource to create, own and keep in sync
with a rule that lives in this repository. The build-step version is
recommended first because all of its logic is committed here.

**One part is unverified.** The `decide` step filters build history with
`--filter="status=SUCCESS AND substitutions._TIER=full"`. Whether `gcloud
builds list` supports filtering on a substitution key could not be checked
without a live project. If it does not, the fallbacks in order of
preference are a build `tags` entry filtered with `tags='...'`, or a
timestamp object in a **CI-owned** bucket that is not the science bucket.
Confirm this before relying on the deadline half of the rule; the idle half
does not depend on it.

### Slow on a `mighty-colab` version bump -- detected here, run by a human

This is the tier the design cannot honour automatically, and saying so is
the point.

Detection is native: `includedFiles: ["uv.lock", "pyproject.toml"]` on a
push trigger, plus a step that diffs the pinned `mighty-colab` version. The
discrimination between "the lock changed" and "the `mighty-colab` pin
changed" is not native and is a few lines of shell.

Running the slow tier is what fails. The slow selection is eleven tests.
Six need the `claude` CLI, four need a gitignored `.pkl`, and the one that
would actually exercise a `mighty-colab` regression --
`test_object_written_from_colab_is_readable_from_outside` -- provisions a
billing Colab runtime and writes to the science bucket. Under the two
constraints above it skips; without them CI could spend. **Running that
tier green in CI would be a vacuous pass reproduced at trigger scale**,
which is the specific thing this repository has a document about.

So the honest decomposition: CI detects the bump and **fails with an
instruction** to run `make stage2b-test-roundtrip` by hand. The billing run
stays human-gated. The trigger is still the right one -- a `mighty-colab`
bump breaks exactly the real-infrastructure tests while every fast test
stays green, which is a dependency boundary the code does not own -- but
what fires is a notification, not a run.

**A partial automation exists and is not recommended yet.**
`tests/test_mighty_colab_contract.py` is a genuine contract test against a
stub CLI: no session, no billing. Two of its tests need the real
`mighty-colab` binary on `PATH`, which `--group gpu` would provide without
granting any credential. They skip in CI today (baseline entries
`test_status_of_unknown_session_exits_zero_on_stdout` and
`test_exec_default_timeout_is_short_enough_to_need_overriding`).

The blocker is a contradiction in the record. That first test's docstring
says `status` "reads local session state only -- no network, nothing
provisioned", while `PROJECT_MEMORY.md` Part 4 says `sessions`/`status`
"query the backend/API directly". If the second is right, the test needs
Colab auth and would fail in CI for a reason unrelated to what it tests --
a red build from an unrelated cause is its own kind of noise. Resolving
that contradiction is a science-track question, and until it is resolved
the `gpu` group stays out of CI. Installing it would also put
`google-cloud-storage` in the environment and break the capability-absence
assertion, so the two decisions are coupled.

## Provenance capture hooks in CI

Cloud Build does not run Claude Code, so the hooks in
`.claude/hooks/provenance-capture/` never load and there is nothing to
enable or disable. They are registered in `.claude/settings.json`, which
only a Claude Code session reads.

What CI does verify is the parts that are ordinary code:
`test_provenance_capture.py` (the predicate corpus),
`test_provenance_capture_hook.py` (the hook driven as a real subprocess),
`test_provenance_capture_stats.py`, `test_provenance_probe.py`, and
`test_hook_registrations.py` -- which asserts that both tracks'
registrations survive any edit to the shared settings file. All of those
run in CI and pass.

What CI cannot verify is the **wiring**: that a registration actually fires
in a live session. That needs a session by construction, which is
`test_provenance_live_registration.py`, excluded twice over. Its own README
says to run it after any change to `.claude/settings.json` or the scripts
it names, and that instruction is unchanged by CI existing.

`test_provenance_executor_version.py` skips two parametrised cases in CI
because `mighty-colab` is not installed. Both are in the baseline.

## What a human has to create in GCP

None of this exists. Nothing below has been run.

1. **Connect the repository.** Cloud Build → Repositories → connect
   `danbarua/bonsai-2026` via the GitHub App. The source must be a git
   checkout, not an uploaded archive: the Makefile resolves every path from
   `git rev-parse --show-toplevel`, and the `verify` step fails with that
   explanation if `.git` is absent.

2. **Trigger `bonsai-ci-fast`.** Event: push to branch. Branch regex
   `^(stage2b|main)$`. Config: `cloudbuild.yaml`. Substitution `_TIER=fast`.

3. **Trigger `bonsai-ci-full`.** Event: manual. Source: same repository,
   branch `stage2b`. Config: `cloudbuild.yaml`. Substitution `_TIER=full`.
   This is what a poll dispatches to, and what a human runs on demand.

4. **Trigger `bonsai-ci-poll`.** Event: Pub/Sub message on a new topic
   (`bonsai-ci-poll`). Source: same repository, branch `stage2b`. Config:
   `cloudbuild.yaml`. Substitution `_TIER=gated`. Then a Cloud Scheduler
   job publishing to that topic every 15 minutes.

5. **Trigger `bonsai-ci-deps`** (optional, tier 3). Event: push to the same
   branches, with `includedFiles: ["uv.lock", "pyproject.toml"]`.

6. **Service account.** `cloudbuild.yaml` names no `serviceAccount:`, so
   these roles go on whichever account the triggers run as -- the default
   Cloud Build service account unless a trigger names another. `logging.logWriter`
   for `CLOUD_LOGGING_ONLY`, and `cloudbuild.builds.viewer` for the
   `decide` step's history query. Only the **poll** trigger needs
   `cloudbuild.builds.editor`; giving it a dedicated account keeps that
   role off the two that just run tests.

   **Do not grant any role on `bonsai-2026-stage2b-cache`**, and do not
   mount `bonsai-colab-storage-key.json`. An unattended identity with write
   access to the science bucket is not a trade this project makes; if a
   future build needs to keep reports, give it a separate CI-owned bucket.

   **`cloudbuild.builds.editor` is broader than the job needs, and there is
   no narrower role.** Classic Cloud Build has no per-trigger IAM, so
   "may run trigger X" is not expressible -- the role permits creating
   arbitrary builds in the project. Two things bound the consequence rather
   than eliminate it: `tools/ci/ci_targets.py` constrains what any build of
   this config can invoke, and the `decide` step refuses to dispatch when
   `_FULL_TRIGGER` names its own trigger, which is the one configuration
   that would loop and bill build minutes indefinitely.

   Note this is a spend path the spend guard does **not** see:
   `ci_targets.py` derives from `make` invocations, and
   `gcloud builds triggers run` is outside that derivation. It costs build
   minutes, never GPU or bucket writes.

7. **Notifications.** A break is only useful if it reaches the loop. Cloud
   Build → Pub/Sub `cloud-builds` topic → whatever the loops read.

## Bootstrapping the baseline, and the one measurement nobody has taken

`tools/ci/ci_skip_baseline.txt` was measured on macOS/ARM, Python 3.14.6,
in a checkout with no `datasets/`, no cached artifacts, no `gpu` group and
with `claude` and `mighty-colab` removed from `PATH` -- a capability-for-
capability match for the CI environment. The full tier: 1,046 passed, 36
skipped, 11 deselected, in 2 minutes 41 seconds. The fast tier: 807 passed,
3 skipped, in 80 seconds, and the 3 are the subset of the same 36 belonging
to modules that tier collects.

The skipped set held constant at those 36 across three runs taken while
`stage2b` advanced under this worktree three times and the passing count
moved from 1,000 to 1,046. That stability is the argument for comparing
identities: a count-based guard would have needed rebasing three times in
one afternoon.

**The suite has never run on Linux/x86.** This project has already measured
ARM-vs-x86 divergence in the encoder (93.4% of coordinates identical, max 3
ULP) and has a whole entry on device-dependent numerics. Green on macOS
does not establish green on Cloud Build, and the first build is the
measurement, not a confirmation. If the skip set differs there, the vacuity
check fails on the first run -- which is the correct direction. Read the
failure, decide whether each difference is a genuine capability difference,
and regenerate:

```bash
python3 tools/ci/check_suite_not_vacuous.py --junit junit.xml --write-baseline
```

Regenerating it from a fully-populated developer checkout would record zero
skips and pass every subsequent build vacuously. Regenerate from a CI run's
report, or from a run in a checkout with the same capabilities removed.

## Known soft spots

- **`BRANCH_NAME` on a Pub/Sub-invoked build is unverified.** If it is
  empty, the poll fails loudly every 15 minutes rather than dispatching to
  the wrong branch -- fail-closed, but noisy. The `_BRANCH` substitution is
  the fix, set on the poll trigger.
- **`$TRIGGER_NAME`'s availability is likewise unverified.** The
  self-dispatch refusal reads it and does not fire when it is empty, so an
  absent value costs the guard rather than the build.
- The vacuity checker's module-presence assertion (`--require-baseline-modules`)
  runs only on the full tier. The fast tier's equivalent protection is
  `tests/test_stage2b_gcs_makefile.py`, which asserts `STAGE2B_TEST_FILES`
  is complete against the filesystem.
- `ghcr.io/astral-sh/uv:python3.14-bookworm-slim` is a **mutable** tag
  (verified to exist; the version-pinned equivalent for uv 0.11.29 does
  not). The uv version can move under the build. Pin by digest once the
  first build records one.
- `apt-get install git make` runs on every build, adding fixed setup time.
  A prebuilt image in Artifact Registry removes it and adds an image to
  maintain.
- The `decide` step's build-history filter is unverified (above).
- Build-minute arithmetic in the polling section is arithmetic, not
  measurement.
