Simplified Technical English version of `experiments/stage2b_denoising/NEGATIVE_PATH_EVIDENCE.md`.

# Negative-path evidence: what fails, and the test that proves it

Negative-path evidence is proof, in the form of a specific automatic
test, that a safety check actually stops a specific kind of failure —
not just proof that the check exists in the code. Before Stage 4 could
run, a reviewer required evidence for five negative paths, listed in
`STAGE3_PLAN.md`, item 7. All five are now covered, start to finish. The
fifth — refusing a stale artifact — was the last to be covered. Its
underlying mechanism was built first, and only actually put into use
later. It is recorded as covered here only now that `ensure_artifact`'s
trust point actually routes every read through that mechanism.

Every row below names a test file, a test function, and what that test
actually checks. **The function name is the citation.** A separate test,
`tests/test_stage2b_negative_path_evidence.py`, checks that every
function named in this document still exists, so a rename or a deletion
makes the test suite fail, instead of quietly leaving a broken pointer.
Line numbers, where given, are as of the commit that wrote that line, and
they exist only to help a reader navigate, not as a guarantee they stay
correct forever. Where a safety check was confirmed by deliberately
breaking the thing it watches, what was broken and what failed is
recorded either directly in this document, or cited by a commit's unique
ID, never paraphrased.

This table follows two rules, both from working rule 21 in `CLAUDE.md`.
First, a test is cited only where it would actually fail if the
behavior it names stopped working — not merely where its name describes
that behavior. Second, wherever the actual coverage is narrower than
what was demanded, that gap is stated directly in the row, rather than
left for the reader to discover on their own.

---

## Demand 1 — a verification mismatch must produce a nonzero exit code at the top level

The mechanism here is a sentinel — a special marker word a script prints
when it truly finishes successfully. The remote script prints this token
on its own success path, and the Makefile recipe captures the combined
output of the `exec` command and searches for that token. `exec` exiting
with code 0 is not enough on its own, because a script can exit cleanly
without ever reaching the point where it prints its real verdict.

| test | checks |
|---|---|
| `tests/test_mighty_colab_contract.py:259` `test_ladder_missing_sentinel_fails_even_on_a_zero_exit` | Runs the real `stage2b-ladder-stage1` recipe against a stand-in command line whose `exec` step exits with code **0** and prints `nothing useful here`. The recipe's own exit code is 1, and the text `FAILED: ladder stage 1` appears on the standard output. |
| `tests/test_mighty_colab_contract.py:170` `test_a_leak_never_masks_the_scientific_verdict` | The same mismatch, but on `stage2b-verify-gpu`, using the sentinel `NOTHING_USEFUL`: exit code 1, and `FAILED: the GPU ridge gate` on standard output. |

The exit code these tests read is the **recipe's own** exit code, not
`make`'s. `make` itself always exits with code 2 for any recipe failure,
so a helper function, `_run_target`
(`tests/test_mighty_colab_contract.py:120`), reads the real code out of
`make`'s own `*** [target] Error N` line on its error output. Without
this step, "the exit code is nonzero" would be impossible to test
meaningfully, since every failure would otherwise look identical, as
code 2.

The recipe source code is at `Makefile:378-386` (the verify recipe) and
`Makefile:480-488` (the ladder recipe). Both compute
`if [ $rc -ne 0 ] || ! echo "$out" | grep -q <SENTINEL>`, print
`FAILED:`, and set `rc` to 1 whenever `exec` itself already returned 0.

Without a positive control, a refusal test proves nothing on its own.
The positive controls here are `test_healthy_run_exits_zero` (line 152)
and `test_ladder_healthy_run_exits_zero` (line 242): a correct sentinel
produces exit code 0, with no `LEAK WARNING` message.

The opposite case — `exec` exiting with a nonzero code while the
sentinel IS present, meaning a driver that printed its real verdict and
then died anyway — is the other half of the check
(`[ $rc -ne 0 ]`), and it is covered separately:

| test | checks |
|---|---|
| `tests/test_mighty_colab_contract.py` `test_a_nonzero_exec_fails_the_target_even_when_the_sentinel_is_present` | Sets `STUB_EXEC_RC=5` with a correct sentinel, on `stage2b-verify-gpu`: the recipe's exit code is **5**, not 0 and not a generic 1, and the text `exec rc=5` appears on standard output, so the failure can be diagnosed. |
| `tests/test_mighty_colab_contract.py` `test_ladder_nonzero_exec_fails_the_target_even_when_the_sentinel_is_present` | The same check, on `stage2b-ladder-stage1` — the target that spends real money. |

**Deliberate breakage.** The team broke both halves of this mechanism
separately, and observed the exact expected failure each time. Removing
`[ $$rc -ne 0 ] ||` from the verify recipe's condition leaves the exit
code correctly at 5, but drops the `FAILED:` message — the test fails on
its message check. Replacing `|| rc=$$?` with `|| true` leaves the
message intact but makes the target exit with code **0** instead — the
test fails on its exit-code check. Neither break is caught by the
other's check, which is exactly why both checks exist.

**Where this coverage is narrower than it looks.**

- On `stage2b-verify-gpu`, the missing-sentinel case is only ever tested
  *together with* a failing teardown step (`STUB_STOP_RC=7`). This test
  still has real teeth for this specific demand — deleting the sentinel
  search causes `rc` to reach `check_teardown` as 0, which then gets
  promoted to 7, and the check that expects `rc == 1` fails — but the
  case of a missing sentinel entirely on its own is tested only on the
  ladder target.

## Demand 2 — a missing or corrupted artifact must fail, never be silently accepted

Content verification runs on by default, on every GCS file transfer, in
both directions, using `crc32c` — the same checksum GCS itself records
for every object it stores, including composed objects. Every test below
uses a fake, adversarial storage bucket
(`tests/test_stage2b_gcs.py:144-225`) that corrupts data in a way
nothing else in the transfer system can otherwise see.

**Corrupt bytes that arrive as a whole file**

| test | checks |
|---|---|
| `tests/test_stage2b_gcs.py:1597` `test_a_corrupted_download_raises_naming_the_object_and_both_digests` | Raises a `ChecksumMismatchError`, and the error message includes the object's name, the checksum the object is supposed to have, and the checksum of the bytes actually received. |
| `tests/test_stage2b_gcs.py:1611` `test_a_corrupted_download_leaves_nothing_at_the_destination` | Verification runs on the temporary `.part` file, before it is renamed into place: the final destination path does not exist afterward, and neither does the temporary file. |
| `tests/test_stage2b_gcs.py:1628` `test_a_corrupted_download_does_not_overwrite_a_good_local_file` | A good file already at the destination still reads `b"the good copy"` after the failed download attempt. |
| `tests/test_stage2b_gcs.py:1674` `test_a_plain_upload_that_lands_wrong_raises_and_removes_the_object` | Using a bucket that truncates data: raises an error, and the object is **gone** from the bucket afterward (`object_exists` returns `False`). |
| `tests/test_stage2b_gcs.py:1688` `test_a_miscomposed_chunked_upload_is_caught_by_the_content_digest` | Every chunk of a chunked upload is intact and the right length, but composed back together in the wrong order: this raises an error, and the object is absent afterward. Checks on size and existence cannot see this problem; only the checksum can. |
| `tests/test_stage2b_gcs.py:1713` `test_ensure_artifact_verifies_the_artifact_it_downloads` | Tests the resume path — a fresh session picking up what a dead session left behind — and confirms it raises an error, leaving no local file behind. |
| `tests/test_stage2b_gcs.py:1728` `test_ensure_artifact_verifies_what_it_uploads_on_both_routes` | Tests both upload routes, each given a kind of corruption its own earlier checks cannot detect; neither route leaves a bad object behind. |

The deletion on the upload side is the important part: `ensure_artifact`
treats an object's existence as proof its step is already done
(`stage2b_gcs.py:1299`), so a known-bad object must not remain in place,
making that false claim.

**A checksum that is missing, rather than wrong**

`tests/test_stage2b_gcs.py:1641`
`test_an_object_with_no_recorded_digest_is_refused_rather_than_trusted`
uses a bucket that reports no checksum at all. Both `upload_file` and
`download_file` raise a `ChecksumMissingError` (matched on the field
name for the digest), and the download leaves no file behind. This is
what stops "this could not be checked" from quietly becoming "this
passed the check."

**A step that produced nothing**

| test | checks |
|---|---|
| `tests/test_stage2b_gcs.py:804` `test_a_step_whose_producer_writes_nothing_fails_instead_of_recording_completion` | A step that writes no output file raises a `FileNotFoundError`, uploads nothing, and leaves `object_exists` as `False` — so the next run does not mistakenly treat this step as already done. |
| `tests/test_stage2b_gcs.py:816` `test_a_producer_that_raises_leaves_no_object_behind` | A step that raises an error lets that error propagate, and the bucket stays empty. |

**Positive control**: `tests/test_stage2b_gcs.py:1703`
`test_a_chunked_upload_that_composes_correctly_still_passes` checks that
a correctly composed upload still passes. Catching the wrong answer
proves nothing if the right answer also gets rejected.

**Deliberate breakage.** Commit `5f5ff3c2` ("Stage 2B: verify object
content on every GCS transfer") records: *"Mutating the comparison to a
no-op fails the 10 tests that assert a transfer is refused, and no
pre-existing test; the remaining 9 tests pin the checksum itself, which
that mutation does not touch."* This is the record of the guard being
watched fail. The split between refusal tests and checksum tests shows
that this deliberate break really did target the right thing.

**Where this coverage is narrower than it looks.**
`verify_content=False` is a genuine way to bypass this check, and the
team exercises this deliberately with
`tests/test_stage2b_gcs.py:1659`
`test_verification_is_on_by_default_and_can_be_switched_off` (the
corrupted file is allowed through, and the test confirms it really is
corrupt). This bypass is opt-out rather than opt-in, and it is visible
at the place in the code where it is used. No Stage 2B driver ever
passes it. But the actual guarantee is "verification is on unless a
call site specifically asks otherwise," not "verification can never be
turned off."

## Demand 3 — an inner remote failure must survive teardown, not get overwritten by it

This is the **failing**-teardown case. Here, the run's own verdict is 1
(failure), and `stop` (the teardown command) exits with code 7. The only
thing stopping 7 from replacing 1 as the reported result is the
`if [ $rc -eq 0 ]` condition inside `check_teardown` (`Makefile:101`).

| test | checks |
|---|---|
| `tests/test_mighty_colab_contract.py:170` `test_a_leak_never_masks_the_scientific_verdict` | A bad sentinel **together with** `stop` exiting with code 7. The final exit code is **1**, not 7 — so the science's own failure stays the headline. `FAILED: the GPU ridge gate` and `LEAK WARNING` are **both** shown on standard output: the resource leak is reported rather than hidden, and reporting it does not overwrite the real verdict. |

The complementary case — teardown failing while the science succeeded
— is covered by `test_teardown_failure_fails_an_otherwise_successful_target`
(line 158) and its ladder counterpart (line 248): the exit code is 7
(teardown's own code, not a generic 1), `LEAK WARNING` is present, and
the `FAILED:` line is explicitly **absent** — so a billing leak is never
mistakenly reported as a scientific failure.

**Where this coverage is narrower than it looks.**
`test_a_leak_never_masks_the_scientific_verdict` is the **only** test of
this specific demand, and it only runs against `stage2b-verify-gpu`.
There is no test on any ladder target that combines an inner failure
with a failing teardown; the ladder's own teardown-failure test (line
248) pairs a failed teardown with a *successful* run instead, which is
the other, separate case.

## Demand 4 — a successful teardown must not overwrite a real failure verdict

This is the **succeeding**-teardown case, and here the safety mechanism
is stronger than just the outcome it produces: `check_teardown`'s entire
body is guarded by `[ $src -ne 0 ]` (`Makefile:100-102`), so on a
successful teardown, the run's verdict is not merely preserved — the
code path that could ever touch it is never even entered.

| test | checks |
|---|---|
| `tests/test_mighty_colab_contract.py:259` `test_ladder_missing_sentinel_fails_even_on_a_zero_exit` | `STUB_STOP_RC` is left unset, so teardown exits with code 0. The exit code of 1, coming from the run's own verdict, survives all the way to the top level. |
| `tests/test_mighty_colab_contract.py` `test_ladder_refuses_a_dirty_source_closure_before_provisioning` | Exit code 1, with `REFUSING` and `closure` shown on standard output, and `stub] created` **absent** — the run is refused before any cloud session was even provisioned, so there is nothing for teardown to overwrite. This pre-flight check works by examining the driver's own source-file list, not by asking whether the whole code repository is tidy. |
| `tests/test_mighty_colab_contract.py` `test_ladder_proceeds_when_only_unrelated_files_are_uncommitted` | This is the positive control for the row above, and the case that led the team to narrow the check: a clean source-file list, inside an otherwise untidy code repository, must still be allowed to reach provisioning. A refusal test alone proves only that the target *can* refuse, not that it refuses correctly. |
| `tests/test_mighty_colab_contract.py:288` `test_ladder_refuses_an_unpushed_head_before_provisioning` | Exit code 1, with `REFUSING` and `not on any remote` shown on standard output, and `stub] created` absent. This check exists because the cloud runtime fetches one specific pinned commit, so an unpushed local commit would let the cloud session run code that is not actually the code being tested. |

These pre-flight refusals cover the same underlying demand, one step
earlier: both of the failure modes they catch would otherwise show up
looking like a *scientific* result, rather than looking like a mistake
in how the run was launched, and neither one ever reaches the teardown
step at all.

**The first refusal check was narrowed, and the narrowing has its own
safety checks.** It used to check the entire code tree, using
`git status --porcelain`. But uncommitted work outside a driver's own
list of source files can never actually reach the computation, since the
cloud runtime only ever executes one pinned commit — so the old, coarse
check refused perfectly correct runs, while a genuinely dirty source
file was reported as just one line among many others. Two separate
static checks now keep this narrower replacement check honest, and both
are derived automatically rather than kept as a hand-written list:
`test_no_gpu_target_still_gates_on_whole_tree_porcelain` (this check
confirms the old, coarse check has not silently come back, because if it
had, it would run first and make the new, narrower check dead code that
never actually runs) and
`test_every_repo_fetching_gpu_target_runs_the_closure_check` (this check
confirms that any recipe that fetches a pinned commit for the cloud
runtime also asks whether that commit actually contains the driver's own
source files — so any new ladder target automatically gets this
coverage on the day it is written). The team confirmed both of these
checks by deliberately breaking what they watch: restoring the old,
coarse check inside stage 2's recipe makes both tests fail, and the
failure message correctly names stage 2 as the problem.

**Supporting check — a teardown signal must not invent a verdict
either.** `test_ladder_absent_session_is_not_treated_as_a_leak` (line
268) and
`test_a_distinct_absent_code_can_be_declared_without_rewriting_recipes`
(line 296) both give the teardown step a nonzero exit code that a
setting called `STOP_ABSENT_RC` declares to mean "the session was
already gone," and both confirm the final exit code is 0, with no `LEAK
WARNING` message. "Already gone" is the goal state here; only "could not
stop the session" actually costs money.

**Deliberate breakage (covering demands 1, 3, and 4 together).** Commit
`e6398e09` ("Add the Stage 2B ladder stage-1 driver and the targets that
run it", 2026-08-05) records an eleven-item sweep of deliberate breaks:
*"the sentinel grep, the leak check, both pre-flight refusals, an
&&-chained teardown, a recipe that omits --timeout, a local recipe using
the remote env form, an ENV_ the recipe never sets, a "test"-named
staged object, a ragged evolution chunk, and a hoisted cloud import."*
Four of these eleven relate to the demands in this document — the
sentinel search (demand 1), the leak check (demands 3 and 4), and both
pre-flight refusals (demand 4). The other seven relate to unrelated
safety checks, and are not evidence for this document.

Commit `a63dbd87` ("Add the Stage 2B ladder stage-2 driver", 2026-08-06)
records the same discipline applied to that rung's own checks, and notes
that *"two of the new tests were themselves found vacuous on first
breakage (a whole-file substring search that matched an unrelated,
correct call site elsewhere in the same file) and rewritten to target
the specific call site"* — the reason breaking the guard is the real
check, and simply reading it is not enough.

## Demand 5 — stale artifacts must be refused — **COVERED**

The mechanism here — `stage2b_fingerprint.py`, plus the sidecar-manifest
system in `stage2b_gcs.py` — is now fully **adopted**:
`ensure_artifact`'s trust point is itself a `consume_validated` call, so
a resumed step is validated automatically, by construction, rather than
by a driver's author having to remember to add the check. There is only
one route by which bytes travel from GCS into a consumer, and that route
always checks them.

**The wiring, and the tests that hold it together**

| test | checks |
|---|---|
| `tests/test_stage2b_gcs.py` `test_ensure_artifact_refuses_to_resume_from_an_object_with_no_manifest` | The default behavior. An object that exists but carries no manifest file halts the step — and, importantly, the producing step is **not** silently re-run instead. |
| `tests/test_stage2b_gcs.py` `test_ensure_artifact_refuses_when_the_manifest_disagrees_with_the_payload` | Confirms that existence plus a manifest is still not enough on its own: if the object is overwritten behind the manifest's back, exactly as a half-finished regeneration might leave it, the resume attempt raises an error. |
| `tests/test_stage2b_gcs.py` `test_ensure_artifact_refuses_a_fingerprint_the_consumer_did_not_expect` | A consumer that declares what it expects to have produced its input is refused if the recorded producer disagrees, and the error names exactly which field disagreed. |
| `tests/test_stage2b_gcs.py` `test_a_forced_overwrite_never_leaves_a_manifest_describing_the_old_bytes` | Closes a hole in the `force=True` option. A stale sidecar manifest is worse than having none at all — the next read would raise a mismatch error that looks like corruption, rather than looking like a deliberate regeneration — so a forced overwrite either republishes a fresh manifest, or removes the old one, and never leaves a mismatched one behind. |
| `tests/test_stage2b_gcs.py` `test_the_manifest_is_published_only_after_the_payload_verifies` | Confirms the write order: a failed upload leaves neither the payload nor the manifest, so any manifest that does exist always describes a complete, correct object. |
| `tests/test_stage2b_gcs_makefile.py` `test_no_stage2b_script_downloads_around_the_validated_consume_path` | **Derived by reading the code's syntax tree, not kept as a hand-written list**: confirms no Stage 2B script calls `download_file` directly. Any exemption from this rule is named with its reason, and separately checked to confirm it is still actually needed. Confirmed by deliberately breaking it: reintroducing a raw call in `run_ladder_stage2.py` makes this check fail and names that exact file. |

**Positive controls** — a contract that refuses everything is not a
useful contract, so the team also confirms it accepts correct cases:
`test_a_forced_overwrite_with_a_fingerprint_republishes_rather_than_removing`
(a forced regeneration, followed by a resume from the new manifest, with
no complaint), `test_ensure_artifact_permits_cross_stage_reuse_under_content_only`,
and `test_the_pre_contract_optout_is_what_keeps_a_completed_rung_rerunnable`.

**The opt-out, and why it is not a hole in the check.** Ladder stages 1
and 2 wrote every one of their artifacts before this manifest system
existed, so retrofitting manifests onto them now would fabricate
provenance that never actually existed, rather than accurately
recording it. Three call sites pass `require_manifest=False`, each with
an inline comment naming the exact rung it belongs to: the two
`stage_kmnist` staging reads, and stage 2's cross-rung corruption spot-
check. What this option does **not** mean: when a manifest is actually
present, it is still checked, no matter what this option says — the
opt-out only relaxes the requirement that a manifest must exist, never
the check itself. The four staged KMNIST objects now carry manifests
(created with `make stage2b-publish-input-manifests`, sidecar files
only, each one checked against its own local copy first), so those two
reads now validate correctly in practice, while staying able to re-run
against older history.

There is no general-purpose exemption list, and there will not be one.
A future `ManifestMissingError`, raised against pre-contract history, is
the **correct** behavior: it means new code has reached backward into
old data, and the right fix is to decide deliberately, at that exact
call site, rather than widening a central policy where nobody would
notice the change.

**A completed artifact whose provenance no longer matches its consumer**

| test | checks |
|---|---|
| `tests/test_stage2b_gcs.py` `test_a_payload_with_no_manifest_is_refused` | The core of the whole contract: an object that exists but carries no manifest raises a `ManifestMissingError`. Existence alone proves nothing. |
| `tests/test_stage2b_gcs.py` `test_a_payload_edited_after_publication_is_refused` | Bytes changed after publication raise a `ManifestMismatchError`, based on the recorded payload checksum — the object still exists, but that is not enough on its own. |
| `tests/test_stage2b_gcs.py` `test_a_fingerprint_from_a_different_config_is_refused` | A consumer whose expected fingerprint differs, specifically in its `config_digest` value, raises a `FingerprintMismatch` error, naming the exact field that disagreed. |
| `tests/test_stage2b_fingerprint.py` `test_strict_policy_catches_a_changed_digest` | Tested separately for each participating field: any single changed checksum is caught, under the STRICT policy. |
| `tests/test_stage2b_fingerprint.py` `test_strict_policy_catches_a_changed_commit` | A different producing code commit is refused under the STRICT policy. |
| `tests/test_stage2b_fingerprint.py` `test_content_only_still_refuses_a_different_fingerprint_format` | Confirms the relaxed policy only relaxes the requirement about who produced the data — it is not an escape route from the contract's own version rule. |
| `tests/test_stage2b_fingerprint.py` `test_a_non_mapping_recorded_fingerprint_is_reported_not_crashed` | A malformed recorded fingerprint produces a clean refusal, not a program crash — otherwise "provenance that cannot be read" would quietly become "no check at all." |

**Staleness that a whole-file checksum alone cannot see**

| test | checks |
|---|---|
| `tests/test_stage2b_fingerprint.py` `test_revalidation_refuses_a_repo_module_absent_from_the_manifest` | A source file imported *during* the run, but absent from the pre-run source list, is caught by the check that re-validates after the run finishes. This is a case a check that only runs before the run starts cannot see. |
| `tests/test_stage2b_fingerprint.py` `test_revalidation_refuses_a_source_file_that_changed_during_execution` | A participating source file edited while the run was still executing is refused at this post-run re-validation check. |
| `tests/test_stage2b_fingerprint.py` `test_array_manifest_detects_a_single_changed_value` | One changed value, inside one array, inside one file, is caught by the per-array checksum. |
| `tests/test_stage2b_fingerprint.py` `test_array_manifest_distinguishes_nan_bit_patterns` | Two "not a number" (NaN) values that compare as unequal under `==` still differ in their exact bit pattern, and the manifest system sees this difference. |

**Positive controls** — a refusal test alone proves nothing without
these: `test_a_published_artifact_validates_against_its_own_manifest`,
`test_cross_stage_reuse_under_content_only_is_permitted` (stage 2 and
stage 3 deliberately reuse stage 1's topologies; a check keyed uniformly
by commit would incorrectly break this correct behavior), and
`test_the_no_manifest_optout_is_explicit_and_permits_pre_contract_artifacts`
(ladder stages 1 and 2 predate this whole manifest system).

**Deliberate breakage.** Seven fingerprint checks and four manifest
checks were each broken deliberately, and the team observed the specific
expected failure each time. The fourth manifest break is worth recording
specially: it triggered **no failure at all**, because the whole-file
payload checksum caught the corruption first. This exposed
`test_the_per_array_manifest_is_what_survives_a_container_rewrite` as a
test that could never actually fail correctly, in its original form. It
is now rewritten to test the cross-file regeneration case the per-array
manifest actually exists to serve, a case where the whole-file checksum
is useless, because the `np.savez` file format embeds a timestamp inside
the zip archive itself.

**Where this coverage is still narrower than it looks.** This contract
only governs Stage 2B's own data-transport code. Nothing here stops a
future script from importing `google.cloud.storage` directly and never
touching this module at all — the syntax-tree check described above
only scans files under `experiments/stage2b_denoising/*.py` for
`download_file`, so a genuinely separate storage client would be
invisible to it. `tests/test_stage2b_gcs_makefile.py`'s live check
against the real bucket does look for direct `google.cloud` imports,
which narrows this gap without fully closing it.

**Related, but explicitly not coverage for this demand.**
`stage2b_gcs.py` already discards stale **in-progress transfer state**
in several cases: a resume checkpoint from a file of a different size
(`tests/test_stage2b_gcs.py:1036`), from a same-sized but rewritten file
(line 1056), one naming a different object (line 1076), one written
under a different chunk size (line 1093), one that is corrupt on disk
(line 1107); a recorded chunk that has vanished from the bucket
(line 1116), or whose remote size disagrees with the record (line 1134);
and chunks left over from a previous, larger upload attempt, which must
not be composed into the new file (line 1196). Every one of these cases
refuses to trust *leftover, in-progress state from a previous attempt on
the same local file and the same object*. None of them refuse a
**completed artifact whose provenance no longer matches its consumer**,
which is what this demand actually asks for, and which the manifest
tests above now cover. Counting these older checks here would have made
the table look like it covered five demands, while the actual mechanism
for this one did not yet exist.

---

## How much of the recipe surface this covers

The behavioral tests above exercise **two** of the seven `Makefile`
recipes that call `mighty-colab exec`: `stage2b-verify-gpu` and
`stage2b-ladder-stage1`. The other five — `stage2b-verify-cnn-gpu`,
`stage2b-ladder-stage2`, and the three Stage 2A GPU recipes
(`Makefile:162`, `:185`, `:254`) — are covered only by three static
checks that read and parse the Makefile itself:

- `test_makefile_has_gpu_recipes_to_check` (line 58) — fails if the
  parser finds fewer than three `exec` recipes, so the two checks below
  it cannot accidentally pass on an empty set of recipes.
- `test_every_exec_passes_an_explicit_timeout` (line 68) — checks that
  every `exec` line carries a `--timeout` value; without it, the
  30-second default would kill any driver that is still computing
  quietly.
- `test_every_session_creating_recipe_tears_down_unconditionally`
  (line 85) — checks that every recipe which creates a session also
  contains a `stop` command, and that no `stop` command is chained with
  `&&` onto a previous command's success, which would skip it on
  failure.

Two consequences worth stating plainly. First, the `_run_target` helper
takes the target name as a parameter specifically so recipe-shape
behavior can be checked on more than one target, and today two targets
are actually checked this way. Second, the sentinel mechanism itself is
a property of the four Stage 2B targets specifically, not of GPU targets
in general: the three Stage 2A recipes have no sentinel search at all,
and read only `exec`'s own exit status.

## Summary

| # | demand | status | main evidence |
|---|---|---|---|
| 1 | a verification mismatch produces a nonzero top-level exit code | covered | `test_ladder_missing_sentinel_fails_even_on_a_zero_exit` for a clean exit with no real verdict; `test_a_nonzero_exec_fails_the_target_even_when_the_sentinel_is_present` and its ladder counterpart for the opposite case |
| 2 | a missing or corrupted artifact fails, rather than being silently accepted | covered | the transfer-refusal tests, plus `test_an_object_with_no_recorded_digest_is_refused_rather_than_trusted`; `verify_content=False` exists as a visible, deliberate opt-out |
| 3 | an inner remote failure survives teardown | covered | `test_a_leak_never_masks_the_scientific_verdict` — the sole test for this case, on one target |
| 4 | a successful teardown cannot overwrite a failure verdict | covered | `test_ladder_missing_sentinel_fails_even_on_a_zero_exit`, plus both pre-flight refusal tests |
| 5 | stale artifacts are refused | covered | `test_ensure_artifact_refuses_to_resume_from_an_object_with_no_manifest` and the rest of the wiring set, plus `test_no_stage2b_script_downloads_around_the_validated_consume_path` (derived automatically from the code's syntax tree) |
