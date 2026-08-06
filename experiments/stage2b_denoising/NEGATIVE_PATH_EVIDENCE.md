# Negative-path evidence: what fails, and the test that proves it

The pre-Stage-4 package (`STAGE3_PLAN.md`, item 7) owes a reviewer
evidence for five negative paths. Four are covered by tests that exist.
The fifth — stale-artifact refusal — is not, and is marked pending
throughout rather than assembled from adjacent material.

Every row names a test file, a test function, a line number, and what the
assertion actually checks. Line numbers are as of the commit that added
this document; the function names are the stable handle. Where a guard was
confirmed by deliberately breaking what it watches, the commit that
records the breakage is cited by SHA and what it broke is quoted from that
commit, not paraphrased.

Two conventions this table follows, both from CLAUDE.md principle 21. A
test is cited only where it would fail if the named behaviour regressed —
not where its name merely describes the behaviour. And where coverage is
narrower than the demand, the narrowing is stated in the row rather than
left for the reader to discover.

---

## Demand 1 — a verification mismatch produces a nonzero top-level exit

The mechanism is a sentinel string. The remote script prints a token on
its own success path; the recipe captures `exec`'s combined output and
greps for it. `exec` exiting 0 is not sufficient, because a script can
exit cleanly having never reached its verdict.

| test | asserts |
|---|---|
| `tests/test_mighty_colab_contract.py:259` `test_ladder_missing_sentinel_fails_even_on_a_zero_exit` | Drives the real `stage2b-ladder-stage1` recipe with a stub CLI whose `exec` exits **0** and prints `nothing useful here`. Recipe exit code is 1, and `FAILED: ladder stage 1` is on stdout. |
| `tests/test_mighty_colab_contract.py:170` `test_a_leak_never_masks_the_scientific_verdict` | Same mismatch on `stage2b-verify-gpu` (sentinel `NOTHING_USEFUL`): exit 1, `FAILED: the GPU ridge gate` on stdout. |

The exit code these read is the **recipe's**, not `make`'s. `make` exits 2
for any recipe failure regardless, so `_run_target`
(`tests/test_mighty_colab_contract.py:120`) parses the code out of make's
`*** [target] Error N` line on stderr. Without that, "nonzero" would be
untestable — every failure would look like 2.

Recipe source: `Makefile:378-386` (verify) and `Makefile:480-488`
(ladder). Both compute `if [ $rc -ne 0 ] || ! echo "$out" | grep -q
<SENTINEL>`, print `FAILED:`, and promote `rc` to 1 when `exec` itself
returned 0.

Positive controls, without which a refusal test proves nothing:
`test_healthy_run_exits_zero` (:152) and
`test_ladder_healthy_run_exits_zero` (:242) — correct sentinel, exit 0, no
`LEAK WARNING`.

**Where this is narrower than it looks.**

- On `stage2b-verify-gpu` the mismatch is only ever exercised *together
  with* a failing teardown (`STUB_STOP_RC=7`). The test still has teeth
  for this demand — delete the sentinel grep and `rc` reaches
  `check_teardown` as 0, gets promoted to 7, and the `rc == 1` assertion
  fails — but the isolated mismatch case exists only on the ladder target.
- The `[ $rc -ne 0 ]` half of the disjunct is never exercised. The stub
  CLI declares `STUB_EXEC_RC` (`tests/test_mighty_colab_contract.py:105`)
  and **no test sets it**. So "a zero-exit run that never reported its
  verdict fails" is evidenced; "a nonzero `exec` exit propagates to the
  target's exit code" is visible in the recipe (`|| rc=$$?` … `exit $$rc`)
  and is not evidenced by any test.

## Demand 2 — a missing or corrupted artifact fails rather than being silently accepted

Content verification is on by default on every GCS transfer, in both
directions, on `crc32c` — the digest GCS records for every object it
stores, composed objects included. All tests below inject an adversarial
fake bucket (`tests/test_stage2b_gcs.py:144-225`) that corrupts a specific
way nothing else in the transport can see.

**Corrupt bytes that arrive whole**

| test | asserts |
|---|---|
| `tests/test_stage2b_gcs.py:1597` `test_a_corrupted_download_raises_naming_the_object_and_both_digests` | `ChecksumMismatchError`, and the message carries the object name, the digest the object records, and the digest of the bytes actually delivered. |
| `tests/test_stage2b_gcs.py:1611` `test_a_corrupted_download_leaves_nothing_at_the_destination` | Verification runs on the `.part` sidecar, before the rename: the destination path does not exist afterwards, and neither does the sidecar. |
| `tests/test_stage2b_gcs.py:1628` `test_a_corrupted_download_does_not_overwrite_a_good_local_file` | A good file already at the destination still reads `b"the good copy"` after the failed download. |
| `tests/test_stage2b_gcs.py:1674` `test_a_plain_upload_that_lands_wrong_raises_and_removes_the_object` | A truncating bucket: raises, and the object is **gone** from the bucket — `object_exists` is `False`. |
| `tests/test_stage2b_gcs.py:1688` `test_a_miscomposed_chunked_upload_is_caught_by_the_content_digest` | Every part intact, every part the right length, composed in reverse order: raises, object absent. Size and existence checks cannot see this; the digest is the only thing that can. |
| `tests/test_stage2b_gcs.py:1713` `test_ensure_artifact_verifies_the_artifact_it_downloads` | The resumption path — a fresh runtime pulling what a dead session left — raises, and no local file is left behind. |
| `tests/test_stage2b_gcs.py:1728` `test_ensure_artifact_verifies_what_it_uploads_on_both_routes` | Both upload routes, each given the corruption its own earlier checks cannot detect; neither leaves an object. |

The upload-side deletion is the load-bearing part: `ensure_artifact` reads
an object's existence as proof its step is done
(`stage2b_gcs.py:1299`), so an object known to be wrong must not remain
making that claim.

**A digest that is missing rather than wrong**

`tests/test_stage2b_gcs.py:1641`
`test_an_object_with_no_recorded_digest_is_refused_rather_than_trusted` —
against a bucket that reports no checksum, both `upload_file` and
`download_file` raise `ChecksumMissingError` (matched on the digest field
name), and the download leaves no file. This is what stops "could not be
checked" degrading into "no check".

**A step that produced nothing**

| test | asserts |
|---|---|
| `tests/test_stage2b_gcs.py:804` `test_a_step_whose_producer_writes_nothing_fails_instead_of_recording_completion` | A producer that writes no file raises `FileNotFoundError`, uploads nothing, and leaves `object_exists` `False` — the next run does not read the step as done. |
| `tests/test_stage2b_gcs.py:816` `test_a_producer_that_raises_leaves_no_object_behind` | A producer that raises propagates, and the bucket is empty. |

**Positive control**: `tests/test_stage2b_gcs.py:1703`
`test_a_chunked_upload_that_composes_correctly_still_passes` — the check
passes on a correct composition. Catching the wrong answer proves nothing
if the right one is also rejected.

**Deliberate breakage.** Commit `5f5ff3c2` ("Stage 2B: verify object
content on every GCS transfer") records: *"Mutating the comparison to a
no-op fails the 10 that assert a transfer is refused, and no pre-existing
test; the remaining 9 pin the digest itself, which that mutation does not
touch."* That is the guard being watched fail, and the split between
refusal tests and digest tests is what shows the mutation targeted the
right thing.

**Where this is narrower than it looks.** `verify_content=False` is a
genuine bypass, exercised deliberately by
`tests/test_stage2b_gcs.py:1659`
`test_verification_is_on_by_default_and_can_be_switched_off` (the
corrupted file lands, and the test asserts it really is corrupt). It is
opt-out rather than opt-in and visible at the call site, and no Stage 2B
driver passes it — but the guarantee is "on unless a call site asks
otherwise", not "unconditional".

## Demand 3 — an inner remote failure survives teardown

This is the **failing**-teardown case. The run's own verdict is 1, `stop`
exits 7, and the only thing stopping 7 from replacing 1 is the `if [ $rc
-eq 0 ]` guard inside `check_teardown` (`Makefile:101`).

| test | asserts |
|---|---|
| `tests/test_mighty_colab_contract.py:170` `test_a_leak_never_masks_the_scientific_verdict` | Bad sentinel **and** `stop` exiting 7. Exit code is **1**, not 7 — the science's failure stays the headline. `FAILED: the GPU ridge gate` and `LEAK WARNING` are **both** on stdout: the leak is reported rather than swallowed, and reporting it does not cost the verdict. |

The complementary direction — teardown failing while the science
succeeded — is `test_teardown_failure_fails_an_otherwise_successful_target`
(:158) and its ladder twin (:248): exit code 7 (`stop`'s own code, not a
generic 1), `LEAK WARNING` present, and the `FAILED:` line explicitly
**absent**, so a billing leak is never misreported as a scientific
failure.

**Where this is narrower than it looks.** `test_a_leak_never_masks_the_scientific_verdict`
is the **only** test of this demand, and it runs on `stage2b-verify-gpu`
only. There is no ladder-target test combining an inner failure with a
failing teardown; the ladder's teardown-failure test (:248) pairs a failed
teardown with a *successful* run, which is the other case.

## Demand 4 — teardown success cannot overwrite a substantive failure verdict

This is the **succeeding**-teardown case, and the mechanism is stronger
than the outcome: `check_teardown`'s entire body is guarded by `[ $src -ne
0 ]` (`Makefile:100-102`), so on a successful teardown `rc` is not merely
preserved — the code path that could touch it is never entered.

| test | asserts |
|---|---|
| `tests/test_mighty_colab_contract.py:259` `test_ladder_missing_sentinel_fails_even_on_a_zero_exit` | `STUB_STOP_RC` unset, so teardown exits 0. The verdict-derived exit code 1 survives to the top level. |
| `tests/test_mighty_colab_contract.py:278` `test_ladder_refuses_a_dirty_tree_before_provisioning` | Exit 1 with `REFUSING` and `dirty` on stdout, and `stub] created` **absent** — refused before any session was provisioned, so there is no teardown to overwrite anything. |
| `tests/test_mighty_colab_contract.py:288` `test_ladder_refuses_an_unpushed_head_before_provisioning` | Exit 1 with `REFUSING` and `not on any remote`, `stub] created` absent. The runtime fetches one pinned commit, so an unpushed HEAD would run code that is not the code under test. |

The pre-flight refusals are the same demand one step earlier: the two
failure modes they catch would both surface as *scientific* results rather
than mistakes, and neither reaches the teardown path at all.

**Supporting — a teardown signal must not fabricate a verdict either.**
`test_ladder_absent_session_is_not_treated_as_a_leak` (:268) and
`test_a_distinct_absent_code_can_be_declared_without_rewriting_recipes`
(:296) both give `stop` a nonzero code that `STOP_ABSENT_RC` declares to
mean "already absent", and assert exit 0 with no `LEAK WARNING`. "Already
gone" is the goal; only "could not stop" costs money.

**Deliberate breakage (demands 1, 3 and 4).** Commit `e6398e09` ("Add the
Stage 2B ladder stage-1 driver and the targets that run it", 2026-08-05)
records an eleven-item sweep: *"the sentinel grep, the leak check, both
pre-flight refusals, an &&-chained teardown, a recipe that omits
--timeout, a local recipe using the remote env form, an ENV_ the recipe
never sets, a "test"-named staged object, a ragged evolution chunk, and a
hoisted cloud import."* Four of the eleven bear on these demands — the
sentinel grep (demand 1), the leak check (demands 3 and 4), and both
pre-flight refusals (demand 4). The other seven concern unrelated guards
and are not evidence here.

Commit `a63dbd87` ("Add the Stage 2B ladder stage-2 driver", 2026-08-06)
records the same discipline for that rung's guards, and that *"two of the
new tests were themselves found vacuous on first breakage (a whole-file
substring search that matched an unrelated, correct call site elsewhere in
the same file) and rewritten to target the specific call site"* — the
reason breaking the guard is the check, and reading it is not.

## Demand 5 — stale artifacts are refused — **PENDING, NOT COVERED**

No test evidences this, because the mechanism does not exist yet. It is
item 1 of `STAGE3_PLAN.md`'s sequencing: a provenance fingerprint on GCS
artifacts, carrying the static ∪ runtime import closure, with per-artifact
field selection.

Three facts bound what coverage will have to reach, all verified from the
code:

- `ensure_artifact`'s trust point is one line — `stage2b_gcs.py:1299`,
  `if not force and object_exists(...)`. `force=True` bypasses it
  entirely.
- `run_ladder_stage1.py:342` calls `download_file` **directly**, outside
  `ensure_artifact`. A check placed only in `ensure_artifact` does not
  cover that path.
- Per-artifact field selection is required, not optional: stage 2
  deliberately consumes stage 1's `topologies.npz` and the KMNIST IDX
  files staged under stage 1, both written under a different commit. A
  commit-keyed fingerprint applied uniformly would refuse legitimate
  cross-stage reuse.

**Adjacent, and explicitly not coverage.** `stage2b_gcs.py` already
discards stale **transfer state** — a checkpoint from a differently sized
file (`tests/test_stage2b_gcs.py:1036`), from a same-sized but rewritten
file (:1056), naming a different object (:1076), written under a different
chunk size (:1093), corrupt on disk (:1107); a recorded part that vanished
from the bucket (:1116) or whose remote size disagrees (:1134); and parts
left by a previous larger upload that must not be composed in (:1196).
Every one of those refuses *in-flight state left by a previous attempt
against the same local file and object*. None of them refuses a
**completed artifact whose provenance no longer matches its consumer**,
which is what this demand asks for. Counting them here would make the
table read as five covered demands.

---

## Coverage of the recipe surface

The behavioural tests above drive **two** of the seven `Makefile` recipes
that invoke `mighty-colab exec`: `stage2b-verify-gpu` and
`stage2b-ladder-stage1`. The other five — `stage2b-verify-cnn-gpu`,
`stage2b-ladder-stage2`, and the three Stage 2A GPU recipes
(`Makefile:162`, `:185`, `:254`) — are covered only by the three static
checks that parse the Makefile:

- `test_makefile_has_gpu_recipes_to_check` (:58) — fails if the parser
  finds fewer than three `exec` recipes, so the two checks below cannot
  pass vacuously on an empty set.
- `test_every_exec_passes_an_explicit_timeout` (:68) — every `exec` line
  carries `--timeout`; the 30-second default kills any driver that
  computes quietly.
- `test_every_session_creating_recipe_tears_down_unconditionally` (:85) —
  every such recipe contains a `stop`, and no `stop` line is chained onto
  a previous command's success with `&&`.

Two consequences worth stating plainly. First, `_run_target` takes the
target as a parameter precisely so recipe-shape behaviour can be checked
on more than one target, and today two are checked. Second, the sentinel
mechanism is a property of the four Stage 2B targets, not of GPU targets
generally: the three Stage 2A recipes have no sentinel grep at all and
read `exec`'s exit status only.

## Summary

| # | demand | status | primary evidence |
|---|---|---|---|
| 1 | verification mismatch → nonzero top-level exit | covered | `test_ladder_missing_sentinel_fails_even_on_a_zero_exit`; nonzero-`exec` propagation untested |
| 2 | missing or corrupted artifact fails, not silently accepted | covered | the transfer-refusal tests plus `test_an_object_with_no_recorded_digest_is_refused_rather_than_trusted`; `verify_content=False` is a visible opt-out |
| 3 | inner remote failure survives teardown | covered | `test_a_leak_never_masks_the_scientific_verdict` — sole test, one target |
| 4 | teardown success cannot overwrite a failure verdict | covered | `test_ladder_missing_sentinel_fails_even_on_a_zero_exit` plus both pre-flight refusals |
| 5 | stale artifacts are refused | **pending** | none — mechanism under construction |
